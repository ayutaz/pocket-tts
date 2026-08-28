"""Turn MoeSpeech's per-character zips into an aligned training manifest.

    python -m training.scripts.prepare_moespeech --hours 124 --out data/ja

MoeSpeech is 621 hours of 44.1 kHz Japanese character speech, published as one
zip per speaker with an ASR transcript, a duration and a speechMOS score beside
each clip. This script selects speakers, fetches only their zips, filters the
utterances on transcript agreement, concatenates each speaker's clips into
pseudo-long recordings, and runs forced alignment over the result.

Every stage skips work whose output already exists, and writes partial output
to a .partial file that is renamed on completion. The script is meant to run on
a preemptible cloud instance: being killed and re-run must always be safe.
"""

import csv
import json
import logging
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import jiwer
import numpy as np
import sphn
import typer
from huggingface_hub import hf_hub_download

from pocket_tts.utils.text_normalization import normalize_japanese
from training.scripts.prepare_data import align, already_aligned

logger = logging.getLogger("prepare_moespeech")
app = typer.Typer(pretty_exceptions_show_locals=False)

SELECTION_SEED = 0  # so --order random is still reproducible across re-runs
DATASET_REPO = "ayousanz/moe-speech-plus"
# The aligner reads kana readings, so its vocabulary has to be kana: align_data
# refuses a model without hiragana in it rather than reporting every utterance
# of the corpus unalignable, one at a time.
KANA_ALIGN_MODEL = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
EXTRACT_MARKER = ".complete"  # beside the directory, not in it
# The grids probe.json reports retention over. They are candidates to read a
# cutoff off, not cutoffs: nothing here filters anything.
CER_THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
MOS_THRESHOLDS = (0.0, 2.5, 3.0, 3.5, 4.0)
# There is deliberately no default cutoff: it is read off probe.json's retention
# table, and this corpus has never been measured, so any number written here now
# would be a guess that afterwards is indistinguishable from a measurement.
DEFAULT_MAX_CER = None
DEFAULT_MIN_MOS = None


def select_characters(info_csv: Path, hours: float, order: str = "largest") -> list[dict]:
    """The speakers whose combined duration first reaches `hours`.

    Reads the dataset's info.csv, which carries num_files, total_duration_min
    and f0_mean for every character -- 12.5 KB that answers "what should I
    download" without fetching any audio.

    Largest-first is the default because it reaches a given number of hours in
    far fewer zips: 124 hours takes 28 characters this way against roughly 160
    at random, and unpacking is this phase's bottleneck rather than GPU time.
    Pass "random" when speaker diversity matters more than download size.
    """
    with open(info_csv, encoding="utf-8") as f:
        rows = [
            {
                "name": r["name"],
                "num_files": int(r["num_files"]),
                "total_duration_min": float(r["total_duration_min"]),
                "f0_mean": float(r["f0_mean"]),
            }
            for r in csv.DictReader(f)
        ]
    if order == "largest":
        rows.sort(key=lambda r: (-r["total_duration_min"], r["name"]))
    elif order == "random":
        rows.sort(key=lambda r: r["name"])  # a stable base for the shuffle
        random.Random(SELECTION_SEED).shuffle(rows)
    else:
        raise typer.BadParameter(f"--order must be 'largest' or 'random', got {order!r}")

    chosen, minutes = [], 0.0
    for row in rows:
        if minutes >= hours * 60:
            break
        chosen.append(row)
        minutes += row["total_duration_min"]
    return chosen


def download_characters(names: list[str], dest: Path, repo: str = DATASET_REPO) -> list[Path]:
    """Fetch one zip per character, skipping those already present.

    huggingface_hub downloads to a cache and only writes the final path once
    the transfer completes, so a kill mid-download leaves an incomplete file
    in the cache, not here -- but a kill mid-copy into `dest` would leave an
    incomplete file right here, which is exactly what this function must never
    mistake for "already have it". So the copy lands at `<name>.zip.partial`
    first and is renamed into place only once it is whole; `os.replace` is
    atomic on both POSIX and Windows, so there is no window where a reader
    could see a half-renamed file either.
    """
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        local = dest / f"{name}.zip"
        if local.exists():
            logger.info(f"{local.name} already present, skipping")
        else:
            fetched = hf_hub_download(repo, f"{name}.zip", repo_type="dataset")
            partial = dest / f"{name}.zip.partial"
            shutil.copyfile(fetched, partial)
            os.replace(partial, local)
        paths.append(local)
    return paths


def extract_character(zip_path: Path, dest_root: Path) -> Path:
    """Unpack one character's zip into `dest_root/<name>/`, once.

    Thirty gigabytes takes tens of minutes to unpack and the instance can be
    reclaimed in the middle of it, so a re-run has to tell a finished character
    from an interrupted one. The directory existing does not answer that: a
    half-extracted character has a directory too, and trusting it would drop
    every clip the kill arrived before while looking exactly like success.

    Completion is therefore recorded explicitly, as `<name>.complete` written
    only once the last member is on disk. It sits beside the directory rather
    than inside it so the directory holds corpus files and nothing else, and
    callers can iterate it without filtering. A directory without that record
    is deleted and unpacked again rather than resumed -- the member the kill
    interrupted is likely truncated, and the members it never reached are
    missing, neither of which is visible from the outside.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    out = dest_root / zip_path.stem
    marker = dest_root / f"{zip_path.stem}{EXTRACT_MARKER}"
    if marker.exists() and out.is_dir():
        logger.info(f"{out.name} already extracted, skipping")
        return out
    if out.exists():
        logger.info(f"{out.name} was left half-extracted, unpacking it again")
        shutil.rmtree(out)
    marker.unlink(missing_ok=True)  # so a kill mid-unpack cannot leave it lying
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    marker.write_text("", encoding="utf-8")
    return out


def read_annotation(path: Path, speaker: str | None = None) -> dict | None:
    """One clip's annotation, or None if it cannot stand as a training example.

    Every clip ships two independent ASR transcriptions and no manual one, so
    there is nothing to check either against except the other. Their mutual CER
    is what this returns as `cer`: not a measure of the audio, but of where the
    two systems disagree, which is the best available evidence that the text is
    wrong. `transcript` is the anime-whisper one, and it is the reference side
    of that CER, so the number reported is the disagreement measured against
    exactly the string the manifest would carry.

    Both are normalized before they are compared, and the normalized whisper is
    what `transcript` carries -- measuring the raw strings would report a
    disagreement the corpus does not contain. Two Japanese ASRs differ by
    convention far more than they differ by hearing: half-width katakana against
    full-width, full-width digits and latin against ASCII, a space where the
    other put none. NFKC erases exactly those, so on `ＡＢＣです` against
    `ABCです` the raw CER is 0.600 and the strings the manifest would carry are
    identical. Measured raw, every cutoff an operator could plausibly read off
    the retention table would throw those clips away -- and the table itself
    would be a distribution over strings this corpus never contains, which is
    the one artifact the whole choice of cutoffs is made from.

    A field that is absent -- or an empty transcription, which is worse than
    absent, since CER is the edit distance over the reference's length and an
    empty reference makes that number mean nothing -- drops the clip rather
    than being defaulted. A defaulted
    duration reaches the manifest as a window the audio does not contain, and
    the loader reads silence and trains on it as speech, which nothing
    downstream can detect.

    A JSON that is not an object at all is dropped the same way. The scan that
    calls this picks up every `.json` under the extract root, and a character
    zip may well ship an index or a metadata file among the per-clip ones; a
    list or a bare number parses without complaint and then has no `.get`, so
    without this check one such file raises AttributeError -- not a decode
    error, not caught by the caller -- and ends a pass 40 minutes in.

    `id` and `wav` come from the path because the JSON carries neither: the
    audio is the file's sibling. So is the speaker missing from the JSON, but
    it does not come from the path -- the caller passes it, because only the
    caller knows it. `_scan_annotations` walks each selected speaker's root
    with `rglob`, which is to say it is built for the clips to sit anywhere
    below `extracted/<name>/`, and `path.parent.name` is the speaker only in
    the one layout where they sit directly in it. Nobody has unpacked a real
    MoeSpeech zip; put the clips one directory down -- `<name>/wav/clip.json`,
    an ordinary way to pack a zip -- and every clip of every character is
    labelled `wav`. Everything downstream then trusts that label: `concatenate`
    compares it to refuse a mixed list and would see one speaker where there
    are two, joining two voices into one file and teaching the model that the
    prompt does not decide the voice; the split would hold out a label rather
    than a character; and `audio/<speaker>.wav` would collide between them.
    None of it raises.

    `path.parent.name` stays as the answer for a caller that names no speaker,
    which is the flat layout a fixture lays out and nothing real.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    whisper = data.get("anime_whisper_transcription")
    parakeet = data.get("parakeet_jp_transcription")
    duration = data.get("duration")
    mos = data.get("speechMOS")
    if not whisper or not parakeet or duration is None or mos is None:
        return None
    whisper, parakeet = normalize_japanese(whisper), normalize_japanese(parakeet)
    return {
        "id": path.stem,
        "speaker": path.parent.name if speaker is None else speaker,
        "wav": path.with_suffix(".wav"),
        "duration": float(duration),
        "transcript": whisper,
        # Normalization can empty a transcription the check above kept, and the
        # scan drops such a clip a line later. Its CER is never read, and there
        # is no meaningful one to compute -- nothing to divide the distance by
        # -- so it is reported as the total disagreement it is.
        "cer": jiwer.cer(whisper, parakeet) if whisper else 1.0,
        "mos": float(mos),
    }


def _scan_annotations(
    root: Path, skipped: Counter, names: list[str] | None = None
) -> Iterator[dict]:
    """Every usable annotation under `root`, normalized, in path order.

    `names` bounds the walk to those speakers' directories under `root`, and is
    how --hours bounds what a run *uses* rather than only what it fetches. The
    extract root accumulates: it is the directory a larger run's speakers were
    unpacked into, and asking for fewer hours afterwards has to mean fewer
    speakers or the flag does nothing at all. Passing None walks everything,
    which is what a fixture with its clips laid out flat wants.

    The probe and the selection walk the same 400,000 files and must agree on
    which of them are annotations at all, or the retention table an operator
    reads their cutoffs off describes a different corpus from the one those
    cutoffs are then applied to. So both walk through here -- and so, for the
    same reason, does the drop below. read_annotation normalizes the transcript
    it returns, and normalizing can empty one that was not empty; a clip with
    no text left has to be dropped, and done on the selection's side alone that
    drop would land after the probe had already counted the clip and promised
    it in the retention table. The table would then over-promise, which is the
    one thing it may not do: it is the only artifact an operator reads when
    choosing cutoffs.

    Path order is part of the contract, not an accident of `rglob`. The stage
    after this one concatenates a speaker's clips into pseudo-long recordings
    and writes a manifest of offsets into them; taken in filesystem order, a
    re-run after a kill builds a different recording and the offsets no longer
    describe the audio.

    The scan is `rglob` because the extract root holds one directory per
    character; `glob` would report an empty corpus for the real data while
    passing on any fixture that keeps its JSON flat. Because it is `rglob`, the
    speaker is this loop's to say and not `read_annotation`'s to infer: a zip
    that puts its clips in a subdirectory of the character's would otherwise
    label every one of them after that subdirectory. So the name of the root
    being walked is handed over with each path.

    A file that cannot be read is counted in `skipped` and passed over rather
    than raising: one bad JSON in 400,000 must not end a pass that takes 40
    minutes to reach it. Counted, though -- a pass that quietly returns 300,000
    of 400,000 clips looks exactly like a corpus that was only ever 300,000
    long. The clips normalization empties are counted for that same reason:
    otherwise a manifest short of what probe.json promised is indistinguishable
    from a corpus that was smaller all along. `skipped` belongs to the caller
    because this is a generator: an exhausted one cannot report anything back,
    and the probe has to publish those numbers.
    """
    roots = [(root, None)] if names is None else [(root / n, n) for n in sorted(set(names))]
    for speaker, path in (
        (name, p) for scan_root, name in roots for p in sorted(scan_root.rglob("*.json"))
    ):
        try:
            row = read_annotation(path, speaker)
        except (OSError, ValueError, TypeError) as e:
            skipped["unreadable"] += 1
            logger.debug(f"{path}: unreadable ({e})")
            continue
        if row is None:
            skipped["incomplete"] += 1
            continue
        if not row["transcript"]:
            skipped["blank"] += 1
            logger.debug(f"{path}: nothing left of the transcript after normalization")
            continue
        yield row


def select_utterances(
    root: Path, max_cer: float, min_mos: float, names: list[str] | None = None
) -> Iterator[dict]:
    """The utterances worth training on, out of everything under `root`.

    This is where the probe's measurements become decisions. Both cutoffs are
    required rather than defaulted -- see DEFAULT_MAX_CER above -- because the
    right values are a property of this corpus, which nothing has measured yet.

    An utterance is kept when the two ASRs agree closely enough (`cer` at most
    `max_cer`) and the audio scores well enough (`mos` at least `min_mos`).
    Both comparisons are inclusive, matching the retention table exactly, so
    the count an operator read there is the count they get.

    The transcript arrives already normalized, with the same function
    align_data.py applies under --segmenter japanese. That keeps the manifest,
    the tokenizer corpus and a user's inference input in one distribution; a
    mismatch between them raises nothing and shows up only as a model that
    never quite becomes intelligible. It happens in `read_annotation`, before
    the CER is measured against it, so that `cer` scores the string the
    manifest will carry rather than one nothing keeps -- and the clips
    normalization empties are dropped by `_scan_annotations`, which the probe
    shares, so its counts and this one's agree. The two cutoffs are the only
    thing this function decides.
    """
    # The counts go nowhere here on purpose: the probe already reported them to
    # the operator over this same tree, and repeating them would read as a
    # second, different set of unusable files.
    skipped: Counter = Counter()
    for row in _scan_annotations(root, skipped, names):
        if row["cer"] > max_cer or row["mos"] < min_mos:
            continue
        yield {
            "id": row["id"],
            "speaker": row["speaker"],
            "wav": row["wav"],
            "duration": row["duration"],
            "transcript": row["transcript"],
            "cer": row["cer"],
            "mos": row["mos"],
        }


def _distribution(values: list[float]) -> dict:
    """min / percentiles / median / max / mean of one measured quantity.

    The percentiles are the point: a mean says a corpus of 5-second clips and a
    corpus of 1-second clips with a few 60-second ones are the same corpus, and
    the cutoffs chosen next depend entirely on telling those apart.
    """
    keys = ["min", "p1", "p5", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max", "mean"]
    if not values:
        return dict.fromkeys(keys)
    ordered = sorted(values)
    stats = {"min": ordered[0], "max": ordered[-1], "mean": sum(ordered) / len(ordered)}
    for key, q in [
        ("p1", 1),
        ("p5", 5),
        ("p10", 10),
        ("p25", 25),
        ("median", 50),
        ("p75", 75),
        ("p90", 90),
        ("p95", 95),
        ("p99", 99),
    ]:
        stats[key] = _percentile(ordered, q)
    return {k: stats[k] for k in keys}


def _percentile(ordered: list[float], q: float) -> float:
    """The q-th percentile of an already sorted list, interpolating between the
    two neighbouring samples -- so the median of an even-length list is the
    midpoint of the middle pair rather than an arbitrary one of them."""
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _retention(rows: list[tuple[float, float, float]]) -> list[dict]:
    """How much of the corpus each candidate pair of cutoffs would leave.

    A distribution alone does not answer the question actually being asked,
    which is not "how bad is the corpus" but "how many hours are left if I
    demand this much agreement". Hours are reported next to counts because the
    target this pipeline is given is in hours, and a cutoff that keeps 80% of
    the clips can still keep far less than 80% of the audio.
    """
    table = []
    for max_cer in CER_THRESHOLDS:
        for min_mos in MOS_THRESHOLDS:
            kept = [d for d, cer, mos in rows if cer <= max_cer and mos >= min_mos]
            table.append(
                {
                    "max_cer": max_cer,
                    "min_mos": min_mos,
                    "kept": len(kept),
                    "fraction": len(kept) / len(rows) if rows else 0.0,
                    "hours": sum(kept) / 3600,
                }
            )
    return table


def probe_utterances(root: Path, names: list[str] | None = None) -> dict:
    """Measure every annotation under `root`. Decide nothing about any of them.

    The returned dict is what `probe.json` holds: how many utterances there
    are, the distribution of clip length, mutual CER and speechMOS, and a table
    of how many clips and hours survive each candidate pair of cutoffs. No
    threshold is applied here and none is recommended -- this exists so the
    next stage's cutoffs come from the corpus rather than from a guess.

    `names` bounds it to those speakers, exactly as the selection is bounded:
    the two have to describe one corpus, and the extract root can hold speakers
    this run did not ask for.

    Files that cannot be read, and clips normalization leaves no text of, are
    skipped and counted by `_scan_annotations`, which the selection pass shares
    so that both see the same corpus -- the retention table below therefore
    promises a count the selection actually delivers. Each kind is reported
    under its own name, because "the manifest is smaller than the table said"
    has three different causes and only the counts tell them apart.

    Only the three measured numbers are retained per clip, not the row: the
    transcriptions of a whole corpus do not need to be in memory at once for
    this, and at this scale that is gigabytes.
    """
    skipped: Counter = Counter()
    rows = [(r["duration"], r["cer"], r["mos"]) for r in _scan_annotations(root, skipped, names)]
    unreadable, incomplete, blank = (skipped["unreadable"], skipped["incomplete"], skipped["blank"])
    dropped = unreadable + incomplete + blank
    if dropped:
        logger.warning(
            f"skipped {dropped} of {len(rows) + dropped} json files: {unreadable} unreadable, "
            f"{incomplete} not usable as an annotation, {blank} with nothing left of the "
            f"transcript after normalization"
        )
    return {
        "count": len(rows),
        "unreadable": unreadable,
        "incomplete": incomplete,
        "blank": blank,
        "hours": sum(d for d, _, _ in rows) / 3600,
        "duration": _distribution([d for d, _, _ in rows]),
        "cer": _distribution([c for _, c, _ in rows]),
        "mos": _distribution([m for _, _, m in rows]),
        "retention": _retention(rows),
    }


def _joined_path(out_wav: Path, index: int) -> Path:
    """The `index`-th pseudo-long file of one speaker's run.

    The first one is the name the caller gave, exactly: a caller that names its
    output has to be able to find it again without knowing how many files the
    run happened to need. The rest are numbered beside it.
    """
    if index == 0:
        return out_wav
    return out_wav.with_name(f"{out_wav.stem}_{index:03d}{out_wav.suffix}")


def _write_joined(parts: list, sample_rate: int, path: Path) -> None:
    """Write one pseudo-long recording, and let nothing see it half-written.

    A truncated wav is still a valid wav -- it is simply shorter than the
    manifest says -- so every offset past the point the kill arrived reads as
    silence or fails to seek, and a re-run that finds the file sitting there
    cannot tell it from a finished one. So the samples land beside the name and
    are renamed in only once all of them are there; `os.replace` is atomic on
    both POSIX and Windows.
    """
    joined = np.concatenate(parts, axis=1)
    partial = path.with_name(f"{path.name}.partial")
    sphn.write_wav(str(partial), joined[0], sample_rate)
    os.replace(partial, path)


def concatenate(utterances: list[dict], out_wav: Path, target_sec: float) -> list[dict]:
    """Join one speaker's clips into pseudo-long recordings, and say where each went.

    The median character's mean clip is 5.8 seconds and the loader keeps a
    second of audio on either side of the cut it makes, so what training would
    otherwise see is under five seconds of target audio. Joining a speaker's
    clips end to end gives the loader room to cut. The returned entries are
    what the manifest is written out of -- one per clip, in the order they were
    handed over, each naming the file its audio landed in and the window inside
    that file it occupies.

    Every offset is measured from the samples actually laid down, not from the
    annotation's `duration`, and no silence is inserted between clips. A gap,
    or a length taken on trust from a JSON somebody else's tool wrote, slides
    every later clip in the file by the difference, and nothing downstream can
    detect that: the manifest still parses, every offset is still inside a real
    file, and the model simply learns from speech that does not match its text.

    A run is cut into as many files as it takes for none of them to run past
    `target_sec`. What that bounds is this function: `_write_joined` holds a
    whole file in memory to concatenate it, and a preemption costs the file in
    flight. It is not what bounds either reader -- `align_data.read_window` and
    the loader's `_load_window` both read the window a row names and never the
    file it sits in, so neither one's memory goes with this number. The floor
    is that joining has to be worth doing: at 120 s the corpus's 5.8-second
    median clip means about twenty per file, and 120 s is four times the
    loader's `max_duration_sec`, so a file stays long against anything read out
    of it. Clips themselves are never cut, so one longer than the target gets a
    file to itself. Offsets restart at zero in each file, being offsets into
    the file they name.

    Clips are never joined across speakers, and a mixed list raises here rather
    than being grouped: the caller groups, and everything after this treats a
    cut anywhere inside one of these files as one voice -- the loader takes one
    side of the cut as the voice prompt for the other, so a file holding two
    speakers would teach the model that the prompt does not decide the voice.
    That one is a caller's bug and stops the run; a clip whose audio is
    missing or truncated, or recorded at another rate, or in a different number
    of channels, is corpus data and costs only that clip. Raising there would
    end the run at the same clip on every re-run -- the selection is
    deterministic and read back off `utterances.jsonl` -- with no way past it
    but hand-editing that file, which is the one thing this script is built not
    to need. A skipped clip simply never reaches `entries`,
    which is already what the manifest should say about it.

    The audio is left at its own 44.1 kHz; the loader resamples as it reads.
    A re-run lays the same clips down in the same order and writes the same
    bytes to the same names, so being killed anywhere in here costs only the
    files that had not been renamed into place yet.
    """
    speakers = {u["speaker"] for u in utterances}
    if len(speakers) > 1:
        raise ValueError(f"concatenate() takes one speaker at a time, got {sorted(speakers)}")

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    parts: list = []  # the clips of the file being built, at its own sample rate
    cursor = 0  # samples already laid down in that file
    written = 0  # files finished, which is the index of the one being built
    sample_rate = None
    channels = None  # of the first readable clip, like the rate beside it

    skipped = 0
    mismatched = 0  # clips whose audio is not the length the annotation claims
    who = next(iter(speakers), out_wav.stem)

    def skip(wav_path, why) -> None:
        nonlocal skipped
        skipped += 1
        if skipped % 100 == 1:  # one line per hundred: this can be the whole corpus
            logger.warning(f"not joining ({skipped} so far): {wav_path}: {why}")

    for u in utterances:
        try:
            wav, sr = sphn.read(str(u["wav"]))
        except (ValueError, OSError) as e:
            # The wav is a path read_annotation derived from the json's name and
            # nothing has opened until now, so this is where a missing or
            # truncated clip first shows up -- the plan lists broken wavs among
            # the expected surprises of the first real run.
            skip(u["wav"], e)
            continue
        if sample_rate is None:
            sample_rate, channels = sr, wav.shape[0]
        elif sr != sample_rate:
            # Joining it anyway would play it at the wrong speed and put every
            # offset after it in the file out by a factor nothing reports.
            skip(u["wav"], f"{sr} Hz among {sample_rate} Hz clips")
            continue
        elif wav.shape[0] != channels:
            # Skipped for the same reason as the rate, and urgently: this one
            # does not merely mislabel its own clip. `np.concatenate` refuses
            # arrays that disagree on every axis but the one it joins, and it
            # runs inside `_write_joined`, outside the try above -- so a single
            # stereo clip among mono ones raises where nothing catches it and
            # ends the whole run, at the same clip on every re-run, with no way
            # past it but hand-editing utterances.jsonl.
            skip(u["wav"], f"{wav.shape[0]} channels among {channels}-channel clips")
            continue
        n_samples = wav.shape[1]
        target_samples = round(target_sec * sample_rate)
        if parts and cursor + n_samples > target_samples:
            _write_joined(parts, sample_rate, _joined_path(out_wav, written))
            parts, cursor, written = [], 0, written + 1
        duration = n_samples / sample_rate
        if abs(duration - u["duration"]) > 0.1:
            # Not an error and not corrected: the audio is what was written, so
            # the audio is what the manifest describes. Worth seeing, though --
            # it means the corpus's own durations cannot be trusted elsewhere,
            # and that is a fact about the corpus rather than about any one
            # clip, so it is counted here and reported once at the end. Per
            # clip it is debug: 400,000 of these is not something anyone reads.
            mismatched += 1
            logger.debug(f"{u['wav']}: annotated {u['duration']:.2f}s, audio {duration:.2f}s")
        entries.append(
            {
                "id": u["id"],
                "speaker": u["speaker"],
                "transcript": u["transcript"],
                "path": str(_joined_path(out_wav, written)),
                # Six decimals rather than three because this number is a
                # key, not a display: align_data resumes on (path, start), so
                # two clips whose combined length rounds to the same
                # millisecond would be one utterance to `_resume_done` and the
                # second would be dropped from a resumed alignment without a
                # word. Microseconds cannot collide for any real clip.
                "start": round(cursor / sample_rate, 6),
                "duration": round(duration, 3),
            }
        )
        parts.append(wav)
        cursor += n_samples

    if parts:
        _write_joined(parts, sample_rate, _joined_path(out_wav, written))
    if skipped:
        # Said once at the end as well as every hundredth time above: this is
        # how many clips the manifest is short of the selection, and a speaker
        # whose first clip set the rate can lose every clip after it that way.
        logger.warning(
            f"{who}: {skipped} of {len(utterances)} clips "
            "could not be joined and are not in the manifest"
        )
    if mismatched:
        # One line, at info, because the per-clip lines are debug and there can
        # be 400,000 of them: without this the operator either sees nothing or
        # sees an unreadable flood, and the number that matters -- how much of
        # this corpus's own metadata disagrees with its audio -- is in neither.
        logger.info(
            f"{who}: {mismatched} of {len(utterances)} clips are more than 0.1s from their "
            "annotated duration; the manifest describes the audio that was written"
        )
    return entries


def split_by_speaker(entries: list[dict], valid_hours: float) -> tuple[list, list]:
    """Hold out whole speakers, never single utterances, for the valid set.

    Split by utterance and the same voice stands on both sides. The validation
    loss then measures a voice the model has already fitted, which is precisely
    what it exists not to do: it is the number that decides when to stop, and it
    would report a model that has memorized these speakers as a model that has
    learned to speak Japanese. Held out whole, a valid speaker is a voice the
    weights have never seen, and the loss over it means what it is read as.

    An utterance-level split also breaks the eval protocol outright. Scoring a
    voice means cloning it from one utterance and synthesizing another, so a
    held-out speaker needs at least two -- and a random utterance-level split
    over a corpus whose tail is speakers with a handful of lines produces plenty
    with exactly one. Speakers with a single utterance are therefore never held
    out. They are not dropped either: one utterance is a fine training example
    and only a useless evaluation one, so it goes to train.

    Speakers are taken smallest-first until `valid_hours` is reached. Every hour
    held out is an hour not trained on, and this granularity is one whole
    speaker: the smallest eligible ones buy the most distinct voices per held-out
    hour, and keep the overshoot past the target down to the size of one small
    speaker rather than one of the largest.

    The result is a pure function of the entries -- no seed, no shuffle, no
    clock. This script is re-run after being killed, and a split that moved
    between runs would put a speaker the previous run validated on into this
    run's training set, at which point the loss over the rest of that set is
    quietly optimistic for the remainder of the run and nothing says so.
    """
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_speaker[entry["speaker"]].append(entry)
    seconds = {s: sum(e["duration"] for e in rows) for s, rows in by_speaker.items()}

    # Name breaks the ties, so equal-length speakers order the same way twice.
    eligible = sorted(
        (s for s, rows in by_speaker.items() if len(rows) > 1), key=lambda s: (seconds[s], s)
    )
    held_out: set[str] = set()
    target, taken = valid_hours * 3600, 0.0
    for speaker in eligible:
        if taken >= target:
            break
        held_out.add(speaker)
        taken += seconds[speaker]

    train = [e for e in entries if e["speaker"] not in held_out]
    valid = [e for e in entries if e["speaker"] in held_out]
    # Counted rather than merely skipped: a speaker with one utterance is
    # dropped from what may be held out, silently and on every run, and how
    # much of the corpus that is is a number the operator has no other way to
    # see. It is only alarming when the valid set also came up short, which is
    # the warning below; here it is one info line about the corpus's shape.
    singletons = sum(1 for rows in by_speaker.values() if len(rows) == 1)
    logger.info(
        f"held out {len(held_out)} speakers ({taken / 3600:.2f}h, {len(valid)} utterances) "
        f"of {len(by_speaker)}; {len(train)} utterances left to train on; {singletons} "
        "speakers have a single utterance and were never eligible"
    )
    if taken < target:
        # Not an error -- a small corpus simply has less to hold out -- but the
        # valid set is smaller than asked for and that has to be said out loud.
        logger.warning(
            f"only {taken / 3600:.2f}h of the {valid_hours:.2f}h asked for could be held out: "
            f"{singletons} of {len(by_speaker)} speakers have a single utterance and cannot "
            "be evaluated on"
        )
    return train, valid


def write_manifest(entries: list[dict], path: Path) -> int:
    """Write one JSON object per line, the format the loader reads, and say how
    many.

    Each entry is written whole. `training/dataloader.py` picks the fields it
    needs out of the object by name -- `path`, `duration`, `transcript`, `start`
    -- and ignores the rest, so carrying `id` and `speaker` along costs a few
    bytes a line and makes the manifest answerable afterwards: which speakers
    ended up held out, and which clip a bad sample came from, are questions the
    four fields the loader reads cannot answer.

    UTF-8 is explicit because this machine's default is cp932, and so is
    `ensure_ascii=False` -- the transcripts are Japanese and a manifest nobody
    can read with `head` is a manifest nobody checks. The two go together: with
    the escapes on, the file would be ASCII and the encoding would not matter
    until the day something wrote a raw character into it, far from here.

    The lines land beside the name and are renamed in once they are all there.
    A manifest cut short by a kill is still valid JSONL -- every line parses and
    every path exists -- so nothing downstream can tell it from a finished one,
    and a re-run that skips this stage because the file is present would train
    on however much of the corpus the kill let through.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    written = 0
    with open(partial, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
    os.replace(partial, path)
    logger.info(f"{path}: {written} utterances")
    return written


def _write_json(obj: dict, path: Path) -> None:
    """Write one JSON document, and let nothing see it half-written.

    The same reason as everywhere else here: a truncated `characters.json` or
    `probe.json` still sits under the name the next run tests for, and being
    taken for finished is exactly what must not happen to either. Both are read
    by a person -- one is the table the cutoffs are chosen off, the other is how
    one checks which speakers a run actually took -- so they are indented, and
    written with the escapes off so the names stay Japanese.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with open(partial, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(partial, path)


def _read_jsonl(path: Path) -> list[dict]:
    """The rows of something this script wrote earlier, so a re-run continues
    from them instead of walking 400,000 files again."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _same_cutoffs(path: Path, cutoffs: dict) -> bool:
    """Whether `path` already records exactly the cutoffs this run was given.

    Read twice: stage 5 writes the file only when it would change, so an
    unchanged pair does not touch the timestamp everything below is gated on;
    and stage 0 asks the same question to tell a re-run that will rebuild the
    manifests from one that will not. An unreadable or absent file answers no,
    which rewrites it and refuses the run -- both of them the cheap direction.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8")) == cutoffs
    except (OSError, ValueError):
        return False


def _stale(output: Path, inputs: list[Path]) -> Path | None:
    """Whichever of `inputs` was written after `output` was built, if any.

    Every stage below skips work whose output is already on disk; that is how a
    preempted run costs only the stage in flight. Existence alone is the wrong
    question the moment an option changes, because the way to redo a stage is
    to delete its artifact: delete `utterances.jsonl` and re-run under a
    stricter --min-mos and every later stage still finds its own output sitting
    there and skips, so `train.jsonl` goes on describing the selection that was
    just rejected while the run reports success over it. Nothing in the tree
    says the two disagree, and the model trains on the rejected one.

    So an output is reusable only while it is at least as new as everything it
    was built from. Equal timestamps count as fresh: file clocks are coarser
    than these stages are fast -- 15 ms on Windows -- so two artifacts of one
    run can share a timestamp, and treating that as stale would mean never
    skipping anything. A clock that jumps backwards costs a rebuild, which is
    the harmless direction to be wrong in.
    """
    if not output.exists():
        return None
    built = output.stat().st_mtime
    moved = [p for p in inputs if p.exists() and p.stat().st_mtime > built]
    return max(moved, key=lambda p: p.stat().st_mtime, default=None)


def _japanese_segmenter_error() -> Exception | None:
    """None if align_data can build the Japanese segmenter, else why it cannot.

    Asked through `SEGMENTERS` rather than by importing fugashi, so the answer
    cannot come to disagree with what stage 8 will actually construct: building
    the tagger needs the dictionary as well as the wrapper, and both live in
    the `japanese` dependency group. An `align_data` that will not import at
    all is a different failure and is left to raise as itself.
    """
    from training.scripts.align_data import SEGMENTERS

    build = SEGMENTERS["japanese"]  # outside the try: a missing key is a bug here, not there
    try:
        build()
    except Exception as e:
        return e
    return None


def require_japanese_segmenter(will_align: bool) -> None:
    """Refuse a run that cannot finish, before it spends the day finding out.

    Stage 8 segments with MeCab, which `align_data._japanese_segmenter` imports
    lazily, out of a dependency group that is not part of a plain `uv sync`:

        uv sync --group japanese

    Without it the run downloads, unpacks, probes, selects, concatenates and
    writes both manifests -- tens of gigabytes and hours of a paid instance --
    and only then fails. Worse, it fails through `prepare_data.align`, which
    runs the aligner in a subprocess, so what surfaces is a CalledProcessError
    naming a return code rather than the missing package.

    `will_align` is the caller's answer to "will stage 8 actually align", which
    is not the same question as "was this run given both cutoffs". A run with no
    cutoffs stops at the probe and never reaches the aligner; so does a re-run
    over a finished tree, since `align` skips an output it finds already there
    -- and that command is the one an operator retypes after a preemption, so it
    is typed over trees in every state, complete ones included. The caller asks
    `prepare_data.already_aligned` rather than restating when `align` skips, so
    the two cannot come to disagree.

    Neither of those runs is refused -- but both are told, because the probe run
    is the long one and the group can be installed while it downloads.

    What the two alignments being there cannot say on its own is whether they
    will survive the run: change the cutoffs over a finished tree and both are
    on disk when this is asked, then thrown away in stage 8 as older than the
    manifests stage 7 has since rewritten. Predicting that from here would mean
    restating stages 5 and 7, which is the drift `will_align` is passed to
    avoid -- so instead the caller answers the cheap half of it, and this guard
    over-approximates: it is only quiet when the alignments are there *and* the
    cutoffs on disk are the ones just given, which is to say when nothing can
    be rebuilt. Changed cutoffs are refused, which is the documented main path
    for changing them, and refusing costs thirty seconds of the install.

    What that still lets through -- a hand-deleted `utterances.jsonl`, another
    --align-model, another shard count -- `require_segmenter_to_align` catches
    one line before each `align`, where the answer is no longer a prediction.
    """
    error = _japanese_segmenter_error()
    if error is not None:
        install = "install it with: uv sync --group japanese"
        if not will_align:
            logger.warning(
                f"the Japanese segmenter is not available ({error}); this run does not reach "
                f"the aligner and does not need it, but a run that has to align does -- "
                f"{install}"
            )
            return
        logger.error(
            f"stage 8 aligns with the Japanese segmenter, which is not available ({error}). "
            f"{install}. Nothing has been fetched or written: this is checked before the "
            "first stage precisely so it is not found out after 30 GB and several hours."
        )
        raise typer.Exit(1) from error


def require_segmenter_to_align(aligned: Path) -> None:
    """The guard above, asked again where the answer is finally knowable.

    Stage 0 has to guess, because whether stage 8 aligns depends on what stages
    5 and 7 will do to the manifests, and predicting those from up there is the
    drift `will_align` exists to avoid. It therefore over-approximates: a run
    given both cutoffs is refused unless nothing can be rebuilt. Here, one line
    before `align`, the question is no longer a prediction -- every manifest has
    been written and every stale alignment discarded, so `already_aligned` is
    the whole answer.

    The two guards do not cost the same and are not tuned the same way. A run
    refused for nothing costs the operator thirty seconds of
    `uv sync --group japanese`; a run waved through that cannot finish costs a
    day of a paid instance and surfaces as a CalledProcessError out of the
    aligner's subprocess, naming a return code rather than a package. So stage
    0 leans towards refusing, and this catches whatever it still let past -- a
    hand-deleted `utterances.jsonl`, a changed --align-model, a changed shard
    count -- at the cost of a clean stop instead of that.
    """
    if already_aligned(aligned):
        return
    error = _japanese_segmenter_error()
    if error is None:
        return
    logger.error(
        f"{aligned} has to be aligned with the Japanese segmenter, which is not available "
        f"({error}). Install it with: uv sync --group japanese, and run this again -- "
        "everything above the alignment is on disk and will not be redone."
    )
    raise typer.Exit(1)


def _reusable(outputs: list[Path], inputs: list[Path], keeping: str) -> bool:
    """Whether every one of `outputs` still stands for the work it records.

    All of them or none: stage 7 writes two manifests that only mean anything
    together, and a rebuild of one from inputs the other no longer matches is
    the same silent disagreement `_stale` exists to prevent.
    """
    if not all(p.exists() for p in outputs):
        return False
    for output in outputs:
        moved = _stale(output, inputs)
        if moved is not None:
            logger.info(f"{moved} is newer than {output}, which is rebuilt rather than kept")
            return False
    logger.info(f"{', '.join(str(p) for p in outputs)} up to date, {keeping}")
    return True


@app.command()
def main(
    out: Annotated[
        str, typer.Option(help="where every artifact of this run is written")
    ] = "data/ja",
    hours: Annotated[float, typer.Option(help="hours of speech to pick speakers for")] = 124,
    order: Annotated[
        str, typer.Option(help="'largest' for the fewest zips, 'random' for the most voices")
    ] = "largest",
    max_cer: Annotated[
        float | None,
        typer.Option(
            help="keep utterances whose two ASR transcripts disagree by at most this. "
            "Read it off probe.json's retention table; there is no default"
        ),
    ] = DEFAULT_MAX_CER,
    min_mos: Annotated[
        float | None,
        typer.Option(
            help="keep utterances scoring at least this speechMOS. Read it off "
            "probe.json's retention table; there is no default"
        ),
    ] = DEFAULT_MIN_MOS,
    valid_hours: Annotated[
        float, typer.Option(help="hours held out for validation, whole speakers at a time")
    ] = 1.0,
    target_sec: Annotated[
        float,
        typer.Option(
            help="longest pseudo-recording to build out of one speaker's clips. Long enough "
            "that joining is worth doing and well past the loader's max_duration_sec; short "
            "enough that a preemption costs one small file, since this stage builds each in "
            "memory"
        ),
    ] = 120.0,
    zips: Annotated[
        str | None,
        typer.Option(
            help="where the downloaded zips are kept (default: <out>/zips). Naming one "
            "directory across runs keeps a larger --hours from re-fetching what a "
            "smaller one already has"
        ),
    ] = None,
    align_shards: Annotated[
        int, typer.Option(help="parallel alignment processes (one GPU each)")
    ] = 1,
    align_model: Annotated[
        str, typer.Option(help="CTC model the aligner reads; it must have kana in its vocabulary")
    ] = KANA_ALIGN_MODEL,
    repo: Annotated[str, typer.Option(help="the dataset's HuggingFace repo")] = DATASET_REPO,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="log every clip a stage passes over")
    ] = False,
) -> None:
    """Eight stages from a dataset name to an aligned training manifest.

    Every stage skips work whose output is already on disk, so this command is
    re-run rather than resumed. That is the whole design: the instance it runs
    on is preemptible, and being killed must cost only the stage in flight.

    It stops once on purpose, after the probe. The two transcript cutoffs have
    no default because nothing has ever measured this corpus; the probe is what
    measures it. Read them off `probe.json` and run the same command again with
    --max-cer and --min-mos, and it carries on from there.

    What a stage skips on is its output and the timestamps of its inputs, not
    the options it was given -- with one exception. --max-cer and --min-mos are
    written to `cutoffs.json` and counted among the selection's inputs, so a
    changed pair rebuilds the selection, the offsets, both manifests and both
    alignments on its own. Nothing else on disk records them, and without that
    a stricter cutoff would rewrite `utterances.jsonl` and change nothing else:
    the run would report success over the selection it had just replaced.

    Every other option -- --target-sec, --valid-hours, --order -- is recorded
    nowhere, so changing one re-runs nothing. Delete that stage's artifact to
    redo it under a new value; deleting is the only way to say so, and it is
    deliberate, since the alternative is a stage that quietly redoes 30 GB of
    work. Deleting one is enough: what was built out of it is rebuilt with it,
    so removing `utterances.jsonl` alone carries through `entries/`, `audio/`,
    `train.jsonl`, `valid.jsonl` and both alignments.

    --hours is the third case. `characters.json` is reused whenever it exists,
    whatever --hours now says, and a mismatch is only warned about: delete that
    file to select speakers again, and everything below re-runs against the new
    selection.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Named up here, before anything is written, because stage 0 has to ask
    # about the last stage's outputs: whether the aligner will run at all is
    # what decides whether a missing segmenter is fatal or merely worth saying.
    out_dir = Path(out)
    zip_dir = Path(zips) if zips else out_dir / "zips"
    extract_root = out_dir / "extracted"
    train_manifest, valid_manifest = out_dir / "train.jsonl", out_dir / "valid.jsonl"
    train_aligned = out_dir / "train_aligned.jsonl"
    valid_aligned = out_dir / "valid_aligned.jsonl"
    # Stage 5 is what writes these; they are named here because stage 0 has to
    # ask whether stage 8's inputs are about to be rebuilt underneath it.
    cutoffs_json = out_dir / "cutoffs.json"
    cutoffs = {"max_cer": max_cer, "min_mos": min_mos}

    # 0. The one thing that can fail for a reason no stage below can fix. It is
    #    asked before stage 1 because the answer never changes mid-run and the
    #    stage that needs it is the last one: see require_japanese_segmenter.
    #    A run without both cutoffs stops at the probe and never aligns, and so
    #    does one whose two alignments are already on disk *and* whose cutoffs
    #    are the ones that built them -- align() skips an output it finds
    #    finished, and that is asked of align()'s own predicate rather than
    #    restated here. Both are warned rather than refused.
    #    The cutoffs are half of that question because changing them rebuilds
    #    both manifests in stage 7 and so discards both alignments in stage 8:
    #    the alignments being there says nothing about whether they will still
    #    be there when the aligner is reached. Anything this still lets past is
    #    caught by require_segmenter_to_align, one line before each align().
    require_japanese_segmenter(
        will_align=max_cer is not None
        and min_mos is not None
        and not (
            all(already_aligned(p) for p in (train_aligned, valid_aligned))
            and _same_cutoffs(cutoffs_json, cutoffs)
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Which speakers. 12.5 KB of info.csv decides what the other seven
    #    stages will ever touch, and answers it without fetching any audio.
    characters_json = out_dir / "characters.json"
    if characters_json.exists():
        chosen = json.loads(characters_json.read_text(encoding="utf-8"))
        if chosen["hours_requested"] != hours:
            # Silently re-selecting would orphan whatever is already unpacked
            # under a different set of speakers; silently ignoring --hours
            # would be worse still, so say which of the two won.
            logger.warning(
                f"{characters_json} holds the selection for --hours "
                f"{chosen['hours_requested']:g} and is being reused for --hours {hours:g}; "
                "delete it to select speakers again"
            )
    else:
        info_csv = Path(hf_hub_download(repo, "info.csv", repo_type="dataset"))
        characters = select_characters(info_csv, hours, order)
        chosen = {
            "hours_requested": hours,
            "hours_selected": sum(c["total_duration_min"] for c in characters) / 60,
            "order": order,
            "characters": characters,
        }
        _write_json(chosen, characters_json)
    names = [c["name"] for c in chosen["characters"]]
    logger.info(f"{len(names)} speakers, {chosen['hours_selected']:.1f}h of audio to fetch")

    # 2 + 3. Fetch and unpack, speaker by speaker. Both stages decide per
    #        speaker what they already have -- this is where the tens of
    #        minutes are, and where a preemption would otherwise cost most.
    for zip_path in download_characters(names, zip_dir, repo=repo):
        extract_character(zip_path, extract_root)

    # 4. Measure. Decides nothing: probe_utterances returns the numbers and
    #    deliberately does not write them, so measuring stays independent of
    #    where the file goes. Gated like every stage below rather than on the
    #    file merely being there: a speaker unpacked after it was written makes
    #    it incomplete, and incomplete is exactly wrong for a retention table.
    #    Its hours column is the column --hours is expressed in, and it is the
    #    one artifact an operator reads to choose the two cutoffs -- describing
    #    a smaller corpus than the one those cutoffs are then applied to is the
    #    disagreement the whole staleness check exists to prevent.
    #    A measurement made before a speaker was unpacked does not cover them,
    #    so the completion markers of stage 3 are what this is measured against
    #    -- the selected speakers' markers, since those are the speakers walked.
    #    characters.json is an input for the other direction: the extract root
    #    only ever grows, so a narrower selection adds no marker at all, and
    #    without it a run asked for fewer hours would keep a table measured
    #    over speakers it is no longer training on.
    unpacked = [extract_root / f"{name}{EXTRACT_MARKER}" for name in names]
    probe_json = out_dir / "probe.json"
    probe_inputs = [*unpacked, characters_json]
    if not _reusable([probe_json], probe_inputs, "keeping the measurements it holds"):
        _write_json(probe_utterances(extract_root, names), probe_json)

    # 5. Decide, or stop. The cutoffs are a property of this corpus and the
    #    line above is the first thing that has ever measured it, so there is
    #    nothing to default them to -- see DEFAULT_MAX_CER.
    if max_cer is None or min_mos is None:
        logger.error(
            f"read the retention table in {probe_json} and run this again with --max-cer "
            "and --min-mos. Neither has a default: nothing had measured this corpus until "
            "the line above, and a number guessed now would afterwards be indistinguishable "
            "from a measured one. Everything up to here is on disk and will not be redone."
        )
        raise typer.Exit(1)
    #    The pair is written down beside the selection they made, and is one of
    #    its inputs: nothing else on disk records them, so without this a
    #    re-run under a stricter cutoff would find utterances.jsonl sitting
    #    there and reuse it -- silently, and with no way afterwards for either
    #    the script or the operator to tell which cutoffs a given train.jsonl
    #    was built under. Written only when they differ, so an unchanged pair
    #    does not touch the file and nothing downstream is rebuilt.
    if not _same_cutoffs(cutoffs_json, cutoffs):
        _write_json(cutoffs, cutoffs_json)
    utterances_jsonl = out_dir / "utterances.jsonl"
    selection_inputs = [*unpacked, characters_json, cutoffs_json]
    if _reusable([utterances_jsonl], selection_inputs, "keeping the utterances it holds"):
        utterances = _read_jsonl(utterances_jsonl)
    else:
        # `wav` arrives as a Path, which json.dumps refuses; every reader of
        # this file passes it to sphn.read, which takes the string just as well.
        utterances = [
            {**u, "wav": str(u["wav"])}
            for u in select_utterances(extract_root, max_cer, min_mos, names)
        ]
        write_manifest(utterances, utterances_jsonl)

    # 6. Join each speaker's clips into pseudo-long recordings. The grouping
    #    happens here because concatenate() refuses a mixed list rather than
    #    grouping one: a file holding two voices would teach the model that the
    #    prompt does not decide the voice, and nothing downstream could see it.
    #    Each speaker's offsets are recorded under their own name, so a kill
    #    costs the speaker in flight rather than all of them.
    #    What a run stops naming, it deletes. `audio/` is tens of gigabytes at
    #    124 hours of 44.1 kHz, deleting an artifact and re-running is the
    #    documented way to change any option, and the disk on a preemptible
    #    instance is fixed -- filling it mid-run costs the run. A tighter cutoff
    #    leaves a speaker fewer clips and so fewer files than the last run
    #    wrote, and can drop a speaker out of the selection altogether; nothing
    #    reads what is left over, since every manifest names its files, but it
    #    is never freed either.
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for utterance in utterances:
        by_speaker[utterance["speaker"]].append(utterance)
    #    The braces to the belt above. Every guard from here down compares this
    #    label and none of them can check it: concatenate refuses a mixed list
    #    by comparing labels, so one wrong label shared by two characters is a
    #    mixed list that passes -- two voices in one file, the loader taking one
    #    as the prompt for the other, the split holding out a label instead of a
    #    character, and audio/<speaker>.wav colliding between them. Nothing
    #    raises and nothing downstream can see it. The one thing that can be
    #    checked is that every label is a character this run selected, which is
    #    true of a correct label by construction; it is asked here, before a
    #    single wav is joined or deleted.
    unexpected = sorted(set(by_speaker) - set(names))
    if unexpected:
        logger.error(
            f"the selection holds speakers that were never selected: {unexpected}. These are "
            "the directory names under the extract root, so the layout of the zips is not "
            "what the scan assumes -- joining them would mix characters into one recording. "
            "Nothing has been joined."
        )
        raise typer.Exit(1)
    audio_dir = out_dir / "audio"
    for part in sorted((out_dir / "entries").glob("*.jsonl")):
        if part.stem in by_speaker:
            continue
        # The offsets file is what says which audio was this speaker's, so it
        # is read before it goes; deriving the names from the speaker instead
        # would delete by prefix, and one speaker's name can begin another's.
        for row in _read_jsonl(part):
            wav = Path(row["path"])
            if wav.parent == audio_dir:
                wav.unlink(missing_ok=True)
        part.unlink()
        logger.info(f"{part.stem} is no longer selected; their offsets and audio are removed")
    entries: list[dict] = []
    for speaker in sorted(by_speaker):
        part = out_dir / "entries" / f"{speaker}.jsonl"
        if _reusable([part], [utterances_jsonl], f"keeping {speaker}'s offsets"):
            entries += _read_jsonl(part)
            continue
        base = audio_dir / f"{speaker}.wav"
        rows = concatenate(by_speaker[speaker], base, target_sec)
        # The files are numbered from the name upwards, so everything from the
        # count this run wrote onwards is what a wider previous run left.
        index = len({row["path"] for row in rows})
        while (surplus := _joined_path(base, index)).exists():
            surplus.unlink()
            logger.info(f"{surplus} is past what {speaker} now needs; removed")
            index += 1
        write_manifest(rows, part)
        entries += rows

    # 7. Split off the valid speakers and write the two manifests. Both or
    #    neither: a kill between them leaves train.jsonl describing a corpus
    #    valid.jsonl was never held out of.
    #    The selection is an input here alongside the offsets built out of it,
    #    because the loop above only rewrites an entries file for a speaker the
    #    selection still holds. Tighten a cutoff until a speaker keeps nothing
    #    and their file is left where it lay; tighten it until nothing at all
    #    survives and not one entries file moves. Measured against those alone
    #    the split would find every input unmoved and go on naming utterances
    #    that had just been excluded -- with utterances.jsonl beside it saying
    #    otherwise, the alignment below kept for the same reason, and the run
    #    reporting success over the manifest the operator had rejected.
    written_entries = sorted((out_dir / "entries").glob("*.jsonl"))
    if not _reusable(
        [train_manifest, valid_manifest],
        [utterances_jsonl, *written_entries],
        "keeping the split they hold",
    ):
        train, valid = split_by_speaker(entries, valid_hours)
        write_manifest(train, train_manifest)
        write_manifest(valid, valid_manifest)

    # 8. Align. prepare_data's align() already streams into a .partial that
    #    --resume picks up, renames its output in only once a pass has finished
    #    -- the sharded merge included -- and starts over rather than resuming
    #    a leftover written under a different --align-shards, so it is reused
    #    here rather than reimplemented. The segmenter is not an
    #    option: this manifest is Japanese, and "whitespace" over a language
    #    written without spaces returns one word per utterance -- the aligner
    #    emits a single span, the loader finds no cut point, and the voice
    #    prompt quietly comes from the utterance being predicted.
    #    That skipping is on the output's name, and --resume continues whatever
    #    the .partial holds without asking which manifest produced it, so an
    #    alignment older than the manifest it claims to align has to be thrown
    #    away here rather than kept or continued -- it describes rows the split
    #    above has since rewritten, and nothing in the file says so.
    for aligned, manifest in ((train_aligned, train_manifest), (valid_aligned, valid_manifest)):
        produced = [aligned, aligned.with_suffix(".partial")]
        produced += sorted(aligned.parent.glob(f"{aligned.stem}.shard*"))
        for leftover in produced:
            if _stale(leftover, [manifest]):
                logger.info(f"{leftover} was aligned from an older {manifest.name}; discarding it")
                leftover.unlink()
    #    And the question stage 0 could only guess at is asked again here,
    #    where every manifest has been written and every stale alignment
    #    discarded, so that a hole left up there costs a clean stop rather than
    #    a CalledProcessError out of the aligner's subprocess hours from now.
    require_segmenter_to_align(train_aligned)
    align(
        train_manifest,
        train_aligned,
        align_shards,
        align_model,
        "training manifest",
        segmenter="japanese",
    )
    require_segmenter_to_align(valid_aligned)
    align(valid_manifest, valid_aligned, 1, align_model, "valid manifest", segmenter="japanese")
    logger.info(f"Done. Training on {train_aligned.resolve()} and {valid_aligned.resolve()}")


if __name__ == "__main__":
    app()
