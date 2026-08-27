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
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import jiwer
import numpy as np
import sphn
import typer
from huggingface_hub import hf_hub_download

from pocket_tts.utils.text_normalization import normalize_japanese

logger = logging.getLogger("prepare_moespeech")
app = typer.Typer(pretty_exceptions_show_locals=False)

SELECTION_SEED = 0  # so --order random is still reproducible across re-runs
DATASET_REPO = "ayousanz/moe-speech-plus"
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


def read_annotation(path: Path) -> dict | None:
    """One clip's annotation, or None if it cannot stand as a training example.

    Every clip ships two independent ASR transcriptions and no manual one, so
    there is nothing to check either against except the other. Their mutual CER
    is what this returns as `cer`: not a measure of the audio, but of where the
    two systems disagree, which is the best available evidence that the text is
    wrong. Both transcriptions are kept so a later stage can choose between
    them; `transcript` names the anime-whisper one because it is the reference
    side of that CER, so the number reported is the disagreement measured
    against exactly the string the manifest would carry.

    A field that is absent -- or an empty transcription, which for jiwer is
    worse than absent, since CER divides by the reference's length and raises
    on an empty one -- drops the clip rather than being defaulted. A defaulted
    duration reaches the manifest as a window the audio does not contain, and
    the loader reads silence and trains on it as speech, which nothing
    downstream can detect.

    A JSON that is not an object at all is dropped the same way. The scan that
    calls this picks up every `.json` under the extract root, and a character
    zip may well ship an index or a metadata file among the per-clip ones; a
    list or a bare number parses without complaint and then has no `.get`, so
    without this check one such file raises AttributeError -- not a decode
    error, not caught by the caller -- and ends a pass 40 minutes in.

    `id`, `speaker` and `wav` come from the path because the JSON carries none
    of them: the audio is the file's sibling and the speaker is the directory
    the zip was unpacked into.
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
    return {
        "id": path.stem,
        "speaker": path.parent.name,
        "wav": path.with_suffix(".wav"),
        "duration": float(duration),
        "transcript": whisper,
        "whisper": whisper,
        "parakeet": parakeet,
        "cer": jiwer.cer(whisper, parakeet),
        "mos": float(mos),
    }


def _scan_annotations(root: Path, skipped: Counter) -> Iterator[dict]:
    """Every usable annotation under `root`, normalized, in path order.

    The probe and the selection walk the same 400,000 files and must agree on
    which of them are annotations at all, or the retention table an operator
    reads their cutoffs off describes a different corpus from the one those
    cutoffs are then applied to. So both walk through here -- and so, for the
    same reason, does the normalization. Normalizing can empty a transcript
    that was not empty, and a clip with no text left has to be dropped; done on
    the selection's side alone, that drop would land after the probe had
    already counted the clip and promised it in the retention table. The table
    would then over-promise, which is the one thing it may not do: it is the
    only artifact an operator reads when choosing cutoffs.

    Path order is part of the contract, not an accident of `rglob`. The stage
    after this one concatenates a speaker's clips into pseudo-long recordings
    and writes a manifest of offsets into them; taken in filesystem order, a
    re-run after a kill builds a different recording and the offsets no longer
    describe the audio.

    The scan is `rglob` because the extract root holds one directory per
    character; `glob` would report an empty corpus for the real data while
    passing on any fixture that keeps its JSON flat.

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
    for path in sorted(root.rglob("*.json")):
        try:
            row = read_annotation(path)
        except (OSError, ValueError, TypeError) as e:
            skipped["unreadable"] += 1
            logger.debug(f"{path}: unreadable ({e})")
            continue
        if row is None:
            skipped["incomplete"] += 1
            continue
        row["transcript"] = normalize_japanese(row["transcript"])
        if not row["transcript"]:
            skipped["blank"] += 1
            logger.debug(f"{path}: nothing left of the transcript after normalization")
            continue
        yield row


def select_utterances(root: Path, max_cer: float, min_mos: float) -> Iterator[dict]:
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
    never quite becomes intelligible. It happens in `_scan_annotations` rather
    than here so that the clips normalization empties are dropped from the
    probe's counts too -- see there -- and the two cutoffs are the only thing
    this function decides.
    """
    # The counts go nowhere here on purpose: the probe already reported them to
    # the operator over this same tree, and repeating them would read as a
    # second, different set of unusable files.
    skipped: Counter = Counter()
    for row in _scan_annotations(root, skipped):
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


def probe_utterances(root: Path) -> dict:
    """Measure every annotation under `root`. Decide nothing about any of them.

    The returned dict is what `probe.json` holds: how many utterances there
    are, the distribution of clip length, mutual CER and speechMOS, and a table
    of how many clips and hours survive each candidate pair of cutoffs. No
    threshold is applied here and none is recommended -- this exists so the
    next stage's cutoffs come from the corpus rather than from a guess.

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
    rows = [(r["duration"], r["cer"], r["mos"]) for r in _scan_annotations(root, skipped)]
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
    `target_sec` -- what that bounds is the alignment pass, which holds a whole
    file at once. Clips themselves are never cut, so one longer than the target
    gets a file to itself. Offsets restart at zero in each file, being offsets
    into the file they name.

    Clips are never joined across speakers, and a mixed list raises here rather
    than being grouped: the caller groups, and everything after this treats a
    cut anywhere inside one of these files as one voice -- the loader takes one
    side of the cut as the voice prompt for the other, so a file holding two
    speakers would teach the model that the prompt does not decide the voice.

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

    for u in utterances:
        wav, sr = sphn.read(str(u["wav"]))
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            # Concatenating these would play one of them at the wrong speed and
            # put every offset after it out by a factor nothing reports.
            raise ValueError(f"{u['wav']}: {sr} Hz among {sample_rate} Hz clips")
        n_samples = wav.shape[1]
        target_samples = round(target_sec * sample_rate)
        if parts and cursor + n_samples > target_samples:
            _write_joined(parts, sample_rate, _joined_path(out_wav, written))
            parts, cursor, written = [], 0, written + 1
        duration = n_samples / sample_rate
        if abs(duration - u["duration"]) > 0.1:
            # Not an error and not corrected: the audio is what was written, so
            # the audio is what the manifest describes. Worth seeing, though --
            # it means the corpus's own durations cannot be trusted elsewhere.
            logger.debug(f"{u['wav']}: annotated {u['duration']:.2f}s, audio {duration:.2f}s")
        entries.append(
            {
                "id": u["id"],
                "speaker": u["speaker"],
                "transcript": u["transcript"],
                "path": str(_joined_path(out_wav, written)),
                "start": round(cursor / sample_rate, 3),
                "duration": round(duration, 3),
            }
        )
        parts.append(wav)
        cursor += n_samples

    if parts:
        _write_joined(parts, sample_rate, _joined_path(out_wav, written))
    return entries
