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
from pathlib import Path

import jiwer
import typer
from huggingface_hub import hf_hub_download

logger = logging.getLogger("prepare_moespeech")
app = typer.Typer(pretty_exceptions_show_locals=False)

SELECTION_SEED = 0  # so --order random is still reproducible across re-runs
DATASET_REPO = "ayousanz/moe-speech-plus"
EXTRACT_MARKER = ".complete"  # beside the directory, not in it
# The grids probe.json reports retention over. They are candidates to read a
# cutoff off, not cutoffs: nothing here filters anything.
CER_THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
MOS_THRESHOLDS = (0.0, 2.5, 3.0, 3.5, 4.0)


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

    The scan is `rglob` because the extract root holds one directory per
    character; `glob` would report an empty corpus for the real data while
    passing on any fixture that keeps its JSON flat.

    A file that cannot be read is counted and skipped rather than raising: one
    bad JSON in 400,000 must not end a pass that takes 40 minutes to reach it.
    Counted, though, and logged -- a pass that quietly returns 300,000 of
    400,000 clips looks exactly like a corpus that was only ever 300,000 long.
    Only the three measured numbers are retained per clip, not the row: the
    transcriptions of a whole corpus do not need to be in memory at once for
    this, and at this scale that is gigabytes.
    """
    rows: list[tuple[float, float, float]] = []
    unreadable = incomplete = 0
    for path in sorted(root.rglob("*.json")):
        try:
            row = read_annotation(path)
        except (OSError, ValueError, TypeError) as e:
            unreadable += 1
            logger.debug(f"{path}: unreadable ({e})")
            continue
        if row is None:
            incomplete += 1
            continue
        rows.append((row["duration"], row["cer"], row["mos"]))
    if unreadable or incomplete:
        logger.warning(
            f"skipped {unreadable + incomplete} of {len(rows) + unreadable + incomplete} "
            f"json files: {unreadable} unreadable, {incomplete} not usable as an annotation"
        )
    return {
        "count": len(rows),
        "unreadable": unreadable,
        "incomplete": incomplete,
        "hours": sum(d for d, _, _ in rows) / 3600,
        "duration": _distribution([d for d, _, _ in rows]),
        "cer": _distribution([c for _, c, _ in rows]),
        "mos": _distribution([m for _, _, m in rows]),
        "retention": _retention(rows),
    }
