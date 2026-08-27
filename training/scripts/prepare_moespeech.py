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
import logging
import os
import random
import shutil
from pathlib import Path

import typer
from huggingface_hub import hf_hub_download

logger = logging.getLogger("prepare_moespeech")
app = typer.Typer(pretty_exceptions_show_locals=False)

SELECTION_SEED = 0  # so --order random is still reproducible across re-runs
DATASET_REPO = "ayousanz/moe-speech-plus"


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
