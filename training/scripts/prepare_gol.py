"""Turn GOL's per-game tars into an aligned training manifest.

GOL (`midralab/gol-dataset`) is 10,654 hours of 48 kHz Japanese visual-novel
speech -- 7,405,094 utterances across 19,349 speakers and 596 games, 7 TB in
all, published as one tar per game. That is five times MoeSpeech, and shaped
differently in the two ways that matter here: the download unit is a whole
work rather than one character, and the transcripts come from a single ASR
pass, so the mutual-CER filter that carried MoeSpeech has no counterpart.

This script follows the same stages as prepare_moespeech, and the same rule:
every stage skips work whose output already exists, and writes partial output
to a .partial file that is renamed on completion. The instance it runs on is
preemptible, so being killed and re-run must always be safe.
"""

import csv
import logging
import os
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger("prepare_gol")

DATASET_REPO = "midralab/gol-dataset"
EXTRACT_MARKER = ".complete"  # beside the directory, not in it


def select_games(metadata_tsv: Path, hours: float) -> list[dict]:
    """The games whose combined duration first reaches `hours`.

    Reads the dataset's metadata.tsv, which carries the game, speaker, text,
    path and duration of every one of the 7.4 million utterances -- 1.68 GB
    that answers "which tars should I download" without fetching a byte of
    audio, the same job info.csv does for MoeSpeech.

    The unit is what differs. MoeSpeech publishes one zip per character, so
    selecting speakers and selecting downloads were the same act; a GOL tar is
    a whole game, median 17.4 hours and 35 speakers, so a game is taken or left
    whole and the speakers come along with it. Filtering speakers is a later
    stage's job, on data already on disk.

    Largest-first for the same reason as there, only more so: a tar averages
    11 GB, and 1,000 hours is 18 of them taken this way -- 200 GB and 2,451
    speakers. Ties break on game_id so that an interrupted run asks for the
    same tars when it resumes.

    Accumulation streams: 596 games and 19,349 speakers fit in memory, the
    7.4 million rows do not.
    """
    seconds: defaultdict[str, float] = defaultdict(float)
    speakers: defaultdict[str, set[str]] = defaultdict(set)
    utterances: Counter[str] = Counter()
    with open(metadata_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            game = row["game_id"]
            seconds[game] += float(row["duration"])  # the column is seconds
            speakers[game].add(row["speaker"])
            utterances[game] += 1

    games = [
        {
            "game_id": game,
            "hours": total / 3600,
            "speakers": len(speakers[game]),
            "utterances": utterances[game],
        }
        for game, total in seconds.items()
    ]
    games.sort(key=lambda g: (-g["hours"], g["game_id"]))

    chosen, taken = [], 0.0
    for game in games:
        if taken >= hours:
            break
        chosen.append(game)
        taken += game["hours"]
    return chosen


def download_games(ids: list[str], dest: Path, repo: str = DATASET_REPO) -> list[Path]:
    """Fetch one tar per game, skipping those already present.

    This and `extract_game` below are `prepare_moespeech.download_characters`
    and `extract_character` with the archive format changed, and they are
    copied rather than shared. Sharing them would mean handing each an archive
    parameter -- a suffix on one side and a callable on the other, since
    `zipfile.ZipFile(p)` and `tarfile.open(p)` are not interchangeable and only
    the tar side takes the `filter=` that keeps a member from being written
    outside `dest_root`. That is a caller-supplied extraction policy in the one
    place it must not be optional, bought for about ten lines. The guarantees
    are what actually carry across, and they live in these docstrings, which
    are corpus-specific anyway: 11 GB of a whole work here against 30 GB of one
    character there. A shared pair would have kept the code and lost the
    reasons.

    huggingface_hub downloads to a cache and only writes the final path once
    the transfer completes, so a kill mid-download leaves an incomplete file in
    the cache, not here -- but a kill mid-copy into `dest` would leave an
    incomplete file right here, which is exactly what this function must never
    mistake for "already have it". So the copy lands at `<game_id>.tar.partial`
    first and is renamed into place only once it is whole; `os.replace` is
    atomic on both POSIX and Windows, so there is no window where a reader
    could see a half-renamed file either. At a median 11 GB per tar that copy
    is minutes long, and it is one of 607 of them.
    """
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for game_id in ids:
        local = dest / f"{game_id}.tar"
        if local.exists():
            logger.info(f"{local.name} already present, skipping")
        else:
            fetched = hf_hub_download(repo, f"{game_id}.tar", repo_type="dataset")
            partial = dest / f"{game_id}.tar.partial"
            shutil.copyfile(fetched, partial)
            os.replace(partial, local)
        paths.append(local)
    return paths


def extract_game(tar_path: Path, dest_root: Path) -> Path:
    """Unpack one game's tar into `dest_root/<game_id>/`, once.

    Eleven gigabytes takes many minutes to unpack and the instance can be
    reclaimed in the middle of it, so a re-run has to tell a finished game from
    an interrupted one. The directory existing does not answer that: a
    half-extracted game has a directory too, and trusting it would drop every
    utterance the kill arrived before while looking exactly like success.

    Completion is therefore recorded explicitly, as `<game_id>.complete`
    written only once the last member is on disk. It sits beside the directory
    rather than inside it so the directory holds corpus files and nothing else,
    and callers can walk it without filtering. A directory without that record
    is deleted and unpacked again rather than resumed -- the member the kill
    interrupted is likely truncated, and the members it never reached are
    missing, neither of which is visible from the outside.

    The tar carries its own `<game_id>/` at the top, so what lands is
    `dest_root/<game_id>/<game_id>/<speaker>/`. The double nesting is left
    alone. Undoing it would mean rewriting 11 GB to produce a tree whose last
    directory is a speaker name, which is the shape that invites a later stage
    to take the speaker from the path; metadata.tsv states the speaker in its
    own column, and every scan below is `rglob`, so the depth costs nothing.

    `filter="data"` is not tidiness. These tars come from a third-party repo,
    and without it a member named `../../something`, or one with an absolute
    path, is written where it says -- outside `dest_root` entirely, on a machine
    this pipeline runs unattended on. It is also the default from 3.14 on, so
    stating it keeps this stable across the interpreter as well.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    out = dest_root / tar_path.stem
    marker = dest_root / f"{tar_path.stem}{EXTRACT_MARKER}"
    if marker.exists() and out.is_dir():
        logger.info(f"{out.name} already extracted, skipping")
        return out
    if out.exists():
        logger.info(f"{out.name} was left half-extracted, unpacking it again")
        shutil.rmtree(out)
    marker.unlink(missing_ok=True)  # so a kill mid-unpack cannot leave it lying
    with tarfile.open(tar_path) as t:
        t.extractall(out, filter="data")
    marker.write_text("", encoding="utf-8")
    return out
