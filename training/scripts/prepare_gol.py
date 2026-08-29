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

# _distribution rather than a second copy of it: these two scripts are one
# pipeline in two files, the survey in docs/ prints their percentiles in the
# same table, and two definitions of "median" would eventually disagree in a
# way nobody would think to check.
from training.scripts.prepare_moespeech import _distribution

logger = logging.getLogger("prepare_gol")

DATASET_REPO = "midralab/gol-dataset"
EXTRACT_MARKER = ".complete"  # beside the directory, not in it
# The floors the retention table is reported over. They are candidates to read
# a cutoff off, not cutoffs: nothing here filters anything, and
# `select_speakers` takes both of its floors as required arguments so that no
# number can reach the corpus without somebody having read the table first.
UTTERANCE_FLOORS = (1, 2, 3, 5, 10, 20, 50, 100)
SECONDS_FLOORS = (0, 60, 300, 600, 1800, 3600, 7200)


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


def _speaker_totals(metadata_tsv: Path, game_ids: list[str]) -> dict[str, tuple[int, float, int]]:
    """Every speaker of `game_ids`, with the utterances, seconds and games they hold.

    One scan feeding both the probe and the selection, for the reason
    `prepare_moespeech._scan_annotations` is shared the same way: the retention
    table is only worth reading if the count it promises is the count the
    selection then delivers, and two scans that drift apart cannot promise
    that.

    Bounded to the games actually taken. The extract root accumulates across
    runs, so a later run asking for fewer games would otherwise go on measuring
    -- and selecting -- the speakers of the earlier one.

    The speaker comes from metadata.tsv's own column and never from a path.
    That it also names a directory is an artefact of how the tars are packed,
    and what `extract_game` leaves on disk is double-nested besides.

    7.4 million rows do not fit in memory. 19,349 speakers do, so the file is
    streamed and only the three totals are kept.
    """
    wanted = set(game_ids)
    utterances: Counter[str] = Counter()
    seconds: defaultdict[str, float] = defaultdict(float)
    games: defaultdict[str, set[str]] = defaultdict(set)
    with open(metadata_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            game = row["game_id"]
            if game not in wanted:
                continue
            speaker = row["speaker"]
            utterances[speaker] += 1
            seconds[speaker] += float(row["duration"])  # the column is seconds
            games[speaker].add(game)
    return {s: (n, seconds[s], len(games[s])) for s, n in utterances.items()}


def select_speakers(
    metadata_tsv: Path, game_ids: list[str], min_utterances: int, min_seconds: float
) -> list[str]:
    """The speakers worth keeping, out of the games this run took.

    Both floors are required rather than defaulted, for the same reason the
    MoeSpeech cutoffs are: the right values are a property of this corpus, they
    are read off `probe_speakers`'s retention table, and a number written here
    now would afterwards be indistinguishable from a measurement.

    A speaker has to clear both, because the two reject different things.
    `min_utterances` rejects the speaker who cannot be evaluated at all -- the
    protocol clones a voice from one utterance and synthesizes another, so a
    speaker with a single clip has nothing to hold out and `concatenate` has
    nothing to join it to; 3,922 of GOL's 19,349 are in exactly that state.
    `min_seconds` rejects the bit part, who has a dozen lines and forty seconds
    of voice inside them. Neither floor catches the other's case, and either
    one alone lets its own through.

    Both comparisons are inclusive, matching the retention table exactly, so
    the count an operator read there is the count they get.

    The result is sorted. `speakers.json` is written out of it and read back on
    the re-run after a preemption, and the held-out split is taken off that
    file; the order 7.4 million rows happen to arrive in is not an order, and a
    resumed run that held out a different set of speakers would make the
    validation loss incomparable across the kill.
    """
    totals = _speaker_totals(metadata_tsv, game_ids)
    return sorted(
        speaker
        for speaker, (utterances, seconds, _games) in totals.items()
        if utterances >= min_utterances and seconds >= min_seconds
    )


def _speaker_retention(totals: list[tuple[int, float, int]]) -> list[dict]:
    """How many speakers, and how much audio, each pair of floors would leave.

    Both columns are reported because on this corpus they answer opposite
    questions about the same number. A one-hour floor keeps 2,095 of 19,349
    speakers, which reads as throwing the corpus away, and 9,493 of 10,654
    hours, which is keeping 89% of it. Either column on its own would be read
    as a verdict on the other.
    """
    total_seconds = sum(s for _, s, _ in totals)
    table = []
    for min_utterances in UTTERANCE_FLOORS:
        for min_seconds in SECONDS_FLOORS:
            kept = [s for n, s, _ in totals if n >= min_utterances and s >= min_seconds]
            table.append(
                {
                    "min_utterances": min_utterances,
                    "min_seconds": min_seconds,
                    "kept": len(kept),
                    "speaker_fraction": len(kept) / len(totals) if totals else 0.0,
                    "hours": sum(kept) / 3600,
                    "hours_fraction": sum(kept) / total_seconds if total_seconds else 0.0,
                }
            )
    return table


def probe_speakers(metadata_tsv: Path, game_ids: list[str]) -> dict:
    """Measure the speakers of `game_ids`. Decide nothing about any of them.

    This exists because 19,349 is a trap. The median speaker in the whole
    corpus has 0.6 minutes of audio, 3,922 have a single utterance, and the
    2,095 with an hour or more hold 89% of it. A held-out split designed
    around the headline count would be designed around speakers that cannot be
    evaluated and cannot be concatenated. So this reports the shape and applies
    nothing: the floors come out of the table below, not out of this file.

    The distribution is per speaker and in minutes -- 0.6 is a number an
    operator can read, 0.01 hours is not -- and the medians are the point. The
    mean minutes per speaker on this corpus describes a speaker who does not
    exist, because the same 2,095 that hold the audio also hold the mean.

    `speakers_in_more_than_one_game` measures rather than assumes the one thing
    about speaker identity this pipeline cannot check for itself. The id is
    taken from the metadata column and treated as the speaker; if two games
    reuse an id, their two characters merge into one, and `concatenate` would
    join them into a file the loader reads as a single voice -- taking one side
    of a cut as the voice prompt for the other, and so teaching the model that
    the prompt does not decide the voice. The tree nests speakers under
    `<game_id>/`, which is a hint that ids may be game-local, and nothing in
    the dataset settles it. It is cheap to count and expensive to be wrong
    about, so it is counted and reported to the operator.

    Bounded to the same games as `select_speakers`, through the same scan, so
    that the table and the selection describe one corpus.
    """
    totals = _speaker_totals(metadata_tsv, game_ids)
    rows = list(totals.values())
    spanning = sum(1 for _, _, games in rows if games > 1)
    if spanning:
        logger.warning(
            f"{spanning} of {len(rows)} speaker ids appear in more than one game -- if these "
            f"are different characters sharing an id, joining their clips would put two voices "
            f"in one file, and everything downstream reads such a file as one speaker"
        )
    return {
        "speakers": len(rows),
        "utterances": sum(n for n, _, _ in rows),
        "hours": sum(s for _, s, _ in rows) / 3600,
        "single_utterance_speakers": sum(1 for n, _, _ in rows if n == 1),
        "speakers_in_more_than_one_game": spanning,
        "utterances_per_speaker": _distribution([float(n) for n, _, _ in rows]),
        "minutes_per_speaker": _distribution([s / 60 for _, s, _ in rows]),
        "retention": _speaker_retention(rows),
    }
