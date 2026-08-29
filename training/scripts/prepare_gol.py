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
from collections import Counter, defaultdict
from pathlib import Path


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
