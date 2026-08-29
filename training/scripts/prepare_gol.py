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
import json
import logging
import os
import shutil
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Annotated

import typer
from huggingface_hub import hf_hub_download

# The same normalization the MoeSpeech path applies, so the two corpora reach
# the tokenizer and the loader as one distribution rather than two.
from pocket_tts.utils.text_normalization import normalize_japanese

# The aligner and its skip predicate, shared with the MoeSpeech script and with
# the English path before it: align() already streams into a .partial that
# --resume picks up and renames its output in only once a pass has finished.
from training.scripts.prepare_data import align, already_aligned

# Borrowed rather than copied: these two scripts are one pipeline in two files.
# _distribution because the survey in docs/ prints their percentiles in the same
# table and two definitions of "median" would eventually disagree in a way
# nobody would think to check; _read_jsonl and write_manifest because they are
# the two halves of one format and a second reader could drift from the writer;
# and split_by_speaker because holding speakers out of the two corpora
# separately is the one thing the merge below exists to prevent.
#
# The rest are main's, and they are the stage machinery rather than the corpus:
# _reusable, _stale and _same_cutoffs are what "skip work whose output is
# already there" means, and a second definition of when an artifact is out of
# date is the one kind of drift neither script could survive; _write_json and
# _joined_path are the shapes of files the other script's readers also open;
# concatenate refuses a mixed-speaker list, which is the guard stage 7 leans on;
# the two segmenter checks answer a question about this machine and not about
# either corpus; and KANA_ALIGN_MODEL is the vocabulary the aligner has to have,
# which is a fact about Japanese.
from training.scripts.prepare_moespeech import (
    KANA_ALIGN_MODEL,
    _distribution,
    _joined_path,
    _percentile,
    _read_jsonl,
    _reusable,
    _same_cutoffs,
    _stale,
    _write_json,
    concatenate,
    require_japanese_segmenter,
    require_segmenter_to_align,
    split_by_speaker,
    write_manifest,
)

logger = logging.getLogger("prepare_gol")
app = typer.Typer(pretty_exceptions_show_locals=False)

DATASET_REPO = "midralab/gol-dataset"
EXTRACT_MARKER = ".complete"  # beside the directory, not in it
# The floors the retention table is reported over. They are candidates to read
# a cutoff off, not cutoffs: nothing here filters anything, and
# `select_speakers` takes both of its floors as required arguments so that no
# number can reach the corpus without somebody having read the table first.
UTTERANCE_FLOORS = (1, 2, 3, 5, 10, 20, 50, 100)
SECONDS_FLOORS = (0, 60, 300, 600, 1800, 3600, 7200)
# The smallest validation set that can say when to stop. Phase 1 held out one
# speaker; see `split_across_corpora` for what that cost and why this is a hard
# floor rather than a target.
MIN_VALID_SPEAKERS = 20
# The cutoffs the score retention table is reported over, and the
# normalizations it is reported under. Quantiles rather than absolute scores
# because an alignment log-probability has no scale known before the corpus is
# measured; see `_score_retention`. All three normalizations, because which one
# the distribution supports is the question this stage exists to ask and not
# one it may answer; see `probe_scores`.
SCORE_QUANTILES = (0, 1, 5, 10, 25, 50, 75, 90)
NORMALIZATIONS = ("raw", "per_frame", "per_token")


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

    Largest-first for the same reason as there, only more so: 1,000 hours is 18
    tars taken this way, and 2,451 speakers. Ties break on game_id so that an
    interrupted run asks for the same tars when it resumes.

    Those 18 are **660 GB**, not the 198 that 18 times the 11 GB median would
    suggest. The median is the wrong statistic for a largest-first selection --
    the tars this picks average about 37 GB. The corpus is 7,019 GB over 10,654
    hours, so 0.66 GB per hour is the number to size a disk by, and `main`'s
    docstring carries what the peak becomes once the cache, the tars, the
    extracted tree and the joined audio all exist at once.

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


def _members_inside(archive: tarfile.TarFile, dest_root: Path) -> Iterator[tarfile.TarInfo]:
    """`archive`'s members, refusing any that would land outside `dest_root`.

    This is the containment `filter="data"` provides, done here rather than
    left to that keyword because the keyword does not exist before 3.10.12 and
    pyproject asks for ">= 3.10,<3.15". On 3.10.0 through 3.10.11 -- which is
    what a rented image can perfectly well come with -- passing it is a
    TypeError on every single extraction, so the run would download 660 GB and
    unpack none of it, and it would fail there and not here. The check
    therefore runs on every interpreter, and the keyword is added on top of it
    where it exists rather than being the only thing between a third-party tar
    and the filesystem.

    Refused rather than skipped. A tar holding a member that names somewhere
    else is not a tar this corpus can be built out of, and a game quietly short
    of clips is the state the completion marker exists to prevent; the caller
    writes no marker when this raises, so the next run unpacks it again.

    Stricter than `data` in one place: only regular files and directories go
    through. A GOL tar holds wavs, and a symlink or a device node in one is a
    thing to refuse outright rather than to resolve carefully.
    """
    root = Path(dest_root).resolve()
    for member in archive:
        if not (member.isfile() or member.isdir()):
            raise tarfile.ExtractError(
                f"{member.name!r} is neither a file nor a directory. These tars hold wavs, "
                "and this one is unpacked unattended on a rented box."
            )
        target = (root / member.name).resolve()
        if target != root and root not in target.parents:
            raise tarfile.ExtractError(
                f"{member.name!r} would be written to {target}, which is outside {root}."
            )
        yield member


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

    Containment is not tidiness. These tars come from a third-party repo, and
    unfiltered, a member named `../../something`, or one with an absolute path,
    is written where it says -- outside `dest_root` entirely, on a machine this
    pipeline runs unattended on. `_members_inside` is what refuses that, and it
    refuses it on every interpreter; `filter="data"` is asked for as well
    wherever `extractall` has it, which is 3.10.12 on and the default from 3.14,
    for the mode bits and ownership it strips that this does not.
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
        # `tarfile.data_filter` exists exactly where the `filter=` keyword does,
        # so it is what the interpreter is asked through -- and it is asked
        # about, not assumed, because passing the keyword where it is absent is
        # a TypeError rather than a warning.
        if hasattr(tarfile, "data_filter"):
            t.extractall(out, _members_inside(t, out), filter="data")
        else:
            t.extractall(out, _members_inside(t, out))
    marker.write_text("", encoding="utf-8")
    return out


def _speaker_key(game_id: str, speaker: str) -> str:
    """The identity everything downstream treats as one voice.

    Ruling G9: the game is part of it. Measured over the whole metadata, 3,522
    of the 19,349 speaker ids appear in more than one game and those ids hold
    61% of the corpus; the widest spans 115 games at 457 utterances of 1.8
    seconds. Five short lines per game is not a prolific actor with a role, it
    is a bucket for unnamed characters or system lines -- and an id that
    *might* mean two people is enough to defeat `concatenate`'s cross-speaker
    guard, because that guard compares the label and not the voice. Two
    characters merged under one label become one joined file, and the loader
    takes one side of a cut in that file as the voice prompt for the other.

    Keying them apart costs 748 hours of 10,654 (7%) at a two-utterance floor
    and yields 2,819 speakers of an hour or more against 2,095 -- 35% more.
    Speaker diversity is the axis the intermediate run exists to test, so the
    safe option is also the better one.

    A composite string rather than a tuple, because `concatenate` and
    `split_by_speaker` are reused from prepare_moespeech and treat the speaker
    as an opaque label; a string keeps them working untouched and keeps the
    game readable in the manifest.
    """
    return f"{game_id}:{speaker}"


def _speaker_totals(
    metadata_tsv: Path, game_ids: list[str]
) -> tuple[dict[str, tuple[int, float]], dict[str, int]]:
    """Every speaker of `game_ids` with the utterances and seconds they hold,
    and, beside it, how many games each bare id turned up in.

    Keyed by `_speaker_key`, so one id used by two games is two speakers here.
    The second return value is what says how often that happens; it is counted
    over bare ids, since a composite key spans one game by construction.

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

    7.4 million rows do not fit in memory. The 19,349 ids and the 30,193 keys
    they make between them do, so the file is streamed and only the totals
    are kept.
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
            key = _speaker_key(game, row["speaker"])
            utterances[key] += 1
            seconds[key] += float(row["duration"])  # the column is seconds
            games[row["speaker"]].add(game)  # the bare id, which is the thing that spans
    totals = {key: (n, seconds[key]) for key, n in utterances.items()}
    return totals, {speaker: len(g) for speaker, g in games.items()}


def select_speakers(
    metadata_tsv: Path, game_ids: list[str], min_utterances: int, min_seconds: float
) -> list[str]:
    """The speakers worth keeping, out of the games this run took.

    Each one is a `_speaker_key`, so what comes back is `<game_id>:<speaker>`
    and one id shared by two games is two entries. See that function for why.

    Both floors are required rather than defaulted, for the same reason the
    MoeSpeech cutoffs are: the right values are a property of this corpus, they
    are read off `probe_speakers`'s retention table, and a number written here
    now would afterwards be indistinguishable from a measurement.

    A speaker has to clear both, because the two reject different things.
    `min_utterances` rejects the speaker who cannot be evaluated at all -- the
    protocol clones a voice from one utterance and synthesizes another, so a
    speaker with a single clip has nothing to hold out and `concatenate` has
    nothing to join it to; 3,922 of GOL's 19,349 ids are in exactly that
    state, and keying them apart by game can only make more of them.
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
    totals, _spanning = _speaker_totals(metadata_tsv, game_ids)
    return sorted(
        speaker
        for speaker, (utterances, seconds) in totals.items()
        if utterances >= min_utterances and seconds >= min_seconds
    )


def _speaker_retention(totals: list[tuple[int, float]]) -> list[dict]:
    """How many speakers, and how much audio, each pair of floors would leave.

    Both columns are reported because on this corpus they answer opposite
    questions about the same number. A one-hour floor keeps 2,819 of the
    30,193 speakers, which reads as throwing the corpus away, and 8,745 of
    10,654 hours, which is keeping 82% of it. Either column on its own would
    be read as a verdict on the other.
    """
    total_seconds = sum(s for _, s in totals)
    table = []
    for min_utterances in UTTERANCE_FLOORS:
        for min_seconds in SECONDS_FLOORS:
            kept = [s for n, s in totals if n >= min_utterances and s >= min_seconds]
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
    corpus has 0.6 minutes of audio, 3,922 ids have a single utterance, and
    once the ids are keyed apart by game the 2,819 speakers holding an hour or
    more hold 82% of the audio. A held-out split designed around the headline
    count would be designed around speakers that cannot be evaluated and
    cannot be concatenated. So this reports the shape and applies nothing: the
    floors come out of the table below, not out of this file.

    The distribution is per speaker and in minutes -- 0.6 is a number an
    operator can read, 0.01 hours is not -- and the medians are the point. The
    mean minutes per speaker on this corpus describes a speaker who does not
    exist, because the same few thousand that hold the audio hold the mean.

    A speaker is a `_speaker_key`, so `speakers` counts `<game_id>:<speaker>`
    pairs and not bare ids -- 30,193 against 19,349 on the real corpus. Both are
    reported, with `ids_spanning_games` between them, because the gap is the
    whole of ruling G9 and an operator reading only one of the two numbers
    would think the corpus had grown or the survey had been wrong.

    Bounded to the same games as `select_speakers`, through the same scan, so
    that the table and the selection describe one corpus.
    """
    totals, games_per_id = _speaker_totals(metadata_tsv, game_ids)
    rows = list(totals.values())
    spanning = sum(1 for count in games_per_id.values() if count > 1)
    if spanning:
        logger.warning(
            f"{spanning} of {len(games_per_id)} speaker ids appear in more than one game; "
            f"each game's use of an id is kept apart as <game_id>:<speaker>, so these "
            f"{len(rows)} speakers are more than there are ids"
        )
    return {
        "speakers": len(rows),
        "speaker_ids": len(games_per_id),
        "ids_spanning_games": spanning,
        "utterances": sum(n for n, _ in rows),
        "hours": sum(s for _, s in rows) / 3600,
        "single_utterance_speakers": sum(1 for n, _ in rows if n == 1),
        "utterances_per_speaker": _distribution([float(n) for n, _ in rows]),
        "minutes_per_speaker": _distribution([s / 60 for _, s in rows]),
        "retention": _speaker_retention(rows),
    }


def _clip_index(extract_root: Path, game_ids: list[str]) -> dict[tuple[str, str], list[Path]]:
    """Every wav of the games taken, keyed by the game and the file's own name.

    The metadata states each clip's path, and the obvious thing would be to
    join that onto the extract root and stat it. What that path is relative to
    is the trap: `extract_game` unpacks a tar that carries its own `<game_id>/`
    at the top into a directory of the same name, so a clip metadata.tsv calls
    `<game>/<speaker>/x.wav` is at `<root>/<game>/<game>/<speaker>/x.wav`
    (ruling G2). MoeSpeech cost this project a debugging round on exactly that
    kind of assumption, so no depth is assumed here at all: the tree is scanned
    with `rglob` and matched on the one component the metadata and the
    filesystem cannot disagree about, which is the file's own name.

    Names collide, though. Visual novels number a character's lines from one,
    so `0001.wav` under two speakers of one game is ordinary rather than odd,
    and a name-only match would hand both rows whichever clip was found first
    -- one speaker's transcript over the other's voice, in a manifest that
    parses, with every offset inside a real file and nothing downstream able to
    tell. So the value is every clip of that name, and the caller narrows a
    list of more than one by the rest of the path the metadata states -- and
    drops the row when that still leaves more than one, since which of them is
    this row's is then not a question the corpus answers. Kept sorted, so that
    what a `--verbose` run says about such a row is the same list on every
    re-run rather than whatever `rglob` happened to return.

    Bounded to `game_ids`, which bounds both the work and the memory: the
    extract root accumulates across runs, and the 700,000 paths of a
    1,000-hour run fit in memory where a walk per row would be 700,000 walks.
    A game not on disk yet contributes nothing rather than raising -- a run
    killed between tars leaves exactly that, and it is the ordinary state of
    the extract root rather than a corrupt one.
    """
    index: defaultdict[tuple[str, str], list[Path]] = defaultdict(list)
    for game in sorted(set(game_ids)):
        for wav in sorted((Path(extract_root) / game).rglob("*.wav")):
            index[(game, wav.name)].append(wav)
    return index


def gol_utterances(
    metadata_tsv: Path, extract_root: Path, speakers: list[str], game_ids: list[str]
) -> Iterator[dict]:
    """The utterances of `speakers`, shaped the way `concatenate` reads them.

    This is the last stage before the joining and does none of it. GOL's median
    clip is 4.55 seconds against MoeSpeech's 5.46, and the loader keeps a
    second of audio on either side of the cut it makes, so an unjoined clip
    leaves under three seconds of target audio -- concatenation matters more
    here than it did there. But `prepare_moespeech.concatenate` already does
    it, is already tested, and already refuses a mixed-speaker list, so what
    this yields is its input: `id`, `speaker`, `wav`, `duration` and
    `transcript`, which are exactly the keys it reads. Grouping by speaker
    before that call is the caller's job (ruling G4); this yields across all of
    them.

    A speaker is a `_speaker_key`, so `speakers` holds `<game_id>:<speaker>`
    exactly as `select_speakers` returned it, and it is read from
    metadata.tsv's own column rather than from any directory name -- see
    `_clip_index` for where the audio comes from, and `read_annotation` in the
    MoeSpeech script for what taking a speaker off a path costs.

    A row has to clear both bounds. `game_ids` is what a smaller re-run means
    against an extract root that still holds a larger one's games, and
    `speakers` carries `select_speakers`'s floors, which are the whole point of
    the stage before this one: a GOL tar is a whole work and its median 35
    speakers come along with it, most of them below any floor worth setting.

    The transcript is normalized with the same function `align_data.py` applies
    under --segmenter japanese and `prepare_ja_text.py` fits the tokenizer
    through. Those three strings have to be one distribution or the run
    degrades with nothing reporting why. A row left with no text is dropped --
    1,055 of GOL's rows are empty as written and normalization empties more --
    and dropped after normalizing rather than before, since a text of one
    ideographic space is not empty until NFKC has folded it to a space.

    A row whose wav is not on disk is dropped as well. metadata.tsv describes
    all 7,405,094 files while the tars actually taken hold a subset, so this is
    the normal case for the corpus at large and the sign of an interrupted
    extraction inside the games taken. `concatenate` would skip such a clip
    anyway, but only after this had already promised it.

    And so is a row the tree answers more than once, for the opposite reason:
    there the audio is on disk and it is which of it belongs to this row that
    the corpus does not say. Both are counted and reported apart, because they
    mean different things about the tree -- one that a tar is short, the other
    that it holds two clips this walk cannot tell between.

    Order is metadata.tsv's, which is a file's order and so the same on every
    re-run. That matters past tidiness: `concatenate` lays clips down in the
    order it is handed them and writes offsets into the files it builds, so a
    resumed run that reordered them would produce a manifest describing audio
    that is no longer where it says. Nothing sorts here, because 7.4 million
    rows do not fit in memory to be sorted; they are streamed, and only the
    clip paths of the games taken are held.
    """
    wanted_games = set(game_ids)
    wanted_speakers = set(speakers)
    clips = _clip_index(extract_root, game_ids)
    missing = ambiguous = blank = 0
    with open(metadata_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            game = row["game_id"]
            if game not in wanted_games:
                continue
            speaker = _speaker_key(game, row["speaker"])
            if speaker not in wanted_speakers:
                continue
            stated = PurePosixPath(row["file_path"])
            # Narrowed by the speaker directory the metadata itself names, which
            # is a lookup against the corpus's own claim about where it put the
            # file and not a speaker read off a path.
            #
            # Always, and not only when the name is ambiguous. A name that
            # happens to be unique within its game is otherwise taken wherever
            # in that game it sits, so a clip packed under another character's
            # directory is handed to this row -- that character's voice under
            # this row's transcript, in a manifest that parses, with every
            # offset inside a real file and nothing downstream able to see it.
            # The lookup costs the same either way, and what it turns a wrong
            # clip into is one more of the drops this walk already counts.
            tail = "/" + "/".join(stated.parts[-2:])
            found = [p for p in clips.get((game, stated.name), []) if p.as_posix().endswith(tail)]
            if not found:
                missing += 1
                logger.debug(f"{row['file_path']}: no such wav under {Path(extract_root) / game}")
                continue
            if len(found) > 1:
                # Two paths under one game answer to the same stated tail, and
                # matching the whole stated path instead is not the way out:
                # ruling G2 is that no depth may be assumed, so the clip the
                # metadata calls `<game>/<speaker>/x.wav` is legitimately at any
                # depth. A work shipped on two discs unpacks `<game>/<speaker>/`
                # and `<game>/disc2/<speaker>/` into one tree, both hold a
                # `0001.wav`, and both end the way the metadata says.
                #
                # Taking the first of them would take `disc2/`, which sorts
                # first; taking any of them is a guess at which character is
                # speaking, made silently, in a row that parses. So the row is
                # dropped and counted, exactly as a clip that never arrived is.
                # Short of a clip is a state this walk reports and the pipeline
                # survives -- wrong about one is neither.
                ambiguous += 1
                logger.debug(
                    f"{row['file_path']}: {len(found)} wavs under {Path(extract_root) / game} "
                    f"answer to it ({', '.join(p.as_posix() for p in found)}); none is taken"
                )
                continue
            transcript = normalize_japanese(row["text"])
            if not transcript:
                blank += 1
                logger.debug(f"{row['file_path']}: nothing left of the text after normalization")
                continue
            yield {
                # The corpus's own path for the clip with the extension off,
                # rather than the bare stem: the entry `concatenate` builds
                # names the joined file and never this one, so this is the only
                # way back from a suspect manifest row to the audio it came
                # from, and 596 games number their lines the same way.
                "id": stated.with_suffix("").as_posix(),
                "speaker": speaker,
                "wav": found[0],
                "duration": float(row["duration"]),  # the column is seconds
                "transcript": transcript,
            }
    if missing or ambiguous or blank:
        # Once, at the end, rather than per row: there can be hundreds of
        # thousands of these, and the number that matters is how many clips the
        # manifest is short of what the selection promised.
        logger.warning(
            f"{missing} rows name a wav that is not under {extract_root}, {ambiguous} name one "
            f"the tree holds under more than one path, and {blank} have no text left after "
            "normalization; none of the three is in the manifest"
        )


def merge_entries(dirs: list[Path]) -> list[dict]:
    """Every corpus's offset rows as one list, which is the only shape the
    split may be taken over.

    This is the one point where GOL and MoeSpeech meet, and `split_by_speaker`
    is why they have to. It holds out whole speakers, and which speakers it
    holds out is a fact about the list it is handed. Split the two corpora
    separately and there are two held-out sets, neither of which the other
    corpus's training set knows anything about -- so a run trained on both
    would have trained on every speaker the other one held out, and both
    validation losses would be reporting voices the weights had already seen.
    That is precisely the failure the split exists to prevent, reached by
    splitting twice instead of once.

    A directory is an `<out_dir>/entries`, one `.jsonl` per speaker, exactly as
    stage 6 of either script leaves it: `write_manifest` wrote those files and
    `_read_jsonl` reads them back, so nothing here parses a format of its own.
    A `.partial` is not one of them -- `write_manifest` renames a file in only
    once all its lines are there, so what a preemption leaves behind is a
    speaker cut off in the middle, and the resumed run rewrites it under the
    real name.

    The file names are not read for anything. A speaker is the row's own
    column: `<game_id>:<speaker>` on the GOL side (ruling G9) and a bare
    character name on the MoeSpeech one. Taking it off the file name instead is
    the mistake ruling G2 is about, and here it would be worse than a wrong
    label, because a key holding a colon cannot be a Windows file name at all.

    Those two label spaces cannot collide: a GOL key always holds a colon, and
    a MoeSpeech name is a directory name out of a zip, which cannot hold one on
    the machine that unpacked it. That is stated rather than relied on. A label
    arriving from two corpora is refused, because everything downstream
    compares labels and none of it can see a voice -- two characters under one
    label go to the same side of the split as a single speaker, and the eval
    protocol, which clones a voice from one utterance and synthesizes another,
    would clone one of them and score the other against it.

    A named directory that holds no rows is refused rather than contributing
    nothing. Merging one corpus while the operator believes two are in is the
    silent version of that same bug: the manifest parses, the alignment runs,
    the training finishes, and the only sign is a speaker count nobody has a
    second number to compare against.

    Order is the order the corpora were named and, inside one, the file name's.
    Nothing downstream depends on it -- every row names its own file and the
    window inside it -- but a manifest that reorders itself between two runs
    over the same corpus cannot be diffed against the last one, and this script
    is re-run after every preemption.
    """
    merged: list[dict] = []
    first_seen: dict[str, tuple[int, Path]] = {}
    for position, directory in enumerate(dirs):
        directory = Path(directory)
        if not directory.is_dir():
            raise typer.BadParameter(
                f"{directory} is not a directory, so the corpus it names contributes nothing. "
                "Merging the rest would build a manifest out of fewer corpora than were asked "
                "for, and nothing after this could tell."
            )
        rows: list[dict] = []
        for part in sorted(directory.glob("*.jsonl")):
            rows += _read_jsonl(part)
        if not rows:
            raise typer.BadParameter(
                f"{directory} holds no utterances. Either that corpus's run has not reached "
                "its joining stage yet or it wrote them somewhere else; merging the rest "
                "would look exactly like a run over one corpus that succeeded."
            )
        for row in rows:
            was = first_seen.setdefault(row["speaker"], (position, directory))
            if was[0] != position:
                raise typer.BadParameter(
                    f"speaker {row['speaker']!r} is in both {was[1]} and {directory}. Every "
                    "guard from here down compares this label and none of them can see a "
                    "voice, so two characters under one label would be held out together, "
                    "trained together, and scored against each other."
                )
        merged += rows
        logger.info(
            f"{directory}: {len(rows)} utterances from "
            f"{len({row['speaker'] for row in rows})} speakers"
        )
    return merged


def split_across_corpora(
    entries: list[dict], valid_hours: float, minimum: int = MIN_VALID_SPEAKERS
) -> tuple[list[dict], list[dict]]:
    """`split_by_speaker` over the merged corpora, and a refusal when what it
    held out is too small to say when to stop.

    The split itself is `prepare_moespeech.split_by_speaker` untouched. It
    takes the speaker as an opaque label, so a `<game_id>:<speaker>` key and a
    bare character name are the same kind of thing to it, and reusing it is
    what makes one split over both corpora possible at all.

    What is added is the count of voices held out. Phase 1 held out one. Its
    validation loss bottomed at step 7,500 and had doubled by 15,000 while the
    samples over that same stretch kept getting better, and both were true:
    `train.py` takes the sample's voice prompt from a training batch, so the
    samples showed a voice the model had seen and the valid set was the one
    voice it had not. The model went on improving on the 27 voices it saw while
    getting worse on the single one it did not, and one speaker cannot tell
    that apart from noise. For a model whose whole point is cloning a voice it
    has never heard, the unseen number is the one that decides when to stop, so
    there have to be enough of them for it to mean something.

    Refusing rather than warning, because a warning is what phase 1 had:
    `split_by_speaker` already says out loud that it could not hold out the
    hours it was asked for, and the run went its full 15,000 steps over one
    voice regardless. This costs a stop before the aligner rather than a
    40,000-step run whose stopping criterion never existed.

    How many voices a `--valid-hours` buys is a property of the corpus and not
    of that number. The split takes the smallest speakers first, so it is those
    hours divided by the size of the smallest ones -- which the floors
    `select_speakers` was given decide. At a 60-minute floor every speaker is
    at least an hour and ten valid-hours is ten voices; with no floor, the
    1,000-hour selection's smallest speakers are seconds long and the same ten
    hours buy hundreds. Neither knob can be read without the other, so the
    message names both.
    """
    train, valid = split_by_speaker(entries, valid_hours)
    held_out = {entry["speaker"] for entry in valid}
    if len(held_out) < minimum:
        raise typer.BadParameter(
            f"the valid split holds {len(held_out)} speaker(s) and needs at least {minimum}: "
            f"{valid_hours}h bought that few because the smallest speakers in this corpus are "
            "large. Raise --valid-hours, or lower the speaker floors so that smaller speakers "
            "are selected; both change how many voices the same held-out hours buy. One "
            "speaker is what phase 1 held out, and it could not say when to stop."
        )
    return train, valid


def require_held_out_voices(
    valid_aligned: Path, minimum: int = MIN_VALID_SPEAKERS
) -> set[str | None]:
    """The voices left in the manifest training will actually validate on, and
    a refusal when there are too few of them.

    `split_across_corpora` asserts this same floor, and two stages that run
    after it can undo what it asserted. Ruling G7 put the score filter after
    alignment, so between the split and these manifests there are two stages
    that drop rows -- the aligner discards the utterances it could not align,
    and `filter_by_score` discards the ones it scored badly. Both work one
    utterance at a time, and a voice all of whose utterances went is a voice
    gone: nothing counting utterances sees a speaker leave, and the cutoff was
    read off the training distribution in any case, so what it does to the
    held-out set is not a number anybody looked at.

    M2a exists to answer exactly one question -- whether the validation loss
    turns over 20+ voices the weights have never heard. A held-out set that has
    quietly fallen under that is the experiment failing silently, at the end of
    a run whose alignment has already been paid for. An empty one is worse and
    is not detectable further down: a zero-byte manifest is valid JSONL, so the
    loader opens it, the run starts, and the loss the whole thing is read off is
    taken over nothing.

    Read back off the file rather than carried down from the split, because
    what happened in between is the whole question. Streamed and only the
    labels kept: the valid manifest is a fraction of the training one, but it is
    an aligned manifest all the same, and those carry a timestamp per word.

    Asked on every run, not only on the run that wrote the file. Both filtered
    manifests are skipped when they are up to date, and a refusal that a re-run
    walked straight past would be a refusal only the first time.
    """
    voices: set[str | None] = set()
    with open(valid_aligned, encoding="utf-8") as f:
        for line in f:
            voices.add(json.loads(line).get("speaker"))
    if len(voices) < minimum:
        raise typer.BadParameter(
            f"{valid_aligned} holds {len(voices)} speaker(s) and needs at least {minimum}. "
            "The split held out enough and the two stages since then took rows out from "
            "under it one utterance at a time -- the aligner drops what it could not align "
            "and the score cutoff drops what it scored badly -- so whole voices left "
            "without anything counting utterances seeing them go. The cutoff was read off "
            "the training distribution, which is not this one. Lower --min-score, or delete "
            "train.jsonl and valid.jsonl and split again under a larger --valid-hours; the "
            "alignment above is on disk either way. This run's whole readout is the loss "
            "over these voices, and one voice is what phase 1 had."
        )
    return voices


def _scored(row: dict) -> tuple[float, int, int] | None:
    """One aligned row's `(score, frames, tokens)`, or None if it carries no
    score this stage can read.

    Both denominators have to be there and neither may be zero, because both of
    them are divided by. `align_data` emits all three together for every
    utterance it aligns and writes nothing at all for the ones it cannot, so on
    a manifest that came from it this never returns None -- which is exactly
    why the caller must not treat None as a bad alignment. It is a row from
    somewhere else, and the callers say so out loud instead of quietly filing
    it under the thing this stage throws away.
    """
    score, frames, tokens = row.get("score"), row.get("frames"), row.get("tokens")
    if score is None or not frames or not tokens:
        return None
    return score, frames, tokens


def _normalized(score: float, frames: int, tokens: int) -> dict[str, float]:
    """The same alignment score under all three normalizations.

    The raw value is a log-probability summed along the best path, so it grows
    with the number of frames the path runs over and with the number of tokens
    it has to consume. Dividing by either is defensible and they are not the
    same ordering -- a slow speaker and a fast one with the same text differ in
    frames and not in tokens -- so all three are carried until the corpus has
    been looked at. See `probe_scores`.
    """
    return {"raw": score, "per_frame": score / frames, "per_token": score / tokens}


def _score_retention(rows: list[tuple[float, int, int, float]]) -> list[dict]:
    """How much of the corpus each candidate cutoff would leave, under each of
    the three normalizations.

    The cutoffs are quantiles of the corpus's own scores rather than a written
    grid, which is the one real difference from `_retention` on the MoeSpeech
    side. CER lives in [0, 1] and speechMOS in [1, 5], so a grid of absolute
    values could be written there before any measurement. An alignment
    log-probability has no scale known in advance: it depends on the acoustic
    model, on the language, and on how long the utterances happen to be, and
    any absolute grid written here would be a guess wearing the clothes of a
    measurement. A quantile cannot be one -- `min_score` is a number this
    corpus actually holds.

    That does make `kept` nearly the same in every row at a given quantile, by
    construction. The column that tells the three normalizations apart is
    `hours`: they keep the same *number* of clips and a different *set* of
    them. Hours are reported both absolutely and as a fraction because the two
    answer opposite questions -- what is left against the 100-hour floor the
    README cites, and what this cut costs relative to what there was -- and
    because the two diverge here in the expensive direction. A raw cutoff
    drops long clips first, since a long clip sums more negative log-prob, so
    it costs far more hours than it costs clips.

    An empty corpus gets an empty table. There is no cutoff to offer when
    there is nothing to read one off, and a table of zeros would read as a
    measurement of a corpus that scored badly rather than of one that is not
    there yet.
    """
    if not rows:
        return []
    total_seconds = sum(d for _, _, _, d in rows)
    table = []
    for normalization in NORMALIZATIONS:
        values = [(_normalized(s, f, t)[normalization], d) for s, f, t, d in rows]
        ordered = sorted(v for v, _ in values)
        for quantile in SCORE_QUANTILES:
            cutoff = _percentile(ordered, quantile)
            kept = [d for v, d in values if v >= cutoff]
            table.append(
                {
                    "normalization": normalization,
                    "quantile": quantile,
                    "min_score": cutoff,
                    "kept": len(kept),
                    "fraction": len(kept) / len(rows),
                    "hours": sum(kept) / 3600,
                    "hours_fraction": sum(kept) / total_seconds if total_seconds else 0.0,
                }
            )
    return table


def probe_scores(aligned_jsonl: Path) -> dict:
    """Measure the alignment scores in `aligned_jsonl`. Decide nothing.

    This is the stage that replaces MoeSpeech's quality filter, and it is a
    better measurement than the one it replaces. There, two independent ASR
    passes were compared against each other and their mutual CER stood in for
    quality -- an indirect proxy, since it measures where two systems disagreed
    rather than whether either was right. GOL ships one transcription, so that
    proxy does not exist; what does exist is the aligner's own log-probability,
    which measures how well the audio supports the text directly.

    Three normalizations are reported and none is chosen. The raw score scales
    with both frames and tokens, and which of those the distribution wants
    divided out is a question about this corpus that nobody has asked yet.
    Reporting one would be answering it. `align_data` carries `frames` and
    `tokens` on every row precisely so that the answer can wait until here, and
    `filter_by_score` below takes its threshold as a required argument so that
    it can wait past here too.

    The reason that discipline is worth this much apparatus is on the record.
    On MoeSpeech, `--min-mos 3.0` was the obvious default -- it is the middle
    of a five-point scale and it looks like a modest demand. The corpus was
    then measured and its median speechMOS turned out to be 2.281, so that
    default would have cut 124.4 hours to 16.2: seven eighths of the corpus
    discarded, below the 100-hour floor the project's own README cites, by a
    run whose manifest looked entirely normal.

    A row that carries no score is counted under `unscored` and left out of the
    distributions rather than folded in as a very bad one. It is not a badly
    aligned utterance -- those never reach an aligned manifest at all -- and
    scoring it as one would move the percentiles that a cutoff is read off.

    Only the four numbers each row is measured on are retained, not the row.
    An aligned manifest is the largest file this pipeline writes -- every
    utterance carries a timestamp per word -- and a thousand hours of it does
    not need to be in memory at once to have its median taken.
    """
    rows: list[tuple[float, int, int, float]] = []
    unscored = 0
    with open(aligned_jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            got = _scored(row)
            if got is None:
                unscored += 1
                continue
            rows.append((*got, row.get("duration", 0.0)))
    if unscored:
        logger.warning(
            f"{unscored} of {len(rows) + unscored} rows carry no score and were not measured; "
            "align_data writes one for every utterance it aligns, so these came from "
            "somewhere else and the table below does not describe them"
        )

    # One pass per normalization rather than one list of all three, which
    # would hold three floats and a dict per utterance at once. Each still goes
    # through `_normalized`, so there is one definition of what per-frame
    # means and not a second one here to drift from it.
    def measured(normalization: str) -> list[float]:
        return [_normalized(s, f, t)[normalization] for s, f, t, _ in rows]

    return {
        "count": len(rows),
        "unscored": unscored,
        "hours": sum(d for _, _, _, d in rows) / 3600,
        "raw": _distribution(measured("raw")),
        "per_frame": _distribution(measured("per_frame")),
        "per_token": _distribution(measured("per_token")),
        "retention": _score_retention(rows),
    }


def filter_by_score(
    aligned: Path, out: Path, min_score: float | None, normalization: str | None
) -> int:
    """Write the rows of `aligned` whose score clears the floor under
    `normalization`, and say how many that was.

    There is no default threshold and asking for one raises. The threshold is a
    property of this corpus, it is read off `probe_scores`'s retention table,
    and a number written here now would afterwards be indistinguishable from a
    measured one. See that function for what the plausible-looking default did
    on the corpus before this one.

    The normalization has no default either, and it is the same argument
    twice. `probe_scores` reports all three and chooses none, which is only
    worth its apparatus if all three can then be acted on -- a measurement
    favouring per-token is not a measurement if this stage can only cut per
    frame. And the pair cannot be split: the retention table names a
    normalization on every row, the number was read off one of those rows, and
    a per-token cutoff applied per frame is not a stricter or a looser cut but
    a meaningless one. It sits below every per-frame score there is, so it
    keeps the entire corpus and logs that it kept it. Requiring both is what
    makes that mismatch impossible to make silently, the same way requiring
    both speaker floors is.

    Which row was kept is `_normalized`'s answer and not a second definition of
    it here, so the number an operator read off the table is compared the way
    the table computed it.

    The floor is inclusive, matching the retention table exactly, so the count
    an operator read there is the count they get.

    What it cost is reported per manifest and in voices as well as in clips.
    The cutoff is read off the *training* distribution and then applied to a
    valid set nobody measured it against, and the valid set is the whole
    readout of this run: whether it survived the cut is a different question
    from whether the training set did, and one line over both together answers
    neither.

    A row with no score goes through and is counted. `align_data` drops the
    utterances it could not align before they reach this file, so a row without
    one did not come from it; filtering it out here would file a different
    problem under "bad alignment", and it is the expensive direction to get
    wrong, since the alignment has already been paid for and nothing
    downstream would ever say the rows were missing.

    This runs after alignment rather than before it (ruling G7), which is a
    real change from the MoeSpeech stage order and follows from where the score
    comes from: it does not exist until the aligner has run. So the aligner
    runs over utterances this then discards. That is still far cheaper than the
    second ASR pass the mutual-CER filter needed, by about an order of
    magnitude.

    Rows are copied through as the bytes they arrived as, rather than parsed
    and written back. Every field in an aligned row belongs to a later stage --
    the per-word timestamps the DataLoader cuts on above all -- so re-encoding
    them here is nothing but an opportunity to change one; `ensure_ascii` left
    at its default is the concrete way that happens, and it leaves a file that
    is still valid JSON and no longer readable with `head`. Binary mode is also
    the one way this machine's cp932 default cannot reach the transcripts.

    `write_manifest` is not used for the same reason: it takes a list, and this
    is the one file in the pipeline that should never be in memory whole.

    The lines land beside the name and are renamed in once they are all there.
    A manifest a preemption cut short is still valid JSONL -- every line parses
    and every path exists -- so nothing downstream can tell it from a finished
    one, and the re-run that finds it skips this stage.
    """
    if min_score is None:
        raise typer.BadParameter(
            "filter_by_score has no default threshold and needs one. The scale of an "
            "alignment log-probability is a property of this corpus and it has not been "
            "measured: run probe_scores, read a cutoff off its retention table -- which "
            "reports what each one costs in hours and not only in clips -- and pass that, "
            "with the normalization its row names. The corpus before this one is why: "
            "--min-mos 3.0 was the plausible default there and the measured median was "
            "2.281, which would have cut 124.4 hours to 16.2 and written a manifest that "
            "looked entirely normal."
        )
    if normalization not in NORMALIZATIONS:
        raise typer.BadParameter(
            f"{normalization!r} is not one of the normalizations the retention table "
            f"reports ({', '.join(NORMALIZATIONS)}). The cutoff was read off the rows "
            "carrying one of those names and means nothing away from it: applied to the "
            "wrong scale it does not cut harder or softer, it sits below every score "
            "there is and keeps the whole corpus while reporting that it kept it."
        )
    aligned, out = Path(aligned), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f"{out.name}.partial")
    kept = unscored = total = 0
    kept_seconds = total_seconds = 0.0
    # Labels rather than counts, because the two questions differ: a cut that
    # takes half of every voice and one that takes every clip of half the
    # voices remove the same number of utterances.
    voices: set[str | None] = set()
    kept_voices: set[str | None] = set()
    with open(aligned, "rb") as fin, open(partial, "wb") as fout:
        for line in fin:
            row = json.loads(line)
            total += 1
            duration = row.get("duration", 0.0)
            total_seconds += duration
            voices.add(row.get("speaker"))
            got = _scored(row)
            if got is None:
                unscored += 1
            elif _normalized(*got)[normalization] < min_score:
                continue
            fout.write(line)
            kept += 1
            kept_seconds += duration
            kept_voices.add(row.get("speaker"))
    if unscored:
        logger.warning(
            f"{unscored} of {total} rows carry no score to filter on and were kept; "
            "align_data writes one for every utterance it aligns and drops the ones it "
            "cannot, so these came from somewhere else, and discarding them here would "
            "record a different problem as a bad alignment"
        )
    logger.info(
        f"{out.name}: kept {kept} of {total} utterances from {len(kept_voices)} of "
        f"{len(voices)} voices at {normalization} >= {min_score}: "
        f"{kept_seconds / 3600:.1f} of {total_seconds / 3600:.1f} hours"
    )
    os.replace(partial, out)
    return kept


def _entries_name(index: int) -> str:
    """The file one speaker's offsets go in, and the stem of their audio.

    Numbered rather than named after the speaker, which is the one place this
    script cannot follow prepare_moespeech. A GOL key is `<game_id>:<speaker>`
    (ruling G9) and a colon is not a legal Windows file name at all -- and
    stripping it would merge two speakers whose keys differ only there, which is
    the mislabelling every guard below compares by label and none of them can
    see.

    The index is the speaker's position in the sorted selection, so it moves
    whenever the selection does. What keeps a moved index from being handed a
    previous run's file is that the selection is rewritten first, which makes
    every offsets file older than it; and because file clocks are coarser than
    these stages are fast, `main` reads back whose rows are actually in the file
    rather than trusting that alone.
    """
    return f"{index:04d}"


@app.command()
def main(
    out: Annotated[
        str, typer.Option(help="where every artifact of this run is written")
    ] = "data/ja-gol",
    hours: Annotated[float, typer.Option(help="hours of speech to pick whole games for")] = 1000.0,
    min_utterances: Annotated[
        int | None,
        typer.Option(
            help="keep speakers with at least this many utterances. Read it off "
            "speakers_probe.json's retention table; there is no default"
        ),
    ] = None,
    min_seconds: Annotated[
        float | None,
        typer.Option(
            help="keep speakers holding at least this many seconds of audio. Read it off "
            "speakers_probe.json's retention table; there is no default"
        ),
    ] = None,
    min_score: Annotated[
        float | None,
        typer.Option(
            help="keep utterances the aligner scored at least this well. Read it off "
            "scores.json's retention table, together with the normalization its row "
            "names; there is no default"
        ),
    ] = None,
    score_normalization: Annotated[
        str | None,
        typer.Option(
            help="which scale --min-score is on: one of " + ", ".join(NORMALIZATIONS) + ". It "
            "is the retention table's own column and the cutoff means nothing away from it"
        ),
    ] = None,
    valid_hours: Annotated[
        float,
        typer.Option(
            help="hours held out for validation, whole speakers at a time. How many voices "
            f"that buys is a property of the corpus, and at least {MIN_VALID_SPEAKERS} of "
            "them are required"
        ),
    ] = 10.0,
    target_sec: Annotated[
        float,
        typer.Option(
            help="longest pseudo-recording to build out of one speaker's clips. Long enough "
            "that joining is worth doing and well past the loader's max_duration_sec; short "
            "enough that a preemption costs one small file, since this stage builds each in "
            "memory"
        ),
    ] = 120.0,
    moespeech: Annotated[
        str | None,
        typer.Option(
            help="the --out of a prepare_moespeech run, whose entries are merged with this "
            "run's before the split. Both corpora have to be split at once or each one's "
            "held-out speakers are in the other's training set"
        ),
    ] = None,
    tars: Annotated[
        str | None,
        typer.Option(
            help="where the downloaded tars are kept (default: <out>/tars). Naming one "
            "directory across runs keeps a larger --hours from re-fetching what a smaller "
            "one already has"
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
    """Nine stages from a dataset name to a filtered, aligned training manifest.

    Every stage skips work whose output is already on disk, so this command is
    re-run rather than resumed. That is the whole design: a 1,000-hour selection
    is 660 GB of tars, the instance it runs on is preemptible, and being killed
    must cost only the stage in flight.

    Size the disk for four copies of that, because four of them coexist. The
    hub cache holds every tar `hf_hub_download` fetched and nothing here frees
    it; `tars/` holds the copy `download_games` made; `extracted/` holds the
    same audio unpacked; and `audio/` holds it again, joined into 48 kHz
    pseudo-recordings. That is roughly 660 + 660 + 660 + 660, about 2.6 TB at
    peak for a 1,000-hour run, against the 660 GB of the download alone. Only
    one of the four can be given back mid-run: once `tars/` holds every tar,
    nothing reads the hub cache again and deleting it frees 660 GB before the
    extraction that needs it. Running out mid-run costs the run, and this
    docstring is the only number an operator renting the disk sizes it by.

    It stops twice on purpose, and neither stop is a failed run. The first is
    after the speaker probe: GOL's median speaker has 0.6 minutes of audio and
    3,922 of its ids have a single utterance, so a floor guessed rather than
    read off `speakers_probe.json` would be attached to whatever the tars
    happened to hold. The second is after the score probe, which cannot come any
    earlier -- the score is the aligner's own and does not exist until it has
    run (ruling G7), so the alignment is paid for over utterances the cut then
    discards. Read the table, pass the flag, and run the same command again.

    What a stage skips on is its output and the timestamps of its inputs, not
    the options it was given -- with three exceptions. Two of them are what the
    stops above force onto the command line a second time: the speaker floors go
    to `floors.json` and the score cutoff to `score_cutoff.json`, and each is
    counted among the inputs of what it decided, so a changed value rebuilds
    everything below it on its own. Nothing else on disk records either, and
    without that a stricter floor would rewrite `speakers.json` and change
    nothing else: the run would report success over the selection it had just
    replaced.

    The third is --moespeech, which goes to `corpora.json` for that reason and
    a sharper one. The entries it names were written by the other script's run,
    so they are older than a train.jsonl this one has already left behind:
    supplying the flag at the second of these three invocations, having
    forgotten it at the first, moves nothing the split is gated on, and the run
    prints Done over a manifest built from one corpus while the command line
    named two. Taking it away again is the same silence with the rows still in.

    Every other option -- --target-sec, --valid-hours -- is
    recorded nowhere, so changing one re-runs nothing. Delete that stage's
    artifact to redo it under a new value; deleting is the only way to say so,
    and it is deliberate, since the alternative is a stage that quietly redoes
    660 GB of work. Deleting one is enough: what was built out of it is rebuilt
    with it, so removing `utterances.jsonl` alone carries through `entries/`,
    `audio/`, both manifests, both alignments and both filtered manifests.

    --hours is the third case. `games.json` is reused whenever it exists,
    whatever --hours now says, and a mismatch is only warned about: delete that
    file to select games again, and everything below re-runs against it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Named up here, before anything is written, because stage 0 has to ask
    # about a later stage's outputs: whether the aligner will run at all is what
    # decides whether a missing segmenter is fatal or merely worth saying.
    out_dir = Path(out)
    tar_dir = Path(tars) if tars else out_dir / "tars"
    extract_root = out_dir / "extracted"
    games_json = out_dir / "games.json"
    speakers_probe_json = out_dir / "speakers_probe.json"
    floors_json = out_dir / "floors.json"
    speakers_json = out_dir / "speakers.json"
    utterances_jsonl = out_dir / "utterances.jsonl"
    entries_dir, audio_dir = out_dir / "entries", out_dir / "audio"
    train_manifest, valid_manifest = out_dir / "train.jsonl", out_dir / "valid.jsonl"
    # The aligner's own output, and then what the cut leaves of it. The filtered
    # pair keeps the `_aligned` name because that is the name the training
    # configs read and prepare_moespeech writes: the file training reads has to
    # mean one thing across both pipelines. `_scored` is the intermediate, and
    # names what stage 8 adds and stage 9 reads.
    train_scored, valid_scored = out_dir / "train_scored.jsonl", out_dir / "valid_scored.jsonl"
    scores_json = out_dir / "scores.json"
    score_cutoff_json = out_dir / "score_cutoff.json"
    train_aligned, valid_aligned = out_dir / "train_aligned.jsonl", out_dir / "valid_aligned.jsonl"
    corpora_json = out_dir / "corpora.json"
    floors = {"min_utterances": min_utterances, "min_seconds": min_seconds}
    score_cutoff = {"min_score": min_score, "normalization": score_normalization}

    # 0. The one thing that can fail for a reason no stage below can fix, asked
    #    before stage 1 because the answer never changes mid-run and the stage
    #    that needs it is the eighth: see require_japanese_segmenter. A run
    #    without both floors stops at the speaker probe and never aligns, and so
    #    does one whose two alignments are already on disk *and* whose floors are
    #    the ones that built them -- align() skips an output it finds finished,
    #    and that is asked of align()'s own predicate rather than restated here.
    #    The score cutoff is deliberately not part of the question: it is read
    #    off a table the aligner has to run to produce, so a run without it
    #    aligns like any other. Anything this still lets past is caught by
    #    require_segmenter_to_align, one line before each align().
    require_japanese_segmenter(
        will_align=min_utterances is not None
        and min_seconds is not None
        and not (
            all(already_aligned(p) for p in (train_scored, valid_scored))
            and _same_cutoffs(floors_json, floors)
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Which games. 1.68 GB of metadata.tsv decides what the other eight
    #    stages will ever touch, and answers it without fetching a byte of audio.
    #    Fetched on every run rather than only when the selection is made, which
    #    is where this differs from the MoeSpeech script: three stages below read
    #    this file, not one. The hub serves it out of its cache after the first
    #    time.
    metadata_tsv = Path(hf_hub_download(repo, "metadata.tsv", repo_type="dataset"))
    if games_json.exists():
        chosen = json.loads(games_json.read_text(encoding="utf-8"))
        if chosen["hours_requested"] != hours:
            # Silently re-selecting would orphan the tars already unpacked under
            # a different set of games; silently ignoring --hours would be worse
            # still, so say which of the two won.
            logger.warning(
                f"{games_json} holds the selection for --hours "
                f"{chosen['hours_requested']:g} and is being reused for --hours {hours:g}; "
                "delete it to select games again"
            )
    else:
        games = select_games(metadata_tsv, hours)
        chosen = {
            "hours_requested": hours,
            "hours_selected": sum(g["hours"] for g in games),
            "games": games,
        }
        _write_json(chosen, games_json)
    game_ids = [g["game_id"] for g in chosen["games"]]
    logger.info(
        f"{len(game_ids)} games, {chosen['hours_selected']:.1f}h of audio to fetch "
        f"({sum(g['speakers'] for g in chosen['games'])} speaker ids in them)"
    )

    # 2 + 3. Fetch every tar, then unpack them. `download_games` returns a list
    #        rather than yielding, so nothing is unpacked until all of the tars
    #        are on disk -- the two stages do not overlap. Each still decides per
    #        game what it already has, which is what a re-run after a preemption
    #        rests on: this is where the hours are, at a median 11 GB a tar.
    for tar_path in download_games(game_ids, tar_dir, repo=repo):
        extract_game(tar_path, extract_root)

    # 4. Measure the speakers, and decide nothing about them. This reads only
    #    metadata.tsv, so it is bounded by games.json and by nothing on disk --
    #    it describes the games this run took whether or not their tars have
    #    finished unpacking.
    if not _reusable([speakers_probe_json], [games_json], "keeping the measurements it holds"):
        _write_json(probe_speakers(metadata_tsv, game_ids), speakers_probe_json)

    # 5. Decide, or stop. Both floors are a property of this corpus and the line
    #    above is what measures it, so there is nothing to default them to.
    if min_utterances is None or min_seconds is None:
        logger.error(
            f"read the retention table in {speakers_probe_json} and run this again with "
            "--min-utterances and --min-seconds. Neither has a default: GOL's median speaker "
            "has 0.6 minutes of audio and 3,922 of its ids have a single utterance, so a "
            "number guessed now would afterwards be indistinguishable from a measured one. "
            "Everything up to here is on disk and will not be redone."
        )
        raise typer.Exit(1)
    #    The pair is written down beside the selection they make, and is one of
    #    its inputs: nothing else on disk records them, so without this a re-run
    #    under a stricter floor would find speakers.json sitting there and reuse
    #    it -- silently, and with no way afterwards for either the script or the
    #    operator to tell which floors a given train.jsonl was built under.
    #    Written only when they differ, so an unchanged pair does not touch the
    #    file and nothing downstream is rebuilt.
    if not _same_cutoffs(floors_json, floors):
        _write_json(floors, floors_json)
    if _reusable([speakers_json], [games_json, floors_json], "keeping the speakers it holds"):
        speakers = json.loads(speakers_json.read_text(encoding="utf-8"))["speakers"]
    else:
        speakers = select_speakers(metadata_tsv, game_ids, min_utterances, min_seconds)
        _write_json({**floors, "speakers": speakers}, speakers_json)
    logger.info(f"{len(speakers)} speakers clear both floors")

    # 6. Gather their utterances. The completion markers are inputs alongside the
    #    selection: the extract root is what this walks for the audio, and a game
    #    unpacked after the walk makes it short of exactly the clips that arrived
    #    late -- which is what a re-run after a kill between two tars leaves.
    markers = [extract_root / f"{game}{EXTRACT_MARKER}" for game in game_ids]
    gather_inputs = [speakers_json, *markers]
    if _reusable([utterances_jsonl], gather_inputs, "keeping the utterances it holds"):
        utterances = _read_jsonl(utterances_jsonl)
    else:
        # `wav` arrives as a Path, which json.dumps refuses; posix separators
        # rather than str(), because a manifest written on Windows is read on
        # Linux, where a backslash is part of the name rather than a separator.
        utterances = [
            {**u, "wav": u["wav"].as_posix()}
            for u in gol_utterances(metadata_tsv, extract_root, speakers, game_ids)
        ]
        write_manifest(utterances, utterances_jsonl)

    # 7. Join each speaker's clips into pseudo-long recordings. The grouping
    #    happens here because concatenate() refuses a mixed list rather than
    #    grouping one (ruling G4): a file holding two voices would teach the
    #    model that the prompt does not decide the voice, and nothing downstream
    #    could see it. Each speaker's offsets are recorded under their own
    #    number, so a kill costs the speaker in flight rather than all of them.
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for utterance in utterances:
        by_speaker[utterance["speaker"]].append(utterance)
    #    The braces to the belt in `gol_utterances`. Every guard from here down
    #    compares this label and none of them can check it, so the one thing that
    #    can be checked is that every label is a speaker this run selected --
    #    true of a correct label by construction, and asked before a single wav
    #    is joined or deleted.
    unexpected = sorted(set(by_speaker) - set(speakers))
    if unexpected:
        logger.error(
            f"the selection holds speakers that were never selected: {unexpected[:10]}. The "
            "speaker is metadata.tsv's own column and is keyed by game (ruling G9), so this "
            "is a key built differently in two places -- joining them would mix characters "
            "into one recording. Nothing has been joined."
        )
        raise typer.Exit(1)
    numbered = {speaker: _entries_name(i) for i, speaker in enumerate(sorted(by_speaker))}
    #    What a run stops naming, it deletes. `audio/` is hundreds of gigabytes
    #    at 1,000 hours of 48 kHz, deleting an artifact and re-running is the
    #    documented way to change any option, and the disk on a preemptible
    #    instance is fixed -- filling it mid-run costs the run.
    for part in sorted(entries_dir.glob("*.jsonl")):
        if part.stem in set(numbered.values()):
            continue
        # The offsets file is what says which audio was this speaker's, so it is
        # read before it goes; deriving the names from the number instead would
        # miss the files `_joined_path` numbered beside the first one.
        for row in _read_jsonl(part):
            wav = Path(row["path"])
            if wav.parent == audio_dir:
                wav.unlink(missing_ok=True)
        part.unlink()
        logger.info(f"{part.stem} is no longer selected; their offsets and audio are removed")
    for speaker, index in numbered.items():
        part = entries_dir / f"{index}.jsonl"
        if _reusable([part], [utterances_jsonl], f"keeping {speaker}'s offsets"):
            held = {row["speaker"] for row in _read_jsonl(part)}
            if held == {speaker}:
                continue
            # A number is not a name. `index` is this speaker's position in the
            # sorted selection, and it moves whenever the selection does; what
            # keeps a moved number from being handed the previous run's file is
            # that the selection is rewritten first and so is newer than every
            # offsets file. File clocks are coarser than these stages are fast
            # and equal timestamps count as fresh, so that is a race rather than
            # a guarantee -- and losing it would write one speaker's offsets
            # under another's label, which every guard below compares and none
            # of them can see.
            logger.warning(
                f"{part} holds {sorted(held)} rather than {speaker!r}; the selection has "
                "moved under it and it is being rebuilt"
            )
        base = audio_dir / f"{index}.wav"
        rows = concatenate(by_speaker[speaker], base, target_sec)
        # The files are numbered from the name upwards, so everything from the
        # count this run wrote onwards is what a wider previous run left.
        surplus_index = len({row["path"] for row in rows})
        while (surplus := _joined_path(base, surplus_index)).exists():
            surplus.unlink()
            logger.info(f"{surplus} is past what {speaker} now needs; removed")
            surplus_index += 1
        write_manifest(rows, part)

    # 8. Merge the corpora and split off the valid speakers. This is the one
    #    point where GOL and MoeSpeech meet, and it has to be one point: split
    #    them separately and each corpus's held-out speakers are in the other
    #    corpus's training set, so both validation losses report voices the
    #    weights have already seen. Both manifests or neither -- a kill between
    #    them leaves train.jsonl describing a corpus valid.jsonl was never held
    #    out of.
    corpora = [entries_dir]
    if moespeech:
        corpora.append(Path(moespeech) / "entries")
    #    Which corpora are in the merge is recorded beside the manifests it
    #    decides and counted among their inputs, exactly as the floors and the
    #    score cutoff are -- and for a sharper reason than either. The other
    #    script's entries were written by another run, so by the time this
    #    command is typed for the second of the three times its two stops force,
    #    they are older than the train.jsonl this one already left behind.
    #    Gated on those files alone, --moespeech remembered late moves nothing
    #    and the run prints Done over a split that never saw it; --moespeech
    #    dropped late is the same silence with its rows still in.
    #    Posix, because this is compared against what a later run writes and a
    #    path spelled with backslashes cannot be compared with one read on Linux.
    named_corpora = {"corpora": [d.as_posix() for d in corpora]}
    if not _same_cutoffs(corpora_json, named_corpora):
        _write_json(named_corpora, corpora_json)
    written_entries = [part for directory in corpora for part in sorted(directory.glob("*.jsonl"))]
    if not _reusable(
        [train_manifest, valid_manifest],
        [utterances_jsonl, corpora_json, *written_entries],
        "keeping the split they hold",
    ):
        train, valid = split_across_corpora(merge_entries(corpora), valid_hours)
        write_manifest(train, train_manifest)
        write_manifest(valid, valid_manifest)

    # 9. Align. prepare_data's align() already streams into a .partial that
    #    --resume picks up, renames its output in only once a pass has finished
    #    -- the sharded merge included -- and starts over rather than resuming a
    #    leftover written under a different --align-shards, so it is reused here
    #    rather than reimplemented. The segmenter is not an option: this manifest
    #    is Japanese, and "whitespace" over a language written without spaces
    #    returns one word per utterance -- the aligner emits a single span, the
    #    loader finds no cut point, and the voice prompt quietly comes from the
    #    utterance being predicted.
    #    That skipping is on the output's name, and --resume continues whatever
    #    the .partial holds without asking which manifest produced it, so an
    #    alignment older than the manifest it claims to align has to be thrown
    #    away here rather than kept or continued.
    for scored, manifest in ((train_scored, train_manifest), (valid_scored, valid_manifest)):
        produced = [scored, scored.with_suffix(".partial")]
        produced += sorted(scored.parent.glob(f"{scored.stem}.shard*"))
        for leftover in produced:
            if _stale(leftover, [manifest]):
                logger.info(f"{leftover} was aligned from an older {manifest.name}; discarding it")
                leftover.unlink()
    require_segmenter_to_align(train_scored)
    align(
        train_manifest,
        train_scored,
        align_shards,
        align_model,
        "training manifest",
        segmenter="japanese",
    )
    require_segmenter_to_align(valid_scored)
    align(valid_manifest, valid_scored, 1, align_model, "valid manifest", segmenter="japanese")

    # 10. Measure what the aligner thought of the audio, then cut on a number
    #     read off that. This is where GOL's stage order differs from
    #     MoeSpeech's and why (ruling G7): the score does not exist until the
    #     aligner has run, so the alignment above was paid for over utterances
    #     this discards -- still an order of magnitude cheaper than the second
    #     ASR pass the mutual-CER filter needed. Measured over the training
    #     manifest, which is where the cutoff has to hold; the valid one is a
    #     fraction of the size and is then cut by the same number.
    if not _reusable([scores_json], [train_scored], "keeping the measurements it holds"):
        _write_json(probe_scores(train_scored), scores_json)
    if min_score is None or score_normalization is None:
        logger.error(
            f"read the retention table in {scores_json} and run this again with "
            "--min-score and --score-normalization. Neither has a default: an alignment "
            "log-probability has no scale known before the corpus is measured, so a number "
            "guessed now would afterwards be indistinguishable from a measured one -- and "
            "the table reports what each cutoff costs in hours and not only in clips. The "
            "normalization comes with it because the table names one on every row and the "
            "number means nothing away from it: a per-token cutoff applied per frame sits "
            "below every per-frame score there is and keeps the whole corpus. Everything up "
            "to here, the alignment included, is on disk and will not be redone."
        )
        raise typer.Exit(1)
    #     Recorded beside the manifests it decides, for the same reason the
    #     floors are: nothing else on disk says which cutoff a given
    #     train_aligned.jsonl was written under, and without it a re-run under a
    #     new one would find that file sitting there, newer than the scored
    #     manifest it came from, and report success over the cut just replaced.
    if not _same_cutoffs(score_cutoff_json, score_cutoff):
        _write_json(score_cutoff, score_cutoff_json)
    #     Both manifests, under the one number. The valid loss is the whole
    #     readout of this run, and a valid set kept under a different rule from
    #     the training set is not comparable with it. Separately rather than as a
    #     pair, because the two are independent files of independent inputs and a
    #     kill between them should cost the second one alone; a changed cutoff
    #     moves score_cutoff.json, which is an input to both.
    for scored, filtered, what in (
        (train_scored, train_aligned, "training"),
        (valid_scored, valid_aligned, "valid"),
    ):
        if _reusable([filtered], [scored, score_cutoff_json], f"keeping the {what} manifest"):
            continue
        filter_by_score(scored, filtered, min_score, score_normalization)
    # 11. The twenty-voice floor again, on the file training will actually read.
    #     Stage 8 asserted it over the split, and the two stages between there
    #     and here drop rows one utterance at a time, so a voice can leave the
    #     held-out set entire with nothing per-utterance noticing. Asked outside
    #     the loop above so that a re-run which skipped the filter still asks it.
    logger.info(f"{len(require_held_out_voices(valid_aligned))} voices held out and validated on")
    logger.info(f"Done. Training on {train_aligned.resolve()} and {valid_aligned.resolve()}")


if __name__ == "__main__":
    app()
