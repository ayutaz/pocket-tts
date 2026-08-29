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
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import typer
from huggingface_hub import hf_hub_download

# The same normalization the MoeSpeech path applies, so the two corpora reach
# the tokenizer and the loader as one distribution rather than two.
from pocket_tts.utils.text_normalization import normalize_japanese

# Borrowed rather than copied: these two scripts are one pipeline in two files.
# _distribution because the survey in docs/ prints their percentiles in the same
# table and two definitions of "median" would eventually disagree in a way
# nobody would think to check; _read_jsonl because it is the other half of the
# write_manifest that wrote those files and a second reader could drift from the
# writer; and split_by_speaker because holding speakers out of the two corpora
# separately is the one thing the merge below exists to prevent. write_manifest
# itself is main's, at the point where the split is written out.
from training.scripts.prepare_moespeech import _distribution, _read_jsonl, split_by_speaker

logger = logging.getLogger("prepare_gol")

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
    list of more than one by the rest of the path the metadata states. Kept
    sorted, so that narrowing starts from a stated order rather than from
    whatever `rglob` happened to return.

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
    missing = blank = 0
    with open(metadata_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            game = row["game_id"]
            if game not in wanted_games:
                continue
            speaker = _speaker_key(game, row["speaker"])
            if speaker not in wanted_speakers:
                continue
            stated = PurePosixPath(row["file_path"])
            found = clips.get((game, stated.name), [])
            if len(found) > 1:
                # Narrowed by the speaker directory the metadata itself names,
                # which is a lookup against the corpus's own claim about where
                # it put the file and not a speaker read off a path. Two clips
                # of one name under one speaker of one game would take two
                # directories of that name at different depths, which is not a
                # tree a tar can hold.
                tail = "/" + "/".join(stated.parts[-2:])
                found = [p for p in found if p.as_posix().endswith(tail)]
            if not found:
                missing += 1
                logger.debug(f"{row['file_path']}: no such wav under {Path(extract_root) / game}")
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
    if missing or blank:
        # Once, at the end, rather than per row: there can be hundreds of
        # thousands of these, and the number that matters is how many clips the
        # manifest is short of what the selection promised.
        logger.warning(
            f"{missing} rows name a wav that is not under {extract_root} and {blank} have no "
            "text left after normalization; neither is in the manifest"
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
