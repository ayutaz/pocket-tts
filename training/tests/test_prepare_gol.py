"""Building a training manifest out of GOL.

GOL is five times MoeSpeech and shaped differently: the download unit is a
whole work rather than one character, and its transcripts come from a single
ASR pass, so the mutual-CER filter that carried MoeSpeech has no counterpart.
"""

import csv
import inspect
import io
import json
import logging
import os
import shutil
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from training.scripts.prepare_gol import (
    EXTRACT_MARKER,
    MIN_VALID_SPEAKERS,
    _speaker_key,
    filter_by_score,
    gol_utterances,
    merge_entries,
    probe_scores,
    probe_speakers,
    select_games,
    select_speakers,
    split_across_corpora,
)
from training.scripts.prepare_moespeech import (
    _read_jsonl,
    concatenate,
    split_by_speaker,
    write_manifest,
)


def _metadata(tmp_path, rows, text: str = "あ"):
    """A stand-in for GOL's metadata.tsv: game_id, speaker, text, path, duration.

    A row is `(game, speaker, duration)`, optionally followed by the wav's file
    name and then by that row's own text. The stages that only read the
    metadata need neither -- every clip may as well be called x.wav -- and the
    stage that goes to the audio needs both.
    """
    p = tmp_path / "metadata.tsv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["game_id", "speaker", "text", "file_path", "duration"])
        for row in rows:
            g, s, d = row[:3]
            name = row[3] if len(row) > 3 else "x.wav"
            w.writerow([g, s, row[4] if len(row) > 4 else text, f"{g}/{s}/{name}", d])
    return p


def test_largest_first_reaches_the_target_with_fewest_tars(tmp_path):
    """Measured on the real metadata: 1,000 hours is 18 tars taken largest-first.
    A tar is 11 GB on average, so the ordering decides hours of download.

    Every signal here except duration points at the other game: `a-many-short`
    has more utterances, more speakers, sorts first by id and comes first in
    the file. Only summed duration picks `z-one-long`. A fixture where the
    biggest game also happened to be the first or the busiest would pass under
    a sort on any of those -- or under no sort at all -- and a size-blind order
    needs about 57 tars for the 1,000 hours that cost 18 here, some 600 GB of
    download that buys nothing.
    """
    rows = [("a-many-short", f"s{i}", 60.0) for i in range(30)]
    rows += [("z-one-long", "s", 7200.0)] * 2
    (chosen,) = select_games(_metadata(tmp_path, rows), hours=1.0)
    assert chosen["game_id"] == "z-one-long"


def test_selection_stops_once_the_target_is_reached(tmp_path):
    md = _metadata(tmp_path, [(f"g{i}", "s", 3600.0) for i in range(5)])
    assert len(select_games(md, hours=1.0)) == 1


def test_a_game_reports_its_speakers_and_utterances(tmp_path):
    """Both decide what the next stage can use: the speaker filter needs the
    counts, and a tar of one speaker is worth less than a tar of thirty."""
    md = _metadata(tmp_path, [("g", "a", 60.0), ("g", "a", 60.0), ("g", "b", 60.0)])
    (g,) = select_games(md, hours=100.0)
    assert (g["speakers"], g["utterances"]) == (2, 3)


def test_ties_break_on_game_id_so_a_resumed_run_asks_for_the_same_tars(tmp_path):
    """Games of equal length must come out in a stated order, not in whatever
    order the accumulating dict happens to hold them.

    Asserting instead that two calls agree would pass for any pure function,
    including one that leaves equal games in hash order: within one process the
    two calls agree, and across processes they do not. A resumed run would then
    re-download tars it already has and skip ones it does not, at 11 GB each.
    """
    md = _metadata(tmp_path, [(g, "s", 3600.0) for g in ["gc", "ga", "gb"]])
    assert [g["game_id"] for g in select_games(md, hours=2.0)] == ["ga", "gb"]


def test_the_hours_column_is_seconds(tmp_path):
    """metadata.tsv's duration is in seconds and --hours is in hours. The same
    unit confusion cost a test in the MoeSpeech pipeline."""
    md = _metadata(tmp_path, [("g", "s", 3600.0)])
    assert len(select_games(md, hours=0.5)) == 1
    assert select_games(md, hours=100.0)[0]["hours"] == 1.0


FETCHED = b"gol\x00tar bytes"


def _fake_hub(tmp_path):
    """A stand-in for huggingface_hub whose cache is *not* the destination.

    The real `hf_hub_download` returns a path inside its own cache, and moving
    those bytes into `dest` is the only reason `download_games` exists. A fake
    that wrote straight into `dest` would make that move invisible: the fetched
    file and the destination file would be the same file, so an implementation
    that transferred nothing at all would still pass.
    """
    cache = tmp_path / "hf_cache"
    cache.mkdir()
    dest = tmp_path / "tars"
    calls = []

    def fake_fetch(repo_id, filename, **kw):
        calls.append((repo_id, filename, kw.get("repo_type")))
        p = cache / filename
        p.write_bytes(FETCHED)
        return str(p)

    return dest, calls, fake_fetch


def test_download_skips_a_tar_already_present(tmp_path, monkeypatch):
    """Re-running after a preemption must not re-fetch 11 GB it already has.

    The bytes already on disk differ from the ones the fake serves, so "skipped"
    is told apart from "fetched again and overwritten with the same thing" --
    which a fixture whose two files held identical bytes could not do, and which
    costs exactly the download this branch exists to avoid.
    """
    from training.scripts import prepare_gol as m

    dest, calls, fake_fetch = _fake_hub(tmp_path)
    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    dest.mkdir()
    (dest / "aaa.tar").write_bytes(b"already here")

    paths = m.download_games(["aaa", "bbb"], dest, repo="fake/repo")

    # The repo and repo_type are asserted, not just the filename: these tars
    # live in a dataset repo, and without repo_type="dataset" the hub resolves
    # the name in the model namespace and every fetch 404s.
    assert calls == [("fake/repo", "bbb.tar", "dataset")], calls
    assert (dest / "aaa.tar").read_bytes() == b"already here"
    assert (dest / "bbb.tar").read_bytes() == FETCHED
    assert [p.name for p in paths] == ["aaa.tar", "bbb.tar"]


def test_download_returns_a_path_for_every_requested_game(tmp_path, monkeypatch):
    """A caller that gets fewer paths than it asked for silently trains on a
    smaller corpus than intended. One tar is already present so the list spans
    both branches -- forgetting to append the skipped one is the likeliest way
    to lose a path, and on the re-run after a preemption that is where nearly
    every game sits. It is the middle one, so a list of the right length is not
    also in the right order by accident."""
    from training.scripts import prepare_gol as m

    dest, calls, fake_fetch = _fake_hub(tmp_path)
    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    dest.mkdir()
    (dest / "bbb.tar").write_bytes(b"already here")

    paths = m.download_games(["aaa", "bbb", "ccc"], dest, repo="fake/repo")

    assert [p.name for p in paths] == ["aaa.tar", "bbb.tar", "ccc.tar"]
    assert [p.parent for p in paths] == [dest, dest, dest]
    assert calls == [("fake/repo", "aaa.tar", "dataset"), ("fake/repo", "ccc.tar", "dataset")]


def test_download_does_not_trust_a_leftover_partial(tmp_path, monkeypatch):
    """A kill mid-copy leaves `<game_id>.tar.partial`, not `<game_id>.tar`.
    Mistaken for a finished download -- or promoted into place as if it were a
    transfer to resume -- the run would unpack a truncated 11 GB tar, and every
    re-run after it would find the same file there and do the same."""
    from training.scripts import prepare_gol as m

    dest, calls, fake_fetch = _fake_hub(tmp_path)
    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    dest.mkdir()
    (dest / "aaa.tar.partial").write_bytes(b"truncated")

    paths = m.download_games(["aaa"], dest, repo="fake/repo")

    assert calls == [("fake/repo", "aaa.tar", "dataset")], calls
    assert [p.name for p in paths] == ["aaa.tar"]
    assert (dest / "aaa.tar").read_bytes() == FETCHED
    assert not (dest / "aaa.tar.partial").exists()


def test_a_kill_mid_copy_leaves_nothing_under_the_finished_name(tmp_path, monkeypatch):
    """The window `.partial` exists for, entered rather than staged.

    The test above starts from a partial some earlier run left behind; this one
    is killed during the copy itself, which is the only way to tell a copy that
    lands at `.partial` from one that writes straight to `<game_id>.tar`.
    Written straight there, a preemption partway through 11 GB leaves a
    truncated file under exactly the name every later run tests for, and the
    next run unpacks it as though it were whole.
    """
    from training.scripts import prepare_gol as m

    dest, _calls, fake_fetch = _fake_hub(tmp_path)
    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)

    def killed(src, dst, *a, **kw):
        Path(dst).write_bytes(FETCHED[:3])  # part of it lands, then the instance goes
        raise KeyboardInterrupt

    # Only the copy is put back afterwards, not everything: monkeypatch.undo()
    # would take the hub fake with it and the re-run below would go to the real
    # hub for a repo that does not exist.
    real_copyfile = shutil.copyfile
    monkeypatch.setattr(m.shutil, "copyfile", killed)
    with pytest.raises(KeyboardInterrupt):
        m.download_games(["aaa"], dest, repo="fake/repo")
    monkeypatch.setattr(m.shutil, "copyfile", real_copyfile)

    assert not (dest / "aaa.tar").exists(), "the kill left something under the finished name"

    paths = m.download_games(["aaa"], dest, repo="fake/repo")  # the re-run
    assert paths[0].read_bytes() == FETCHED


def test_download_creates_the_destination_directory(tmp_path, monkeypatch):
    """The stage runs on a fresh instance where `dest` does not exist yet."""
    from training.scripts import prepare_gol as m

    dest, _calls, fake_fetch = _fake_hub(tmp_path)
    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    assert not dest.exists()

    paths = m.download_games(["aaa"], dest, repo="fake/repo")

    assert paths[0].read_bytes() == FETCHED


def _make_tar(path, names):
    """A tar holding `names`, each a one-byte file.

    Built member by member rather than from a directory on disk, so the member
    names are exactly what is written here: a GOL tar carries its own
    `<game_id>/` at the top, and that is the whole subject of the first test
    below.
    """
    with tarfile.open(path, "w") as t:
        for n in names:
            info = tarfile.TarInfo(n)
            info.size = 1
            t.addfile(info, io.BytesIO(b"x"))
    return path


def test_extract_keeps_the_tars_own_tree_below_the_game_directory(tmp_path):
    """A GOL tar holds `<game_id>/<speaker>/<file>.wav` and is unpacked into
    `<dest_root>/<game_id>/`, so what lands is double-nested. Pinned here for
    two reasons.

    The path is where the resumability tests below hand-build a half-extracted
    game. Left unpinned, a change to the naming would send those fixtures
    somewhere the code never looks, and they would go on passing while testing
    nothing.

    And the nesting itself must not be tidied away. Strip the top component and
    the tree becomes `<game_id>/<speaker>/`, whose last directory is a speaker
    name -- exactly the shape that invites a later stage to read the speaker off
    the path. It comes from metadata.tsv's own column. Two speakers are in the
    fixture rather than one so that a flattening implementation is caught by
    what the tree looks like and not only by how deep it is.
    """
    from training.scripts.prepare_gol import extract_game

    root = tmp_path / "extracted"
    t = _make_tar(tmp_path / "g.tar", ["g/spk/a.wav", "g/other/b.wav"])

    out = extract_game(t, root)

    assert out == root / "g"
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.wav")) == [
        "g/other/b.wav",
        "g/spk/a.wav",
    ]


def test_a_member_that_points_outside_the_game_directory_is_refused(tmp_path):
    """These tars come from a third-party repo and are unpacked unattended on a
    rented box. Left unfiltered, a member named `../escaped.wav` is written
    exactly where it says -- outside the game's directory, into the extract root
    the later stages rglob, and a `../../` one outside `dest_root` altogether.
    Refusing costs the tar; not refusing costs whatever the path names.
    """
    from training.scripts.prepare_gol import extract_game

    root = tmp_path / "extracted"
    t = _make_tar(tmp_path / "g.tar", ["g/spk/a.wav", "../escaped.wav"])

    with pytest.raises(tarfile.TarError):
        extract_game(t, root)

    assert not (root / "escaped.wav").exists()


def test_extraction_is_safe_on_the_oldest_interpreter_this_project_runs_on(tmp_path, monkeypatch):
    """`filter="data"` is a keyword `extractall` only grew in 3.10.12, and
    pyproject asks for ">= 3.10". On 3.10.0 through 3.10.11 passing it is a
    TypeError on every single extraction: the whole download paid for, the
    instance rented, and not one game unpacked -- and it would only be found on
    the rented box, since the machine this suite runs on has the keyword.

    So the containment those tars need does not come from the keyword. It is
    checked here, on every interpreter, and the keyword is added on top where it
    exists. `tarfile.data_filter` is what its presence is asked through, and
    taking that away is what an older interpreter looks like from in here.
    """
    from training.scripts.prepare_gol import extract_game

    real_extractall = tarfile.TarFile.extractall

    def without_the_keyword(self, path=".", members=None, *, numeric_owner=False):
        """`extractall`'s signature before 3.10.12. Deleting `tarfile.data_filter`
        alone would not do: `filter="data"` is resolved through a dict of names
        rather than through that attribute, so the call would go on working here
        and the test would pass against the very code it exists to reject."""
        return real_extractall(self, path, members, numeric_owner=numeric_owner)

    monkeypatch.setattr(tarfile.TarFile, "extractall", without_the_keyword)
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    root = tmp_path / "extracted"

    out = extract_game(_make_tar(tmp_path / "g.tar", ["g/spk/a.wav"]), root)
    assert (out / "g" / "spk" / "a.wav").read_text() == "x"

    escaping = _make_tar(tmp_path / "h.tar", ["h/spk/a.wav", "../escaped.wav"])
    with pytest.raises(tarfile.TarError):
        extract_game(escaping, root)
    assert not (root / "escaped.wav").exists()


def test_the_completion_marker_sits_beside_the_directory_not_inside_it(tmp_path):
    """So the directory holds corpus files and nothing else, and every later
    stage can rglob it without excluding a name. Written inside, the marker
    would be picked up by those scans and would have to be filtered out in each
    of them -- and the one that forgot would carry it into the manifest."""
    from training.scripts.prepare_gol import EXTRACT_MARKER, extract_game

    root = tmp_path / "extracted"
    t = _make_tar(tmp_path / "g.tar", ["g/spk/a.wav"])

    out = extract_game(t, root)

    assert (root / f"g{EXTRACT_MARKER}").exists()
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == [
        "g",
        "g/spk",
        "g/spk/a.wav",
    ]


def test_extract_skips_a_game_already_done(tmp_path):
    """Unpacking 11 GB is many minutes; re-running must not redo it."""
    from training.scripts.prepare_gol import extract_game

    root = tmp_path / "extracted"
    t = _make_tar(tmp_path / "g.tar", ["g/spk/a.wav"])
    out = extract_game(t, root)
    (out / "g" / "spk" / "a.wav").write_text("edited")  # prove it is not rewritten

    extract_game(t, root)

    assert (out / "g" / "spk" / "a.wav").read_text() == "edited"


def test_an_interrupted_extraction_is_redone(tmp_path):
    """A directory that exists but is incomplete must not be taken for a
    finished one, or the game arrives short by however many members the kill
    came before and nothing says so."""
    from training.scripts.prepare_gol import extract_game

    root = tmp_path / "extracted"
    t = _make_tar(tmp_path / "g.tar", ["g/spk/a.wav", "g/spk/b.wav"])
    half = root / "g"
    (half / "g" / "spk").mkdir(parents=True)
    (half / "g" / "spk" / "a.wav").write_text("partial")  # no completion marker

    out = extract_game(t, root)

    assert out == half, "the half-extracted fixture was never the directory under test"
    assert sorted(p.name for p in out.rglob("*.wav")) == ["a.wav", "b.wav"]
    assert (out / "g" / "spk" / "a.wav").read_text() == "x"  # the tar's byte, not the stub


def test_a_stale_member_does_not_survive_the_redo(tmp_path):
    """Redoing an interrupted game rebuilds it rather than filling in the gaps:
    whatever the killed attempt left behind goes first. Merged in instead, a
    clip the kill truncated would be indistinguishable from a whole one and
    would stay in the corpus for every run after."""
    from training.scripts.prepare_gol import extract_game

    root = tmp_path / "extracted"
    t = _make_tar(tmp_path / "g.tar", ["g/spk/a.wav"])
    half = root / "g"
    (half / "g" / "spk").mkdir(parents=True)
    (half / "g" / "spk" / "stale.wav").write_text("truncated")

    out = extract_game(t, root)

    assert out == half, "the half-extracted fixture was never the directory under test"
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.wav")) == ["g/spk/a.wav"]


def test_a_kill_during_unpacking_leaves_no_completion_marker(tmp_path, monkeypatch):
    """The marker has to mean "the last member is on disk". Written before the
    unpack instead, a preemption mid-extract leaves a half-extracted game that
    every later run skips, and the corpus shrinks with nothing to show for it.

    The tests above either run to completion or start from a hand-made
    directory, so none of them enters the window between the first member
    landing and the marker being written -- which is the whole window a
    preemption can arrive in. This one is killed inside it.
    """
    from training.scripts.prepare_gol import EXTRACT_MARKER, extract_game

    root = tmp_path / "extracted"
    t = _make_tar(tmp_path / "g.tar", ["g/spk/a.wav", "g/spk/b.wav", "g/spk/c.wav"])

    def killed(self, path=None, *a, **kw):
        self.extract("g/spk/a.wav", path)  # one member lands, then the instance goes
        raise KeyboardInterrupt

    monkeypatch.setattr(tarfile.TarFile, "extractall", killed)
    with pytest.raises(KeyboardInterrupt):
        extract_game(t, root)
    monkeypatch.undo()
    assert not (root / f"g{EXTRACT_MARKER}").exists(), "the kill left a completion marker"

    out = extract_game(t, root)  # the re-run
    assert sorted(p.name for p in out.rglob("*.wav")) == ["a.wav", "b.wav", "c.wav"]


def test_a_marker_without_its_directory_is_not_trusted(tmp_path, monkeypatch):
    """A finished game whose directory is later reclaimed for disk space leaves
    the marker behind. The marker alone is not evidence -- and the stale one has
    to go before the re-unpack starts, or a kill during that re-unpack leaves
    marker-plus-half-a-directory, which every later run then skips.

    Every other test here starts from either an empty root or a marker-less
    directory, so none of them reaches the state "marker present, directory
    absent" -- the one state in which these two guards do any work at all.
    """
    from training.scripts.prepare_gol import EXTRACT_MARKER, extract_game

    root = tmp_path / "extracted"
    t = _make_tar(tmp_path / "g.tar", ["g/spk/a.wav", "g/spk/b.wav", "g/spk/c.wav"])
    shutil.rmtree(extract_game(t, root))  # the directory goes, the marker stays
    assert (root / f"g{EXTRACT_MARKER}").exists(), "the fixture did not leave a stale marker"

    def killed(self, path=None, *a, **kw):
        self.extract("g/spk/a.wav", path)
        raise KeyboardInterrupt

    monkeypatch.setattr(tarfile.TarFile, "extractall", killed)
    # Not raising would mean the marker alone was trusted and this call returned
    # a directory that does not exist -- rglob over which yields nothing, so the
    # game reads as zero utterances downstream instead of failing.
    with pytest.raises(KeyboardInterrupt):
        extract_game(t, root)
    monkeypatch.undo()
    assert not (root / f"g{EXTRACT_MARKER}").exists(), "the stale marker outlived the kill"

    out = extract_game(t, root)
    assert sorted(p.name for p in out.rglob("*.wav")) == ["a.wav", "b.wav", "c.wav"]


def test_a_speaker_with_one_utterance_is_never_selected(tmp_path):
    """One utterance cannot be evaluated -- the protocol clones a voice from one
    and synthesizes another -- and cannot be concatenated with itself either.
    3,922 of GOL's 19,349 speakers are in this state.

    Both speakers here hold the same ten minutes, so the only thing telling
    them apart is how many pieces it arrives in.
    """
    md = _metadata(tmp_path, [("g", "solo", 600.0), ("g", "pair", 300.0), ("g", "pair", 300.0)])
    assert select_speakers(md, ["g"], min_utterances=2, min_seconds=0.0) == ["g:pair"]


def test_the_retention_table_reports_what_each_floor_keeps(tmp_path):
    """The output that decides the next task's defaults. Measured on the real
    metadata: a 1-hour floor keeps 2,095 of 19,349 speakers and 89% of the audio."""
    rows = [("g", f"s{i}", 3600.0) for i in range(10)] + [("g", f"t{i}", 10.0) for i in range(90)]
    md = _metadata(tmp_path, rows + rows)  # two utterances each
    stats = probe_speakers(md, ["g"])
    assert stats["speakers"] == 100
    assert any(r["min_seconds"] == 3600 and r["kept"] == 10 for r in stats["retention"]), stats


def test_selection_is_bounded_to_the_games_that_were_taken(tmp_path):
    """The extract root accumulates. Asking for fewer games later has to mean
    fewer speakers, or the flag does nothing -- the same failure the MoeSpeech
    pipeline had to be corrected for."""
    rows = [("kept", "a", 60.0)] * 2 + [("dropped", "b", 60.0)] * 2
    md = _metadata(tmp_path, rows)
    assert select_speakers(md, ["kept"], min_utterances=2, min_seconds=0.0) == ["kept:a"]


def _row(stats, min_utterances, min_seconds):
    """The single retention row at these floors.

    Unpacked rather than indexed, so a grid that listed a pair of floors
    twice would fail here instead of quietly reporting the first of two
    rows that an operator would then read as the only one.
    """
    (row,) = [
        r
        for r in stats["retention"]
        if (r["min_utterances"], r["min_seconds"]) == (min_utterances, min_seconds)
    ]
    return row


def test_the_floor_is_the_speakers_total_seconds_not_its_minutes(tmp_path):
    """`min_seconds` and `duration` are both seconds; the distribution is
    reported in minutes because 0.6 is a readable number and 0.01 hours is not.
    Three clips of twenty seconds is one minute exactly, so an implementation
    comparing the minutes against the seconds floor would drop this speaker --
    and on the real corpus would drop very nearly everyone, a floor of an hour
    having become a demand for sixty. The same unit confusion already cost a
    test in the MoeSpeech pipeline.
    """
    md = _metadata(tmp_path, [("g", "s", 20.0)] * 3)
    assert select_speakers(md, ["g"], min_utterances=2, min_seconds=60.0) == ["g:s"]
    assert select_speakers(md, ["g"], min_utterances=2, min_seconds=61.0) == []


def test_a_speaker_has_to_clear_both_floors_and_not_either(tmp_path):
    """Each floor rejects something the other admits, which is why there are
    two. `chatty` is a bit part: eight lines of five seconds, plenty of clips
    and forty seconds of voice to clone from. `terse` has three minutes in two
    clips, and two clips is a held-out speaker that can be evaluated exactly
    once. Only `good` clears both.

    Under `or` all three survive; under either floor by itself, two do.
    """
    rows = [("g", "chatty", 5.0)] * 8 + [("g", "terse", 90.0)] * 2 + [("g", "good", 50.0)] * 4
    md = _metadata(tmp_path, rows)
    assert select_speakers(md, ["g"], min_utterances=3, min_seconds=150.0) == ["g:good"]


def test_the_table_promises_the_count_the_selection_delivers(tmp_path):
    """An operator reads a row off the retention table and hands those two
    numbers straight to the selection. If the table compares inclusively and
    the selection does not, the corpus arrives short of what was chosen and
    nothing anywhere says so. `on-the-floor` sits exactly on both grid values,
    which is the only place the two can disagree.
    """
    rows = [("g", "on-the-floor", 600.0)] * 6  # 3,600 s exactly, in six clips
    rows += [("g", "thin", 1000.0)] * 3  # clears the utterances, not the seconds
    rows += [("g", "over", 500.0)] * 9
    rows += [("g", "once", 45.0)]
    md = _metadata(tmp_path, rows)

    row = _row(probe_speakers(md, ["g"]), min_utterances=3, min_seconds=3600)
    kept = select_speakers(md, ["g"], min_utterances=3, min_seconds=3600)

    assert kept == ["g:on-the-floor", "g:over"], "a floor is a floor, not a strict inequality"
    assert row["kept"] == len(kept)


def test_the_probe_counts_speakers_apart_from_their_utterances(tmp_path):
    """Three quantities that a corpus of 19,349 speakers and 7,405,094
    utterances keeps inviting the confusion between. All three differ in the
    fixture: three speakers, thirteen utterances, one hour."""
    rows = [("g", "a", 100.0)] * 7 + [("g", "b", 200.0)] * 4 + [("g", "c", 1050.0)] * 2
    stats = probe_speakers(_metadata(tmp_path, rows), ["g"])
    assert (stats["speakers"], stats["utterances"]) == (3, 13)
    assert stats["hours"] == pytest.approx(1.0)


def test_the_distribution_is_per_speaker_minutes_and_reports_the_median(tmp_path):
    """The number this whole task turns on is a median: 0.6 minutes per speaker,
    against a mean dragged far above it by the 2,095 speakers holding 89% of
    the audio. Report the mean and GOL looks like a corpus of half-hour
    speakers, which is a corpus that does not exist. The fixture is that shape
    in miniature, and its median in minutes (0.3) is not its median in seconds
    (18).
    """
    rows = [("g", "a", 6.0)] + [("g", "b", 9.0)] * 2 + [("g", "c", 1500.0)] * 4
    stats = probe_speakers(_metadata(tmp_path, rows), ["g"])

    minutes = stats["minutes_per_speaker"]
    assert minutes["median"] == pytest.approx(0.3)
    assert minutes["mean"] == pytest.approx((0.1 + 0.3 + 100.0) / 3)
    assert minutes["max"] == pytest.approx(100.0)
    # A second distribution, over a different quantity of the same speakers.
    assert stats["utterances_per_speaker"]["median"] == pytest.approx(2)


def test_the_probe_measures_only_the_games_that_were_taken(tmp_path):
    """The table and the selection have to describe one corpus, or the numbers
    a floor is chosen from are about a corpus this run does not hold. Measured
    over the whole file instead, every row would be about 596 games while the
    run had eighteen tars on disk."""
    rows = [("kept", "a", 100.0)] * 3
    rows += [("dropped", "b", 100.0)] * 5 + [("dropped", "c", 100.0)] * 7
    stats = probe_speakers(_metadata(tmp_path, rows), ["kept"])
    assert (stats["speakers"], stats["utterances"]) == (1, 3)


def test_speakers_come_out_in_a_stated_order(tmp_path):
    """`speakers.json` is written out of this list and read back on the re-run
    after a preemption, and the held-out split is taken off it. Left in whatever
    order the metadata happened to be read in -- or in count order -- a resumed
    run holds out a different set of speakers than the run before it, and the
    validation loss stops being comparable across the kill.

    The fixture's file order, its utterance-count order and its alphabetical
    order are three different orders, so only one of them passes here.
    """
    rows = [("g", "sc", 60.0)] * 4 + [("g", "sa", 60.0)] * 3 + [("g", "sb", 60.0)] * 5
    md = _metadata(tmp_path, rows)
    keys = select_speakers(md, ["g"], min_utterances=2, min_seconds=0.0)
    assert keys == ["g:sa", "g:sb", "g:sc"]


def test_the_table_reports_the_audio_a_floor_keeps_and_not_only_the_speakers(tmp_path):
    """The row the next task takes its defaults off reads two ways at once: on
    the real metadata a one-hour floor keeps 11% of the speakers and 89% of the
    audio. A table of speaker counts alone would make that floor look like it
    threw the corpus away, which is exactly backwards. The two fractions here
    are 5% and 97%, so no assertion can take one for the other.
    """
    rows = [("g", "big", 3600.0)] * 2
    rows += [("g", f"t{i}", 5.0) for i in range(20) for _ in range(2)]
    stats = probe_speakers(_metadata(tmp_path, rows), ["g"])

    row = _row(stats, min_utterances=2, min_seconds=3600)
    assert row["kept"] == 1
    assert row["hours"] == pytest.approx(2.0)
    assert row["speaker_fraction"] == pytest.approx(1 / 21)
    assert row["hours_fraction"] == pytest.approx(7200 / 7400)


def test_the_probe_names_how_many_speakers_have_a_single_utterance(tmp_path):
    """3,922 of GOL's 19,349, and the reason the headline speaker count cannot
    be designed around: they are corpus size that is not usable size.

    `solo` holds more audio than anyone else in the fixture, so what is counted
    here is the utterance count and not "speakers with little audio".
    """
    rows = [("g", "solo", 900.0), ("g", "alone", 60.0)]
    rows += [("g", "pair", 30.0)] * 2 + [("g", "trio", 30.0)] * 3
    stats = probe_speakers(_metadata(tmp_path, rows), ["g"])
    assert (stats["speakers"], stats["single_utterance_speakers"]) == (4, 2)


def test_the_probe_counts_speaker_keys_apart_from_speaker_ids(tmp_path):
    """The gap between the two is the whole of ruling G9: 30,193 keys against
    19,349 ids on the real corpus, because 3,522 ids appear in more than one
    game. An operator shown only one of the numbers would read the corpus as
    having grown, or the survey as having been wrong.

    Four keys, three ids and one id that spans -- four different numbers, so no
    assertion here can be satisfied by the quantity next to it. Bounded to a
    single game nothing spans anything, which is what separates "spans two
    games" from "appears more than once".
    """
    rows = [("g1", "shared", 100.0)] * 2 + [("g2", "shared", 100.0)] * 3
    rows += [("g1", "own", 100.0)] * 4 + [("g2", "another", 100.0)] * 5
    md = _metadata(tmp_path, rows)

    both = probe_speakers(md, ["g1", "g2"])
    assert (both["speakers"], both["speaker_ids"], both["utterances"]) == (4, 3, 14)
    assert both["ids_spanning_games"] == 1

    one = probe_speakers(md, ["g1"])
    assert (one["speakers"], one["speaker_ids"], one["ids_spanning_games"]) == (2, 2, 0)


def test_one_speaker_id_in_two_games_is_two_speakers(tmp_path):
    """Ruling G9, pinned by its side effect. 3,522 of GOL's 19,349 ids appear
    in more than one game and hold 61% of the audio; the widest spans 115 games
    at 457 utterances of 1.8 seconds, which is the shape of a bucket for
    unnamed characters rather than of a prolific actor with a role. An id that
    *might* mean two people is enough: `concatenate`'s cross-speaker guard
    compares the label, not the voice, so a merged label puts two voices in one
    file and everything downstream reads that file as one speaker.

    `aoi` has three utterances in one game and four in the other, and the two
    counts differ so that a merged key is distinguishable from either half
    rather than only from their sum. Merged, it is one speaker of seven
    utterances and 700 seconds; kept apart it is two of three and four, 300 and
    400 -- so each floor below finds nothing here and would find something
    under a bare-id key.
    """
    rows = [("g1", "aoi", 100.0)] * 3 + [("g2", "aoi", 100.0)] * 4
    md = _metadata(tmp_path, rows)

    assert select_speakers(md, ["g1", "g2"], min_utterances=2, min_seconds=0.0) == [
        "g1:aoi",
        "g2:aoi",
    ]
    assert select_speakers(md, ["g1", "g2"], min_utterances=5, min_seconds=0.0) == []
    assert select_speakers(md, ["g1", "g2"], min_utterances=2, min_seconds=500.0) == []


def test_a_game_that_matched_nothing_measures_an_empty_corpus(tmp_path):
    """A mistyped game id is the likeliest way to get here, and the probe is the
    first thing that runs over the selection. Reporting zero rather than
    dividing by an empty corpus keeps the failure where an operator can read it,
    instead of ending the run in a ZeroDivisionError four stages early."""
    md = _metadata(tmp_path, [("g", "a", 60.0)] * 2)
    stats = probe_speakers(md, ["typo"])
    assert (stats["speakers"], stats["utterances"], stats["hours"]) == (0, 0, 0.0)
    assert all(r["kept"] == 0 and r["speaker_fraction"] == 0.0 for r in stats["retention"])
    assert stats["minutes_per_speaker"]["median"] is None
    assert select_speakers(md, ["typo"], min_utterances=2, min_seconds=0.0) == []


def _wav(path, seconds=1.0, hz=440.0, sr=48000):
    """A pure tone at GOL's own 48 kHz, which nothing here converts (G6)."""
    import numpy as np
    import sphn

    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sphn.write_wav(str(path), (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32), sr)
    return path


def _at(directory, name="x.wav", seconds=1.0, hz=440.0):
    """One clip, at whatever depth the caller names."""
    directory.mkdir(parents=True, exist_ok=True)
    return _wav(directory / name, seconds, hz)


def _clip(root, game, speaker, name="x.wav", seconds=1.0, hz=440.0):
    """One clip where extract_game leaves it: `<root>/<game>/<game>/<speaker>/`.

    The tar carries its own game directory at the top and is unpacked into one,
    so the tree is double-nested (G2).
    """
    return _at(root / game / game / speaker, name, seconds, hz)


def test_an_utterance_carries_what_concatenate_needs(tmp_path):
    """`concatenate` is reused unmodified, so this asserts through it rather
    than against a list of key names: a dict carrying the right keys over the
    wrong values joins nothing, and the keys it reads are the whole interface.

    Two clips, so the second entry's offset says the first one's audio was
    actually laid down -- through a single clip `start` is 0.0 whatever the
    function was handed.
    """
    md = _metadata(tmp_path, [("game", "spk", 1.5, "x.wav"), ("game", "spk", 1.5, "y.wav")])
    root = tmp_path / "extracted"
    _clip(root, "game", "spk", "x.wav", 1.5, 440.0)
    _clip(root, "game", "spk", "y.wav", 1.5, 660.0)

    us = list(gol_utterances(md, root, ["game:spk"], ["game"]))
    entries = concatenate(us, tmp_path / "joined.wav", target_sec=120.0)
    assert [e["id"] for e in entries] == [u["id"] for u in us]
    assert {e["speaker"] for e in entries} == {"game:spk"}
    assert [e["start"] for e in entries] == [0.0, 1.5]


def test_the_duration_is_the_one_the_corpus_claims(tmp_path):
    """`concatenate` measures every offset off the samples it lays down and
    uses this number for one thing only: to report how far the corpus's own
    metadata is from its audio. Measuring the wav here instead would hand it
    two copies of the same measurement, and that signal -- "this corpus's
    durations cannot be trusted elsewhere" -- would read as a clean corpus
    however wrong the metadata was.

    So the row claims five seconds over one second of audio, and the claim is
    what comes out. A fixture whose column agreed with its audio could not tell
    the two apart, and the column is in seconds besides: the same unit
    confusion already cost a test on the game selection above.
    """
    md = _metadata(tmp_path, [("game", "spk", 5.0)])
    root = tmp_path / "extracted"
    _clip(root, "game", "spk", "x.wav", seconds=1.0)

    (u,) = list(gol_utterances(md, root, ["game:spk"], ["game"]))
    assert u["duration"] == 5.0


def test_the_speaker_is_the_composite_key_so_two_games_are_two_voices(tmp_path):
    """Ruling G9, asserted through the consequence rather than the string.

    3,522 of GOL's 19,349 speaker ids appear in more than one game and hold 61%
    of the corpus. `concatenate`'s cross-speaker guard compares the label, so a
    bare id shared by two games passes that guard and two voices land in one
    joined file -- and the loader then takes one side of a cut in it as the
    voice prompt for the other. Raising here is what says the composite key
    reached the guard at all.

    The game ids and the speaker id are textually different, so a key built the
    wrong way round, or out of one half, is not the key built out of both.
    """
    md = _metadata(tmp_path, [("alpha", "voice", 1.0), ("beta", "voice", 1.0)])
    root = tmp_path / "extracted"
    _clip(root, "alpha", "voice")
    _clip(root, "beta", "voice")

    both = list(gol_utterances(md, root, ["alpha:voice", "beta:voice"], ["alpha", "beta"]))
    assert sorted(u["speaker"] for u in both) == ["alpha:voice", "beta:voice"]
    with pytest.raises(ValueError, match="one speaker at a time"):
        concatenate(both, tmp_path / "joined.wav", target_sec=120.0)

    (one,) = list(gol_utterances(md, root, ["alpha:voice"], ["alpha", "beta"]))
    assert one["speaker"] == "alpha:voice"


def test_the_transcript_is_normalized(tmp_path):
    """The manifest transcript, the tokenizer corpus and a user's inference
    input have to be one distribution, exactly as on the MoeSpeech path; a
    mismatch raises nothing and shows up only as a model that never quite
    becomes intelligible.

    The text folds on two axes at once -- fullwidth latin to ASCII and an
    ideographic space to a collapsed one -- and neither is what the row says,
    so a transcript passed through untouched is a different string.
    """
    md = _metadata(tmp_path, [("game", "spk", 1.0)], text="ＡＢＣ　です")
    root = tmp_path / "extracted"
    _clip(root, "game", "spk")

    (u,) = list(gol_utterances(md, root, ["game:spk"], ["game"]))
    assert u["transcript"] == "ABC です"


def test_a_row_whose_text_normalizes_away_is_not_a_training_example(tmp_path):
    """1,055 of GOL's 7.4M rows carry an empty text column, and normalization
    empties others: an ideographic space is text until NFKC folds it to a space
    and the collapse strips it. A clip with no text left trains the model on
    speech it is given no reason for, and align_data has nothing to segment.

    One row survives beside it, so a function that yielded both -- or neither --
    is distinguishable from one that drops only the blank. The blank row's text
    is not empty as written, so a check made before normalization keeps it.
    """
    md = _metadata(
        tmp_path,
        [("game", "spk", 1.0, "blank.wav", "　"), ("game", "spk", 1.0, "kept.wav", "こんにちは")],
    )
    root = tmp_path / "extracted"
    _clip(root, "game", "spk", "blank.wav")
    _clip(root, "game", "spk", "kept.wav")

    us = list(gol_utterances(md, root, ["game:spk"], ["game"]))
    assert [(u["transcript"], u["wav"].name) for u in us] == [("こんにちは", "kept.wav")]


def test_an_utterance_whose_wav_is_missing_is_dropped(tmp_path):
    """metadata.tsv describes all 7,405,094 files; the tars actually taken hold
    a subset, and an interrupted extraction holds less than that. A row without
    its audio is not a training example -- `concatenate` would skip it, but only
    after the manifest had already promised it.

    One row of the two is on disk, so dropping everything is as visibly wrong
    as dropping nothing.
    """
    md = _metadata(tmp_path, [("game", "spk", 1.0, "here.wav"), ("game", "spk", 1.0, "gone.wav")])
    root = tmp_path / "extracted"
    _clip(root, "game", "spk", "here.wav")

    us = list(gol_utterances(md, root, ["game:spk"], ["game"]))
    assert [u["wav"].name for u in us] == ["here.wav"]


def test_the_clip_is_found_wherever_the_tar_put_it(tmp_path):
    """Ruling G2: what `extract_game` leaves is `<root>/<game>/<game>/<speaker>/`
    because the tar carries its own game directory at the top, and MoeSpeech
    already cost this project a round of debugging by having its clips one
    directory below where the code assumed. So nothing here derives a depth:
    one clip sits where the real tars put it and another one directory further
    down, and both are found.

    The whole path is asserted and not the file name: a pass that joined the
    stated path onto the extract root and never looked would name a wav that is
    not there, and every name in it would still be right.
    """
    md = _metadata(tmp_path, [("game", "spk", 1.0, "flat.wav"), ("game", "spk", 1.0, "deeper.wav")])
    root = tmp_path / "extracted"
    flat = _clip(root, "game", "spk", "flat.wav")
    deeper = _at(root / "game" / "game" / "disc2" / "spk", "deeper.wav")

    us = list(gol_utterances(md, root, ["game:spk"], ["game"]))
    assert [u["wav"] for u in us] == [flat, deeper]


def test_the_speaker_never_comes_from_the_directory_name(tmp_path):
    """Ruling G2 again, and the trap MoeSpeech's `read_annotation` documents:
    take the speaker from a directory name and a corpus that packs its clips one
    level down labels every character after whatever directory happens to be
    there. `concatenate` then sees one speaker where there are two and joins
    them, the split holds out a label rather than a character, and none of it
    raises.

    metadata.tsv states the speaker in its own column, so the tree decides only
    where the audio is. What separates the two here is ruling G9: the speaker
    is `<game_id>:<speaker>`, assembled from two of the metadata's columns, and
    no directory in this tree is named that. The clip does sit under a `spk`
    directory, because that is where the metadata says it is and the lookup now
    holds the corpus to its own claim -- see the test below for why. So the
    directory name is available to be taken and is still not the answer.
    """
    md = _metadata(tmp_path, [("game", "spk", 1.0)])
    root = tmp_path / "extracted"
    _clip(root, "game", "spk", "x.wav")

    (u,) = list(gol_utterances(md, root, ["game:spk"], ["game"]))
    assert u["speaker"] == "game:spk"
    assert u["speaker"] != Path(u["wav"]).parent.name


def test_two_speakers_sharing_a_file_name_keep_their_own_audio(tmp_path):
    """Visual novels number a character's lines from one, so the same file name
    under two speakers of one game is the ordinary case rather than an odd one.

    Matched on the file name alone both rows get whichever clip was found
    first: one speaker's transcript over the other's voice, in a manifest that
    parses and offsets that are all inside real files. Nothing downstream can
    detect that, so it is asserted here on the paths themselves.
    """
    md = _metadata(tmp_path, [("game", "ann", 1.0, "0001.wav"), ("game", "bob", 1.0, "0001.wav")])
    root = tmp_path / "extracted"
    _clip(root, "game", "ann", "0001.wav", hz=440.0)
    _clip(root, "game", "bob", "0001.wav", hz=880.0)

    us = list(gol_utterances(md, root, ["game:ann", "game:bob"], ["game"]))
    assert [(u["speaker"], u["wav"]) for u in us] == [
        ("game:ann", root / "game" / "game" / "ann" / "0001.wav"),
        ("game:bob", root / "game" / "game" / "bob" / "0001.wav"),
    ]


def test_a_clip_that_is_not_where_the_metadata_says_is_missing_rather_than_theirs(tmp_path):
    """The test above narrows two candidates down by the path the metadata
    states. One candidate was never narrowed at all, and a name unique within a
    game was therefore accepted wherever in the game it happened to sit.

    So a clip under the wrong speaker -- a game whose tar packs one character's
    lines under another's directory, or a name reused across speakers where only
    one of the two clips was extracted -- is handed to this row as if it were
    its own: that character's voice under this row's transcript, in a manifest
    that parses, with every offset inside a real file, and nothing downstream
    able to see it. The metadata states where the clip should be; matching that
    costs the same lookup whether there is one candidate or two.

    Counted as missing instead, which is a number the run already reports at the
    end of the walk. The corpus is then short a clip rather than wrong about
    one, and short is the direction this pipeline can survive.
    """
    md = _metadata(tmp_path, [("game", "ann", 1.0, "0001.wav")])
    root = tmp_path / "extracted"
    # The only 0001.wav in the game, and it is not ann's.
    _clip(root, "game", "bob", "0001.wav")

    assert list(gol_utterances(md, root, ["game:ann"], ["game"])) == []


def test_only_the_games_this_run_took_are_read(tmp_path, caplog):
    """The extract root accumulates across runs: it is where a larger run's
    games were unpacked, and asking for fewer hours afterwards has to mean
    fewer games or the flag does nothing at all.

    Both games are on disk and both speakers are offered, so the game bound is
    the only thing that can separate them.

    What is reported is asserted beside what is yielded, because dropping a row
    for the right reason and dropping it for the wrong one look identical from
    outside: a game outside `game_ids` has no clips indexed either, so its rows
    fall out of the audio lookup whatever this bound does -- and are then
    counted as audio the extraction is missing. On the real metadata that is
    6.7 million of 7.4 million rows, a warning that reads like a failed run
    over a run that took exactly what it was asked for.
    """
    md = _metadata(tmp_path, [("kept", "spk", 1.0), ("older", "spk", 1.0)])
    root = tmp_path / "extracted"
    _clip(root, "kept", "spk")
    _clip(root, "older", "spk")

    with caplog.at_level(logging.WARNING, logger="prepare_gol"):
        us = list(gol_utterances(md, root, ["kept:spk", "older:spk"], ["kept"]))
    assert [u["speaker"] for u in us] == ["kept:spk"]
    assert [r.message for r in caplog.records] == []


def test_a_speaker_the_floors_rejected_is_not_read(tmp_path):
    """A GOL tar is a whole work and its median 35 speakers come along with it,
    most of them below `select_speakers`'s floors. Those floors are the entire
    point of the stage before this one, so a speaker on disk but not in the
    selection is not an utterance.

    Both speakers are in the taken game and both have their audio, so the
    speaker bound is the only thing that can separate them.
    """
    md = _metadata(tmp_path, [("game", "lead", 1.0), ("game", "extra", 1.0)])
    root = tmp_path / "extracted"
    _clip(root, "game", "lead")
    _clip(root, "game", "extra")

    us = list(gol_utterances(md, root, ["game:lead"], ["game"]))
    assert [u["speaker"] for u in us] == ["game:lead"]


def test_utterances_come_out_in_metadata_order(tmp_path):
    """`concatenate` lays clips down in the order it is handed them and writes
    a manifest of offsets into the files it builds, so a re-run after a
    preemption has to hand them over in that same order or every offset
    describes different audio -- and nothing downstream can tell.

    metadata.tsv's own order is that order, and it is not the filesystem's:
    `b.wav` is written first here and sorts second, so a pass that returned
    whatever `rglob` found, or that sorted by path, comes out the other way
    round.
    """
    md = _metadata(tmp_path, [("game", "spk", 1.0, "b.wav"), ("game", "spk", 1.0, "a.wav")])
    root = tmp_path / "extracted"
    _clip(root, "game", "spk", "b.wav")
    _clip(root, "game", "spk", "a.wav")

    us = list(gol_utterances(md, root, ["game:spk"], ["game"]))
    assert [u["wav"].name for u in us] == ["b.wav", "a.wav"]


def test_the_id_names_the_clip_in_the_whole_corpus(tmp_path):
    """The manifest entry `concatenate` builds names the joined file, never the
    clip that went into it, so `id` is the only way back from a suspect row to
    the wav it came from. A bare file stem is not that: 596 games number their
    lines the same way and `0001` names one clip in each of them.
    """
    md = _metadata(tmp_path, [("alpha", "spk", 1.0, "0001.wav"), ("beta", "spk", 1.0, "0001.wav")])
    root = tmp_path / "extracted"
    _clip(root, "alpha", "spk", "0001.wav")
    _clip(root, "beta", "spk", "0001.wav")

    us = list(gol_utterances(md, root, ["alpha:spk", "beta:spk"], ["alpha", "beta"]))
    assert [u["id"] for u in us] == ["alpha/spk/0001", "beta/spk/0001"]


def test_a_game_that_was_never_extracted_costs_only_its_own_rows(tmp_path):
    """A run killed between tars leaves exactly this, and it is the ordinary
    state of the extract root rather than a corrupt one. The game that is there
    has to still produce its utterances, instead of the pass ending on the one
    that is not.
    """
    md = _metadata(tmp_path, [("here", "spk", 1.0), ("never", "spk", 1.0)])
    root = tmp_path / "extracted"
    _clip(root, "here", "spk")

    us = list(gol_utterances(md, root, ["here:spk", "never:spk"], ["here", "never"]))
    assert [u["speaker"] for u in us] == ["here:spk"]


def _entries(directory, rows):
    """An `entries/` directory of the shape stage 6 of either script leaves.

    `rows` maps a speaker to the durations of that speaker's clips, and one
    file per speaker is written with `write_manifest`, which is the function
    that writes the real ones -- so what is read back here is the format, not a
    test's idea of it.

    The files are numbered rather than named after the speaker they hold. A GOL
    key holds a colon, which is not a legal Windows file name, and the speaker
    is a column of every row in any case: never the name of the file the row
    sits in.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, (speaker, durations) in enumerate(sorted(rows.items())):
        write_manifest(
            [
                {
                    "id": f"{speaker}/{j:04d}",
                    "speaker": speaker,
                    "path": (directory.parent / "audio" / f"{i:04d}.wav").as_posix(),
                    "start": float(j),
                    "duration": duration,
                    "transcript": "あ",
                }
                for j, duration in enumerate(durations)
            ],
            directory / f"{i:04d}.jsonl",
        )
    return directory


def test_the_two_corpora_are_split_as_one(tmp_path):
    """A speaker held out of GOL but present in MoeSpeech would be in both
    splits, and neither corpus's own split can see the other.

    `split_by_speaker` takes the smallest speakers first, and "smallest" is a
    fact about the list it is handed. Over the union below, all four GOL
    speakers are smaller than every MoeSpeech one, so the half hour held out is
    bought entirely out of GOL. Split the two corpora separately at the same
    half hour and MoeSpeech would hold out アリス as well -- so アリス being
    trained on is what tells one merged split from two separate ones, and the
    other three MoeSpeech characters being there at all is what tells it from a
    merge that quietly dropped a corpus.
    """
    gol = _entries(tmp_path / "gol" / "entries", {f"g1:{c}": [300.0, 300.0] for c in "abcd"})
    moe = _entries(
        tmp_path / "moe" / "entries",
        {n: [3000.0, 3000.0] for n in ("アリス", "ボブ", "キャロル", "デイジー")},
    )

    # Through split_across_corpora because that is what stage 7 will call. Its
    # own floor is what the two tests below are about; this one is about which
    # speakers are held out, so it is turned down out of the way.
    train, valid = split_across_corpora(merge_entries([gol, moe]), valid_hours=0.5, minimum=1)

    held_out = {e["speaker"] for e in valid}
    assert held_out == {"g1:a", "g1:b", "g1:c"}, held_out
    trained = {e["speaker"] for e in train}
    assert trained == {"g1:d", "アリス", "ボブ", "キャロル", "デイジー"}, trained


def test_every_corpus_named_reaches_the_merge(tmp_path):
    """The merge is where a corpus can vanish with nothing downstream noticing:
    a manifest built out of one of the two parses, aligns and trains, and the
    only sign is a speaker count nobody has a second number to compare against.

    The rows are asserted by identity rather than by count, and their order is
    asserted too: the corpora arrive in the order they were named and, inside
    one, in the order of the file names. A merge that dropped a corpus, read
    one twice, or gathered the rows through a set fails here.
    """
    gol = _entries(tmp_path / "gol" / "entries", {"g1:a": [10.0], "g2:b": [20.0, 30.0]})
    moe = _entries(tmp_path / "moe" / "entries", {"アリス": [40.0]})

    merged = merge_entries([gol, moe])

    assert [e["id"] for e in merged] == ["g1:a/0000", "g2:b/0000", "g2:b/0001", "アリス/0000"]
    assert [e["duration"] for e in merged] == [10.0, 20.0, 30.0, 40.0]
    assert [e["id"] for e in merge_entries([moe, gol])] == [
        "アリス/0000",
        "g1:a/0000",
        "g2:b/0000",
        "g2:b/0001",
    ]


def test_a_corpus_that_names_no_rows_is_refused(tmp_path):
    """Contributing nothing is the silent form of the bug above. An `entries`
    directory that was never written -- a MoeSpeech run pointed at another
    --out, or one that has not got that far -- would merge into exactly the
    manifest a one-corpus run produces, and the run would report success over
    it."""
    gol = _entries(tmp_path / "gol" / "entries", {"g1:a": [10.0, 10.0]})
    # The two cases are told apart in the message, because they are different
    # mistakes: one --out is wrong, the other has not got this far yet.
    with pytest.raises(typer.BadParameter, match="not a directory"):
        merge_entries([gol, tmp_path / "moe" / "entries"])
    empty = tmp_path / "empty" / "entries"
    empty.mkdir(parents=True)
    with pytest.raises(typer.BadParameter, match="no utterances"):
        merge_entries([gol, empty])


def test_the_speaker_is_the_rows_own_column_and_not_the_file_name(tmp_path):
    """The same trap as ruling G2, one stage later. Here it is worse than a
    wrong label: a GOL key holds a colon, so it cannot be a Windows file name
    at all, and a merge that read the speaker off the file name would rename
    every GOL speaker to whatever the writer had had to substitute."""
    entries = tmp_path / "gol" / "entries"
    entries.mkdir(parents=True)
    write_manifest(
        [{"id": "x", "speaker": "g1:主人公", "duration": 10.0}] * 2, entries / "decoy.jsonl"
    )

    assert {e["speaker"] for e in merge_entries([entries])} == {"g1:主人公"}


def test_a_gol_key_can_never_be_mistaken_for_a_moespeech_name(tmp_path):
    """Ruling G9. A GOL speaker is `<game_id>:<speaker>` and a MoeSpeech one is
    a bare character name, so no key of one corpus can equal a key of the
    other: the GOL side always holds a colon, and a MoeSpeech name is a
    directory name out of a zip, which cannot hold one on the machine that
    unpacked it. Both are opaque strings to `split_by_speaker`, which is why it
    is reused untouched -- and why nothing else would notice if the two label
    spaces ever did meet. Stated here rather than left to luck."""
    assert ":" in _speaker_key("g1", "主人公")
    gol = _entries(tmp_path / "gol" / "entries", {_speaker_key("g1", "主人公"): [10.0, 10.0]})
    moe = _entries(tmp_path / "moe" / "entries", {"主人公": [10.0, 10.0]})

    speakers = {e["speaker"] for e in merge_entries([gol, moe])}

    assert speakers == {"g1:主人公", "主人公"}, "one corpus's key was taken for the other's"


def test_a_label_that_turns_up_in_two_corpora_is_refused(tmp_path):
    """Every guard from here down compares this label and none of them can see
    a voice. Two characters sharing one label go to the same side of the split
    as a single speaker, so the held-out count is wrong, and the eval protocol
    -- clone a voice from one utterance, synthesize another -- would clone one
    of them and score the other against it."""
    a = _entries(tmp_path / "a" / "entries", {"アリス": [10.0, 10.0]})
    b = _entries(tmp_path / "b" / "entries", {"アリス": [10.0, 10.0]})
    with pytest.raises(typer.BadParameter, match="アリス"):
        merge_entries([a, b])
    # The same directory named twice is that collision with itself, and would
    # otherwise put every one of its rows into the manifest two times over.
    with pytest.raises(typer.BadParameter, match="アリス"):
        merge_entries([a, a])


def test_a_manifest_a_kill_left_half_written_is_not_merged(tmp_path):
    """`write_manifest` lands its lines beside the name and renames the file in
    only once they are all there, so a preemption leaves a `.partial` among the
    finished ones. Its rows are a speaker cut off in the middle, and the run
    that resumes rewrites them under the real name."""
    gol = _entries(tmp_path / "gol" / "entries", {"g1:a": [10.0, 10.0]})
    (gol / "0001.jsonl.partial").write_text(
        '{"id": "g1:b/0000", "speaker": "g1:b", "duration": 10.0}\n', encoding="utf-8"
    )

    assert {e["speaker"] for e in merge_entries([gol])} == {"g1:a"}


def test_the_valid_split_has_many_speakers(tmp_path):
    """Phase 1's validation set was one speaker, and one speaker cannot say
    when to stop -- its loss bottomed at 7,500 while the samples kept
    improving, because the samples use a training voice and the valid set does
    not. For a model whose point is voice cloning, the unseen number is the one
    that matters, so there have to be enough of them to mean something.

    Non-empty is asserted before the count and the utterances per speaker
    after it: an empty valid set satisfies "no speaker is in both splits" and
    "every valid speaker has more than one utterance" without holding anyone
    out, and that is the pair of vacuous assertions phase 1 shipped.
    """
    gol = _entries(tmp_path / "gol" / "entries", {f"g1:s{i:03d}": [400.0] * 3 for i in range(60)})
    moe = _entries(tmp_path / "moe" / "entries", {f"c{i:03d}": [1800.0] * 3 for i in range(10)})

    train, valid = split_by_speaker(merge_entries([gol, moe]), valid_hours=10.0)

    counts = Counter(e["speaker"] for e in valid)
    assert counts, "nothing was held out; everything below is vacuous"
    assert train, "nothing was left to train on"
    assert len(counts) >= 20, counts
    assert all(n > 1 for n in counts.values()), counts
    assert set(counts) & {e["speaker"] for e in train} == set()


def test_a_valid_split_of_too_few_speakers_is_refused(tmp_path):
    """A warning is what phase 1 had. `split_by_speaker` already says out loud
    that it could not hold out the hours it was asked for, and the run went on
    for 15,000 steps over a validation set of one voice regardless, so the
    count is checked where it can still stop the run."""
    entries = [{"speaker": f"s{i:03d}", "duration": 3600.0} for i in range(25) for _ in range(2)]
    with pytest.raises(typer.BadParameter, match="valid-hours"):
        split_across_corpora(entries, valid_hours=1.0)

    _, valid = split_across_corpora(entries, valid_hours=40.0)

    assert len({e["speaker"] for e in valid}) == 20


def test_how_many_speakers_the_valid_hours_buy_is_a_property_of_the_corpus(tmp_path):
    """The split takes the smallest speakers first, so what a --valid-hours
    buys is those hours divided by the size of the smallest speakers -- which
    the speaker floors set. At a 60-minute floor every speaker is an hour and
    ten valid-hours is ten voices, half of what M2a needs; the same ten hours
    over a corpus with no floor buys hundreds. Neither knob can be read without
    the other, which is why the refusal names both."""
    hour_long = [{"speaker": f"s{i:03d}", "duration": 1800.0} for i in range(40) for _ in range(2)]

    _, ten = split_by_speaker(hour_long, valid_hours=10.0)
    assert len({e["speaker"] for e in ten}) == 10

    with pytest.raises(typer.BadParameter):
        split_across_corpora(hour_long, valid_hours=10.0)
    _, twenty = split_across_corpora(hour_long, valid_hours=20.0)
    assert len({e["speaker"] for e in twenty}) == 20


def test_the_two_manifests_together_hold_every_merged_row(tmp_path):
    """The stage end to end: merge both corpora, split the union, write the two
    manifests the loader reads. Every row of both corpora is in exactly one of
    them -- a row dropped between the merge and the manifests is training data
    thrown away with nothing saying so, and a row in both is a valid utterance
    that was trained on."""
    gol = _entries(tmp_path / "gol" / "entries", {f"g1:s{i:03d}": [400.0] * 3 for i in range(60)})
    moe = _entries(tmp_path / "moe" / "entries", {f"c{i:03d}": [1800.0] * 3 for i in range(10)})
    merged = merge_entries([gol, moe])

    train, valid = split_across_corpora(merged, valid_hours=10.0)
    write_manifest(train, tmp_path / "train.jsonl")
    write_manifest(valid, tmp_path / "valid.jsonl")

    read = {
        name: [
            json.loads(line)
            for line in (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for name in ("train", "valid")
    }
    assert read["train"] and read["valid"]
    assert len({e["speaker"] for e in read["valid"]}) >= 20
    ids = {name: [e["id"] for e in rows] for name, rows in read.items()}
    assert sorted(ids["train"] + ids["valid"]) == sorted(e["id"] for e in merged)
    assert set(ids["train"]) & set(ids["valid"]) == set()
    assert {e["speaker"] for e in read["train"]} & {e["speaker"] for e in read["valid"]} == set()
    # The transcripts stay Japanese on a machine whose default encoding is
    # cp932, and unescaped, so the manifest can be read with `head`.
    assert "あ" in (tmp_path / "valid.jsonl").read_text(encoding="utf-8")


def _aligned(tmp_path, rows, name="aligned.jsonl"):
    """A stand-in for what `align_data.py` writes: the manifest it was handed,
    with the alignment's own score and both of that score's denominators added.

    A row is `(score, frames, tokens)`, optionally followed by that clip's
    duration. Any of the three may be None, which leaves that key off the row
    entirely -- which is what a real aligned manifest never holds, and what
    this stage therefore has to treat as something other than a bad alignment.

    Written with `write_manifest`, the function that writes the real ones, so
    what is read back here is the format and not a test's idea of it.
    """
    entries = []
    for i, row in enumerate(rows):
        entry = {
            "id": f"g:s/{i:04d}",
            "speaker": "g:s",
            "path": (Path(tmp_path) / "audio" / f"{i:04d}.wav").as_posix(),
            "start": 0.0,
            "duration": row[3] if len(row) > 3 else 1.0,
            "transcript": "あいう",
        }
        for key, value in zip(("score", "frames", "tokens"), row[:3]):
            if value is not None:
                entry[key] = value
        entry["words"] = [{"word": "あいう", "start": 0.0, "end": 0.5}]
        entries.append(entry)
    path = Path(tmp_path) / name
    write_manifest(entries, path)
    return path


# The four rows the three normalizations disagree about, written down once
# because several tests below need that disagreement and building it is fiddly.
# The raw score, the per-frame score and the per-token score rank these four in
# three different orders, so the better half under any one of them is a
# different pair from the better half under either other:
#
#         raw   frames  tokens  per_frame  per_token  seconds
#   r1  -10.0        5       2      -2.00      -5.00      4.0
#   r2  -20.0       40       1      -0.50     -20.00      1.0
#   r3  -30.0       30      20      -1.00      -1.50      8.0
#   r4  -40.0       10      16      -4.00      -2.50      2.0
#
# Best two by raw: r1 and r2. By per_frame: r2 and r3. By per_token: r3 and r4.
# Frames and tokens move independently of each other and of the score, which is
# the thing a batch of one cannot show: the aligner scores per utterance but
# runs a batch at a time, and three wrong denominators once survived a suite
# whose fixtures could not tell an item's own frames from the batch's padding.
DISAGREEING = [(-10.0, 5, 2, 4.0), (-20.0, 40, 1, 1.0), (-30.0, 30, 20, 8.0), (-40.0, 10, 16, 2.0)]


def _score_row(stats, normalization, quantile):
    """The single retention row for this normalization at this quantile.

    Unpacked rather than indexed, so a table that listed a pair twice would
    fail here instead of quietly reporting the first of two rows that an
    operator would then read as the only one. The same discipline as `_row`.
    """
    (row,) = [
        r
        for r in stats["retention"]
        if (r["normalization"], r["quantile"]) == (normalization, quantile)
    ]
    return row


def test_the_probe_reports_all_three_normalizations(tmp_path):
    """Raw log-prob scales with both frames and tokens, and which one the
    distribution supports is not knowable before measuring it. Reporting one
    would be choosing it.

    These two rows are deliberately the trivial fixture: they are proportional
    in all three normalizations, so nothing here can tell the three apart. That
    is all this test claims -- that three are reported. That they are three
    different measurements is the next test's job, and it needs a fixture this
    one would pass without.
    """
    stats = probe_scores(_aligned(tmp_path, [(-100.0, 200, 10), (-50.0, 100, 5)]))
    assert {"raw", "per_frame", "per_token"} <= set(stats)


def test_each_normalization_divides_by_its_own_denominator(tmp_path):
    """Three measurements, not one reported three times.

    The largest row differs under each: r1 by raw, r2 by per_frame, r3 by
    per_token -- so a distribution copied from one key into another lands a
    value that belongs to a different clip. The six numbers asserted here are
    all distinct, which is what makes swapping any two keys visible.
    """
    stats = probe_scores(_aligned(tmp_path, DISAGREEING))

    assert (stats["raw"]["min"], stats["raw"]["max"]) == (-40.0, -10.0)
    assert (stats["per_frame"]["min"], stats["per_frame"]["max"]) == (-4.0, -0.5)
    assert (stats["per_token"]["min"], stats["per_token"]["max"]) == (-20.0, -1.5)
    assert stats["raw"]["median"] == pytest.approx(-25.0)
    assert stats["per_frame"]["median"] == pytest.approx(-1.5)
    assert stats["per_token"]["median"] == pytest.approx(-3.75)


def test_the_probe_counts_the_corpus_it_measured(tmp_path):
    """The hours are the question this stage is asked -- the README's floor is
    100 of them -- and clips are not hours: three clips here, ninety seconds."""
    rows = [(-10.0, 5, 2, 60.0), (-20.0, 40, 1, 20.0), (-30.0, 30, 20, 10.0)]
    stats = probe_scores(_aligned(tmp_path, rows))
    assert stats["count"] == 3
    assert stats["hours"] == pytest.approx(90.0 / 3600)


def test_the_probe_does_not_measure_a_row_it_could_not_score(tmp_path):
    """A row with no score is not a badly aligned row; it is a row this stage
    cannot say anything about. Folding it into the distribution as a very bad
    score, or as a zero, would move the percentiles a cutoff is read off.

    Both would show here: -inf would take the minimum, 0.0 would take the
    maximum, and either would make the count three.
    """
    rows = [(-10.0, 4, 3), (-60.0, 20, 8), (None, None, None)]
    stats = probe_scores(_aligned(tmp_path, rows))

    assert (stats["count"], stats["unscored"]) == (2, 1)
    assert (stats["raw"]["min"], stats["raw"]["max"]) == (-60.0, -10.0)


def test_the_retention_table_names_a_cutoff_for_each_normalization(tmp_path):
    """A retention table over one normalization is that normalization chosen.
    The cutoffs are quantiles of the corpus's own scores rather than a written
    grid, because a log-probability has no scale known in advance -- the way
    CER and speechMOS did on the corpus this stage replaces.
    """
    stats = probe_scores(_aligned(tmp_path, DISAGREEING))
    assert {r["normalization"] for r in stats["retention"]} == {"raw", "per_frame", "per_token"}
    assert _score_row(stats, "raw", 0)["min_score"] == -40.0
    assert _score_row(stats, "per_frame", 0)["min_score"] == -4.0
    assert _score_row(stats, "per_token", 0)["min_score"] == -20.0


def test_the_hours_a_cutoff_leaves_depend_on_which_normalization_it_is(tmp_path):
    """The column that tells the three tables apart is the hours, not the count.

    A quantile cutoff keeps the same number of clips whichever normalization it
    is taken over -- two of these four, in all three rows below. Which two
    differs, and so the audio does: five seconds, nine, or ten out of the same
    fifteen. An operator comparing the three tables on `kept` would see three
    identical tables and conclude the normalization did not matter.
    """
    stats = probe_scores(_aligned(tmp_path, DISAGREEING))
    raw, per_frame, per_token = (
        _score_row(stats, n, 50) for n in ("raw", "per_frame", "per_token")
    )

    assert (raw["kept"], per_frame["kept"], per_token["kept"]) == (2, 2, 2)
    assert raw["hours"] == pytest.approx(5.0 / 3600)
    assert per_frame["hours"] == pytest.approx(9.0 / 3600)
    assert per_token["hours"] == pytest.approx(10.0 / 3600)


def test_the_table_reports_the_hours_a_cutoff_costs_and_not_only_the_clips(tmp_path):
    """Dropping a quarter of the clips is not dropping a quarter of the audio,
    and here the gap points the expensive way: the worst-scoring clips are the
    longest, so a cutoff that looks like it costs 25% of the corpus costs 40%
    of it. The MoeSpeech table reports hours beside counts for the same reason,
    and the target this pipeline is given is in hours.

    The four clips are 40, 30, 20 and 10 seconds, worst-scoring first.
    """
    # The tokens run the other way from the frames, so per_token ranks these
    # four almost in reverse and a table built on the wrong denominator keeps a
    # different three of them. Proportional tokens would have made this fixture
    # pass under either normalization, which is the coincidence that hides a
    # wrong denominator.
    rows = [
        (-80.0, 20, 40, 40.0),  # per_frame -4.0, per_token -2.0
        (-33.0, 11, 3, 30.0),  # per_frame -3.0, per_token -11.0
        (-54.0, 27, 6, 20.0),  # per_frame -2.0, per_token -9.0
        (-14.0, 14, 2, 10.0),  # per_frame -1.0, per_token -7.0
    ]
    row = _score_row(probe_scores(_aligned(tmp_path, rows)), "per_frame", 25)

    assert (row["kept"], row["fraction"]) == (3, 0.75)
    assert row["hours"] == pytest.approx(60.0 / 3600)
    assert row["hours_fraction"] == pytest.approx(0.6)


def test_a_manifest_with_nothing_to_measure_offers_no_cutoffs(tmp_path):
    """The aligner is the longest stage in the pipeline and the instance it
    runs on is preemptible, so an aligned manifest with no rows in it yet is an
    ordinary state to find rather than a corrupt one.

    A cutoff here is a quantile of the corpus's own scores, so with no scores
    there is no cutoff to offer -- and an empty table is the only honest answer.
    A table of rows saying a threshold of 0.0 keeps 0 clips would read as a
    measurement of a corpus that scored badly.
    """
    stats = probe_scores(_aligned(tmp_path, []))

    assert (stats["count"], stats["unscored"], stats["hours"]) == (0, 0, 0.0)
    assert stats["retention"] == []
    assert stats["per_frame"]["median"] is None


def test_no_default_threshold(tmp_path):
    """The corpus has never been measured. A number written here now would
    afterwards be indistinguishable from a measured one -- which is the failure
    ruling R1 of the MoeSpeech plan existed to prevent, and which the speechMOS
    measurement later vindicated: `--min-mos 3.0` was the plausible default
    there, and the corpus's median turned out to be 2.281, so it would have cut
    124.4 hours to 16.2 and written a manifest that looked entirely normal.

    Nothing may be written before the refusal either. A run that refused after
    leaving an output behind would be skipped by the next one, which is how
    every stage in this script decides it has already run.
    """
    aligned = _aligned(tmp_path, [(-10.0, 10, 5)])
    with pytest.raises(typer.BadParameter):
        filter_by_score(aligned, tmp_path / "kept.jsonl", None, "per_frame")
    assert not (tmp_path / "kept.jsonl").exists()


def test_a_better_alignment_is_the_one_that_survives(tmp_path):
    """The score is a log-probability, so less negative is better and the
    threshold is a floor. Reversed, this filter keeps exactly the clips it
    exists to throw away, and the manifest is the same size either way."""
    rows = [(-5.0, 10, 3), (-90.0, 9, 4)]  # per_frame -0.5 and -10.0
    out = tmp_path / "kept.jsonl"

    assert filter_by_score(_aligned(tmp_path, rows), out, -1.0, "per_frame") == 1
    assert [r["id"] for r in _read_jsonl(out)] == ["g:s/0000"]


def test_the_filter_cuts_under_the_normalization_it_was_given(tmp_path):
    """`probe_scores` reports all three and chooses none, which is only worth
    the apparatus if all three can then be acted on. A filter that could apply
    one of them would leave a measurement favouring either other one with
    nothing to do about it -- and would take a number read off those rows and
    silently keep the whole corpus, since a per-token cutoff is far below every
    per-frame score there is.

    One threshold, three answers, so the denominator cannot hide. At -1.75 the
    per-frame cut keeps r2 and r3; the same number per token keeps only r3, and
    raw keeps nothing at all -- a raw log-probability over a whole utterance is
    nowhere near -1.75. A fixture whose frames and tokens were proportional
    would give one answer three times.
    """
    kept = {}
    for normalization in ("raw", "per_frame", "per_token"):
        out = tmp_path / f"{normalization}.jsonl"
        assert filter_by_score(_aligned(tmp_path, DISAGREEING), out, -1.75, normalization) == len(
            _read_jsonl(out)
        )
        kept[normalization] = [r["id"] for r in _read_jsonl(out)]

    assert kept["per_frame"] == ["g:s/0001", "g:s/0002"], kept
    assert kept["per_token"] == ["g:s/0002"], kept
    assert kept["raw"] == [], kept


def test_a_normalization_the_probe_does_not_report_is_refused(tmp_path):
    """The three names are the retention table's own column, and the number
    passed beside one of them was read off the rows carrying it. A fourth name
    -- or one of these three misspelt -- is a cutoff whose meaning nobody can
    recover afterwards, so it is refused rather than defaulted to per-frame,
    which would silently apply a threshold to the scale it was not read off.
    """
    aligned = _aligned(tmp_path, [(-1.0, 10, 3)])

    with pytest.raises(typer.BadParameter, match="per_token"):
        filter_by_score(aligned, tmp_path / "kept.jsonl", -1.0, "per-frame")

    assert not (tmp_path / "kept.jsonl").exists()


def test_the_table_promises_the_count_the_filter_delivers(tmp_path):
    """An operator reads a cutoff off the retention table and hands that number
    straight to the filter. If the table compares inclusively and the filter
    does not, the corpus arrives short of what was chosen and nothing says so.

    Five rows put the 25th percentile exactly on the second-smallest score
    rather than between two of them, which is the only place the two
    comparisons can disagree -- and it is where an operator lands, because
    every cutoff this table offers is one of the corpus's own values.
    """
    rows = [
        (-50.0, 10, 3),  # per_frame -5.0
        (-28.0, 7, 8),  # per_frame -4.0, and the 25th percentile of the five
        (-39.0, 13, 5),  # per_frame -3.0
        (-18.0, 9, 12),  # per_frame -2.0
        (-11.0, 11, 2),  # per_frame -1.0
    ]
    aligned = _aligned(tmp_path, rows)
    row = _score_row(probe_scores(aligned), "per_frame", 25)

    assert row["min_score"] == -4.0, "the cutoff has to be a value the corpus actually holds"
    assert row["kept"] == 4, "a floor is a floor, not a strict inequality"
    # The row's own two columns, handed over together: `min_score` is only a
    # cutoff alongside the `normalization` it was measured under, and reading
    # one off the table without the other is the mistake the filter now refuses.
    kept = filter_by_score(aligned, tmp_path / "kept.jsonl", row["min_score"], row["normalization"])
    assert kept == row["kept"]


def test_a_row_without_a_score_is_kept_and_counted(tmp_path, caplog):
    """align_data emits no score for an utterance it could not align, and those
    are already absent from the aligned manifest. A row that somehow has none is
    a different problem and must not be silently filtered as a bad alignment.

    Dropping it is the plausible reading -- `row.get("score", -inf)` is one
    character of carelessness -- and it is wrong in the expensive direction:
    the alignment has already been paid for, and a manifest quietly missing
    rows nothing ever explained is what this whole stage exists to avoid. So it
    goes through, and it is said out loud.
    """
    rows = [(-10.0, 10, 5), (-99.0, 9, 3), (None, None, None)]
    out = tmp_path / "kept.jsonl"

    with caplog.at_level(logging.WARNING, logger="prepare_gol"):
        kept = filter_by_score(_aligned(tmp_path, rows), out, -2.0, "per_frame")

    assert kept == 2
    assert [r["id"] for r in _read_jsonl(out)] == ["g:s/0000", "g:s/0002"]
    assert any("no score" in r.getMessage() for r in caplog.records), caplog.text


def test_a_row_with_a_score_but_no_denominator_is_kept_too(tmp_path):
    """The score is what the filter reads, but frames is what it divides by,
    and that division is where a row missing half of what it needs turns into
    an exception in the middle of a stage that has already cost GPU-hours.

    Both rows carry the same -10.0, so a filter falling back to the raw score
    would drop the second one against a per-frame threshold.
    """
    rows = [(-10.0, 10, 4), (-10.0, None, 4)]
    out = tmp_path / "kept.jsonl"

    assert filter_by_score(_aligned(tmp_path, rows), out, -2.0, "per_frame") == 2
    assert [r["id"] for r in _read_jsonl(out)] == ["g:s/0000", "g:s/0001"]


def test_the_filter_changes_nothing_but_which_rows_are_present(tmp_path):
    """The rows go through byte for byte. The aligned manifest is the largest
    file this pipeline writes and every field in it belongs to a later stage --
    the word timestamps the loader cuts on above all -- so this stage rewriting
    them is only a chance to change one. `ensure_ascii` left at its default is
    the concrete way that happens: the file stays valid JSON and stops being
    readable with `head`, which is how anyone here checks a manifest at all.
    """
    aligned = _aligned(tmp_path, DISAGREEING)
    lines = aligned.read_bytes().splitlines(keepends=True)
    out = tmp_path / "kept.jsonl"

    filter_by_score(aligned, out, -1.75, "per_frame")

    assert out.read_bytes() == lines[1] + lines[2]


def test_a_kill_mid_filter_leaves_nothing_under_the_finished_name(tmp_path, monkeypatch):
    """A manifest a preemption cut short is still valid JSONL -- every line
    parses and every path exists -- so nothing downstream can tell it from a
    finished one, and the re-run finds it sitting there and skips the stage. So
    the lines land beside the name and are renamed in only once they are there.
    """
    from training.scripts import prepare_gol as m

    def killed(src, dst):
        raise KeyboardInterrupt

    aligned = _aligned(tmp_path, DISAGREEING)
    out = tmp_path / "kept.jsonl"
    monkeypatch.setattr(m.os, "replace", killed)

    with pytest.raises(KeyboardInterrupt):
        m.filter_by_score(aligned, out, -1.75, "per_frame")

    assert not out.exists(), "the kill left something under the finished name"


# ---------------------------------------------------------------------------
# The nine stages as one command.
#
# Everything below runs `main`. Only what would reach the network or the GPU is
# faked -- the hub fetch, the two download/extract stages and the aligner -- so
# a recording says that main wired the real functions together in the real
# order rather than that it called mocks in one, and the manifests it leaves
# behind can be read for what the wiring decided.

# A speaker is two clips of half a second, which the fake aligner then scores
# one of well and one of badly. Half a second because there are fifty of them:
# main holds `split_across_corpora`'s twenty-speaker floor, so a corpus small
# enough to be cheap is a corpus this command refuses.
CLIP_SEC = 0.5
CORPUS = {
    "g-big": [f"s{i:02d}" for i in range(13)],
    "g-small": [f"s{i:02d}" for i in range(13, 25)],
}
# g-big is 13.6s of audio and g-small 12.0s, so this takes both and a smaller
# number takes only the first. Expressed in hours because --hours is.
HOURS = 0.005
# The floors that reject `solo` (one utterance) and `quiet` (0.1s in total) and
# nobody else. Both are needed: neither catches the other's speaker.
FLOORS = {"min_utterances": 2, "min_seconds": 0.3}
# 21.5 seconds, which buys 22 of the 25 one-second speakers -- two more than the
# floor, so a test that means "the floor was applied" has to say so itself
# rather than reading it off a split that only just cleared it.
VALID_HOURS = 21.5 / 3600
# Half a second of clip and 0.75s of target, so each speaker's two clips land in
# two files: the second offset restarts at zero in a file of its own, which a
# single-file speaker could not show.
TARGET_SEC = 0.75
# What the fake aligner scores a row, and the cutoff between them. Ten frames
# apiece, so per-frame the two are -0.1 and -0.9 and the cutoff sits between.
GOOD_SCORE, BAD_SCORE = -1.0, -9.0
ALIGN_FRAMES, ALIGN_TOKENS = 10, 3
CUTOFF = {"min_score": -0.5, "score_normalization": "per_frame"}
KANA_ALIGNER = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"


def _corpus_rows():
    """Every row of the fake metadata.tsv, in the order the file holds them."""
    rows = []
    for game, speakers in CORPUS.items():
        for speaker in speakers:
            rows += [
                (game, speaker, CLIP_SEC, "a.wav", "こんにちは"),
                (game, speaker, CLIP_SEC, "b.wav", "こんにちは"),
            ]
    # One utterance, so `min_utterances` rejects them and `min_seconds` does not.
    rows.append(("g-big", "solo", CLIP_SEC, "a.wav", "こんにちは"))
    # A tenth of a second in two clips, so `min_seconds` rejects them and
    # `min_utterances` does not. Neither floor alone keeps both of these out.
    rows += [
        ("g-big", "quiet", 0.05, "a.wav", "こんにちは"),
        ("g-big", "quiet", 0.05, "b.wav", "こんにちは"),
    ]
    return rows


def _unpack_game(dest_root, game, skip=()):
    """One game's clips where `extract_game` leaves them, marker and all.

    Double-nested, because the tar carries its own `<game_id>/` at the top and
    is unpacked into a directory of the same name (G2). A fake that flattened
    that would let a `main` reading the speaker off the last directory pass.

    `skip` names `(game, speaker)` pairs whose clips are left out while the
    marker is written anyway, which is the one thing a caller cannot tell from
    the outside: metadata.tsv describes clips the tars on disk may not hold, so
    a game a kill cut short looks exactly like a game the corpus is short of.
    """
    for game_id, speaker, seconds, name, _text in _corpus_rows():
        if game_id != game or (game_id, speaker) in skip:
            continue
        _at(dest_root / game / game / speaker, name, seconds, hz=440.0)
    (dest_root / f"{game}{EXTRACT_MARKER}").write_text("", encoding="utf-8")
    return dest_root / game


def _recording(calls, name, func):
    """`func`, with every call to it written down under `name`."""

    def wrapper(*args, **kwargs):
        calls[name].append((args, kwargs))
        return func(*args, **kwargs)

    return wrapper


def _tree(root):
    """Every file under `root`, by relative path, with its bytes."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _age(*roots, seconds=60.0):
    """Push everything under `roots` back in time by `seconds`.

    Whether an artifact was built before or after its inputs is what every skip
    in `main` turns on, and two runs of this fixture are milliseconds apart
    while the clock a timestamp comes from is coarser than that (about 15 ms on
    Windows). A tie reads as up to date -- deliberately, or nothing would ever
    be skipped -- so a test that means "this was built by an earlier run" has to
    say so rather than race the clock.

    Every root gets the same timestamp, and the plural is why: two calls a
    microsecond apart leave the second tree newer than the first, so ageing the
    two corpora one after the other makes the MoeSpeech entries newer than the
    manifests built out of them and rebuilds everything below the split. That
    reads exactly like the artifact under test having been rebuilt for the
    reason the test is about.
    """
    past = time.time() - seconds
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            os.utime(path, (past, past))


def _pipeline(tmp_path, monkeypatch):
    """`main` with everything off this machine replaced by a recording fake.

    The fakes for `download_games`, `extract_game` and `align` each reproduce
    the one behaviour of their real counterpart this script leans on: doing
    nothing when their output is already there. Those three decide for
    themselves what to skip -- their own tests prove they do -- so a fake that
    recorded the call instead of the work would report a re-run as redoing
    everything when it redid nothing. Every other stage runs for real.

    The aligner's fake is the one that has to do more than skip. Stage 9 reads
    the score `align_data` writes on each row, and there is no score until the
    aligner has run (G7), so a fake that wrote an empty file would leave the two
    stages after it with nothing to measure or to cut on.
    """
    from training.scripts import prepare_data
    from training.scripts import prepare_gol as m

    calls = defaultdict(list)
    cache = tmp_path / "hub"
    cache.mkdir()
    out_dir = tmp_path / "gol"
    # The MoeSpeech run this one merges with, as stage 6 of that script leaves
    # it. Ten minutes a clip, so those two are the largest speakers in the union
    # and the split never reaches them: what is held out is GOL's, which is what
    # the assertions below can then be specific about.
    moe_dir = tmp_path / "moe"
    _entries(moe_dir / "entries", {"アリス": [600.0, 600.0], "ボブ": [600.0, 600.0]})
    # The audio those offsets name. Nothing here reads it -- the aligner is
    # faked -- but the manifests this run writes name it, and a test that walks
    # them has to be able to tell "the split named a file that is not there"
    # from "the fixture never wrote one".
    for i in range(2):
        _at(moe_dir / "audio", f"{i:04d}.wav", CLIP_SEC)

    def fake_fetch(repo_id, filename, **kw):
        calls["metadata.tsv"].append((repo_id, filename, kw.get("repo_type")))
        _metadata(cache, _corpus_rows())
        return str(cache / filename)

    # Two recordings apiece, and they say different things. `*_called` is that
    # main reached the stage at all, `download`/`extract` that the stage found
    # work to do. Only the pair pins the contract: main calls both on every run
    # and they decide for themselves what to skip, so a count of the work alone
    # would be satisfied by a main() that stopped calling them.
    def fake_download(ids, dest, repo=None):
        calls["download_called"].append((tuple(ids), repo))
        dest.mkdir(parents=True, exist_ok=True)
        paths = []
        for game_id in ids:
            path = dest / f"{game_id}.tar"
            if not path.exists():
                calls["download"].append((game_id, repo))
                path.write_bytes(b"gol\x00tar")
            paths.append(path)
        return paths

    # Read at call time, so a test that adds to it before the first run gets a
    # game whose clips are partly absent -- an extraction a kill cut short --
    # and one that clears it before the second gets the rest of them.
    missing = set()

    def fake_extract(tar_path, dest_root):
        calls["extract_called"].append(tar_path.name)
        out = dest_root / tar_path.stem
        marker = dest_root / f"{tar_path.stem}{EXTRACT_MARKER}"
        if marker.exists() and out.is_dir():
            return out
        calls["extract"].append(tar_path.name)
        return _unpack_game(dest_root, tar_path.stem, skip=missing)

    # Bound against the real align()'s signature, so a call main() could not
    # actually make -- a misspelled keyword, an argument too many -- fails here
    # rather than being recorded as if it had worked.
    signature = inspect.signature(prepare_data.align)

    # Speakers every one of whose utterances the aligner scores badly, read at
    # call time so a test can add to it between two runs. Whole voices, where
    # the `bad` below is per utterance: the filter is per utterance too, and a
    # voice whose every clip aligns badly is how a per-utterance cut removes a
    # whole speaker from the held-out set with nothing per-utterance noticing.
    badly_aligned: set[str] = set()

    def fake_align(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        out = Path(bound.arguments["out"])
        if out.exists():
            return
        calls["align"].append(dict(bound.arguments))
        rows = []
        for entry in _read_jsonl(Path(bound.arguments["manifest"])):
            # The second clip of every speaker aligns badly and the first does
            # not, so the cut below takes half of every voice rather than whole
            # speakers -- which is what tells a filter that ran from a split
            # that happened to drop the same rows.
            bad = entry["id"].endswith(("b", "0001")) or entry["speaker"] in badly_aligned
            rows.append(
                {
                    **entry,
                    "score": BAD_SCORE if bad else GOOD_SCORE,
                    "frames": ALIGN_FRAMES,
                    "tokens": ALIGN_TOKENS,
                    "words": [{"word": entry["transcript"], "start": 0.0, "end": 0.1}],
                }
            )
        write_manifest(rows, out)

    # Whether the japanese dependency group is installed is a property of the
    # machine, exactly like the hub and the aligner, and CI syncs without it.
    def fake_segmenter_check(will_align):
        calls["segmenter_check"].append((will_align, len(calls["download_called"])))

    real_segmenter_check = m.require_japanese_segmenter
    real_late_check = m.require_segmenter_to_align

    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    monkeypatch.setattr(m, "download_games", fake_download)
    monkeypatch.setattr(m, "extract_game", fake_extract)
    monkeypatch.setattr(m, "align", fake_align)
    monkeypatch.setattr(m, "require_japanese_segmenter", fake_segmenter_check)
    monkeypatch.setattr(m, "require_segmenter_to_align", lambda aligned: None)
    for name in (
        "select_games",
        "probe_speakers",
        "select_speakers",
        "gol_utterances",
        "concatenate",
        "merge_entries",
        "split_across_corpora",
        "probe_scores",
        "filter_by_score",
    ):
        monkeypatch.setattr(m, name, _recording(calls, name, getattr(m, name)))

    def run(**overrides):
        options = {
            "out": str(out_dir),
            "hours": HOURS,
            "valid_hours": VALID_HOURS,
            "target_sec": TARGET_SEC,
            "moespeech": str(moe_dir),
            "repo": "fake/repo",
        }
        options.update(overrides)
        return m.main(**options)

    def age(seconds=60.0):
        """Push both corpora back in time, not only this run's own tree.

        The MoeSpeech entries are one of the split's inputs, so ageing the GOL
        tree alone leaves them newer than the manifests built out of them.
        """
        _age(out_dir, moe_dir, seconds=seconds)

    return SimpleNamespace(
        module=m,
        calls=calls,
        out=out_dir,
        moe=moe_dir,
        run=run,
        age=age,
        missing=missing,
        badly_aligned=badly_aligned,
        real_segmenter_check=real_segmenter_check,
        real_late_check=real_late_check,
    )


# Every stage, run once, in the order main runs them. Written down so that the
# assertions below can say `FIRST_RUN | {...}` and mean "and nothing else moved".
FIRST_RUN = {
    "segmenter_check": 1,
    "metadata.tsv": 1,
    "select_games": 1,
    "download_called": 1,
    "download": 2,
    "extract_called": 2,
    "extract": 2,
    "probe_speakers": 1,
    "select_speakers": 1,
    "gol_utterances": 1,
    "concatenate": 25,
    "merge_entries": 1,
    "split_across_corpora": 1,
    "align": 2,
    "probe_scores": 1,
    "filter_by_score": 2,
}


def test_the_pipeline_skips_stages_whose_output_exists(tmp_path, monkeypatch):
    """The property the whole script is designed around: re-running after a
    preemption must not redo finished work.

    The instance this runs on is reclaimed without warning, so the command is
    typed again -- often, over a 7 TB corpus. Every stage is asserted to have
    run once and then not again, and the tree it left is asserted byte-identical
    afterwards: a stage that redid its work shows up as a second recording, and
    one that rewrote its output from different inputs as different bytes.

    "Skipped" and "never ran" are kept apart deliberately. `download_called` and
    `extract_called` go up on the second run while `download` and `extract` do
    not, so a main() that had simply stopped calling those two stages would fail
    here -- and it is exactly that call, over a tar the first run never reached,
    that a resumed download depends on.
    """
    p = _pipeline(tmp_path, monkeypatch)

    p.run(**FLOORS, **CUTOFF)

    done = {stage: len(c) for stage, c in p.calls.items()}
    assert done == FIRST_RUN, done
    # And it operated on what it was told to. The repo is an option, the tars
    # are the selected games', and each stage is handed the previous one's
    # output; a stage that ran the right number of times over the wrong file is
    # the failure a count cannot see.
    assert p.calls["metadata.tsv"] == [("fake/repo", "metadata.tsv", "dataset")]
    assert sorted(p.calls["download"]) == [("g-big", "fake/repo"), ("g-small", "fake/repo")]
    assert sorted(p.calls["extract"]) == ["g-big.tar", "g-small.tar"]
    before = _tree(p.out)
    assert (p.out / "train_aligned.jsonl").exists()

    p.run(**FLOORS, **CUTOFF)

    assert {stage: len(c) for stage, c in p.calls.items()} == FIRST_RUN | {
        "download_called": 2,
        "extract_called": 4,
        # Asked once per run, before anything else: it costs an import and the
        # answer can change between runs, which is the point of asking again.
        "segmenter_check": 2,
        # metadata.tsv is fetched every run rather than only when the selection
        # is made -- three stages read it, not one -- and the hub serves it out
        # of its own cache after the first time.
        "metadata.tsv": 2,
    }
    assert _tree(p.out) == before


def test_the_run_stops_until_the_speaker_floors_have_been_chosen(tmp_path, monkeypatch, caplog):
    """Nothing had measured this corpus, so neither floor has a default.

    The speaker probe is what produces the table they are read off, so the run
    downloads, unpacks and measures, then stops and names the file to read and
    the two flags to pass. Guessing instead is what `--min-mos 3.0` did on the
    corpus before this one: the plausible default, against a measured median of
    2.281, would have cut 124.4 hours to 16.2.
    """
    p = _pipeline(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO)

    with pytest.raises(typer.Exit):
        p.run()

    assert (p.out / "speakers_probe.json").exists()
    assert not p.calls["select_speakers"], "no speaker may be selected on a floor nobody chose"
    assert not p.calls["align"]
    said = "\n".join(record.message for record in caplog.records)
    assert "speakers_probe.json" in said, said
    assert "--min-utterances" in said and "--min-seconds" in said, said


def test_supplying_the_floors_resumes_from_the_speaker_probe(tmp_path, monkeypatch):
    """The stop above is a stage boundary, not a failed run.

    The operator reads the table and types the command again with the two
    flags. Two hundred gigabytes fetched and unpacked is what must not be paid
    for a second time.
    """
    p = _pipeline(tmp_path, monkeypatch)
    with pytest.raises(typer.Exit):
        p.run()
    probe = (p.out / "speakers_probe.json").read_bytes()

    p.run(**FLOORS, **CUTOFF)

    assert len(p.calls["probe_speakers"]) == 1, "the speakers were measured twice"
    assert len(p.calls["extract"]) == 2, "the tars were unpacked twice"
    assert len(p.calls["download"]) == 2, "the tars were fetched twice"
    assert (p.out / "speakers_probe.json").read_bytes() == probe
    assert (p.out / "train_aligned.jsonl").exists()


def test_the_run_stops_until_the_score_cutoff_has_been_chosen(tmp_path, monkeypatch, caplog):
    """The second stop, and the one that says where the score comes from.

    An alignment log-probability has no scale known before the corpus is
    measured, and it does not exist at all until the aligner has run (G7). So
    this run aligns -- which is the expensive stage -- and only then measures,
    stops, and asks. Nothing is filtered on a number nobody read off a table.
    """
    p = _pipeline(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO)

    with pytest.raises(typer.Exit):
        p.run(**FLOORS)

    assert len(p.calls["align"]) == 2, "the score is the aligner's, so it has to have run"
    assert (p.out / "scores.json").exists()
    assert not p.calls["filter_by_score"], "nothing may be cut on a threshold nobody chose"
    assert not (p.out / "train_aligned.jsonl").exists()
    said = "\n".join(record.message for record in caplog.records)
    assert "scores.json" in said, said
    assert "--min-score" in said and "--score-normalization" in said, said


def test_supplying_the_cutoff_resumes_from_the_score_probe(tmp_path, monkeypatch):
    """And the alignment -- the most expensive stage in the script -- is not
    redone to apply a number that only ever reads its output."""
    p = _pipeline(tmp_path, monkeypatch)
    with pytest.raises(typer.Exit):
        p.run(**FLOORS)
    scores = (p.out / "scores.json").read_bytes()

    p.run(**FLOORS, **CUTOFF)

    assert len(p.calls["align"]) == 2, "the corpus was aligned twice"
    assert len(p.calls["probe_scores"]) == 1, "the scores were measured twice"
    assert (p.out / "scores.json").read_bytes() == scores
    assert (p.out / "train_aligned.jsonl").exists()


def test_the_filter_reads_the_aligners_output_and_leaves_it_alone(tmp_path, monkeypatch):
    """Ruling G7, as the two files it produces.

    The MoeSpeech pipeline filters before it aligns; this one cannot, because
    the score is the aligner's own and does not exist any earlier. So the
    aligner runs over utterances that are then discarded, and both files stay on
    disk: the scored one is what a cutoff can be re-read off without paying for
    the alignment again, and the filtered one is what training reads.

    Asserted as the scored manifest holding rows the filtered one does not -- a
    filter that ran before the aligner, or over the unaligned manifest, has
    nothing to cut on and would leave the two the same length.
    """
    p = _pipeline(tmp_path, monkeypatch)

    p.run(**FLOORS, **CUTOFF)

    scored = _read_jsonl(p.out / "train_scored.jsonl")
    filtered = _read_jsonl(p.out / "train_aligned.jsonl")
    assert len(scored) == 10 and len(filtered) == 5, (len(scored), len(filtered))
    # Every row of the scored manifest carries what the cut is made on, and the
    # rows that survived are the ones that cleared it.
    assert {row["score"] for row in scored} == {GOOD_SCORE, BAD_SCORE}
    assert {row["score"] for row in filtered} == {GOOD_SCORE}
    # Read from the aligner's output, written beside it, and neither is the
    # other: a filter pointed at its own output truncates the corpus on every
    # re-run, and one pointed at train.jsonl reads rows that have no score.
    (args, _kwargs) = p.calls["filter_by_score"][0]
    assert [Path(a).name for a in args[:2]] == ["train_scored.jsonl", "train_aligned.jsonl"], args
    assert (args[2], args[3]) == (CUTOFF["min_score"], CUTOFF["score_normalization"]), args
    # Both manifests are cut, not just the training one. The valid loss is the
    # entire readout of this run, and a valid set selected under a different
    # rule from the training set is not comparable with it.
    assert len(_read_jsonl(p.out / "valid_scored.jsonl")) == 44
    assert len(_read_jsonl(p.out / "valid_aligned.jsonl")) == 22


def test_every_speaker_is_joined_on_their_own(tmp_path, monkeypatch):
    """Ruling G4: `concatenate` refuses a mixed-speaker list rather than
    grouping one, so the grouping is main's.

    A file holding two voices would teach the model that the prompt does not
    decide the voice, and nothing downstream could see it -- the manifest
    parses, every offset is inside a real file, and the loader takes one side of
    a cut as the prompt for the other. The guard inside `concatenate` compares
    labels, so what it catches is main handing it a mixed list; what it cannot
    catch is main never handing it anything at all.
    """
    p = _pipeline(tmp_path, monkeypatch)

    p.run(**FLOORS, **CUTOFF)

    joined = []
    for args, _kwargs in p.calls["concatenate"]:
        utterances, out_wav, target_sec = args
        speakers = {u["speaker"] for u in utterances}
        assert len(speakers) == 1, (out_wav, sorted(speakers))
        assert target_sec == TARGET_SEC, args
        joined += sorted(speakers)
    # Every selected speaker, once each: a grouping that dropped one, or ran the
    # same one twice, is the failure a per-call check cannot see.
    assert len(joined) == 25 and len(set(joined)) == 25, sorted(joined)
    assert set(joined) == {
        _speaker_key(game, speaker) for game, speakers in CORPUS.items() for speaker in speakers
    }


def test_the_speaker_floors_keep_the_speakers_out_of_everything_below(tmp_path, monkeypatch):
    """`solo` has one utterance and `quiet` has a tenth of a second, and each is
    rejected by one floor and not the other.

    A speaker with a single clip cannot be evaluated -- the protocol clones a
    voice from one utterance and synthesizes another -- and has nothing to be
    joined to; a speaker with a dozen seconds is a bit part. Neither may reach
    the audio, the manifests or the alignment, and the whole point of two floors
    is that either one alone lets the other's speaker through.
    """
    p = _pipeline(tmp_path, monkeypatch)

    p.run(**FLOORS, **CUTOFF)

    rejected = {"g-big:solo", "g-big:quiet"}
    for name in ("utterances.jsonl", "train.jsonl", "valid.jsonl", "train_aligned.jsonl"):
        speakers = {row["speaker"] for row in _read_jsonl(p.out / name)}
        assert not (speakers & rejected), (name, sorted(speakers & rejected))
    assert json.loads((p.out / "speakers.json").read_text(encoding="utf-8"))["speakers"] == sorted(
        _speaker_key(game, speaker) for game, speakers in CORPUS.items() for speaker in speakers
    )


def test_the_manifests_name_the_audio_the_run_selected(tmp_path, monkeypatch):
    """What the run leaves behind is read, not just counted.

    Every way this pipeline can be miswired ends in a file of the right name:
    the split handed the wrong manifest, --target-sec replaced by its default,
    the transcripts never normalized, a speaker dropped by a `=` where a `+=`
    belongs. None of them raises. They show up here, as a manifest describing
    different audio.
    """
    p = _pipeline(tmp_path, monkeypatch)

    p.run(**FLOORS, **CUTOFF)

    train = _read_jsonl(p.out / "train.jsonl")
    valid = _read_jsonl(p.out / "valid.jsonl")
    # Whole speakers are held out, across both corpora at once, and the split
    # takes the smallest first -- so the twenty-two voices it buys are GOL's and
    # the two ten-minute MoeSpeech characters are trained on.
    assert {e["speaker"] for e in valid} == {
        _speaker_key("g-big", f"s{i:02d}") for i in range(13)
    } | {_speaker_key("g-small", f"s{i:02d}") for i in range(13, 22)}
    assert {e["speaker"] for e in train} == {
        "g-small:s22",
        "g-small:s23",
        "g-small:s24",
        "アリス",
        "ボブ",
    }
    # Half a second a clip against --target-sec 0.75, so a speaker's two clips
    # land in two files and the second offset restarts at zero inside its own.
    mine = [e for e in train if e["speaker"] == "g-small:s24"]
    assert [(Path(e["path"]).name, e["start"], e["duration"]) for e in mine] == [
        ("0024.wav", 0.0, 0.5),
        ("0024_001.wav", 0.0, 0.5),
    ], mine
    # The id is the corpus's own path for the clip, which is the only way back
    # from a suspect row to the audio it was cut out of.
    assert [e["id"] for e in mine] == ["g-small/s24/a", "g-small/s24/b"], mine
    for entry in train + valid:
        assert Path(entry["path"]).exists(), entry
    assert {e["transcript"] for e in train + valid} == {"こんにちは", "あ"}


def test_both_corpora_reach_the_split_and_the_run_says_so(tmp_path, monkeypatch, caplog):
    """The merge is where a corpus can vanish with nothing downstream noticing.

    A manifest built out of GOL alone parses, aligns and trains, and the only
    sign is a speaker count nobody has a second number to compare against. So
    the MoeSpeech entries directory is named on the command line and merged with
    this run's own, and both of them are asserted to be in what came out.
    """
    p = _pipeline(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO)

    p.run(**FLOORS, **CUTOFF)

    (args, _kwargs) = p.calls["merge_entries"][0]
    assert [Path(d).parent.name for d in args[0]] == ["gol", "moe"], args
    everyone = {row["speaker"] for row in _read_jsonl(p.out / "train.jsonl")}
    everyone |= {row["speaker"] for row in _read_jsonl(p.out / "valid.jsonl")}
    assert {"アリス", "ボブ"} <= everyone, sorted(everyone)
    assert len(everyone) == 27, sorted(everyone)


def test_a_corpus_too_small_to_hold_out_twenty_voices_is_refused(tmp_path, monkeypatch):
    """The floor phase 1 did not have, asked of the whole command.

    Phase 1 held out one speaker. Its validation loss bottomed at 7,500 and had
    doubled by 15,000, and one voice cannot tell that apart from noise -- while
    the samples, whose prompt `train.py` takes from a training batch, kept
    getting better. For a model whose point is cloning an unseen voice, the
    unseen number is the one that says when to stop.

    Asserted through `main` rather than through `split_across_corpora`, which
    has its own test: that function takes the floor as an argument with a
    default, so a main() that passed `minimum=1` would satisfy every test the
    function has and still start a 40,000-step run with no stopping criterion.
    """
    p = _pipeline(tmp_path, monkeypatch)

    # 0.003 hours is less than g-big alone, so only that game is taken and only
    # its thirteen speakers reach the split.
    with pytest.raises(typer.BadParameter, match=str(MIN_VALID_SPEAKERS)):
        p.run(**FLOORS, **CUTOFF, hours=0.003)

    assert not (p.out / "train.jsonl").exists()
    assert not p.calls["align"], "a run with no stopping criterion reached the aligner"


def test_alignment_uses_the_japanese_segmenter_and_a_kana_model(tmp_path, monkeypatch):
    """align_data refuses a model without hiragana in its vocabulary, but only
    at run time on the instance -- catching it here costs nothing.

    The segmenter matters as much and refuses nothing: `whitespace` over a
    language written without spaces returns one word per utterance, so the
    aligner emits a single span, the loader finds no cut point, and the voice
    prompt silently comes from the utterance being predicted.
    """
    p = _pipeline(tmp_path, monkeypatch)

    p.run(**FLOORS, **CUTOFF, align_shards=3)

    assert len(p.calls["align"]) == 2, p.calls["align"]
    for call in p.calls["align"]:
        assert call["segmenter"] == "japanese", call
        assert call["model"] == KANA_ALIGNER, call
    # Each manifest is aligned from itself, and the valid one is aligned too:
    # the loader reads `words` on either side, and an unaligned valid set is
    # scored differently from the set it is compared against.
    assert [(Path(c["manifest"]).name, Path(c["out"]).name) for c in p.calls["align"]] == [
        ("train.jsonl", "train_scored.jsonl"),
        ("valid.jsonl", "valid_scored.jsonl"),
    ], p.calls["align"]
    # --align-shards is a count of GPUs and belongs to the long pass; the valid
    # manifest is a fraction of the size and is aligned in one process.
    assert [c["shards"] for c in p.calls["align"]] == [3, 1], p.calls["align"]


def test_a_kill_between_the_gathering_and_the_joining_resumes_from_the_file(tmp_path, monkeypatch):
    """The preemption `utterances.jsonl` exists to survive.

    That file is the stream over 7.4 million metadata rows against every wav on
    disk, and a kill just after it was renamed into place -- before a single
    speaker's clips had been joined -- is the case the whole design is for.
    Proving the run carries on from it means leaving it and taking away
    everything built after: with every artifact present, a stage reading the
    file back is indistinguishable from one that never read it.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**FLOORS, **CUTOFF)
    uninterrupted = _tree(p.out)

    shutil.rmtree(p.out / "entries")
    shutil.rmtree(p.out / "audio")
    for name in (
        "train.jsonl",
        "valid.jsonl",
        "train_scored.jsonl",
        "valid_scored.jsonl",
        "train_aligned.jsonl",
        "valid_aligned.jsonl",
    ):
        (p.out / name).unlink()

    p.run(**FLOORS, **CUTOFF)

    assert len(p.calls["gol_utterances"]) == 1, "the corpus was walked a second time"
    assert len(p.calls["concatenate"]) == 50, "the clips the kill cost were not joined"
    # The rows read back off disk are the rows the first run held, so the tree
    # the second one finishes with is the tree it would have finished with.
    assert _tree(p.out) == uninterrupted


def test_a_kill_between_the_joining_and_the_split_resumes_from_the_entries(tmp_path, monkeypatch):
    """One stage further down, and the same property one level deeper.

    Joining is the hours of audio work. A kill after the last speaker's offsets
    were written but before the split must not redo it, and the offsets it reads
    back have to be the ones on disk -- a split over nothing at all writes two
    manifests that parse and describe no training data.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**FLOORS, **CUTOFF)
    uninterrupted = _tree(p.out)

    for name in (
        "train.jsonl",
        "valid.jsonl",
        "train_scored.jsonl",
        "valid_scored.jsonl",
        "train_aligned.jsonl",
        "valid_aligned.jsonl",
    ):
        (p.out / name).unlink()

    p.run(**FLOORS, **CUTOFF)

    assert len(p.calls["concatenate"]) == 25, "the audio was joined a second time"
    assert len(p.calls["split_across_corpora"]) == 2, "the split the kill cost was not redone"
    assert _tree(p.out) == uninterrupted


def test_a_changed_score_cutoff_refilters_without_aligning_again(tmp_path, monkeypatch):
    """Changing the cutoff is the documented way to change the cut, and nothing
    else on disk records what the last one was.

    Without that record the re-run finds `train_aligned.jsonl` sitting there,
    newer than everything it was built from, and reuses it -- so the run reports
    success over the cut the operator has just replaced, with no way afterwards
    for anyone to tell which threshold a given manifest was written under. And
    the alignment, which is the day of GPU time, must not be redone for it.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**FLOORS, **CUTOFF)
    assert len(_read_jsonl(p.out / "train_aligned.jsonl")) == 5
    p.age()

    # -1.0 is below the worse of the two scores per frame (-0.9), so nothing is
    # cut this time and every row of the scored manifest survives.
    p.run(**FLOORS, min_score=-1.0, score_normalization="per_frame")

    assert len(_read_jsonl(p.out / "train_aligned.jsonl")) == 10
    assert len(_read_jsonl(p.out / "valid_aligned.jsonl")) == 44
    assert len(p.calls["align"]) == 2, "the corpus was aligned a second time"
    assert len(p.calls["probe_scores"]) == 1, "the scores did not change and were measured twice"
    assert len(p.calls["filter_by_score"]) == 4, "the manifests still hold the rejected cut"


def test_changed_floors_rebuild_everything_they_decided(tmp_path, monkeypatch):
    """The other recorded pair, and it reaches further down.

    The speaker floors decide the selection, which decides the offsets, the
    audio, both manifests and both alignments. Nothing else on disk records
    them, so a re-run under a looser floor that only rewrote `speakers.json`
    would leave `train.jsonl` describing the selection the operator had just
    replaced -- with the alignment beside it, and the run reporting success.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**FLOORS, **CUTOFF)
    assert "g-big:solo" not in {r["speaker"] for r in _read_jsonl(p.out / "utterances.jsonl")}
    p.age()

    # One utterance is now enough, so `solo` joins the corpus; `quiet` is still
    # below the seconds floor, which is what tells one floor from the other.
    p.run(min_utterances=1, min_seconds=FLOORS["min_seconds"], **CUTOFF)

    trained = {r["speaker"] for r in _read_jsonl(p.out / "train.jsonl")}
    assert "g-big:solo" in trained, sorted(trained)
    assert "g-big:quiet" not in trained, sorted(trained)
    assert len(p.calls["select_speakers"]) == 2, "the floors were not applied again"
    assert len(p.calls["gol_utterances"]) == 2, "the selection still holds the old floors"
    assert len(p.calls["concatenate"]) == 51, "the audio still holds the old selection"
    assert len(p.calls["align"]) == 4, "the alignments still describe the old manifests"
    assert len(p.calls["filter_by_score"]) == 4, "the training manifests were not rebuilt"


def test_an_entries_file_holding_another_speaker_is_not_reused(tmp_path, monkeypatch):
    """The offsets files are numbered, not named after their speaker, and this
    is what pays for that.

    A GOL key holds a colon, which is not a legal Windows file name, so the file
    a speaker's offsets go in is `<index>.jsonl` -- the index they have in the
    sorted selection. That index moves whenever the selection does, and the only
    thing standing between a moved index and a reused file is that the selection
    is rewritten first, which makes every offsets file stale. File clocks are
    coarser than these stages are fast, and equal timestamps count as fresh, so
    that is a race rather than a guarantee.

    Constructed as the race losing: the file is made NEWER than the selection,
    which is the one state where the timestamps say reuse it. Only reading whose
    rows are actually in it can catch that, and getting it wrong writes one
    speaker's offsets under another's label -- two voices in one manifest entry,
    which every guard below compares by label and none of them can see.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**FLOORS, **CUTOFF)
    uninterrupted = _tree(p.out)
    p.age()

    # g-small:s24's rows, written under g-big:s00's number and left newer than
    # the selection, which is the state the timestamps read as "reuse it".
    write_manifest(_read_jsonl(p.out / "entries" / "0024.jsonl"), p.out / "entries" / "0000.jsonl")

    p.run(**FLOORS, **CUTOFF)

    assert len(p.calls["concatenate"]) == 26, "the mislabelled offsets file was reused"
    assert _tree(p.out) == uninterrupted


def test_a_moespeech_corpus_that_moved_rebuilds_the_split(tmp_path, monkeypatch):
    """The two corpora are prepared by two commands, and the other one goes on
    running after this one has split.

    prepare_moespeech is re-run after its own preemptions, and every re-run that
    reaches its joining stage rewrites the offsets files this merge reads. A
    split measured against GOL's own entries alone sees none of its inputs move,
    so `train.jsonl` goes on describing the MoeSpeech corpus as it was -- fewer
    speakers, or a selection under cutoffs since replaced -- while both
    directories on disk say otherwise and the run reports success over it.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**FLOORS, **CUTOFF)
    assert "キャロル" not in {r["speaker"] for r in _read_jsonl(p.out / "train.jsonl")}
    p.age()

    # What the other command reaching its joining stage leaves behind: the same
    # directory, with a speaker in it that was not there before.
    _entries(
        p.moe / "entries",
        {"アリス": [600.0, 600.0], "ボブ": [600.0, 600.0], "キャロル": [600.0, 600.0]},
    )
    _at(p.moe / "audio", "0002.wav", CLIP_SEC)

    p.run(**FLOORS, **CUTOFF)

    trained = {r["speaker"] for r in _read_jsonl(p.out / "train.jsonl")}
    assert "キャロル" in trained, sorted(trained)
    assert len(p.calls["split_across_corpora"]) == 2, "the split still holds the older corpus"
    assert len(p.calls["align"]) == 4, "the alignments still describe the older split"
    assert len(p.calls["concatenate"]) == 25, "GOL's own audio was joined a second time"


def test_a_game_unpacked_after_the_gathering_reaches_the_manifests(tmp_path, monkeypatch):
    """An interrupted extraction is invisible from inside the gather.

    metadata.tsv describes all 7,405,094 clips while the tars actually on disk
    hold a subset, so a row naming a wav that is not there is dropped and
    counted rather than raised on -- which is right, and which means a game the
    kill cut short looks exactly like a game whose clips the corpus does not
    have. `extract_game` redoes such a game on the next run, and the completion
    marker it then writes is the only thing on disk that says the walk is now
    short of clips that have since arrived.

    Without the markers among the gather's inputs, `utterances.jsonl` is reused,
    those clips never reach a manifest, and nothing anywhere reports that the
    corpus is smaller than the tars that were paid for.
    """
    p = _pipeline(tmp_path, monkeypatch)
    # The kill arrived partway through g-small: its last speaker's clips were
    # never written, and the marker the fake writes afterwards claims otherwise.
    p.missing.add(("g-small", "s24"))

    p.run(**FLOORS, **CUTOFF)

    assert "g-small:s24" not in {r["speaker"] for r in _read_jsonl(p.out / "utterances.jsonl")}
    assert len(p.calls["concatenate"]) == 24
    p.age()

    # The re-run finds the game half-extracted and unpacks it again, which is
    # `extract_game`'s own behaviour and has its own tests; what is asked here
    # is what the stage below does about it.
    p.missing.clear()
    shutil.rmtree(p.out / "extracted" / "g-small")
    (p.out / "extracted" / f"g-small{EXTRACT_MARKER}").unlink()

    p.run(**FLOORS, **CUTOFF)

    assert len(p.calls["extract"]) == 3, "the half-extracted game was not unpacked again"
    assert len(p.calls["gol_utterances"]) == 2, "the walk still holds the clips the kill cost"
    assert "g-small:s24" in {r["speaker"] for r in _read_jsonl(p.out / "utterances.jsonl")}
    assert "g-small:s24" in {r["speaker"] for r in _read_jsonl(p.out / "train.jsonl")}


def test_a_speaker_label_no_selected_speaker_answers_to_stops_the_run(
    tmp_path, monkeypatch, caplog
):
    """The braces to `gol_utterances`'s belt: labels are checked once, here.

    Every guard below this point compares the speaker label and not one of them
    can check it. `concatenate` refuses a mixed list by comparing labels, so a
    single wrong label shared by two characters *is* a mixed list and passes --
    two voices joined into one recording, the loader taking one side of a cut as
    the voice prompt for the other, the split holding out a label instead of a
    character, and one audio file written twice under one name.

    A key built the same way in both places is one of the selected speakers by
    construction, so that is what is checked, before a single wav is joined or
    deleted. It is the last thing standing if `_speaker_key` and the selection
    ever come apart -- which is a live risk here and not a hypothetical, because
    the key is assembled from two columns rather than read from one.
    """
    p = _pipeline(tmp_path, monkeypatch)
    gathering = p.module.gol_utterances

    def mislabelled(*args, **kwargs):
        # What taking the speaker off the directory name would produce on a
        # double-nested tree: every clip of every speaker under one label that
        # is not a speaker.
        for utterance in gathering(*args, **kwargs):
            yield {**utterance, "speaker": "wav"}

    monkeypatch.setattr(p.module, "gol_utterances", mislabelled)
    caplog.set_level(logging.ERROR)

    with pytest.raises(typer.Exit):
        p.run(**FLOORS, **CUTOFF)

    assert not p.calls["concatenate"], "two speakers were joined under one label"
    assert not (p.out / "audio").exists()
    said = "\n".join(record.message for record in caplog.records)
    assert "wav" in said, said


def test_a_corpus_named_only_on_a_later_run_reaches_the_split(tmp_path, monkeypatch):
    """--moespeech is supplied at whichever invocation the operator remembers.

    This command stops twice on purpose, so it is typed three times, and the
    entries the other script wrote are older than this run's train.jsonl by the
    time the third one is typed. Gated on those entries files alone nothing
    looks stale: the split is skipped, `Done.` is printed, and the manifest goes
    on describing one corpus while the command line names two -- which is the
    exact failure `merge_entries` refuses an empty directory to prevent, reached
    by not merging at all. Taking the flag away again is the same silence in the
    other direction, and the MoeSpeech rows stay in a manifest that no longer
    claims them.
    """
    p = _pipeline(tmp_path, monkeypatch)

    p.run(**FLOORS, **CUTOFF, moespeech=None)
    assert "アリス" not in {r["speaker"] for r in _read_jsonl(p.out / "train.jsonl")}
    p.age()

    p.run(**FLOORS, **CUTOFF)

    everyone = {r["speaker"] for r in _read_jsonl(p.out / "train.jsonl")}
    everyone |= {r["speaker"] for r in _read_jsonl(p.out / "valid.jsonl")}
    assert {"アリス", "ボブ"} <= everyone, sorted(everyone)
    assert len(p.calls["split_across_corpora"]) == 2, "the corpus that was added never merged"
    p.age()

    p.run(**FLOORS, **CUTOFF, moespeech=None)

    left = {r["speaker"] for r in _read_jsonl(p.out / "train.jsonl")}
    left |= {r["speaker"] for r in _read_jsonl(p.out / "valid.jsonl")}
    assert not {"アリス", "ボブ"} & left, sorted(left)
    assert len(p.calls["split_across_corpora"]) == 3, "the corpus that was dropped is still in"
    # The audio is not rebuilt for either change: which corpora are merged is a
    # fact about the split and about nothing above it, and this stage is
    # hundreds of gigabytes at 1,000 hours.
    assert len(p.calls["concatenate"]) == 25, "GOL's own audio was joined again"


def test_a_valid_set_the_score_filter_shrank_below_the_floor_is_refused(tmp_path, monkeypatch):
    """The twenty-voice floor is asserted at the split, and two stages after it
    can undo what it asserted.

    Ruling G7 put the score filter after alignment, so between the floor and the
    manifests training actually reads there are two per-utterance stages that
    drop rows: the aligner discards what it cannot align, and this cut discards
    what it scored badly. Per-utterance is the trap -- a voice whose every clip
    goes is a voice gone, and nothing counting utterances sees a voice leave.

    M2a exists to answer one question, whether the validation loss turns over
    20+ voices the weights have never heard. A held-out set that quietly falls
    under that is the experiment failing silently, at the end of a run whose
    alignment has already been paid for.

    Three invocations, as the two stops force. The first aligns and stops at the
    score probe, which is where valid.jsonl first exists to be read.
    """
    p = _pipeline(tmp_path, monkeypatch)

    with pytest.raises(typer.Exit):
        p.run(**FLOORS)
    held_out = sorted({row["speaker"] for row in _read_jsonl(p.out / "valid.jsonl")})
    assert len(held_out) == 22, held_out

    # Five held-out voices align badly enough to be cut whole, which takes the
    # set to 17. The aligner is re-run over the valid manifest alone, which is
    # what a rented instance re-run after the aligner was improved would do.
    p.badly_aligned.update(held_out[:5])
    (p.out / "valid_scored.jsonl").unlink()

    with pytest.raises(typer.BadParameter, match=str(MIN_VALID_SPEAKERS)):
        p.run(**FLOORS, **CUTOFF)

    assert len({r["speaker"] for r in _read_jsonl(p.out / "valid_aligned.jsonl")}) == 17
    filtered = len(p.calls["filter_by_score"])
    p.age()

    # And again on the next invocation, which finds both filtered manifests up
    # to date and skips the stage that wrote them. A refusal only the run that
    # happened to write the file makes is one an operator gets past by typing
    # the same command twice.
    with pytest.raises(typer.BadParameter, match=str(MIN_VALID_SPEAKERS)):
        p.run(**FLOORS, **CUTOFF)

    assert len(p.calls["filter_by_score"]) == filtered, "the manifests were not the reused ones"


def test_a_valid_set_the_filter_emptied_is_refused(tmp_path, monkeypatch):
    """The same floor at its far end. A cutoff above every score this corpus
    holds leaves a valid manifest of zero bytes, which is still valid JSONL:
    every line in it parses and every path in it exists, so the loader opens it,
    the run starts, and the number the whole experiment is read off is taken
    over nothing at all.
    """
    p = _pipeline(tmp_path, monkeypatch)

    with pytest.raises(typer.BadParameter, match=str(MIN_VALID_SPEAKERS)):
        p.run(**FLOORS, min_score=0.0, score_normalization="per_frame")

    assert (p.out / "valid_aligned.jsonl").read_bytes() == b""


def test_the_run_says_what_the_cut_cost_each_manifest_apart(tmp_path, monkeypatch, caplog):
    """The cutoff is read off `probe_scores(train_scored)` -- the training
    distribution -- and then applied to a valid set nobody measured it against.
    What it costs there is the number M2a turns on, so the two manifests are
    reported apart, by name, and in voices as well as in clips.

    One line saying "kept 27 of 54" over both together cannot answer either
    question an operator has here: whether the cut was the one they read off the
    table, and whether the held-out set survived it.
    """
    p = _pipeline(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO)

    p.run(**FLOORS, **CUTOFF)

    said = [r.message for r in caplog.records if r.message.startswith(("train_al", "valid_al"))]
    assert len(said) == 2, said
    train_line, valid_line = said
    # Every voice loses its second clip and keeps its first, so the cut costs
    # half the clips and no voices at all -- which is the pair of numbers that
    # tells "this cut is survivable" from "this cut emptied the held-out set",
    # and which a count of utterances alone cannot.
    assert "kept 5 of 10 utterances from 5 of 5 voices" in train_line, train_line
    assert "kept 22 of 44 utterances from 22 of 22 voices" in valid_line, valid_line
