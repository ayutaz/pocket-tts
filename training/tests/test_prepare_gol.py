"""Building a training manifest out of GOL.

GOL is five times MoeSpeech and shaped differently: the download unit is a
whole work rather than one character, and its transcripts come from a single
ASR pass, so the mutual-CER filter that carried MoeSpeech has no counterpart.
"""

import csv
import io
import shutil
import tarfile
from pathlib import Path

import pytest

from training.scripts.prepare_gol import probe_speakers, select_games, select_speakers


def _metadata(tmp_path, rows, text: str = "あ"):
    """A stand-in for GOL's metadata.tsv: game_id, speaker, text, path, duration."""
    p = tmp_path / "metadata.tsv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["game_id", "speaker", "text", "file_path", "duration"])
        for g, s, d in rows:
            w.writerow([g, s, text, f"{g}/{s}/x.wav", d])
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
