"""Building a training manifest out of MoeSpeech.

Everything here guards one property: the manifest must describe the audio
truthfully. A wrong `start` or `duration` does not raise -- it trains the model
on speech that does not match its text, and the only symptom is a model that
never quite becomes intelligible.
"""

import csv
import shutil

import pytest
import typer

from training.scripts.prepare_moespeech import select_characters


def _info_csv(tmp_path, rows):
    """A stand-in for the dataset's info.csv: name, num_files, minutes, f0."""
    path = tmp_path / "info.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "num_files", "total_duration_min", "f0_mean"])
        w.writerows(rows)
    return path


def test_selection_stops_once_the_target_is_reached(tmp_path):
    info = _info_csv(
        tmp_path, [("a", 100, 60.0, 300.0), ("b", 100, 60.0, 300.0), ("c", 100, 60.0, 300.0)]
    )
    chosen = select_characters(info, hours=1.0, order="largest")
    assert len(chosen) == 1, chosen


def test_the_target_is_hours_and_the_column_is_minutes(tmp_path):
    """total_duration_min is minutes; --hours is hours. Two hours of 30-minute
    speakers is four of them, not one. Every other case here is satisfied by a
    single speaker, so this is the only test that walks the accumulator far
    enough to notice a lost `* 60` or a summed `num_files`."""
    rows = [(f"c{i}", 100, 30.0, 300.0) for i in range(10)]
    chosen = select_characters(_info_csv(tmp_path, rows), hours=2.0, order="largest")
    assert len(chosen) == 4, chosen
    assert sum(c["total_duration_min"] for c in chosen) == 120.0


def test_a_japanese_speaker_name_survives_the_read(tmp_path):
    """Default encoding here is cp932, and every name in this dataset is
    Japanese. Read without an explicit utf-8 the name comes back as mojibake
    rather than raising, and Task 2 turns it straight into a zip URL."""
    info = _info_csv(tmp_path, [("ずんだもん", 100, 600.0, 300.0)])
    (chosen,) = select_characters(info, hours=1.0, order="largest")
    assert chosen["name"] == "ずんだもん"


def test_largest_first_takes_the_biggest(tmp_path):
    info = _info_csv(tmp_path, [("small", 10, 6.0, 300.0), ("big", 100, 600.0, 300.0)])
    (chosen,) = select_characters(info, hours=1.0, order="largest")
    assert chosen["name"] == "big"


def test_selection_is_deterministic(tmp_path):
    """A re-run after an interrupted download must ask for the same zips."""
    rows = [(f"c{i}", 100, 30.0, 300.0) for i in range(20)]
    info = _info_csv(tmp_path, rows)
    first = select_characters(info, hours=5.0, order="largest")
    second = select_characters(info, hours=5.0, order="largest")
    assert [c["name"] for c in first] == [c["name"] for c in second]


def test_random_order_is_also_deterministic(tmp_path):
    """Reproducibility does not depend on which ordering was chosen."""
    rows = [(f"c{i}", 100, 30.0, 300.0) for i in range(20)]
    info = _info_csv(tmp_path, rows)
    a = select_characters(info, hours=5.0, order="random")
    b = select_characters(info, hours=5.0, order="random")
    assert [c["name"] for c in a] == [c["name"] for c in b]


def test_asking_for_more_than_exists_returns_everything(tmp_path):
    """Rather than raising: the caller asked for an upper bound, not a promise."""
    info = _info_csv(tmp_path, [("a", 10, 6.0, 300.0)])
    assert len(select_characters(info, hours=1000.0, order="largest")) == 1


def test_an_unknown_order_is_refused(tmp_path):
    """A typo in --order must not silently fall through to info.csv's own
    order, which would look just as deterministic on re-run."""
    info = _info_csv(tmp_path, [("a", 10, 6.0, 300.0)])
    with pytest.raises(typer.BadParameter):
        select_characters(info, hours=1.0, order="larget")


FETCHED = b"PK\x03\x04fake"


def _fake_hub(tmp_path):
    """A stand-in for huggingface_hub whose cache is *not* the destination.

    The real `hf_hub_download` returns a path inside its own cache, and moving
    those bytes into `dest` is the only reason `download_characters` exists. A
    fake that wrote straight into `dest` would make that move invisible: the
    fetched file and the destination file would be the same file, so an
    implementation that transferred nothing at all would still pass.
    """
    cache = tmp_path / "hf_cache"
    cache.mkdir()
    dest = tmp_path / "zips"
    calls = []

    def fake_fetch(repo_id, filename, **kw):
        calls.append((repo_id, filename, kw.get("repo_type")))
        p = cache / filename
        p.write_bytes(FETCHED)
        return str(p)

    return dest, calls, fake_fetch


def test_download_skips_what_is_already_complete(tmp_path, monkeypatch):
    """Re-running after an interrupt must not re-fetch 5 GB it already has."""
    from training.scripts import prepare_moespeech as m

    dest, calls, fake_fetch = _fake_hub(tmp_path)
    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    dest.mkdir()
    (dest / "aaa.zip").write_bytes(b"PK\x03\x04already here")

    paths = m.download_characters(["aaa", "bbb"], dest, repo="fake/repo")

    # The repo and repo_type are asserted, not just the filename: these zips
    # live in a dataset repo, and without repo_type="dataset" the hub resolves
    # the name in the model namespace and every fetch 404s.
    assert calls == [("fake/repo", "bbb.zip", "dataset")], calls
    assert (dest / "aaa.zip").read_bytes() == b"PK\x03\x04already here"
    assert (dest / "bbb.zip").read_bytes() == FETCHED
    assert [p.name for p in paths] == ["aaa.zip", "bbb.zip"]


def test_download_returns_a_path_for_every_requested_character(tmp_path, monkeypatch):
    """A caller that gets fewer paths than it asked for would silently train on
    a smaller corpus than intended. One zip is already present so the list spans
    both branches -- forgetting to append the skipped one is the likeliest way
    to lose a path, and on a re-run that is where nearly every speaker sits."""
    from training.scripts import prepare_moespeech as m

    dest, calls, fake_fetch = _fake_hub(tmp_path)
    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    dest.mkdir()
    (dest / "bbb.zip").write_bytes(b"PK\x03\x04already here")

    paths = m.download_characters(["aaa", "bbb", "ccc"], dest, repo="fake/repo")

    assert [p.name for p in paths] == ["aaa.zip", "bbb.zip", "ccc.zip"]
    assert [p.parent for p in paths] == [dest, dest, dest]
    assert calls == [("fake/repo", "aaa.zip", "dataset"), ("fake/repo", "ccc.zip", "dataset")]


def test_download_does_not_trust_a_leftover_partial(tmp_path, monkeypatch):
    """A kill mid-copy leaves `<name>.zip.partial`, not `<name>.zip`. If that
    partial were mistaken for a completed download -- or promoted into place as
    if resuming it -- the corpus would silently train on a truncated zip
    forever, and re-running would never fix it."""
    from training.scripts import prepare_moespeech as m

    dest, calls, fake_fetch = _fake_hub(tmp_path)
    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    dest.mkdir()
    (dest / "aaa.zip.partial").write_bytes(b"PK\x03\x04truncated")

    paths = m.download_characters(["aaa"], dest, repo="fake/repo")

    assert calls == [("fake/repo", "aaa.zip", "dataset")], calls
    assert [p.name for p in paths] == ["aaa.zip"]
    assert (dest / "aaa.zip").read_bytes() == FETCHED
    assert not (dest / "aaa.zip.partial").exists()


def test_download_creates_the_destination_directory(tmp_path, monkeypatch):
    """The stage runs on a fresh instance where `dest` does not exist yet."""
    from training.scripts import prepare_moespeech as m

    dest, _calls, fake_fetch = _fake_hub(tmp_path)
    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    assert not dest.exists()

    paths = m.download_characters(["aaa"], dest, repo="fake/repo")

    assert paths[0].read_bytes() == FETCHED


def _make_zip(path, names):
    """A zip holding `names`, each a tiny file."""
    import zipfile

    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(n, "x")
    return path


def test_extract_writes_every_member(tmp_path):
    from training.scripts.prepare_moespeech import extract_character

    z = _make_zip(tmp_path / "spk.zip", ["a.wav", "a.json", "b.wav", "b.json"])
    out = extract_character(z, tmp_path / "extracted")
    # Where it unpacks to is pinned here because the resumability tests below
    # hand-build a half-extracted directory at this exact path. Left unpinned, a
    # change to the naming would send those fixtures somewhere the code never
    # looks, and they would go on passing while testing nothing.
    assert out == tmp_path / "extracted" / "spk"
    assert sorted(p.name for p in out.iterdir()) == ["a.json", "a.wav", "b.json", "b.wav"]


def test_extract_skips_a_character_already_done(tmp_path):
    """Unpacking 30 GB is tens of minutes; re-running must not redo it."""
    from training.scripts.prepare_moespeech import extract_character

    z = _make_zip(tmp_path / "spk.zip", ["a.wav"])
    out = extract_character(z, tmp_path / "extracted")
    (out / "a.wav").write_text("edited")  # prove it is not rewritten
    extract_character(z, tmp_path / "extracted")
    assert (out / "a.wav").read_text() == "edited"


def test_an_interrupted_extraction_is_redone(tmp_path):
    """The failure this guards: a directory that exists but is incomplete must
    not be mistaken for a finished one, or the corpus silently shrinks."""
    from training.scripts.prepare_moespeech import extract_character

    z = _make_zip(tmp_path / "spk.zip", ["a.wav", "b.wav"])
    half = tmp_path / "extracted" / "spk"
    half.mkdir(parents=True)
    (half / "a.wav").write_text("partial")  # no completion marker

    out = extract_character(z, tmp_path / "extracted")
    assert out == half, "the half-extracted fixture was never the directory under test"
    assert sorted(p.name for p in out.iterdir() if p.suffix == ".wav") == ["a.wav", "b.wav"]
    assert (out / "a.wav").read_text() == "x"  # the zip's byte, not the fixture's stub


def test_a_stale_member_does_not_survive_the_redo(tmp_path):
    """Redoing an interrupted character rebuilds it rather than filling in the
    gaps: whatever the killed attempt left behind is discarded first. Merged
    in instead, a truncated clip would be indistinguishable from a whole one
    and would stay in the corpus for every run after."""
    from training.scripts.prepare_moespeech import extract_character

    z = _make_zip(tmp_path / "spk.zip", ["a.wav"])
    half = tmp_path / "extracted" / "spk"
    half.mkdir(parents=True)
    (half / "stale.wav").write_text("truncated")

    out = extract_character(z, tmp_path / "extracted")
    assert out == half, "the half-extracted fixture was never the directory under test"
    assert sorted(p.name for p in out.iterdir()) == ["a.wav"]


def test_a_kill_during_unpacking_leaves_no_completion_marker(tmp_path, monkeypatch):
    """The marker has to mean "the last member is on disk". Written before the
    unpack instead, a preemption mid-extract leaves a half-extracted speaker
    that every later run skips, and the corpus shrinks with nothing to show.

    The tests above either run to completion or start from a hand-made
    directory, so none of them enters the window between the first member
    landing and the marker being written -- which is the whole window a
    preemption can arrive in. This one is killed inside it.
    """
    import zipfile as zf

    from training.scripts.prepare_moespeech import extract_character

    z = _make_zip(tmp_path / "spk.zip", ["a.wav", "b.wav", "c.wav"])
    root = tmp_path / "extracted"

    def killed(self, path=None, *a, **kw):
        self.extract("a.wav", path)  # one member lands, then the instance goes
        raise KeyboardInterrupt

    monkeypatch.setattr(zf.ZipFile, "extractall", killed)
    with pytest.raises(KeyboardInterrupt):
        extract_character(z, root)
    monkeypatch.undo()

    out = extract_character(z, root)  # the re-run
    assert sorted(p.name for p in out.iterdir()) == ["a.wav", "b.wav", "c.wav"]


def test_a_marker_without_its_directory_is_not_trusted(tmp_path, monkeypatch):
    """A completed character whose directory is later reclaimed for disk space
    leaves the marker behind. The marker alone is not evidence -- and the stale
    one has to go before the re-unpack starts, or a kill during that re-unpack
    leaves marker-plus-half-a-directory, which every later run then skips.

    Every other test here starts from either an empty root or a marker-less
    directory, so none of them reaches the state "marker present, directory
    absent" -- the one state in which these two guards do any work at all.
    """
    import zipfile as zf

    from training.scripts.prepare_moespeech import extract_character

    z = _make_zip(tmp_path / "spk.zip", ["a.wav", "b.wav", "c.wav"])
    root = tmp_path / "extracted"
    shutil.rmtree(extract_character(z, root))  # the directory goes, the marker stays
    assert (root / "spk.complete").exists(), "the fixture did not leave a stale marker"

    def killed(self, path=None, *a, **kw):
        self.extract("a.wav", path)
        raise KeyboardInterrupt

    monkeypatch.setattr(zf.ZipFile, "extractall", killed)
    # Not raising would mean the marker alone was trusted and this call returned
    # a directory that does not exist -- Path.rglob on which yields nothing, so
    # the speaker reads as zero utterances downstream instead of failing.
    with pytest.raises(KeyboardInterrupt):
        extract_character(z, root)
    monkeypatch.undo()
    assert not (root / "spk.complete").exists(), "the stale marker outlived the kill"

    out = extract_character(z, root)
    assert sorted(p.name for p in out.iterdir()) == ["a.wav", "b.wav", "c.wav"]
