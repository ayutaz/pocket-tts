"""Building a training manifest out of MoeSpeech.

Everything here guards one property: the manifest must describe the audio
truthfully. A wrong `start` or `duration` does not raise -- it trains the model
on speech that does not match its text, and the only symptom is a model that
never quite becomes intelligible.
"""

import csv
import json
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


def _annotation(tmp_path, name, whisper, parakeet, duration=5.0, mos=3.5):
    p = tmp_path / f"{name}.json"
    p.write_text(
        json.dumps(
            {
                "anime_whisper_transcription": whisper,
                "parakeet_jp_transcription": parakeet,
                "duration": duration,
                "speechMOS": mos,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return p


def test_identical_transcripts_score_zero_cer(tmp_path):
    """Two ASRs agreeing is the strongest signal available that the transcript
    is right -- there is no manual transcription to compare against."""
    from training.scripts.prepare_moespeech import read_annotation

    a = read_annotation(_annotation(tmp_path, "a", "こんにちは", "こんにちは"))
    assert a["cer"] == 0.0


def test_disagreeing_transcripts_score_high_cer(tmp_path):
    from training.scripts.prepare_moespeech import read_annotation

    a = read_annotation(_annotation(tmp_path, "a", "こんにちは", "全然違う文章です"))
    assert a["cer"] > 0.5


def test_cer_is_measured_against_the_transcript_the_manifest_will_carry(tmp_path):
    """Which of the two ASRs is the reference decides the number, not just its
    sign. jiwer normalises the edit distance by the reference's length, so the
    two orders stop agreeing the moment the systems disagree about *length* --
    which is the commonest ASR failure, one of them truncating.

    The pair here is deliberately lopsided: fifteen characters against two.
    Measured against the whisper side, which is the string `transcript` carries
    into the manifest, the disagreement is 13/15 and the clip sits inside the
    `max_cer=1.0` row of the retention grid. Measured the other way round it is
    6.5 -- outside every row in the grid -- so the swap would move the whole
    retention table and the whole reported CER distribution with it. Both
    orders are `> 0.5` here, and both are `> 0` on any pair that disagrees at
    all, so only an assertion naming the value has a side to it.
    """
    from training.scripts.prepare_moespeech import read_annotation

    a = read_annotation(_annotation(tmp_path, "a", "こんにちは今日はいい天気ですね", "こん"))
    assert a["transcript"] == "こんにちは今日はいい天気ですね"
    assert a["cer"] == pytest.approx(13 / 15), a["cer"]  # reversed, this pair is 6.5


def test_an_annotation_missing_a_field_is_dropped(tmp_path):
    """Rather than defaulting: a missing duration would become a wrong
    manifest entry, and the loader would read a window that is not there."""
    from training.scripts.prepare_moespeech import read_annotation

    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"anime_whisper_transcription": "あ"}), encoding="utf-8")
    assert read_annotation(p) is None


def test_an_empty_transcription_is_dropped(tmp_path):
    """An empty string is missing too, and worse than missing: jiwer.cer
    divides by the reference's length and raises on an empty reference, so one
    such clip ends the whole pass rather than costing one utterance."""
    from training.scripts.prepare_moespeech import read_annotation

    assert read_annotation(_annotation(tmp_path, "a", "", "こんにちは")) is None
    assert read_annotation(_annotation(tmp_path, "b", "こんにちは", "")) is None


def test_a_json_that_is_not_an_object_is_dropped(tmp_path):
    """Not every `.json` under the extract root is an annotation -- a character
    zip may ship an index or a metadata file among the per-clip ones. A list or
    a bare number parses without complaint and then has no `.get`, so treating
    it as an annotation raises AttributeError rather than returning None."""
    from training.scripts.prepare_moespeech import read_annotation

    for content in ("[]", "3", '"text"', "null"):
        p = tmp_path / "index.json"
        p.write_text(content, encoding="utf-8")
        assert read_annotation(p) is None, content


def test_an_annotation_names_its_clip_and_its_speaker(tmp_path):
    """Neither is in the JSON: the wav is its sibling and the speaker is the
    directory the zip was unpacked into. Later stages concatenate by wav and
    group by speaker, so getting these from anywhere else is not possible."""
    from training.scripts.prepare_moespeech import read_annotation

    spk = tmp_path / "ずんだもん"
    spk.mkdir()
    a = read_annotation(_annotation(spk, "clip_0001", "あ", "あ"))
    assert a["id"] == "clip_0001"
    assert a["speaker"] == "ずんだもん"
    assert a["wav"] == spk / "clip_0001.wav"


def test_retention_excludes_on_mos(tmp_path):
    """The output that decides the next task's defaults -- the MOS half of it,
    which is where `--min-mos` will be read from.

    Every clip here has identical transcriptions, so CER is exactly 0.0 for all
    ten and passes every row of the grid: the only thing that can separate
    seven clips from three is the `mos >= min_mos` comparison. That isolation
    is the point. With a default MOS shared by the whole fixture the retention
    numbers come out right for the wrong reason -- the CER predicate produces
    them and an inverted MOS comparison changes nothing -- and probe.json then
    reports that a 2.0-MOS corpus survives `--min-mos 4.0`.

    The row is looked up rather than searched for, because a 7 somewhere in a
    40-row table says nothing about which cutoffs produced it.
    """
    from training.scripts.prepare_moespeech import probe_utterances

    for i in range(10):
        _annotation(tmp_path, f"u{i}", "こんにちは", "こんにちは", mos=4.2 if i < 7 else 2.0)

    stats = probe_utterances(tmp_path)
    assert stats["count"] == 10
    row = next(r for r in stats["retention"] if r["max_cer"] == 1.0 and r["min_mos"] == 3.5)
    assert row["kept"] == 7, row
    assert row["fraction"] == 0.7, row


def test_retention_keeps_the_clips_sitting_exactly_on_the_cer_cutoff(tmp_path):
    """The CER half, at the boundary. `max_cer=0.0` -- the two ASRs agree
    exactly -- is the most informative row in the table and the first one a
    reader goes to, and it is also the only row where `<=` and `<` differ for
    a corpus of perfectly agreeing clips: strict, it reports kept=0, which
    reads as "nothing here is trustworthy" for a corpus in which seven clips
    are as trustworthy as this dataset can show.

    MOS is held above the row's `min_mos` of 0.0 for all ten, so the CER
    comparison is the only thing doing any excluding.
    """
    from training.scripts.prepare_moespeech import probe_utterances

    for i in range(10):
        _annotation(tmp_path, f"u{i}", "こんにちは", "こんにちは" if i < 7 else "違う", mos=4.2)

    stats = probe_utterances(tmp_path)
    assert stats["count"] == 10
    row = next(r for r in stats["retention"] if r["max_cer"] == 0.0 and r["min_mos"] == 0.0)
    assert row["kept"] == 7, row


def test_probe_reports_hours_of_audio_and_not_seconds_of_it(tmp_path):
    """`hours` is the column an operator actually reads: the question put to
    this table is not "what fraction survives" but "does --max-cer 0.1 still
    leave me the hundred hours I was asked for". It is also the only number in
    probe.json carried in a different unit from the field it is summed from,
    and a seconds- or minutes-for-hours slip is invisible in the file itself --
    every row simply reads uniformly larger, the strictest cutoff appears to
    clear the target, and the shortfall surfaces only after the corpus has been
    built. The same seam has caught this file once already on the other side,
    where info.csv's column is minutes and `--hours` is hours.

    Durations differ between the kept clips and the dropped ones (7 x 30 s
    agreeing, 3 x 90 s not) so that the corpus total, the kept total, and any
    count-times-average stand-in for either are three different numbers: the
    corpus is 480 s and the `max_cer=0.0` row keeps 210 s of it, which is
    exactly the point `_retention` exists to make -- 70% of the clips is 44% of
    the audio.
    """
    from training.scripts.prepare_moespeech import probe_utterances

    for i in range(10):
        agrees = i < 7
        _annotation(
            tmp_path,
            f"u{i}",
            "こんにちは",
            "こんにちは" if agrees else "違う",
            duration=30.0 if agrees else 90.0,
            mos=4.2,
        )

    stats = probe_utterances(tmp_path)
    assert stats["hours"] == pytest.approx(480.0 / 3600)
    row = next(r for r in stats["retention"] if r["max_cer"] == 0.0 and r["min_mos"] == 0.0)
    assert row["kept"] == 7, row
    assert row["hours"] == pytest.approx(210.0 / 3600), row


def test_probe_reports_a_distribution_and_not_just_an_average(tmp_path):
    """The next stage reads its cutoffs off these numbers. A median that is
    really a mean reads a skewed corpus as a symmetric one and moves every
    cutoff with it, without anything failing to say so."""
    from training.scripts.prepare_moespeech import probe_utterances

    for i, seconds in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]):
        _annotation(tmp_path, f"u{i}", "あ", "あ", duration=seconds)

    dist = probe_utterances(tmp_path)["duration"]
    assert dist["min"] == 0.0
    assert dist["max"] == 100.0
    assert dist["median"] == 4.5, dist  # the mean of these is 13.6
    ordered = [dist[k] for k in ("min", "p10", "p25", "median", "p75", "p90", "max")]
    assert ordered == sorted(ordered), dist


def test_probe_reports_each_quantity_under_its_own_name(tmp_path):
    """cer and mos are two floats on the same row, and reporting one under the
    other's name is invisible in probe.json: CER lives in [0, ~1.6] and
    speechMOS in [1, 5], so a transposed file reads as a corpus with alarming
    transcripts and implausibly good audio rather than as a bug -- and the next
    stage sets both cutoffs off exactly these numbers.

    The two are given disjoint ranges here, CER over [0.0, 1.0] and MOS over
    [1.0, 5.0] with nothing below 1.0, so no swap survives all four bounds.
    """
    from training.scripts.prepare_moespeech import probe_utterances

    _annotation(tmp_path, "u0", "あい", "あい", mos=1.0)  # cer 0.0
    _annotation(tmp_path, "u1", "あい", "うえ", mos=5.0)  # cer 1.0, both characters wrong
    _annotation(tmp_path, "u2", "あい", "あえ", mos=3.0)  # cer 0.5

    stats = probe_utterances(tmp_path)
    assert stats["cer"]["min"] == 0.0, stats["cer"]
    assert stats["cer"]["max"] == 1.0, stats["cer"]
    assert stats["mos"]["min"] == 1.0, stats["mos"]
    assert stats["mos"]["max"] == 5.0, stats["mos"]


def test_probe_walks_the_per_character_directories(tmp_path):
    """The corpus is one directory per character under the extract root. A scan
    of the root's own files finds nothing there and reports an empty dataset
    without failing, which looks exactly like a corpus that was never fetched."""
    from training.scripts.prepare_moespeech import probe_utterances

    for speaker in ("aaa", "bbb"):
        d = tmp_path / speaker
        d.mkdir()
        _annotation(d, "u0", "あ", "あ")

    assert probe_utterances(tmp_path)["count"] == 2


def test_probe_survives_a_corrupt_json(tmp_path):
    """One bad file in 400,000 must not end a 40-minute pass."""
    from training.scripts.prepare_moespeech import probe_utterances

    _annotation(tmp_path, "good", "あ", "あ")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert probe_utterances(tmp_path)["count"] == 1


def test_probe_survives_a_json_that_is_valid_but_is_not_an_annotation(tmp_path):
    """The corrupt-json case above only reaches the decode-error branch. A file
    that parses fine and is not an object -- the index json a character zip
    ships beside its clips -- reaches the field lookups instead and raises
    AttributeError, which is not a decode error and would end the pass over
    one file that was never a clip in the first place."""
    from training.scripts.prepare_moespeech import probe_utterances

    _annotation(tmp_path, "good", "あ", "あ")
    (tmp_path / "index.json").write_text("[]", encoding="utf-8")

    stats = probe_utterances(tmp_path)
    assert stats["count"] == 1
    assert stats["incomplete"] == 1, stats
    assert stats["unreadable"] == 0, stats


def test_probe_counts_what_it_skipped(tmp_path):
    """Surviving a bad file silently is its own defect: a pass over 400,000
    clips that reports 300,000 and no error is indistinguishable from a corpus
    that was only ever 300,000 clips long."""
    from training.scripts.prepare_moespeech import probe_utterances

    _annotation(tmp_path, "good", "あ", "あ")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "short.json").write_text(json.dumps({"duration": 1.0}), encoding="utf-8")

    stats = probe_utterances(tmp_path)
    assert stats["count"] == 1
    assert stats["unreadable"] == 1
    assert stats["incomplete"] == 1
