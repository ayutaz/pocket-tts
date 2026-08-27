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


def _partial_annotation(tmp_path, name, **fields):
    """An annotation with only the fields named -- what a truncated file has."""
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(fields, ensure_ascii=False), encoding="utf-8")
    return p


def test_an_annotation_missing_a_field_is_dropped(tmp_path):
    """The shape a file cut off mid-write actually has: everything after the
    first key gone at once. It says nothing about the individual guards -- the
    first one to fire ends the check and the rest are never reached -- so the
    two tests below take the fields one at a time."""
    from training.scripts.prepare_moespeech import read_annotation

    p = _partial_annotation(tmp_path, "bad", anime_whisper_transcription="あ")
    assert read_annotation(p) is None


def test_an_annotation_without_a_duration_is_dropped(tmp_path):
    """Rather than defaulting: a defaulted duration reaches the manifest as a
    window the audio need not contain, and the loader reads whatever is at
    those offsets -- silence, or the next utterance -- and trains on it as the
    speech the transcript describes. Nothing downstream can detect that.

    Everything else here is present, so this clip is one field away from being
    manifest-ready and only the duration guard can drop it. `probe_utterances`
    would otherwise fold the invented seconds into the hours column an operator
    reads the corpus size off.
    """
    from training.scripts.prepare_moespeech import read_annotation

    p = _partial_annotation(
        tmp_path,
        "no_duration",
        anime_whisper_transcription="あ",
        parakeet_jp_transcription="あ",
        speechMOS=3.5,
    )
    assert read_annotation(p) is None


def test_an_annotation_without_a_mos_is_dropped(tmp_path):
    """The same, for the field the audio-quality cutoff is read off. A
    defaulted speechMOS is worse than a missing one: `--min-mos` exists to keep
    noisy recordings out, and a stand-in score sails through whatever floor the
    operator chose, so the clips that reach the manifest under a strict floor
    are exactly the ones nothing ever measured.
    """
    from training.scripts.prepare_moespeech import read_annotation

    p = _partial_annotation(
        tmp_path,
        "no_mos",
        anime_whisper_transcription="あ",
        parakeet_jp_transcription="あ",
        duration=5.0,
    )
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


def test_utterances_over_the_cer_limit_are_dropped(tmp_path):
    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "agree", "こんにちは", "こんにちは")
    _annotation(tmp_path, "differ", "こんにちは", "全然違う文章です")
    kept = list(select_utterances(tmp_path, max_cer=0.2, min_mos=0.0))
    assert [u["id"] for u in kept] == ["agree"]


def test_utterances_below_the_mos_floor_are_dropped(tmp_path):
    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "clean", "あ", "あ", mos=4.0)
    _annotation(tmp_path, "noisy", "あ", "あ", mos=1.0)
    kept = list(select_utterances(tmp_path, max_cer=1.0, min_mos=3.0))
    assert [u["id"] for u in kept] == ["clean"]


def test_the_kept_transcript_is_normalized(tmp_path):
    """The manifest transcript, the tokenizer corpus and a user's inference
    input have to be the same distribution. align_data normalizes what it
    writes back; this must match, or the two disagree from the start."""
    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "wide", "ＡＢＣです", "ＡＢＣです")
    (u,) = list(select_utterances(tmp_path, max_cer=1.0, min_mos=0.0))
    assert u["transcript"] == "ABCです"


def test_selection_survives_the_files_the_probe_already_survived(tmp_path):
    """The selection pass walks the same 400,000 json files the probe walked,
    and runs after it. A bad file the probe counted and logged must not end
    this pass instead -- the operator has already been told about that file,
    read the probe's numbers, chosen cutoffs from them, and started what is by
    then the expensive half of the run.
    """
    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "good", "あ", "あ")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "index.json").write_text("[]", encoding="utf-8")

    kept = list(select_utterances(tmp_path, max_cer=1.0, min_mos=0.0))
    assert [u["id"] for u in kept] == ["good"]


def test_a_transcript_that_normalization_empties_is_dropped(tmp_path):
    """The emptiness check has to sit after normalization, not before it. A
    transcription of nothing but a full-width space is a string read_annotation
    keeps -- it is not empty, and the two ASRs agree on it perfectly -- and
    normalization then leaves nothing of it. Written out, that is a manifest
    entry with audio and no text: the aligner matches it to nothing and the
    model trains on the silence-or-noise under an empty label.
    """
    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "blank", "\u3000", "\u3000")  # U+3000, ideographic space
    assert list(select_utterances(tmp_path, max_cer=1.0, min_mos=0.0)) == []


def test_the_table_never_promises_a_clip_normalization_will_empty(tmp_path):
    """probe.json is the only artifact an operator reads when choosing cutoffs,
    so every count in it has to be a count of clips they will actually get.

    A transcription of nothing but a full-width space is the case where the two
    can come apart. It is a complete annotation, and the two ASRs agree on it
    character for character, so it clears both cutoffs and lands in the
    retention table; normalization then leaves nothing of it and the selection
    drops it. Counted on one side of that check and dropped on the other, the
    table promises two clips and the manifest holds one -- and a manifest short
    of what the table said is indistinguishable from a corpus that was smaller
    all along.

    Asserted over the whole grid, because a table that over-promises in one row
    over-promises in every row that clip falls into.
    """
    from training.scripts.prepare_moespeech import probe_utterances, select_utterances

    _annotation(tmp_path, "good", "\u3042", "\u3042", mos=4.0)
    _annotation(tmp_path, "blank", "\u3000", "\u3000", mos=4.0)

    table = probe_utterances(tmp_path)["retention"]
    assert table, "no retention table to check against"
    for row in table:
        kept = list(select_utterances(tmp_path, max_cer=row["max_cer"], min_mos=row["min_mos"]))
        assert len(kept) == row["kept"], row
    assert max(row["kept"] for row in table) == 1, table


def test_the_probe_counts_the_clips_normalization_emptied(tmp_path):
    """Dropping them is only half of it. `unreadable` and `incomplete` are
    reported for a reason -- a pass that returns fewer clips than the corpus
    holds and says nothing looks exactly like a smaller corpus -- and a clip
    whose transcript normalization empties is dropped for a third reason that
    deserves its own number rather than being folded into either of theirs.
    """
    from training.scripts.prepare_moespeech import probe_utterances

    _annotation(tmp_path, "good", "\u3042", "\u3042")
    _annotation(tmp_path, "blank", "\u3000", "\u3000")

    stats = probe_utterances(tmp_path)
    assert stats["count"] == 1, stats
    assert stats["blank"] == 1, stats
    assert stats["unreadable"] == 0, stats
    assert stats["incomplete"] == 0, stats


def test_a_clip_sitting_exactly_on_both_cutoffs_is_kept(tmp_path):
    """The cutoffs are ceilings and floors, not strict bounds.

    An operator reads a pair off probe.json's retention table and passes it
    straight in, so the comparison here has to be the comparison _retention
    used. The row that makes this matter is max_cer=0.0: clips where both ASRs
    agree character for character are the most trustworthy in the corpus and a
    large share of it, and a strict `>` there would drop every one of them
    while the table promised they were the ones being kept -- an empty manifest
    from the cutoffs the measurement recommended, with nothing raised.
    """
    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "exact", "あ", "あ", mos=3.0)  # cer 0.0, mos on the floor

    kept = list(select_utterances(tmp_path, max_cer=0.0, min_mos=3.0))
    assert [u["id"] for u in kept] == ["exact"]


def test_selection_keeps_exactly_as_many_clips_as_the_table_promised(tmp_path):
    """The retention table is the whole interface between the probe and the
    choice of cutoffs: the operator reads a count of clips off one of its rows
    and then runs the selection with that row's pair. If the two disagree by so
    much as one comparison, the number they chose from described a different
    corpus from the one they get, and nothing tells them so.

    So assert it over the entire grid rather than at one point. The fixtures
    land on grid values on purpose -- cer exactly 0.0 and 0.2, mos exactly 2.5,
    3.0 and 4.0 -- because a clip strictly inside every row's bounds is kept by
    a strict and an inclusive comparison alike and would prove nothing.
    """
    from training.scripts.prepare_moespeech import probe_utterances, select_utterances

    _annotation(tmp_path, "perfect", "こんにちは", "こんにちは", mos=3.0)
    _annotation(tmp_path, "quiet", "こんにちは", "こんにちは", mos=2.5)
    _annotation(tmp_path, "near", "こんにちは", "こんにちわ", mos=4.0)
    _annotation(tmp_path, "wrong", "こんにちは", "全然違う文章です", mos=4.0)

    table = probe_utterances(tmp_path)["retention"]
    assert table, "no retention table to check against"
    for row in table:
        kept = list(select_utterances(tmp_path, max_cer=row["max_cer"], min_mos=row["min_mos"]))
        assert len(kept) == row["kept"], row


def test_a_kept_utterance_carries_what_the_next_stage_needs(tmp_path):
    """Every field, not just the two the filters are about.

    Nothing downstream validates this dict. The concatenation stage groups by
    `speaker` and refuses a mixed group, so a per-clip value in that field
    turns every speaker into a group of one and defeats the pseudo-long
    recordings entirely; it reads `wav` to find the audio and carries
    `duration` through to the manifest as the window the loader will read. A
    wrong `duration` is the worst of them: it raises nothing, and trains the
    model on speech that does not match its text.

    Every value here is one no constant could have produced. The duration is
    not the fixture's default; the MOS is 2.75, which is neither the default
    nor a grid threshold; and the two transcriptions differ in their last
    character out of five, so `cer` is exactly 0.2 -- where identical
    transcriptions would give 0.0, which a hardcoded zero and a CER call that
    never ran produce just as well. The clip sits under a real speaker
    directory, and the whole dict is compared at once so an extra key -- the
    raw per-ASR transcriptions read_annotation also returns -- is caught too.
    """
    from training.scripts.prepare_moespeech import select_utterances

    speaker = tmp_path / "ずんだもん"
    speaker.mkdir()
    _annotation(speaker, "clip_0001", "こんにちは", "こんにちわ", duration=4.25, mos=2.75)

    (u,) = list(select_utterances(tmp_path, max_cer=1.0, min_mos=0.0))
    assert u == {
        "id": "clip_0001",
        "speaker": "ずんだもん",
        "wav": speaker / "clip_0001.wav",
        "duration": 4.25,
        "transcript": "こんにちは",
        "cer": pytest.approx(0.2),
        "mos": 2.75,
    }


def test_the_selection_yields_its_utterances_in_path_order(tmp_path):
    """A re-run has to produce the same manifest as the run it replaces.

    This script is built to be killed and restarted, and the stage after it
    concatenates each speaker's clips into pseudo-long recordings and writes
    offsets into them. Taken in whatever order the filesystem hands them over,
    a second run lays the same clips down in a different arrangement, and the
    manifest the first run left on disk stops describing the audio -- silently,
    since every offset is still inside a real file.

    Every other selection test here keeps exactly one utterance, which no
    ordering can get wrong. So this one keeps two, and creates them back to
    front, where only an actual sort turns them around again.
    """
    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "clip_0002", "あ", "あ")
    _annotation(tmp_path, "clip_0001", "あ", "あ")

    kept = list(select_utterances(tmp_path, max_cer=1.0, min_mos=0.0))
    assert [u["id"] for u in kept] == ["clip_0001", "clip_0002"]


def _wav(path, seconds, hz, sr=44100):
    """A pure tone, so a window can be identified by its frequency."""
    import numpy as np
    import sphn

    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sphn.write_wav(str(path), (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32), sr)
    return path


def _clips(tmp_path, seconds, tones, speaker="spk"):
    """One utterance per tone, shaped the way select_utterances hands them over."""
    return [
        {
            "id": f"u{i}",
            "wav": _wav(tmp_path / f"u{i}.wav", seconds, hz),
            "duration": seconds,
            "transcript": "あ",
            "speaker": speaker,
        }
        for i, hz in enumerate(tones)
    ]


def _dominant_hz(path, start, duration):
    """The frequency of the window a manifest entry claims its clip lives in.

    Reading out of range raises rather than returning silence, so an entry that
    points past the end of its file fails here too, and loudly."""
    import numpy as np
    import sphn

    wav, sr = sphn.read(path, start_sec=start, duration_sec=duration)
    mono = wav.mean(axis=0)
    freqs = np.fft.rfftfreq(len(mono), d=1 / sr)
    return freqs[np.argmax(np.abs(np.fft.rfft(mono)))]


def test_each_utterance_points_at_its_own_audio(tmp_path):
    """The property everything else rests on. Each source clip is a distinct
    tone, so reading back a window and taking its dominant frequency proves
    whether start/duration point where the manifest claims."""
    import numpy as np
    import sphn

    from training.scripts.prepare_moespeech import concatenate

    tones = [220.0, 440.0, 880.0]
    utts = [
        {
            "id": f"u{i}",
            "wav": _wav(tmp_path / f"u{i}.wav", 2.0, hz),
            "duration": 2.0,
            "transcript": "あ",
            "speaker": "spk",
        }
        for i, hz in enumerate(tones)
    ]
    out = concatenate(utts, tmp_path / "joined.wav", target_sec=60.0)
    assert len(out) == 3

    for entry, hz in zip(out, tones):
        wav, sr = sphn.read(entry["path"], start_sec=entry["start"], duration_sec=entry["duration"])
        mono = wav.mean(axis=0)
        freqs = np.fft.rfftfreq(len(mono), d=1 / sr)
        dominant = freqs[np.argmax(np.abs(np.fft.rfft(mono)))]
        assert abs(dominant - hz) < 5, (entry, dominant, hz)


def test_durations_sum_to_the_file_length(tmp_path):
    """A gap nobody accounted for would put every later start off by it."""
    import sphn

    from training.scripts.prepare_moespeech import concatenate

    utts = [
        {
            "id": f"u{i}",
            "wav": _wav(tmp_path / f"u{i}.wav", 1.5, 440.0),
            "duration": 1.5,
            "transcript": "あ",
            "speaker": "spk",
        }
        for i in range(4)
    ]
    out = concatenate(utts, tmp_path / "joined.wav", target_sec=60.0)
    wav, sr = sphn.read(str(tmp_path / "joined.wav"))
    assert abs(len(wav[0]) / sr - (out[-1]["start"] + out[-1]["duration"])) < 0.05


def test_a_long_run_is_split_into_several_files(tmp_path):
    """target_sec bounds each file, or one speaker becomes one enormous wav."""
    from training.scripts.prepare_moespeech import concatenate

    utts = [
        {
            "id": f"u{i}",
            "wav": _wav(tmp_path / f"u{i}.wav", 2.0, 440.0),
            "duration": 2.0,
            "transcript": "あ",
            "speaker": "spk",
        }
        for i in range(10)
    ]
    out = concatenate(utts, tmp_path / "joined.wav", target_sec=5.0)
    assert len({e["path"] for e in out}) > 1


def test_every_utterance_survives_the_concatenation(tmp_path):
    """Losing one is losing training data with nothing to report it."""
    from training.scripts.prepare_moespeech import concatenate

    utts = [
        {
            "id": f"u{i}",
            "wav": _wav(tmp_path / f"u{i}.wav", 1.0, 440.0),
            "duration": 1.0,
            "transcript": "あ",
            "speaker": "spk",
        }
        for i in range(7)
    ]
    out = concatenate(utts, tmp_path / "joined.wav", target_sec=3.0)
    assert sorted(e["id"] for e in out) == sorted(u["id"] for u in utts)


def test_a_second_file_measures_its_offsets_from_its_own_beginning(tmp_path):
    """The split is where the offsets are most easily wrong.

    An offset is into the file it names, not into the speaker's whole run, so a
    cursor that keeps counting across the split sends the third clip here to
    4.0 seconds inside a file that is 4.0 seconds long. Every start is still a
    plausible number and the manifest still validates; only the audio says
    otherwise. Four distinct tones, and a target that fits two of them, so each
    half is checked against the sound actually written there.
    """
    from training.scripts.prepare_moespeech import concatenate

    tones = [220.0, 330.0, 550.0, 880.0]
    out = concatenate(_clips(tmp_path, 2.0, tones), tmp_path / "joined.wav", target_sec=5.0)
    assert len({e["path"] for e in out}) == 2, out
    assert [e["start"] for e in out] == [0.0, 2.0, 0.0, 2.0], out

    for entry, hz in zip(out, tones):
        dominant = _dominant_hz(entry["path"], entry["start"], entry["duration"])
        assert abs(dominant - hz) < 5, (entry, dominant, hz)


def test_the_first_file_is_the_name_it_was_given(tmp_path):
    """The caller names the output and then has to be able to find it.

    `test_durations_sum_to_the_file_length` above reads `joined.wav` by that
    name, so the first file may not be renamed to `joined_000.wav` for
    tidiness; the rest are numbered from 001 beside it. This is the only test
    that says so, and the numbering is a contract because the stage that
    follows collects these files by name.
    """
    from pathlib import Path

    from training.scripts.prepare_moespeech import concatenate

    out = concatenate(_clips(tmp_path, 2.0, [440.0] * 10), tmp_path / "joined.wav", target_sec=5.0)
    names = [Path(p).name for p in dict.fromkeys(e["path"] for e in out)]
    assert names[0] == "joined.wav"
    assert names[1:] == [f"joined_{i:03d}.wav" for i in range(1, len(names))], names
    assert all((tmp_path / n).is_file() for n in names), names


def test_no_output_file_runs_past_the_target(tmp_path):
    """target_sec is a bound on each file, not an average over them.

    The test above only asks that a long run be split at all, which a split
    every hundredth clip satisfies while still writing files far longer than
    the caller asked for. What the bound is for is the alignment stage, whose
    memory use goes with the length of a single file.
    """
    import sphn

    from training.scripts.prepare_moespeech import concatenate

    out = concatenate(_clips(tmp_path, 2.0, [440.0] * 10), tmp_path / "joined.wav", target_sec=5.0)
    for path in dict.fromkeys(e["path"] for e in out):
        wav, sr = sphn.read(path)
        assert len(wav[0]) / sr <= 5.0 + 1e-6, path


def test_the_offsets_describe_the_audio_and_not_the_annotation(tmp_path):
    """`duration` in the annotation is a number somebody else's tool wrote.

    The offsets have to describe the samples this function actually laid down.
    Trusting the annotation instead costs nothing while the two agree and, the
    moment they do not, slides every later clip in the file by the difference
    -- which is exactly the failure this stage exists to avoid, and it raises
    nothing whatsoever. Here the annotation claims five seconds for a clip of
    one, so a manifest built out of it puts the second clip at 5.0 seconds
    inside a file that is 2.0 seconds long.
    """
    from training.scripts.prepare_moespeech import concatenate

    utts = [
        {
            "id": "u0",
            "wav": _wav(tmp_path / "u0.wav", 1.0, 220.0),
            "duration": 5.0,
            "transcript": "あ",
            "speaker": "spk",
        },
        {
            "id": "u1",
            "wav": _wav(tmp_path / "u1.wav", 1.0, 880.0),
            "duration": 5.0,
            "transcript": "あ",
            "speaker": "spk",
        },
    ]
    out = concatenate(utts, tmp_path / "joined.wav", target_sec=60.0)
    assert [e["start"] for e in out] == [0.0, 1.0], out
    assert [e["duration"] for e in out] == [1.0, 1.0], out
    assert abs(_dominant_hz(out[1]["path"], out[1]["start"], out[1]["duration"]) - 880.0) < 5


def test_clips_from_two_speakers_are_never_joined(tmp_path):
    """Everything downstream assumes one voice per file.

    The loader cuts one of these recordings at an arbitrary point and uses one
    side as the voice prompt for the other, so a file holding two speakers
    teaches the model that the prompt does not determine the voice. The caller
    groups by speaker; this is the guard that turns that intention into
    something enforced, and it has to fire before anything is written rather
    than leave a mixed file on disk beside the exception.
    """
    from training.scripts.prepare_moespeech import concatenate

    utts = [
        {
            "id": "u0",
            "wav": _wav(tmp_path / "u0.wav", 1.0, 220.0),
            "duration": 1.0,
            "transcript": "あ",
            "speaker": "ずんだもん",
        },
        {
            "id": "u1",
            "wav": _wav(tmp_path / "u1.wav", 1.0, 880.0),
            "duration": 1.0,
            "transcript": "あ",
            "speaker": "四国めたん",
        },
    ]
    with pytest.raises(ValueError):
        concatenate(utts, tmp_path / "joined.wav", target_sec=60.0)
    assert not (tmp_path / "joined.wav").exists()


def test_a_concatenated_entry_carries_what_the_manifest_needs(tmp_path):
    """The whole dict, because the manifest is written straight out of it.

    `speaker` and `transcript` are carried through from the selection -- the
    alignment stage reads the transcript and the manifest keeps the speaker --
    while `path`, `start` and `duration` are what this stage decided. `path` is
    a string because it is about to be JSON; the two measurements the selection
    made are not carried, having already done their work there.
    """
    from training.scripts.prepare_moespeech import concatenate

    utts = [
        {
            "id": "clip_0001",
            "speaker": "ずんだもん",
            "duration": 1.25,
            "transcript": "こんにちは",
            "wav": _wav(tmp_path / "clip_0001.wav", 1.25, 440.0),
            "cer": 0.2,
            "mos": 2.75,
        }
    ]
    (entry,) = concatenate(utts, tmp_path / "joined.wav", target_sec=60.0)
    assert entry == {
        "id": "clip_0001",
        "speaker": "ずんだもん",
        "transcript": "こんにちは",
        "path": str(tmp_path / "joined.wav"),
        "start": 0.0,
        "duration": 1.25,
    }


def test_a_speaker_left_with_no_clips_writes_no_file(tmp_path):
    """The cutoffs can empty a speaker, and an empty wav is not a training
    file: the aligner would read a header and no samples, and whatever it made
    of that would go into the manifest."""
    from training.scripts.prepare_moespeech import concatenate

    assert concatenate([], tmp_path / "joined.wav", target_sec=60.0) == []
    assert not (tmp_path / "joined.wav").exists()


def test_a_kill_during_the_write_leaves_no_file_to_be_trusted(tmp_path, monkeypatch):
    """This runs on an instance that can be reclaimed mid-write.

    A half-written `joined.wav` is a valid wav that is merely shorter than the
    manifest says, so every offset past the truncation reads as silence or
    fails to seek -- and a re-run that finds the file sitting there has no way
    to tell it from a finished one. So the samples land beside the name first
    and are renamed in only once all of them are there.
    """
    from pathlib import Path

    import training.scripts.prepare_moespeech as m
    from training.scripts.prepare_moespeech import concatenate

    # Built before write_wav is replaced: the fixture lays its clips down
    # through that same function, so building them under the patch would kill
    # the fixture and never reach concatenate at all.
    clips = _clips(tmp_path, 1.0, [440.0])

    def killed(path, *a, **kw):
        Path(path).write_bytes(b"RIFF")  # what the kill would have left behind
        raise KeyboardInterrupt

    monkeypatch.setattr(m.sphn, "write_wav", killed)
    with pytest.raises(KeyboardInterrupt):
        concatenate(clips, tmp_path / "joined.wav", target_sec=60.0)
    assert not (tmp_path / "joined.wav").exists()
    # The bytes did land -- beside the name, under one a re-run does not trust.
    assert (tmp_path / "joined.wav.partial").read_bytes() == b"RIFF"
