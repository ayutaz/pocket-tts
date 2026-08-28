"""Building a training manifest out of MoeSpeech.

Everything here guards one property: the manifest must describe the audio
truthfully. A wrong `start` or `duration` does not raise -- it trains the model
on speech that does not match its text, and the only symptom is a model that
never quite becomes intelligible.
"""

import csv
import inspect
import json
import logging
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from training.scripts.prepare_moespeech import EXTRACT_MARKER, select_characters


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


def _annotation(tmp_path, name, whisper, parakeet, duration=5.0, mos=3.5, wav=True):
    """One clip's annotation, with the audio it describes beside it by default.

    The scan drops an annotation whose wav is not there, because MoeSpeech ships
    a dated .bak.json twin of every clip and only the audio tells the two apart.
    So a fixture that wants to be *kept* has to have one. It is 50 ms of tone --
    nothing at this stage reads the audio, only whether it exists -- and
    `wav=False` is how a fixture asks to be the twin.
    """
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
    if wav:
        _wav(p.with_suffix(".wav"), 0.05, 440.0)
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


def test_two_transcripts_that_differ_only_in_width_agree(tmp_path):
    """The CER has to be measured over the strings the manifest will carry.

    Two Japanese ASR systems differ by convention far more than they differ by
    hearing, and normalization is exactly what erases the conventions: NFKC
    folds full-width latin and digits onto ASCII, half-width katakana onto
    full-width, and collapses the spacing. Measured raw, each of these pairs
    scores 0.25 to 0.75 -- inside no cutoff an operator would ever pick off the
    retention table -- while the two strings the manifest would hold are the
    same string. Measured after normalization they agree exactly, which is what
    they do.

    It is not only recall. probe.json's CER distribution and its whole
    retention table would be computed over strings this corpus never contains,
    and that table is the single artifact the cutoffs are chosen from.
    """
    from training.scripts.prepare_moespeech import read_annotation, select_utterances

    # Both directions: the conventions land on either side, and normalizing
    # only the reference leaves the number measured against the raw other one.
    pairs = [("ＡＢＣです", "ABCです"), ("123円", "１２３円"), ("ｱｲｳです", "アイウです")]
    for i, (whisper, parakeet) in enumerate(pairs):
        a = read_annotation(_annotation(tmp_path, f"pair{i}", whisper, parakeet))
        assert a["cer"] == 0.0, (whisper, parakeet, a["cer"])

    # And they survive the strictest row of the grid, which is the row those
    # clips belong in: two systems agreeing character for character is the
    # strongest evidence this corpus offers that a transcript is right.
    kept = list(select_utterances(tmp_path, max_cer=0.0, min_mos=0.0))
    assert [u["id"] for u in kept] == ["pair0", "pair1", "pair2"], kept
    assert [u["transcript"] for u in kept] == ["ABCです", "123円", "アイウです"], kept


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
    """An empty string is missing too, and worse than missing: CER is the edit
    distance over the reference's length, so a clip whose reference is empty
    has no disagreement to report and would be filtered on a number that means
    nothing."""
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


def test_the_speaker_is_the_character_and_not_the_directory_the_clip_sits_in(tmp_path):
    """The scan walks each character's root with `rglob`, so the clips may sit
    anywhere below it -- and `<name>/wav/clip.json` is an ordinary way to pack a
    zip. Nobody has unpacked a real MoeSpeech zip, and every other fixture here
    lays the clips flat, which is the one layout where the directory a JSON
    sits in happens to be the character.

    Read off the path in this one, every clip of every character is labelled
    "wav". That is not a cosmetic wrong name: `concatenate` refuses a mixed
    list by comparing exactly this label, so two characters sharing one would
    be joined into a single recording -- the loader takes one side of a cut as
    the voice prompt for the other, which is the property the whole pipeline
    exists to protect -- the split would hold out a label rather than a voice,
    and `audio/<speaker>.wav` would collide between them. None of it raises.

    So the speaker is the root the scan is walking, which the scan knows, and
    not the parent directory, which it does not.
    """
    from training.scripts.prepare_moespeech import probe_utterances, select_utterances

    for name in ("kasumi", "yukino"):
        nested = tmp_path / name / "wav"
        nested.mkdir(parents=True)
        _annotation(nested, "clip_0001", "あ", "あ")
        _annotation(nested, "clip_0002", "い", "い")

    kept = list(select_utterances(tmp_path, max_cer=1.0, min_mos=0.0, names=["kasumi", "yukino"]))

    assert len(kept) == 4, kept
    assert sorted({u["speaker"] for u in kept}) == ["kasumi", "yukino"], kept
    # Two characters, and two of the utterances belong to each -- not one label
    # holding all four, which is what reading the parent directory produces.
    assert sorted(Counter(u["speaker"] for u in kept).values()) == [2, 2], kept
    # And the audio is still the JSON's sibling, wherever the JSON sits.
    for utterance in kept:
        assert utterance["wav"] == utterance["wav"].parent / f"{utterance['id']}.wav"
        assert utterance["wav"].parent.name == "wav"
    # The probe walks the same tree through the same scan, so it counts them too.
    assert probe_utterances(tmp_path, ["kasumi", "yukino"])["count"] == 4


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


def test_the_selection_yields_its_utterances_in_path_order(tmp_path, monkeypatch):
    """A re-run has to produce the same manifest as the run it replaces.

    This script is built to be killed and restarted, and the stage after it
    concatenates each speaker's clips into pseudo-long recordings and writes
    offsets into them. Taken in whatever order the filesystem hands them over,
    a second run lays the same clips down in a different arrangement, and the
    manifest the first run left on disk stops describing the audio -- silently,
    since every offset is still inside a real file.

    Every other selection test here keeps exactly one utterance, which no
    ordering can get wrong. So this one keeps two -- and hands them over in the
    wrong order rather than merely creating them in it. Creation order proves
    nothing here: NTFS keeps directory entries by name, so `rglob` returns them
    sorted whatever order they were written in, and an assertion resting on
    that is an assertion about the filesystem that no change to this code can
    fail. The patch below is checked first, so a walk that arrived sorted
    anyway would be caught rather than quietly making the test vacuous again.
    """
    import pathlib

    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "clip_0001", "あ", "あ")
    _annotation(tmp_path, "clip_0002", "あ", "あ")

    walk = pathlib.Path.rglob
    monkeypatch.setattr(
        pathlib.Path, "rglob", lambda self, pat, **kw: sorted(walk(self, pat, **kw), reverse=True)
    )
    assert [p.name for p in tmp_path.rglob("*.json")] == ["clip_0002.json", "clip_0001.json"]

    kept = list(select_utterances(tmp_path, max_cer=1.0, min_mos=0.0))
    assert [u["id"] for u in kept] == ["clip_0001", "clip_0002"]


def _wav(path, seconds, hz, sr=44100):
    """A pure tone, so a window can be identified by its frequency."""
    import numpy as np
    import sphn

    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sphn.write_wav(str(path), (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32), sr)
    return path


def _stereo_wav(path, seconds, hz, sr=44100):
    """The same tone in two channels: the shape `np.concatenate` refuses."""
    import numpy as np
    import sphn

    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    sphn.write_wav(str(path), np.stack([tone, tone]), sr)
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
    the caller asked for. What the bound is for is this stage: `_write_joined`
    holds a whole file in memory to concatenate it, and a preemption costs the
    file in flight. Not the aligner -- `align_data.read_window` reads the
    window each row names, never the file it sits in.
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


def test_two_clips_too_short_to_round_apart_still_get_different_starts(tmp_path):
    """`start` is a key, not a display. align_data resumes on (path, start) --
    `_entry_key` and `_resume_done` both -- so two clips that land on the same
    pair are one utterance to a resumed alignment, and the second is dropped
    from it without a word. Rounded to milliseconds, any two clips whose
    combined length is under half of one collide; three samples is enough.
    """
    from training.scripts.prepare_moespeech import concatenate

    seconds = 3 / 44100
    utts = [
        {
            "id": f"u{i}",
            "wav": _wav(tmp_path / f"u{i}.wav", seconds, 220.0),
            "duration": seconds,
            "transcript": "あ",
            "speaker": "spk",
        }
        for i in range(3)
    ]

    entries = concatenate(utts, tmp_path / "joined.wav", target_sec=60.0)

    starts = [e["start"] for e in entries]
    assert len(set(starts)) == len(starts), starts


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


def test_a_clip_at_another_sample_rate_is_never_joined_in(tmp_path):
    """Joined anyway it reports nothing, so it is left out -- and only it.

    `np.concatenate` does not care what rate the samples were taken at, and the
    file is written at whichever rate the first clip happened to have. A 22.05
    kHz clip laid into a 44.1 kHz file therefore plays at twice its speed and
    half its length -- so its window holds audio that does not say what its
    transcript says, and every offset after it in the file is out by the
    difference. The manifest still parses, every offset is still inside a real
    file, and the only symptom is a model trained on speech that does not match
    its text. Nothing downstream can detect it, so it cannot be joined in.

    Dropping it is the whole of what is required, though, and raising costs far
    more than it buys: this runs over 400,000 clips on a preemptible instance,
    the selection it reads is deterministic, so a raise here ends every re-run
    at the same clip forever with nothing to do about it but hand-edit
    `utterances.jsonl`. The clip simply does not reach `entries`, which is
    already what the manifest should say about it, and the clip after it takes
    the offset it would have had.
    """
    from training.scripts.prepare_moespeech import concatenate

    utts = [
        {
            "id": "u0",
            "wav": _wav(tmp_path / "u0.wav", 1.0, 220.0, sr=44100),
            "duration": 1.0,
            "transcript": "あ",
            "speaker": "spk",
        },
        {
            "id": "u1",
            "wav": _wav(tmp_path / "u1.wav", 1.0, 220.0, sr=22050),
            "duration": 1.0,
            "transcript": "い",
            "speaker": "spk",
        },
        {
            "id": "u2",
            "wav": _wav(tmp_path / "u2.wav", 1.0, 880.0, sr=44100),
            "duration": 1.0,
            "transcript": "う",
            "speaker": "spk",
        },
    ]
    entries = concatenate(utts, tmp_path / "joined.wav", target_sec=60.0)

    assert [e["id"] for e in entries] == ["u0", "u2"], entries
    # No hole where it was: the offsets are measured from the samples actually
    # laid down, so the clip after it starts where it starts.
    assert [e["start"] for e in entries] == [0.0, 1.0], entries
    assert _dominant_hz(tmp_path / "joined.wav", 1.0, 1.0) == pytest.approx(880.0, abs=2)


def test_a_clip_with_another_channel_count_is_never_joined_in(tmp_path):
    """The rate is guarded and the channel count was not, and this one is worse.

    A clip at the wrong rate joins silently and mislabels its own window. A
    stereo clip among mono ones does not join at all: `np.concatenate` refuses
    arrays that disagree on any axis but the one it joins, and it runs inside
    `_write_joined`, outside the try/except around `sphn.read` -- so the
    ValueError escapes `concatenate` entirely and ends the run.

    That contradicts this function's own contract, which is that a clip that is
    missing, truncated or at another rate "is corpus data and costs only that
    clip". The reason is the same one: 400,000 clips on a preemptible instance,
    read back off a deterministic `utterances.jsonl`, so a raise here ends every
    re-run at the same clip, with nothing to do about it but hand-edit that
    file. And it is not the first clip that decides it -- the mono clips after
    the stereo one are perfectly joinable, and would be lost with it.
    """
    from training.scripts.prepare_moespeech import concatenate

    utts = [
        {
            "id": "u0",
            "wav": _wav(tmp_path / "u0.wav", 1.0, 220.0),
            "duration": 1.0,
            "transcript": "あ",
            "speaker": "spk",
        },
        {
            "id": "u1",
            "wav": _stereo_wav(tmp_path / "u1.wav", 1.0, 440.0),
            "duration": 1.0,
            "transcript": "い",
            "speaker": "spk",
        },
        {
            "id": "u2",
            "wav": _wav(tmp_path / "u2.wav", 1.0, 880.0),
            "duration": 1.0,
            "transcript": "う",
            "speaker": "spk",
        },
    ]

    entries = concatenate(utts, tmp_path / "joined.wav", target_sec=60.0)

    assert [e["id"] for e in entries] == ["u0", "u2"], entries
    assert [e["start"] for e in entries] == [0.0, 1.0], entries
    assert _dominant_hz(tmp_path / "joined.wav", 1.0, 1.0) == pytest.approx(880.0, abs=2)


def test_a_clip_whose_audio_cannot_be_read_costs_only_that_clip(tmp_path):
    """The wav is the first thing here that opens the audio at all.

    `wav` is a path read_annotation derived from the json's name -- nothing has
    checked that it exists, or that what is under it is whole -- and this stage
    runs 400,000 clips deep into a pass that took tens of minutes to reach.
    Raising would end the run at the first broken clip, and end it again at the
    same clip on every re-run: the selection is deterministic and read back off
    `utterances.jsonl`, so there is no way past it short of editing that file
    by hand. Broken wavs are on the plan's own list of what the first real run
    will find.

    Both shapes are here because they arrive differently: a clip whose file was
    never written at all, and one a kill left half-written. sphn refuses both.
    """
    from training.scripts.prepare_moespeech import concatenate

    (tmp_path / "truncated.wav").write_bytes(b"RIFF")  # what a kill leaves behind
    clips = _clips(tmp_path, 1.0, [220.0, 880.0])
    broken = [
        {"id": "gone", "wav": tmp_path / "nothing.wav"},
        {"id": "half", "wav": tmp_path / "truncated.wav"},
    ]
    unreadable = [{"duration": 1.0, "transcript": "あ", "speaker": "spk", **b} for b in broken]
    utts = [clips[0], *unreadable, clips[1]]

    entries = concatenate(utts, tmp_path / "joined.wav", target_sec=60.0)

    assert [e["id"] for e in entries] == ["u0", "u1"], entries
    assert [e["start"] for e in entries] == [0.0, 1.0], entries
    assert _dominant_hz(tmp_path / "joined.wav", 1.0, 1.0) == pytest.approx(880.0, abs=2)


def test_a_clip_with_no_audio_in_it_costs_only_that_clip(tmp_path):
    """A valid header over zero frames clears every guard and then breaks them.

    sphn reads it as shape (1, 0): the read succeeds, its sample rate matches,
    its channel count matches, so nothing above stops it -- and then it lays
    down no samples at all. `cursor` does not advance, so the clip after it is
    handed the same (path, start) as this one, which is the pair align_data's
    `_entry_key` and `_resume_done` use to decide what a resumed alignment has
    already done: the two read as one utterance and the second is dropped
    without a word. The entry it writes for itself is a zero-length window
    besides, which `read_window` would go on to ask sphn for.

    A clip with no audio is corpus data like a truncated one, so it is skipped
    the same way and costs the same one clip -- and the clip after it takes the
    offset it would have had.
    """
    from training.scripts.prepare_moespeech import concatenate

    empty = {
        "id": "silent",
        "wav": _wav(tmp_path / "silent.wav", 0.0, 220.0),
        "duration": 0.0,
        "transcript": "あ",
        "speaker": "spk",
    }
    clips = _clips(tmp_path, 1.0, [220.0, 880.0])

    entries = concatenate([clips[0], empty, clips[1]], tmp_path / "joined.wav", target_sec=60.0)

    assert [e["id"] for e in entries] == ["u0", "u1"], entries
    assert [e["start"] for e in entries] == [0.0, 1.0], entries
    assert all(e["duration"] > 0 for e in entries), entries
    assert _dominant_hz(tmp_path / "joined.wav", 1.0, 1.0) == pytest.approx(880.0, abs=2)


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


def test_the_aligner_reads_the_window_each_entry_names(tmp_path):
    """The handoff this stage exists for: align_data reads back what it writes.

    Every entry here names a window inside a file that holds many of them, and
    the first entry of every pseudo-recording starts at 0.0 -- once per file,
    not once per speaker, so at --target-sec 120 over 5.8-second clips it is
    about one entry in twenty. A reader that treats `start == 0.0` as "this row
    is the whole file" hands the aligner two minutes of somebody else's
    utterances to fit a few seconds of transcript against: the spans come back
    spread over the whole file, the loader finds no cut inside the utterance
    and quietly falls back to taking the voice prompt from the audio it is
    predicting -- the exact failure --segmenter japanese is here to prevent --
    and a wav2vec2 forward over that many frames is where an out-of-memory kill
    takes the whole pass down rather than one utterance.

    Each clip is a distinct tone, so what came back is identified rather than
    merely counted: the window read for the first entry has to be its own two
    seconds at 220 Hz, not the six seconds of all three.
    """
    import numpy as np

    from training.scripts.align_data import read_window
    from training.scripts.prepare_moespeech import concatenate

    tones = [220.0, 440.0, 880.0]
    rows = concatenate(_clips(tmp_path, 2.0, tones), tmp_path / "joined.wav", target_sec=120.0)
    assert [r["start"] for r in rows] == [0.0, 2.0, 4.0], rows

    for row, hz in zip(rows, tones):
        wav, sr = read_window(row)
        mono = wav.mean(axis=0)
        assert len(mono) == pytest.approx(2.0 * sr, abs=2), row
        freqs = np.fft.rfftfreq(len(mono), d=1 / sr)
        assert freqs[np.argmax(np.abs(np.fft.rfft(mono)))] == pytest.approx(hz, abs=2), row


def test_no_speaker_appears_in_both_splits(tmp_path):
    """A speaker in both makes the valid loss optimistic, and the number that
    is supposed to say 'stop training' stops meaning anything."""
    from training.scripts.prepare_moespeech import split_by_speaker

    entries = [{"speaker": f"s{i % 5}", "duration": 600.0} for i in range(50)]
    train, valid = split_by_speaker(entries, valid_hours=1.0)
    # An empty valid set satisfies the disjointness below without holding out
    # anything at all, so say first that something was held out.
    assert valid, "nothing was held out; the disjointness below is vacuous"
    assert train
    assert {e["speaker"] for e in train} & {e["speaker"] for e in valid} == set()


def test_valid_speakers_have_more_than_one_utterance(tmp_path):
    """The eval protocol clones a voice from one utterance and synthesizes
    another, so a speaker with a single entry cannot be scored at all."""
    from training.scripts.prepare_moespeech import split_by_speaker

    entries = [{"speaker": f"s{i % 5}", "duration": 600.0} for i in range(50)]
    _, valid = split_by_speaker(entries, valid_hours=1.0)
    from collections import Counter

    counts = Counter(e["speaker"] for e in valid)
    # `all` over an empty Counter is True: a split that held nobody out would
    # pass this without ever meeting the condition it is about.
    assert counts, "nothing was held out; the condition below is vacuous"
    assert all(c > 1 for c in counts.values()), counts


def test_a_speaker_with_one_utterance_is_kept_but_never_held_out(tmp_path):
    """The corpus's tail is speakers with a handful of lines, and the test above
    is satisfied by a corpus that has none of them.

    Held out, such a speaker cannot be scored: its one utterance is spent on the
    voice prompt and there is nothing left to synthesize. Dropped, it costs
    training data for nothing -- one utterance is a perfectly good training
    example, it is only useless as an evaluation one. So it trains, whatever the
    held-out set still needs: here the target is far more than the corpus holds,
    so an implementation that merely stops early once the hours are reached
    takes the singletons too.
    """
    from training.scripts.prepare_moespeech import split_by_speaker

    alone = [{"speaker": f"one{i}", "duration": 1800.0} for i in range(10)]
    paired = [{"speaker": f"two{i}", "duration": 1800.0} for i in range(4) for _ in range(2)]
    train, valid = split_by_speaker(alone + paired, valid_hours=100.0)
    assert {e["speaker"] for e in valid} == {f"two{i}" for i in range(4)}
    assert {e["speaker"] for e in train} == {f"one{i}" for i in range(10)}


def test_the_split_is_deterministic(tmp_path):
    from training.scripts.prepare_moespeech import split_by_speaker

    entries = [{"speaker": f"s{i % 9}", "duration": 300.0} for i in range(90)]
    a, _ = split_by_speaker(entries, valid_hours=1.0)
    b, _ = split_by_speaker(entries, valid_hours=1.0)
    assert [e["speaker"] for e in a] == [e["speaker"] for e in b]


def test_the_split_survives_the_entries_arriving_in_another_order(tmp_path):
    """The same list twice in one process is satisfied by any pure function --
    including one that holds out whichever speakers it happens to meet first.

    That is the one thing that does move between runs here. A re-run after a
    kill rebuilds the entries from a fresh `rglob("*.json")` over a re-extracted
    tree, and nothing pins the order that scan returns. If the split follows it,
    a speaker the previous run validated on lands in this run's training set,
    and the valid loss is quietly optimistic for the rest of the run.

    c, d and e are the same length on purpose and the target falls inside them:
    the tie-break has to land the same way on both orderings too, which sorting
    on length alone -- stable, so ordered by whoever was seen first -- does not.

    What the shuffle has to move, then, is the order the speakers are first met
    in. That is the order `by_speaker` ends up holding them in, and so the only
    order a dict-order-dependent implementation could be following. Asserting
    that the entry list moved is weaker: two entries per speaker means a
    permutation that merely interleaves the duplicates changes the entry list
    while leaving every speaker first met exactly where it was, and such a
    fixture proves nothing at all while the guard still passes it.
    """
    from training.scripts.prepare_moespeech import split_by_speaker

    def firsts(rows):
        return list(dict.fromkeys(e["speaker"] for e in rows))

    lengths = {"a": 1500.0, "b": 900.0, "c": 600.0, "d": 600.0, "e": 600.0, "f": 2100.0}
    entries = [{"speaker": s, "duration": d} for s, d in lengths.items() for _ in range(2)]
    shuffled = list(entries)
    random.Random(1).shuffle(shuffled)
    assert firsts(shuffled) != firsts(entries), firsts(shuffled)

    held_out = {e["speaker"] for e in split_by_speaker(entries, valid_hours=0.5)[1]}
    assert held_out == {"c", "d"}, held_out
    assert {e["speaker"] for e in split_by_speaker(shuffled, valid_hours=0.5)[1]} == held_out


def test_the_shortest_speakers_are_the_ones_held_out(tmp_path):
    """Every held-out hour is an hour not trained on, so it should buy as many
    distinct voices as it can: smallest first.

    Held out largest-first, the same hour is one voice where it could have been
    two or three, and the overshoot past the target is the size of the largest
    speaker rather than a small one. On MoeSpeech, where speaker lengths span
    orders of magnitude, that is the difference between a valid set of a few
    hundred utterances and one that eats a tenth of the corpus.

    The names run counter to the lengths so that ordering by name -- which every
    equal-length fixture elsewhere in this file also satisfies -- picks a
    different set and is caught here.
    """
    from training.scripts.prepare_moespeech import split_by_speaker

    lengths = {"a": 3000.0, "b": 2400.0, "c": 1800.0, "d": 1200.0, "e": 600.0}
    entries = [{"speaker": s, "duration": d} for s, d in lengths.items() for _ in range(2)]
    _, valid = split_by_speaker(entries, valid_hours=0.5)
    assert {e["speaker"] for e in valid} == {"d", "e"}, valid


def test_a_corpus_that_cannot_fill_the_valid_set_says_so(tmp_path, caplog):
    """A valid set smaller than asked for is a legitimate output -- a corpus
    whose speakers mostly have one utterance each has little that can be
    evaluated on -- and it satisfies every guarantee above vacuously: no speaker
    is in both splits, and every valid speaker has more than one utterance,
    because there is hardly one.

    So this warning is the only thing standing between the operator and a run
    that trains for days with almost nothing to validate on. It has to fire, and
    it has to name how many speakers were unusable, or the near-empty
    valid.jsonl is discovered when the training loop divides by zero instead.

    The corpus is deliberately lopsided -- four speakers with one utterance and
    one with two -- so that the three counts the message could be naming are all
    different numbers: four singletons, five speakers, six entries. A fixture of
    five singleton speakers makes all three of them 5, and then "5 of 5" is
    satisfied by a message counting entries, or speakers twice, or anything
    else; the numbers have to disagree for the assertion to mean the message
    says what it claims to say.
    """
    from training.scripts.prepare_moespeech import split_by_speaker

    entries = [{"speaker": f"s{i}", "duration": 600.0} for i in range(4)]
    entries += [{"speaker": "pair", "duration": 600.0} for _ in range(2)]
    with caplog.at_level(logging.WARNING, logger="prepare_moespeech"):
        train, valid = split_by_speaker(entries, valid_hours=1.0)
    assert {e["speaker"] for e in valid} == {"pair"}
    assert len(valid) == 2
    assert len(train) == 4
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warned) == 1, warned
    assert "0.33h of the 1.00h" in warned[0], warned[0]
    assert "4 of 5 speakers" in warned[0], warned[0]

    # And it stays quiet when the hours asked for were there: a warning on every
    # run is a warning nobody reads by the time it means something.
    caplog.clear()
    full = [{"speaker": f"p{i}", "duration": 3600.0} for i in range(3) for _ in range(2)]
    with caplog.at_level(logging.WARNING, logger="prepare_moespeech"):
        split_by_speaker(full, valid_hours=1.0)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_the_split_keeps_every_entry(tmp_path):
    """Whatever valid does not take, train trains on. A speaker that cannot be
    held out is not thereby unusable, and neither is one the target was already
    met before reaching."""
    from training.scripts.prepare_moespeech import split_by_speaker

    entries = [{"speaker": f"s{i % 7}", "duration": 400.0, "id": f"u{i:03d}"} for i in range(70)]
    entries.append({"speaker": "lonely", "duration": 400.0, "id": "u070"})
    train, valid = split_by_speaker(entries, valid_hours=1.0)
    assert sorted(e["id"] for e in train + valid) == sorted(e["id"] for e in entries)


def test_the_held_out_size_is_asked_for_in_hours(tmp_path):
    """`duration` is seconds and `valid_hours` is hours. Three speakers of
    twenty minutes make the hour; one of them does not, however the units are
    confused on the way."""
    from training.scripts.prepare_moespeech import split_by_speaker

    entries = [{"speaker": f"s{i}", "duration": 600.0} for i in range(9) for _ in range(2)]
    _, valid = split_by_speaker(entries, valid_hours=1.0)
    assert len({e["speaker"] for e in valid}) == 3, valid


def test_the_manifest_is_utf8_and_reloadable(tmp_path):
    """Windows defaults to cp932; a manifest written through it is unreadable
    and the failure appears far from here."""
    from training.scripts.prepare_moespeech import write_manifest

    path = tmp_path / "m.jsonl"
    write_manifest(
        [{"path": "a.wav", "duration": 1.0, "transcript": "こんにちは", "start": 0.0}], path
    )
    with open(path, encoding="utf-8") as f:
        assert json.loads(f.readline())["transcript"] == "こんにちは"


def test_the_manifest_carries_what_the_loader_requires(tmp_path):
    """training/dataloader.py's Entry: path, duration, transcript, start."""
    from training.scripts.prepare_moespeech import write_manifest

    path = tmp_path / "m.jsonl"
    write_manifest([{"path": "a.wav", "duration": 1.0, "transcript": "あ", "start": 2.5}], path)
    with open(path, encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert {"path", "duration", "transcript", "start"} <= set(row)


def test_the_manifest_says_how_many_utterances_it_holds(tmp_path):
    """One JSON object per line, and the count the caller logs comes from the
    lines that were written rather than from the list it handed over."""
    from training.scripts.prepare_moespeech import write_manifest

    path = tmp_path / "m.jsonl"
    rows = [{"path": "a.wav", "duration": 1.0, "transcript": "あ", "start": 0.0}] * 3
    assert write_manifest(rows, path) == 3
    with open(path, encoding="utf-8") as f:
        assert len(f.readlines()) == 3


def test_the_manifest_holds_the_japanese_unescaped(tmp_path):
    """Escaped into ASCII every line still parses, so nothing downstream would
    notice -- but a manifest nobody can read with `head` or grep for a speaker's
    line is a manifest nobody checks, and it is also the version where the
    explicit utf-8 above stops mattering until the day it silently does."""
    from training.scripts.prepare_moespeech import write_manifest

    path = tmp_path / "m.jsonl"
    write_manifest(
        [{"path": "a.wav", "duration": 1.0, "transcript": "こんにちは", "start": 0.0}], path
    )
    assert "こんにちは" in path.read_text(encoding="utf-8")


def test_a_kill_partway_through_the_manifest_leaves_no_manifest(tmp_path, monkeypatch):
    """A half-written manifest is still valid JSONL -- it is merely short.

    Nothing downstream can tell it from a finished one: every line parses, every
    path exists, and a re-run that skips the stage because `train.jsonl` is
    there trains on however much of the corpus the kill let through. So the
    lines land beside the name and are renamed in only once they are all there.
    """
    import training.scripts.prepare_moespeech as m
    from training.scripts.prepare_moespeech import write_manifest

    real, seen = json.dumps, []

    def killed(obj, **kw):
        seen.append(obj)
        if len(seen) > 1:
            raise KeyboardInterrupt
        return real(obj, **kw)

    monkeypatch.setattr(m.json, "dumps", killed)
    rows = [{"path": "a.wav", "duration": 1.0, "transcript": "あ", "start": 0.0}] * 2
    with pytest.raises(KeyboardInterrupt):
        write_manifest(rows, tmp_path / "m.jsonl")
    assert not (tmp_path / "m.jsonl").exists()


# The eight stages, wired together by main() and run end to end.

KANA_ALIGNER = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"

# One speaker's clips, and the two cutoffs every run below is given. Each
# cutoff excludes a different clip, and neither excludes the other's: clip3's
# two ASR transcripts disagree by 0.4 CER, clip4 scores 2.0. So the selection
# --max-cer 0.3 --min-mos 3.0 produces is clip0-2, and it is a set no other
# pair of numbers reaching that filter produces -- 0.0/0.0 keeps clip4, the two
# swapped keeps everything, and either shows up in the manifests below.
#
# The three that survive are two seconds each, so with --target-sec 5.0 a
# speaker's audio does not fit in one file and the manifest has to name two.
CUTOFFS = {"max_cer": 0.3, "min_mos": 3.0}
CLIPS = (
    {"hz": 220.0, "parakeet": "こんにちは", "mos": 4.0},
    {"hz": 440.0, "parakeet": "こんにちは", "mos": 4.0},
    {"hz": 880.0, "parakeet": "こんにちは", "mos": 4.0},
    {"hz": 330.0, "parakeet": "こんばんは", "mos": 4.0},  # 0.4 CER against the reference
    {"hz": 550.0, "parakeet": "こんにちは", "mos": 2.0},
)


def _unpack_speaker(dest_root, name):
    """One speaker's clips under `dest_root`, as stage 3 would leave them."""
    out = dest_root / name
    out.mkdir(parents=True, exist_ok=True)
    for i, clip in enumerate(CLIPS):
        _wav(out / f"clip{i}.wav", 2.0, clip["hz"])
        (out / f"clip{i}.json").write_text(
            json.dumps(
                {
                    "anime_whisper_transcription": "こんにちは",
                    "parakeet_jp_transcription": clip["parakeet"],
                    "duration": 2.0,
                    "speechMOS": clip["mos"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    # Completion is recorded beside the directory, and it is what a later run
    # reads to tell a finished character from an interrupted one.
    (dest_root / f"{name}{EXTRACT_MARKER}").write_text("", encoding="utf-8")
    return out


def _jsonl(path):
    """The rows of a manifest this script wrote."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _age(root, seconds=60.0):
    """Push everything under `root` back in time by `seconds`.

    Whether an artifact was built before or after its inputs is the question
    the guards in main() ask, and two runs of this fixture are milliseconds
    apart while the clock a file's timestamp comes from is coarser than that
    (about 15 ms on Windows). A tie reads as up to date -- deliberately, or
    nothing would ever be skipped -- so a test that means "this was built by an
    earlier run" has to say so rather than race the clock.
    """
    past = time.time() - seconds
    for path in sorted(root.rglob("*")):
        os.utime(path, (past, past))


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


def _pipeline(tmp_path, monkeypatch):
    """`main` with everything off this machine replaced by a recording fake.

    Only the stages that would reach the network are faked -- the two hub
    fetches and the aligner. The probe, the selection, the concatenation, the
    split and the manifests run for real over the five clips of CLIPS per
    speaker, so a recording proves that `main` wired the real functions
    together in the real order rather than that it called mocks in one, and the
    manifests it leaves behind can be read for what the wiring decided.

    The fakes for `download_characters`, `extract_character` and `align` each
    reproduce the one behaviour of their real counterpart this script leans on:
    doing nothing when their output is already there. Those three stages decide
    for themselves what to skip -- their own tests prove they do -- so a fake
    that recorded the call instead of the work would report a re-run as redoing
    everything when it redid nothing.
    """
    from training.scripts import prepare_data
    from training.scripts import prepare_moespeech as m

    calls = defaultdict(list)
    cache = tmp_path / "hub"
    cache.mkdir()
    out_dir = tmp_path / "ja"
    # A list, and read at call time: a test that widens the selection appends
    # to it, which is what the dataset growing a speaker looks like from here.
    speakers = ["aoi", "kaede"]

    def fake_fetch(repo_id, filename, **kw):
        calls["info.csv"].append((repo_id, filename, kw.get("repo_type")))
        path = cache / filename
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "num_files", "total_duration_min", "f0_mean"])
            w.writerows([(name, len(CLIPS), 0.1, 300.0) for name in speakers])
        return str(path)

    # Two recordings apiece, and they say different things. `*_called` is that
    # main reached the stage at all, `download`/`extract` that the stage found
    # work to do. Only the pair pins the contract: main calls both of these on
    # every run and they decide for themselves what to skip, so a count of the
    # work alone would be satisfied by a main() that stopped calling them.
    def fake_download(names, dest, repo=None):
        calls["download_called"].append((tuple(names), repo))
        dest.mkdir(parents=True, exist_ok=True)
        paths = []
        for name in names:
            path = dest / f"{name}.zip"
            if not path.exists():
                calls["download"].append((name, repo))
                path.write_bytes(b"PK\x03\x04")
            paths.append(path)
        return paths

    def fake_extract(zip_path, dest_root):
        calls["extract_called"].append(zip_path.name)
        out = dest_root / zip_path.stem
        marker = dest_root / f"{zip_path.stem}{m.EXTRACT_MARKER}"
        if marker.exists() and out.is_dir():
            return out
        calls["extract"].append(zip_path.name)
        return _unpack_speaker(dest_root, zip_path.stem)

    # Bound against the real align()'s signature, so a call main() could not
    # actually make -- a misspelled keyword, an argument too many -- fails here
    # rather than being recorded as if it had worked.
    signature = inspect.signature(prepare_data.align)

    def fake_align(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        out = Path(bound.arguments["out"])
        if out.exists():
            return
        calls["align"].append(dict(bound.arguments))
        out.write_text("", encoding="utf-8")

    # Whether the japanese dependency group is installed is a property of the
    # machine, exactly like the hub and the aligner, and CI syncs without it.
    # Recorded rather than removed: the call and what had happened by the time
    # it was made are the contract, and its own tests below run it for real.
    def fake_segmenter_check(will_align):
        calls["segmenter_check"].append((will_align, len(calls["download_called"])))

    # Kept reachable behind the fake: the question main() computes is only worth
    # anything if the real check answers it, and one test below puts this back
    # to run exactly that -- a finished tree, re-run on a machine without MeCab.
    real_segmenter_check = m.require_japanese_segmenter
    # The late half of the same check, faked for the same reason -- and as a
    # no-op rather than a recording, so the stage counts above stay counts of
    # the stages. It fires only where stage 0 could not see the work coming,
    # which is one test below and no other.
    real_late_check = m.require_segmenter_to_align

    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    monkeypatch.setattr(m, "download_characters", fake_download)
    monkeypatch.setattr(m, "extract_character", fake_extract)
    monkeypatch.setattr(m, "align", fake_align)
    monkeypatch.setattr(m, "require_japanese_segmenter", fake_segmenter_check)
    monkeypatch.setattr(m, "require_segmenter_to_align", lambda aligned: None)
    for name in ("probe_utterances", "select_utterances", "concatenate", "split_by_speaker"):
        monkeypatch.setattr(m, name, _recording(calls, name, getattr(m, name)))

    def run(**overrides):
        options = {
            "out": str(out_dir),
            "hours": 0.01,  # 0.6 minutes, so both 0.1-minute fixture speakers fit
            "valid_hours": 0.001,  # 3.6 seconds, so one of the two 6s speakers is held out
            "target_sec": 5.0,  # the three kept clips are 6s, so each speaker needs two files
            "repo": "fake/repo",
        }
        options.update(overrides)
        return m.main(**options)

    return SimpleNamespace(
        module=m,
        calls=calls,
        out=out_dir,
        run=run,
        speakers=speakers,
        real_segmenter_check=real_segmenter_check,
        real_late_check=real_late_check,
    )


def test_the_pipeline_skips_stages_whose_output_exists(tmp_path, monkeypatch):
    """The property the whole script is designed around: re-running after a
    preemption must not redo finished work.

    The instance this runs on is reclaimed without warning, so the command is
    typed again -- often. Every stage is asserted to have run once and then
    not again, and the tree it left is asserted byte-identical afterwards: a
    stage that redid its work would show up as a second recording, and one
    that re-wrote its output from different inputs as different bytes.
    """
    p = _pipeline(tmp_path, monkeypatch)

    p.run(**CUTOFFS)

    # Every stage really ran the first time -- otherwise "it did not run again"
    # would be satisfied by a main() that does nothing at all.
    done = {stage: len(c) for stage, c in p.calls.items()}
    assert done == {
        "segmenter_check": 1,
        "info.csv": 1,
        "download_called": 1,
        "download": 2,
        "extract_called": 2,
        "extract": 2,
        "probe_utterances": 1,
        "select_utterances": 1,
        "concatenate": 2,
        "split_by_speaker": 1,
        "align": 2,
    }, done
    # And it operated on what it was told to. The repo is an option, the zips
    # are the selected speakers', and each stage is handed the previous one's
    # output; a stage that ran the right number of times over the wrong file
    # is the failure a count cannot see.
    assert p.calls["info.csv"] == [("fake/repo", "info.csv", "dataset")]
    assert sorted(p.calls["download"]) == [("aoi", "fake/repo"), ("kaede", "fake/repo")]
    assert sorted(p.calls["extract"]) == ["aoi.zip", "kaede.zip"]
    before = _tree(p.out)
    assert (p.out / "train_aligned.jsonl").exists()

    p.run(**CUTOFFS)

    # Both stages were entered again and both found their work already done:
    # skipping is theirs to decide, and main re-offering them the work is what
    # makes a re-run after a kill pick up the speakers that never arrived.
    assert {stage: len(c) for stage, c in p.calls.items()} == done | {
        "download_called": 2,
        "extract_called": 4,
        # Asked once per run, before anything else: it costs an import and the
        # answer can change between runs, which is the point of asking again.
        "segmenter_check": 2,
    }
    assert _tree(p.out) == before


def test_alignment_uses_the_japanese_segmenter_and_a_kana_model(tmp_path, monkeypatch):
    """align_data refuses a model without hiragana in its vocabulary, but only
    at run time on the instance -- catching it here costs nothing.

    The segmenter matters as much and refuses nothing: `whitespace` over a
    language written without spaces returns one word per utterance, so the
    aligner emits a single span, the loader finds no cut point, and the voice
    prompt silently comes from the utterance being predicted. Nothing raises;
    the model simply learns that the prompt does not decide the voice.
    """
    p = _pipeline(tmp_path, monkeypatch)

    p.run(**CUTOFFS, align_shards=3)

    assert len(p.calls["align"]) == 2, p.calls["align"]
    for call in p.calls["align"]:
        assert call["segmenter"] == "japanese", call
        assert call["model"] == KANA_ALIGNER, call
    # Both manifests are aligned, not just the training one: the loader reads
    # `words` on either side and an unaligned valid set is scored differently
    # from the set it is compared against. Each is aligned from itself -- an
    # aligner pointed at utterances.jsonl, which carries `wav` and no `start`,
    # or at the training manifest twice, produces a file of the right name
    # holding alignments of the wrong audio.
    assert [(Path(c["manifest"]).name, Path(c["out"]).name) for c in p.calls["align"]] == [
        ("train.jsonl", "train_aligned.jsonl"),
        ("valid.jsonl", "valid_aligned.jsonl"),
    ], p.calls["align"]
    # --align-shards is a count of GPUs and belongs to the long pass; the valid
    # manifest is one speaker's worth and is aligned in one process.
    assert [c["shards"] for c in p.calls["align"]] == [3, 1], p.calls["align"]


def test_a_missing_segmenter_stops_the_run_before_it_fetches_anything(monkeypatch, caplog):
    """The one dependency a plain `uv sync` does not install, checked first.

    fugashi and unidic-lite are the `japanese` dependency group, and
    align_data imports fugashi lazily inside the segmenter -- so an unprepared
    instance gets through the download, the extraction, the probe, the
    selection, the concatenation and both manifests before anything asks for
    it. That is 30 GB and hours of a paid instance, and what finally surfaces
    is a CalledProcessError from the aligner's subprocess, which names a return
    code and not a package.

    Simulated by making the import fail rather than by uninstalling anything,
    so this runs the same way on a machine that has the group and on CI, which
    syncs without it.
    """
    import sys

    from training.scripts.prepare_moespeech import require_japanese_segmenter

    monkeypatch.setitem(sys.modules, "fugashi", None)
    caplog.set_level(logging.INFO)

    with pytest.raises(typer.Exit):
        require_japanese_segmenter(will_align=True)

    said = "\n".join(record.message for record in caplog.records)
    assert "uv sync --group japanese" in said, said


def test_a_run_that_will_not_align_is_told_rather_than_stopped(monkeypatch, caplog):
    """A run with no cutoffs stops at the probe, so it does not need the group.

    Refusing it would refuse the run the operator has to make first -- the
    probe is what the cutoffs are read off. Saying so anyway is the point: that
    run is the long one, and the group can be installed while it downloads.
    """
    import sys

    from training.scripts.prepare_moespeech import require_japanese_segmenter

    monkeypatch.setitem(sys.modules, "fugashi", None)
    caplog.set_level(logging.INFO)

    require_japanese_segmenter(will_align=False)  # returns rather than raising

    said = "\n".join(record.message for record in caplog.records)
    assert "uv sync --group japanese" in said, said


def test_the_installed_segmenter_satisfies_the_check():
    """The other direction: an instance that is ready must not be turned away.

    Without this the check is only known to fail, and a check that always
    fails would refuse every correctly prepared run -- the same wasted day it
    exists to prevent, in the other direction.
    """
    pytest.importorskip("fugashi", reason="uv sync --group japanese")
    pytest.importorskip("unidic_lite", reason="uv sync --group japanese")

    from training.scripts.prepare_moespeech import require_japanese_segmenter

    require_japanese_segmenter(will_align=True)


def test_the_segmenter_is_checked_before_the_first_stage(tmp_path, monkeypatch):
    """Early is the whole value: after stage 1 the check has already cost hours.

    Pinned as "nothing had been fetched when it was asked", which is what an
    operator gets back for a missing package -- an error in the first second,
    with the disk untouched -- rather than as a call count, which a check made
    at the end of the run would satisfy just as well.
    """
    p = _pipeline(tmp_path, monkeypatch)

    with pytest.raises(typer.Exit):  # no cutoffs: stops at the probe
        p.run()
    assert p.calls["segmenter_check"] == [(False, 0)], p.calls["segmenter_check"]
    p.calls.clear()  # the recording below is the second run's, not both runs'

    p.run(**CUTOFFS)

    assert p.calls["segmenter_check"] == [(True, 0)], p.calls["segmenter_check"]


def test_a_re_run_with_nothing_left_to_align_does_not_need_the_segmenter(
    tmp_path, monkeypatch, caplog
):
    """The check refuses a run that cannot finish -- not one with no work left.

    Recovering from a preemption is retyping the same command, so it is typed
    over trees in every state, a finished one included. Over that tree align()
    finds both outputs already there and skips, the segmenter is never built,
    and the run is a few file-existence checks long. Gated on "were both cutoffs
    given" instead of "will the aligner run", that same command starts exiting 1
    on a machine without the japanese group -- for work it was never going to do,
    on a tree that is already complete.
    """
    import sys

    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    before = _tree(p.out)
    assert (p.out / "train_aligned.jsonl").exists() and (p.out / "valid_aligned.jsonl").exists()

    # The real check this time, on a machine that does not have the group --
    # the same way its own tests simulate one, since CI syncs without it.
    monkeypatch.setattr(p.module, "require_japanese_segmenter", p.real_segmenter_check)
    monkeypatch.setitem(sys.modules, "fugashi", None)
    caplog.set_level(logging.INFO)

    p.run(**CUTOFFS)  # returns rather than raising typer.Exit

    assert _tree(p.out) == before
    # Told, though: the group is still what the next run needs if anything
    # above the alignment ever moves.
    said = "\n".join(record.message for record in caplog.records)
    assert "uv sync --group japanese" in said, said


def test_a_missing_alignment_still_refuses_a_run_without_the_segmenter(tmp_path, monkeypatch):
    """The other half of the same question, over the tree a kill actually leaves.

    Aligning is two calls, so a preemption between them leaves the training
    alignment finished and the valid one absent -- and that re-run does align.
    Answered by "are both there" rather than "is either there", it would be
    waved through and would fail hours later inside the aligner's subprocess.
    """
    import sys

    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    (p.out / "valid_aligned.jsonl").unlink()

    monkeypatch.setattr(p.module, "require_japanese_segmenter", p.real_segmenter_check)
    monkeypatch.setitem(sys.modules, "fugashi", None)

    with pytest.raises(typer.Exit):
        p.run(**CUTOFFS)


def test_changed_cutoffs_over_a_finished_tree_are_refused_again(tmp_path, monkeypatch):
    """Both alignments on disk is not proof that the run will not align.

    Changing the cutoffs over a finished tree is the documented way to change
    them, and it rebuilds everything: stage 5 rewrites the selection, stage 7
    both manifests, and stage 8 then discards both alignments as older than the
    manifests they claim to describe and aligns again. Asked only whether the
    two files exist, this run is waved through and dies hours later inside the
    aligner's subprocess, as a CalledProcessError naming a return code.

    So the guard over-approximates: it is quiet only when the alignments are
    there *and* the cutoffs beside them are the ones just given. The two
    mistakes do not cost the same -- a refusal the run did not need costs
    thirty seconds of `uv sync --group japanese`, and a pass it did not deserve
    costs a day of a paid instance.
    """
    import sys

    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    before = _tree(p.out)

    monkeypatch.setattr(p.module, "require_japanese_segmenter", p.real_segmenter_check)
    monkeypatch.setitem(sys.modules, "fugashi", None)

    with pytest.raises(typer.Exit):
        p.run(max_cer=0.3, min_mos=1.0)  # a different pair from CUTOFFS

    # Refused before the first stage, exactly as an unprepared run is: nothing
    # was rebuilt, so the operator installs the group and retypes the command.
    assert _tree(p.out) == before


def test_the_aligner_is_not_reached_when_only_the_late_guard_can_tell(
    tmp_path, monkeypatch, caplog
):
    """The hole the guard above still leaves, and what it costs.

    Stage 0 cannot see a hand-deleted `utterances.jsonl`: both alignments are
    on disk, the cutoffs beside them are the ones being passed, and by every
    question it can ask nothing will be rebuilt. Then the selection is redone
    because its file is gone, the offsets and both manifests follow it, and
    stage 8 discards both alignments as stale and aligns.

    Which is why the same question is asked again one line before each
    `align()`, where every manifest has been written and the answer is finally
    knowable. What was a CalledProcessError hours from now, naming a return
    code, is a stop that names the package -- and everything above the
    alignment stays on disk, so installing the group and retyping the command
    picks up from there.
    """
    import sys

    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    _age(p.out)
    (p.out / "utterances.jsonl").unlink()  # the one thing stage 0 cannot see

    monkeypatch.setattr(p.module, "require_segmenter_to_align", p.real_late_check)
    monkeypatch.setitem(sys.modules, "fugashi", None)
    caplog.set_level(logging.INFO)

    with pytest.raises(typer.Exit):
        p.run(**CUTOFFS)

    # Stage 0 really did wave it through -- otherwise this proves nothing about
    # the late guard, only that something somewhere refused the run.
    assert p.calls["segmenter_check"][-1] == (False, 1), p.calls["segmenter_check"]
    # And the aligner was never reached: the two calls are the first run's.
    assert len(p.calls["align"]) == 2, p.calls["align"]
    said = "\n".join(record.message for record in caplog.records)
    assert "uv sync --group japanese" in said, said


def test_a_speaker_label_no_character_answers_to_stops_the_run(tmp_path, monkeypatch, caplog):
    """The belt to the braces in `read_annotation`: labels are checked once.

    Every guard below stage 6 compares the speaker label and not one of them
    can check it. `concatenate` refuses a mixed list by comparing labels, so a
    single wrong label shared by two characters *is* a mixed list and passes --
    two voices joined into one recording, the loader taking one side of a cut
    as the voice prompt for the other, the split holding out a label instead of
    a character, and `audio/<speaker>.wav` written twice under one name.

    A label the scan derived correctly is one of the selected characters by
    construction, so that is what is checked, once, before a single wav is
    joined or deleted. It is the last thing standing if the layout of a real
    zip ever surprises the scan again.
    """
    p = _pipeline(tmp_path, monkeypatch)
    selecting = p.module.select_utterances

    def mislabelled(*args, **kwargs):
        # What reading the parent directory off a nested layout produces: every
        # clip of every character under one label that is not a character.
        for utterance in selecting(*args, **kwargs):
            yield {**utterance, "speaker": "wav"}

    monkeypatch.setattr(p.module, "select_utterances", mislabelled)
    caplog.set_level(logging.ERROR)

    with pytest.raises(typer.Exit):
        p.run(**CUTOFFS)

    assert not p.calls["concatenate"], "two characters were joined under one label"
    assert not (p.out / "audio").exists()
    said = "\n".join(record.message for record in caplog.records)
    assert "wav" in said, said


def test_the_run_stops_until_the_thresholds_have_been_chosen(tmp_path, monkeypatch, caplog):
    """Nothing has measured this corpus, so the two cutoffs have no default.

    The probe is what produces the numbers they are read off, so the run does
    everything up to and including it and then stops, naming the file to read
    and the two flags to pass. Filtering on a guessed threshold instead would
    produce a manifest that afterwards is indistinguishable from a measured
    one -- and the guess would be attached to 124 hours of training.
    """
    p = _pipeline(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO)

    with pytest.raises(typer.Exit):
        p.run()

    assert (p.out / "probe.json").exists()
    assert not p.calls["select_utterances"], "nothing may be filtered on a threshold nobody chose"
    assert not p.calls["align"]
    said = "\n".join(record.message for record in caplog.records)
    assert "probe.json" in said, said
    assert "--max-cer" in said and "--min-mos" in said, said


def test_supplying_the_thresholds_resumes_from_the_probe(tmp_path, monkeypatch):
    """The stop above is a stage boundary, not a failed run.

    The operator reads probe.json and types the command again with the two
    flags. Everything before the cutoffs -- 30 GB fetched, unpacked and walked
    -- is on disk and must not be paid for a second time.
    """
    p = _pipeline(tmp_path, monkeypatch)
    with pytest.raises(typer.Exit):
        p.run()
    probe = (p.out / "probe.json").read_bytes()

    p.run(**CUTOFFS)

    assert len(p.calls["probe_utterances"]) == 1, "the corpus was measured twice"
    assert len(p.calls["extract"]) == 2, "the zips were unpacked twice"
    assert not p.calls["info.csv"][1:], "info.csv was fetched twice"
    assert (p.out / "probe.json").read_bytes() == probe
    assert (p.out / "train_aligned.jsonl").exists()


def test_the_manifests_name_the_audio_the_cutoffs_selected(tmp_path, monkeypatch):
    """What the run leaves behind is read, not just counted.

    Every other test here asserts that a stage ran, or that a re-run wrote the
    same bytes twice. Neither notices what those bytes say, and every way this
    pipeline can be miswired ends in a file of the right name: the split handed
    the wrong manifest, the cutoffs arriving in the wrong order or not at all,
    a speaker dropped by a `=` where a `+=` belongs, --target-sec replaced by
    its default. All of them raise nothing. They show up here, as a manifest
    describing different audio.
    """
    p = _pipeline(tmp_path, monkeypatch)

    p.run(**CUTOFFS)

    train = _jsonl(p.out / "train.jsonl")
    valid = _jsonl(p.out / "valid.jsonl")
    # Whole speakers are held out, so no voice stands on both sides, and the
    # training set is what is left rather than empty.
    assert {e["speaker"] for e in train} == {"kaede"}, train
    assert {e["speaker"] for e in valid} == {"aoi"}, valid
    # Both cutoffs reached the filter, and as themselves: clip3's two ASR
    # transcripts disagree by 0.4 CER and clip4 scores 2.0, so --max-cer 0.3
    # and --min-mos 3.0 exclude one each and neither excludes the other's.
    assert [e["id"] for e in train] == ["clip0", "clip1", "clip2"], train
    assert [e["id"] for e in valid] == ["clip0", "clip1", "clip2"], valid
    # Six seconds of clips and --target-sec 5.0, so the third one starts a
    # second file and its offset restarts at zero inside it.
    assert [(Path(e["path"]).name, e["start"], e["duration"]) for e in train] == [
        ("kaede.wav", 0.0, 2.0),
        ("kaede.wav", 2.0, 2.0),
        ("kaede_001.wav", 0.0, 2.0),
    ], train
    assert [Path(e["path"]).name for e in valid] == ["aoi.wav", "aoi.wav", "aoi_001.wav"], valid
    assert sorted(w.name for w in (p.out / "audio").glob("*.wav")) == [
        "aoi.wav",
        "aoi_001.wav",
        "kaede.wav",
        "kaede_001.wav",
    ]
    for entry in train + valid:
        assert Path(entry["path"]).exists(), entry
        assert entry["transcript"] == "こんにちは", entry


def test_a_kill_between_the_selection_and_the_joining_resumes_from_the_file(tmp_path, monkeypatch):
    """The preemption `utterances.jsonl` exists to survive.

    That file is the walk over 400,000 annotations, and a kill just after it
    was renamed into place -- before a single speaker's clips had been joined
    -- is the case the whole stop-and-continue design is for. Proving the run
    carries on from it means leaving it and taking away everything built after
    it: with every artifact present, a stage reading the file back is
    indistinguishable from one that never read it.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    uninterrupted = _tree(p.out)

    shutil.rmtree(p.out / "entries")
    shutil.rmtree(p.out / "audio")
    for name in ("train.jsonl", "valid.jsonl", "train_aligned.jsonl", "valid_aligned.jsonl"):
        (p.out / name).unlink()

    p.run(**CUTOFFS)

    assert len(p.calls["select_utterances"]) == 1, "the corpus was walked a second time"
    assert len(p.calls["concatenate"]) == 4, "the clips the kill cost were not joined"
    # The rows read back off disk are the rows the first run held, so the tree
    # the second one finishes with is the tree it would have finished with.
    assert _tree(p.out) == uninterrupted


def test_a_kill_between_the_joining_and_the_split_resumes_from_the_entries(tmp_path, monkeypatch):
    """One stage further down, and the same property one level deeper.

    Joining is the hours of audio work. A kill after the last speaker's offsets
    were written but before the split must not redo it, and the offsets it
    reads back have to be the ones on disk -- a split over nothing at all
    writes two manifests that parse, and describe no training data.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    uninterrupted = _tree(p.out)

    for name in ("train.jsonl", "valid.jsonl", "train_aligned.jsonl", "valid_aligned.jsonl"):
        (p.out / name).unlink()

    p.run(**CUTOFFS)

    assert len(p.calls["concatenate"]) == 2, "the audio was joined a second time"
    assert len(p.calls["split_by_speaker"]) == 2, "the split the kill cost was not redone"
    assert _tree(p.out) == uninterrupted


def test_a_cutoff_changed_after_the_fact_rebuilds_what_it_decided(tmp_path, monkeypatch):
    """Deleting one artifact carries through everything built out of it.

    The way to redo a stage under new options is to delete its output, and the
    output of the selection is `utterances.jsonl`. If only that stage re-ran,
    `train.jsonl` would go on describing the selection the operator had just
    rejected -- the two files on disk saying different things about which
    utterances were kept, the run reporting success, and the training reading
    the older of the two.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    assert [e["id"] for e in _jsonl(p.out / "train.jsonl")] == ["clip0", "clip1", "clip2"]
    _age(p.out)

    (p.out / "utterances.jsonl").unlink()
    p.run(max_cer=0.3, min_mos=1.0)  # clip4 scores 2.0 and is now kept

    selected = _jsonl(p.out / "utterances.jsonl")
    train = _jsonl(p.out / "train.jsonl")
    valid = _jsonl(p.out / "valid.jsonl")
    assert [e["id"] for e in train] == ["clip0", "clip1", "clip2", "clip4"], train
    # The manifests hold the selection, all of it and nothing else.
    assert {(e["speaker"], e["id"]) for e in train + valid} == {
        (e["speaker"], e["id"]) for e in selected
    }
    assert len(p.calls["concatenate"]) == 4, "the audio still holds the old selection"
    assert len(p.calls["align"]) == 4, "the alignments still describe the old manifests"


def test_a_cutoff_that_keeps_nothing_empties_the_manifests_too(tmp_path, monkeypatch):
    """The same carry-through, in the direction where nothing survives.

    Tighten a cutoff until no utterance passes and stage 6 has no speaker to
    rebuild offsets for, so nothing under `entries/` is touched. A split that
    watched only those files would see none of its inputs move, keep naming
    clips the operator has just excluded, keep the alignment beside them, and
    report success -- while `utterances.jsonl` next to it says the selection is
    empty. Whether a speaker survives cannot be what decides that the split is
    out of date, so the selection itself is one of its inputs.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    assert [e["id"] for e in _jsonl(p.out / "train.jsonl")] == ["clip0", "clip1", "clip2"]
    _age(p.out)

    (p.out / "utterances.jsonl").unlink()
    p.run(max_cer=0.3, min_mos=5.0)  # the best clip scores 4.0, so nothing is kept

    assert _jsonl(p.out / "utterances.jsonl") == []
    assert _jsonl(p.out / "train.jsonl") == [], "the split still holds the rejected selection"
    assert _jsonl(p.out / "valid.jsonl") == [], "the split still holds the rejected selection"
    assert len(p.calls["align"]) == 4, "the alignments still describe the old manifests"


def test_a_speaker_a_wider_selection_adds_reaches_the_manifests(tmp_path, monkeypatch):
    """A larger --hours the next day arrives as a new directory to walk.

    Deleting `characters.json` is the documented way to select speakers again,
    and everything after it is measured against what that selection says. A
    speaker fetched and unpacked by the wider run has to reach the probe -- its
    hours column is what the target is expressed in -- and the manifests:
    skipping on `utterances.jsonl` existing would leave 30 GB of freshly
    fetched audio out of the training manifest, with nothing on disk saying it
    had been left out.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    assert json.loads((p.out / "probe.json").read_text(encoding="utf-8"))["count"] == 10
    _age(p.out)

    p.speakers.append("momo")
    (p.out / "characters.json").unlink()
    p.run(**CUTOFFS)

    assert sorted(p.calls["extract"]) == ["aoi.zip", "kaede.zip", "momo.zip"]
    assert json.loads((p.out / "probe.json").read_text(encoding="utf-8"))["count"] == 15
    assert {e["speaker"] for e in _jsonl(p.out / "utterances.jsonl")} == {"aoi", "kaede", "momo"}
    assert {e["speaker"] for e in _jsonl(p.out / "train.jsonl")} == {"kaede", "momo"}
    assert {e["speaker"] for e in _jsonl(p.out / "valid.jsonl")} == {"aoi"}


def test_a_narrower_selection_leaves_the_speakers_it_dropped_out(tmp_path, monkeypatch):
    """--hours has to bound what a run uses, not only what it fetches.

    The extract root only grows: a speaker unpacked once is there for every
    later run, and asking for fewer hours -- the natural reaction to a run too
    slow for the instance it is on -- adds no file anywhere, so nothing about a
    narrower selection is visible in a timestamp. Walking the root instead of
    the selection therefore means there is no way to ask for fewer speakers at
    all, and the ones dropped are not merely still on disk: they are silently
    in the training manifest, which is worse than the flag doing nothing.

    What they leave behind goes with them. `audio/` is tens of gigabytes at 124
    hours of 44.1 kHz, the disk on a preemptible instance is fixed, and filling
    it mid-run costs the run.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    assert {e["speaker"] for e in _jsonl(p.out / "utterances.jsonl")} == {"aoi", "kaede"}
    _age(p.out)

    (p.out / "characters.json").unlink()
    p.run(**CUTOFFS, hours=0.001)  # 0.06 minutes: one of the two 0.1-minute speakers

    chosen = json.loads((p.out / "characters.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in chosen["characters"]] == ["aoi"], chosen
    assert {e["speaker"] for e in _jsonl(p.out / "utterances.jsonl")} == {"aoi"}
    manifests = _jsonl(p.out / "train.jsonl") + _jsonl(p.out / "valid.jsonl")
    assert {e["speaker"] for e in manifests} == {"aoi"}, manifests
    # The retention table is read in hours, so it may not go on counting a
    # speaker this run will not train on either.
    assert json.loads((p.out / "probe.json").read_text(encoding="utf-8"))["count"] == 5
    assert sorted(f.name for f in (p.out / "entries").glob("*.jsonl")) == ["aoi.jsonl"]
    assert sorted(w.name for w in (p.out / "audio").glob("*.wav")) == ["aoi.wav", "aoi_001.wav"]


def test_new_cutoffs_rebuild_the_selection_they_decided(tmp_path, monkeypatch):
    """The two cutoffs are recorded, and changing them carries all the way down.

    Nothing else on disk says what a manifest was filtered under. Without that
    record a re-run with a stricter pair finds `utterances.jsonl` sitting there
    and reuses it -- and every stage built on it -- so the run reports success
    over the selection the operator has just replaced, with no INFO line and no
    warning; the only trace is that the numbers were typed. Worse than needing
    to remember: afterwards neither the script nor the operator can tell which
    cutoffs a given `train.jsonl` was built under.

    Taken here to the pair that keeps nothing, so the carry-through is visible
    in every artifact at once: the selection empties, both manifests empty with
    it, both alignments are rebuilt, and the audio of the speakers the cutoffs
    emptied is freed rather than left occupying the instance's disk.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    assert json.loads((p.out / "cutoffs.json").read_text(encoding="utf-8")) == CUTOFFS
    assert [e["id"] for e in _jsonl(p.out / "train.jsonl")] == ["clip0", "clip1", "clip2"]
    _age(p.out)

    p.run(max_cer=0.0, min_mos=4.5)  # the best clip scores 4.0, so nothing is kept

    assert json.loads((p.out / "cutoffs.json").read_text(encoding="utf-8")) == {
        "max_cer": 0.0,
        "min_mos": 4.5,
    }
    assert _jsonl(p.out / "utterances.jsonl") == []
    assert _jsonl(p.out / "train.jsonl") == [], "the split still holds the rejected selection"
    assert _jsonl(p.out / "valid.jsonl") == [], "the split still holds the rejected selection"
    assert len(p.calls["align"]) == 4, "the alignments still describe the old manifests"
    assert list((p.out / "audio").glob("*.wav")) == []
    assert list((p.out / "entries").glob("*.jsonl")) == []


def test_a_speaker_needing_fewer_files_takes_the_surplus_with_it(tmp_path, monkeypatch):
    """What a run stops naming, it deletes.

    Rebuilding a speaker's offsets under a longer --target-sec fits their clips
    into fewer pseudo-recordings than the last run wrote, and the files past
    the ones the new manifest names are never touched again. Nothing reads them
    -- every manifest names its files explicitly -- but `audio/` is tens of
    gigabytes at 124 hours of 44.1 kHz, deleting an artifact and re-running is
    the documented way to change any option, and the disk on a preemptible
    instance is fixed. Filling it costs the run.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)  # 6s of clips at --target-sec 5.0: two files per speaker
    assert sorted(w.name for w in (p.out / "audio").glob("*.wav")) == [
        "aoi.wav",
        "aoi_001.wav",
        "kaede.wav",
        "kaede_001.wav",
    ]
    _age(p.out)

    (p.out / "utterances.jsonl").unlink()  # how a re-run under new options is asked for
    p.run(**CUTOFFS, target_sec=10.0)  # now all six seconds fit in one file each

    assert sorted(w.name for w in (p.out / "audio").glob("*.wav")) == ["aoi.wav", "kaede.wav"]
    entries = _jsonl(p.out / "train.jsonl") + _jsonl(p.out / "valid.jsonl")
    assert {Path(e["path"]).name for e in entries} == {"aoi.wav", "kaede.wav"}, entries
    for entry in entries:
        assert Path(entry["path"]).exists(), entry


def test_a_re_run_whose_artifacts_all_share_a_timestamp_redoes_nothing(tmp_path, monkeypatch):
    """The tie the staleness check turns on, which no other test here reaches.

    File clocks are coarser than these stages are fast -- about 15 ms on
    Windows -- so the artifacts of one run routinely share a timestamp, and an
    output is reusable while it is at least as new as its inputs rather than
    strictly newer. Read the other way round, every stage is stale the moment
    it finishes and nothing is ever skipped again: on the real corpus that is
    30 GB of joining and the whole alignment pass, redone on every re-run,
    which is the opposite of what this script is for. Every other fixture here
    pushes its artifacts apart in time to say "an earlier run built this", so
    the tie itself is only reachable by making one.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    done = {stage: len(c) for stage, c in p.calls.items()}
    before = _tree(p.out)

    tied = time.time()
    for path in sorted(p.out.rglob("*")):
        os.utime(path, (tied, tied))

    p.run(**CUTOFFS)

    # The two stages that decide for themselves are entered again and find
    # their work done; nothing else runs at all.
    assert {stage: len(c) for stage, c in p.calls.items()} == done | {
        "download_called": 2,
        "extract_called": 4,
        # Asked once per run, before anything else: it costs an import and the
        # answer can change between runs, which is the point of asking again.
        "segmenter_check": 2,
    }
    assert _tree(p.out) == before


def test_a_kill_between_the_two_manifests_writes_both_again(tmp_path, monkeypatch):
    """Both manifests, or neither.

    They are written one after the other, so a kill in between leaves
    `train.jsonl` describing a corpus `valid.jsonl` was never held out of.
    Taking the half that survived for the whole stage would leave the held-out
    speaker in the training manifest and then write them into the valid one as
    well -- the same voice on both sides of the number that decides when to
    stop, and nothing that raises.
    """
    p = _pipeline(tmp_path, monkeypatch)
    p.run(**CUTOFFS)
    uninterrupted = _tree(p.out)

    # Exactly what a kill between the two write_manifest calls leaves behind.
    for name in ("valid.jsonl", "train_aligned.jsonl", "valid_aligned.jsonl"):
        (p.out / name).unlink()

    p.run(**CUTOFFS)

    assert len(p.calls["split_by_speaker"]) == 2, "the surviving half was taken for the whole"
    assert _tree(p.out) == uninterrupted


def test_a_backup_annotation_is_not_counted_as_a_second_clip(tmp_path):
    """The layout MoeSpeech actually ships, confirmed by unpacking fd6ca23b.zip:
    every clip carries a dated backup of its annotation beside it --
    `fd6ca23b_000.json` next to `fd6ca23b_000.20250706221645.bak.json` -- and
    rglob("*.json") finds both. Counted as clips, a speaker's corpus doubles:
    probe.json reports 248 where 124 exist and its retention table promises
    twice what any cutoff can deliver. The backups then fail to join, because
    `fd6ca23b_000.20250706221645.bak.wav` is not a file anybody ever wrote, and
    the operator reads "124 of 248 clips could not be joined" about a corpus
    that was never 248.

    An annotation whose audio is not there is not a training example, whatever
    its name, so the audio is what decides.
    """
    from training.scripts.prepare_moespeech import probe_utterances

    clips = tmp_path / "spk" / "wav"
    clips.mkdir(parents=True)
    for i in range(3):
        _annotation(clips, f"fd6ca23b_{i:03d}", "こんにちは", "こんにちは")
        _wav(clips / f"fd6ca23b_{i:03d}.wav", 1.0, 440.0)
        # The backup: same content, dated name, and no wav of its own.
        _annotation(
            clips, f"fd6ca23b_{i:03d}.20250706221645.bak", "こんにちは", "こんにちは", wav=False
        )

    stats = probe_utterances(tmp_path, names=["spk"])
    assert stats["count"] == 3, stats
    assert stats["no_audio"] == 3, stats


def test_an_annotation_whose_wav_is_missing_is_reported_under_its_own_name(tmp_path):
    """Told apart from an unreadable file and from one missing a field, because
    "the manifest is smaller than the table said" has different causes and only
    the counts distinguish them."""
    from training.scripts.prepare_moespeech import probe_utterances

    clips = tmp_path / "spk" / "wav"
    clips.mkdir(parents=True)
    _annotation(clips, "has_audio", "あ", "あ")
    _wav(clips / "has_audio.wav", 1.0, 440.0)
    _annotation(clips, "no_audio", "あ", "あ", wav=False)
    (clips / "unreadable.json").write_text("{not json", encoding="utf-8")

    stats = probe_utterances(tmp_path, names=["spk"])
    assert (stats["count"], stats["no_audio"], stats["unreadable"]) == (1, 1, 1), stats
