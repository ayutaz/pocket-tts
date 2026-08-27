"""The Japanese text pipeline: normalization, and collecting a tokenizer corpus.

Everything here guards the same invariant -- the tokenizer corpus, the manifest
transcript and a user's inference input must be the same distribution. Nothing
in training asserts it and sentencepiece never errors, so a break shows up only
as a model that never quite becomes intelligible.
"""

from typer.testing import CliRunner

from pocket_tts.utils.text_normalization import normalize_japanese as normalize
from training.scripts.prepare_ja_text import app

runner = CliRunner()


def test_ellipsis_survives_normalization():
    """Bare NFKC turns U+2026 into three ASCII periods, which is the same damage
    GOL's own "normalized" column does: a run of stops the model reads aloud,
    and three sentence boundaries where there was one."""
    assert normalize("ああ…そうか") == "ああ…そうか"
    assert normalize("……お兄ちゃん") == "……お兄ちゃん"
    assert normalize("‥") == "‥"


def test_width_variants_are_folded():
    """These we do want folded: they are the same utterance spelled two ways,
    and keeping both spends vocabulary and splits the distribution."""
    assert normalize("ＡＢＣと１２３") == "ABCと123"
    assert normalize("ｱｲｳ") == "アイウ"
    assert normalize("！？") == "!?"


def test_japanese_punctuation_is_left_alone():
    assert normalize("こんにちは、世界。") == "こんにちは、世界。"


def test_standalone_dakuten_does_not_inject_a_space():
    """NFKC turns U+309B into SPACE + an orphan combining mark. The emphatic
    spelling is common in this kind of corpus, and a space injected into text
    that has none is worse than leaving the mark as written."""
    assert normalize("え゛っ") == "え゛っ"
    assert normalize("あ゜") == "あ゜"
    assert " " not in normalize("え゛っ")


def test_private_use_characters_are_not_mistaken_for_sentinels():
    """The held-out characters are swapped for noncharacters, not private-use
    code points: legacy carrier emoji live in the PUA and this corpus already
    carries some, so a PUA sentinel would rewrite them into ellipses."""
    for pua in ["", "", ""]:
        assert normalize("あ" + pua + "い") == "あ" + pua + "い"


def test_whitespace_runs_collapse_and_control_chars_go():
    assert normalize("  a   b  ") == "a b"
    assert normalize("a\x00b\x07c") == "abc"


def test_normalize_is_idempotent():
    """It runs on the corpus and again on each transcript; applying it twice
    must not change the answer."""
    for text in ["ああ…そうか", "ＡＢＣ", "こんにちは、世界。", "  a   b  "]:
        assert normalize(normalize(text)) == normalize(text)


# -- corpus builder -----------------------------------------------------------


def _run(*args):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def test_gol_uses_the_raw_column_not_the_damaged_one(tmp_path):
    """GOL's third column rewrites the ellipsis as a run of ideographic full
    stops in 44.6% of rows. Reading it would teach the model to say them."""
    meta = tmp_path / "metadata.csv"
    meta.write_text(
        "GOL-0000001|ああ……そうか|ああ。。。。。。そうか\nGOL-0000002|はい|はい\n", encoding="utf-8"
    )
    out = tmp_path / "corpus.txt"
    _run(out, "--gol", meta)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == ["ああ……そうか", "はい"]
    assert not any("。。" in line for line in lines)


def test_malformed_gol_rows_are_dropped_not_misparsed(tmp_path):
    """A row containing the separator itself cannot be parsed positionally."""
    meta = tmp_path / "metadata.csv"
    meta.write_text("GOL-1|よし|よし\nGOL-2||-ﾟ)|゚\nGOL-3|だめ|だめ\n", encoding="utf-8")
    out = tmp_path / "corpus.txt"
    _run(out, "--gol", meta)
    assert out.read_text(encoding="utf-8").splitlines() == ["よし", "だめ"]


def test_ljspeech_takes_the_last_field(tmp_path):
    meta = tmp_path / "metadata.csv"
    meta.write_text("id_0|spk_a|こんにちは\nid_1|spk_b|さようなら\n", encoding="utf-8")
    out = tmp_path / "corpus.txt"
    _run(out, "--ljspeech", meta)
    assert out.read_text(encoding="utf-8").splitlines() == ["こんにちは", "さようなら"]


def test_sources_combine_and_deduplicate(tmp_path):
    """19% of GOL is stock phrases; leaving them in biases the merges toward
    them without adding any coverage."""
    gol = tmp_path / "gol.csv"
    gol.write_text("G-1|はい|はい\nG-2|はい|はい\nG-3|いいえ|いいえ\n", encoding="utf-8")
    lj = tmp_path / "lj.csv"
    lj.write_text("id_0|spk|はい\nid_1|spk|どうも\n", encoding="utf-8")
    out = tmp_path / "corpus.txt"
    _run(out, "--gol", gol, "--ljspeech", lj)
    assert out.read_text(encoding="utf-8").splitlines() == ["はい", "いいえ", "どうも"]


def test_dedup_can_be_turned_off(tmp_path):
    gol = tmp_path / "gol.csv"
    gol.write_text("G-1|はい|はい\nG-2|はい|はい\n", encoding="utf-8")
    out = tmp_path / "corpus.txt"
    _run(out, "--gol", gol, "--no-dedup")
    assert out.read_text(encoding="utf-8").splitlines() == ["はい", "はい"]


def test_moe_json_emits_both_asr_spellings(tmp_path):
    """Either transcription could end up as a manifest's transcript, so the
    tokenizer has to have seen both."""
    import json

    d = tmp_path / "jsons"
    d.mkdir()
    (d / "a.json").write_text(
        json.dumps(
            {
                "anime_whisper_transcription": "昨夜からずっと…",
                "parakeet_jp_transcription": "昨夜からずっと",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "corpus.txt"
    _run(out, "--moe-json", d)
    assert sorted(out.read_text(encoding="utf-8").splitlines()) == sorted(
        ["昨夜からずっと…", "昨夜からずっと"]
    )


def test_corpus_is_normalized_on_the_way_out(tmp_path):
    """The manifest builder applies the same function, so what the tokenizer
    learns and what it is later asked to encode line up."""
    gol = tmp_path / "gol.csv"
    gol.write_text("G-1|ＡＢＣ…だ|x\n", encoding="utf-8")
    out = tmp_path / "corpus.txt"
    _run(out, "--gol", gol)
    assert out.read_text(encoding="utf-8").splitlines() == ["ABC…だ"]


def test_no_sources_is_an_error(tmp_path):
    result = runner.invoke(app, [str(tmp_path / "corpus.txt")])
    assert result.exit_code != 0
