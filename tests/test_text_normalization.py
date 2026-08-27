"""The one normalization the corpus, the manifests and inference all share.

Three strings have to come from the same distribution or a run degrades with
nothing reporting why: the corpus the tokenizer was fitted on, the transcript
the DataLoader feeds at train time, and the text a user types at inference.
Sentencepiece encodes all three happily whatever they look like, so the only
symptom of a mismatch is a model that never quite becomes intelligible.

This module lives in pocket_tts/ rather than training/ because inference needs
it and the inference package cannot import from the training package.
"""

import pytest

from pocket_tts.utils.text_normalization import NORMALIZERS, normalize_japanese, resolve_normalizer


def test_fullwidth_variants_are_folded():
    """A Japanese IME emits these by default. Unfolded they are characters the
    tokenizer was never fitted on, and they encode as <unk>."""
    assert normalize_japanese("ＡＢＣ１２３？！") == "ABC123?!"
    # Halfwidth katakana folds the other way, into fullwidth -- same reason.
    assert normalize_japanese("ｱｲｳ") == "アイウ"


def test_the_ellipsis_survives():
    """Bare NFKC turns U+2026 into three ASCII periods: a run of stops the
    model reads aloud, and three sentence boundaries where there was one.
    15.6% of corpus utterances end in it."""
    assert normalize_japanese("ああ…") == "ああ…"
    assert normalize_japanese("……お兄ちゃん") == "……お兄ちゃん"
    assert normalize_japanese("‥") == "‥"  # U+2025, the two-dot leader


def test_japanese_punctuation_is_left_alone():
    assert normalize_japanese("こんにちは、世界。") == "こんにちは、世界。"


def test_the_standalone_dakuten_does_not_inject_a_space():
    """NFKC turns U+309B into a SPACE plus an orphan combining mark. The
    emphatic spelling is common in this kind of corpus, and injecting a space
    into a writing system that has none is worse than leaving it."""
    assert normalize_japanese("え゛っ") == "え゛っ"
    assert normalize_japanese("あ゜") == "あ゜"
    assert " " not in normalize_japanese("え゛っ")


def test_private_use_characters_are_not_mistaken_for_sentinels():
    """The held-out characters are swapped for noncharacters, not private-use
    code points: legacy carrier emoji live in the PUA and this corpus already
    carries some, so a PUA sentinel would rewrite them into ellipses."""
    for pua in ["", "", ""]:
        assert normalize_japanese("あ" + pua + "い") == "あ" + pua + "い"


def test_whitespace_runs_collapse_and_control_chars_go():
    assert normalize_japanese("  a   b  ") == "a b"
    assert normalize_japanese("a\x00b\x07c") == "abc"


def test_normalize_is_idempotent():
    """It runs on the corpus and again on each transcript; applying it twice
    must not change the answer."""
    for text in ["ああ…そうか", "ＡＢＣ", "こんにちは、世界。", "  a   b  "]:
        assert normalize_japanese(normalize_japanese(text)) == normalize_japanese(text)


def test_the_registry_resolves_by_name():
    assert resolve_normalizer("japanese") is normalize_japanese
    assert set(NORMALIZERS) == {"japanese"}


def test_no_normalizer_is_the_identity():
    """What every released model gets: its text must reach the tokenizer
    exactly as the caller wrote it."""
    identity = resolve_normalizer(None)
    for text in ["Hello world.", "ＡＢＣ", "  spaced  out  "]:
        assert identity(text) == text


def test_an_unknown_normalizer_is_an_error():
    """A typo in a config must not silently fall back to doing nothing."""
    with pytest.raises(KeyError):
        resolve_normalizer("japanesee")
