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


def test_the_ellipsis_survives():
    """Bare NFKC turns U+2026 into three ASCII periods: a run of stops the
    model reads aloud, and three sentence boundaries where there was one.
    15.6% of corpus utterances end in it."""
    assert normalize_japanese("ああ…") == "ああ…"


def test_the_standalone_dakuten_does_not_inject_a_space():
    """NFKC turns U+309B into a SPACE plus an orphan combining mark. Injecting
    a space into a writing system that has none is worse than leaving it."""
    assert " " not in normalize_japanese("え゛っ")


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
