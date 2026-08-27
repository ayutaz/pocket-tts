"""What the model is actually asked to speak, after the frontend has had it.

The text that reaches the tokenizer has to look like the text the model was
trained on. Nothing checks that -- sentencepiece encodes whatever it is given
-- so a mismatch here surfaces only as a model that never quite becomes
intelligible.
"""

import pytest

from pocket_tts.models.tts_model import prepare_text_prompt
from pocket_tts.utils.text_normalization import TextRules

JA = TextRules(
    normalizer="japanese",
    sentence_boundaries=".!...?。…",
    clause_boundaries=",;:、",
    terminal_punctuation="",
    segment_separator="",
)


def test_english_still_gets_its_period():
    """The behaviour every released model depends on."""
    text, _ = prepare_text_prompt("Hello world", False, False)
    assert text == "Hello world."


def test_japanese_gets_no_period():
    """49.4% of training utterances end with no terminal punctuation at all,
    and only 1.12% end in the full stop. Appending anything -- the ASCII
    period or the Japanese one -- puts the text outside that distribution."""
    text, _ = prepare_text_prompt("こんにちは", False, False, rules=JA)
    assert text == "こんにちは"


def test_japanese_keeps_the_punctuation_it_already_has():
    """Fullwidth ? folds to ASCII, which is the form 17.6% of corpus
    utterances actually end in."""
    text, _ = prepare_text_prompt("元気ですか？", False, False, rules=JA)
    assert text == "元気ですか?"


def test_the_normalizer_runs():
    """A Japanese IME emits fullwidth by default; unfolded it is <unk>."""
    text, _ = prepare_text_prompt("ＡＢＣです", False, False, rules=JA)
    assert text == "ABCです"


def test_no_normalizer_leaves_text_alone():
    """Released models must receive their text exactly as written."""
    text, _ = prepare_text_prompt("ＡＢＣ", False, False)
    assert text.startswith("ＡＢＣ")


def test_empty_text_is_still_an_error():
    with pytest.raises(ValueError):
        prepare_text_prompt("", False, False, rules=JA)


def test_text_that_normalizes_to_empty_is_an_error():
    """A control character survives str.strip() but is stripped out by
    normalize_japanese; either order clears the text, but only running
    normalization first reaches this ValueError instead of text[0] raising
    IndexError on an empty string."""
    with pytest.raises(ValueError):
        prepare_text_prompt("\x01", False, False, rules=JA)
