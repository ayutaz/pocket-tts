"""The metric that can tell a good Japanese generation from a bad one.

Word error rate cannot: Japanese is written without spaces, so an utterance is
one word, and every imperfect generation scores exactly 1.0. Judging the
validation run on WER would mean judging it on a number that is 1.0 whether
the model said almost the right thing or nothing like it.
"""

from types import SimpleNamespace

import jiwer

from training.eval.librispeech import DEFAULT_ASR, build_normalizer, eval_dir_name

REF = "今日はいい天気ですね"
CLOSE = "今日はいい天気ですわ"  # one character differs
UNRELATED = "全然関係のない文章がここにある"


def _args(normalizer: str) -> SimpleNamespace:
    return SimpleNamespace(
        temp=0.7,
        cfg=1.0,
        use_ema=True,
        num_items=None,
        seed=0,
        asr=DEFAULT_ASR,
        prompt_root=None,
        text_normalizer=normalizer,
    )


def test_wer_cannot_tell_these_apart():
    """The reason this task exists. Both score 1.0."""
    assert jiwer.wer(REF, CLOSE) == jiwer.wer(REF, UNRELATED) == 1.0


def test_cer_can():
    assert jiwer.cer(REF, CLOSE) < 0.2
    assert jiwer.cer(REF, UNRELATED) > 1.0


def test_the_english_normalizer_destroys_dakuten():
    """が (U+304C) and か (U+304B) are different sounds. Scoring Japanese
    through this normalizer forgives every voicing error in the language."""
    assert build_normalizer("english")("が") == "か"


def test_the_basic_normalizer_keeps_them():
    assert build_normalizer("basic")("が") == "が"


def test_an_unknown_normalizer_is_an_error():
    import pytest

    with pytest.raises(ValueError):
        build_normalizer("japanese")


def test_the_normalizer_is_in_the_output_directory_name():
    """eval_dir_name's own docstring: anything that changes the numbers belongs
    in the name, or two evals of one checkpoint overwrite each other."""
    assert eval_dir_name(_args("english"), 1000) != eval_dir_name(_args("basic"), 1000)


def test_the_default_name_is_unchanged():
    """Existing eval directories must keep resolving to the same path."""
    assert eval_dir_name(_args("english"), 1000) == "libri_eval_step1000_t0.7_cfg1.0"
