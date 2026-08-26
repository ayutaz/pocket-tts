"""Segmentation behaviour that silently degrades training when it breaks.

The manifest's "word" field is load-bearing twice over: align_data.py aligns it
and training/dataloader.py re-reads it as the text to speak. A segmenter bug
therefore corrupts the training text, not just the timestamps, and nothing
downstream would notice.
"""

import pytest

from training.scripts.align_data import (
    SEGMENTERS,
    _japanese_segmenter,
    _merge_phrases,
    _whitespace_segments,
)


def test_whitespace_segmenter_matches_the_old_split():
    """The English path has to stay bit-for-bit what it was before segmenters."""
    assert _whitespace_segments("  one two   three ") == [
        ("one", "one", True),
        ("two", "two", True),
        ("three", "three", True),
    ]


def test_whitespace_segments_are_all_phrase_heads():
    """So --merge-phrases can never collapse an English manifest."""
    segments = _whitespace_segments("one two three")
    merged = _merge_phrases(
        [{"word": s, "start": 0.0, "end": 1.0} for s, _, _ in segments], [h for _, _, h in segments]
    )
    assert [m["word"] for m in merged] == ["one", "two", "three"]


def test_merge_phrases_glues_function_words_onto_the_word_before():
    timed = [
        {"word": "A", "kana": "a", "start": 0.0, "end": 1.0},
        {"word": "b", "kana": "b", "start": 1.0, "end": 2.0},
        {"word": "C", "kana": "c", "start": 2.0, "end": 3.0},
    ]
    merged = _merge_phrases(timed, [True, False, True])
    assert [m["word"] for m in merged] == ["Ab", "C"]
    assert [m["kana"] for m in merged] == ["ab", "c"]
    # The phrase spans from the head's start to the last glued word's end.
    assert merged[0]["start"] == 0.0 and merged[0]["end"] == 2.0


def test_merge_phrases_keeps_text_when_a_word_has_no_timestamps():
    """An unalignable word must not drop out of the training text."""
    timed = [
        {"word": "A", "start": 0.0, "end": 1.0},
        {"word": "b", "start": None, "end": None},
        {"word": "c", "start": 2.0, "end": 3.0},
    ]
    merged = _merge_phrases(timed, [True, False, False])
    assert [m["word"] for m in merged] == ["Abc"]
    assert merged[0]["start"] == 0.0 and merged[0]["end"] == 3.0


def test_merge_phrases_survives_a_leading_non_head():
    """A transcript starting with punctuation must not lose its first word."""
    timed = [{"word": "a", "start": 0.0, "end": 1.0}, {"word": "B", "start": 1.0, "end": 2.0}]
    assert [m["word"] for m in _merge_phrases(timed, [False, True])] == ["a", "B"]


def test_merge_phrases_does_not_mutate_its_input():
    timed = [{"word": "A", "start": 0.0, "end": 1.0}, {"word": "b", "start": 1.0, "end": 2.0}]
    _merge_phrases(timed, [True, False])
    assert timed[0]["word"] == "A"


def test_segmenter_registry_holds_the_documented_names():
    assert sorted(SEGMENTERS) == ["japanese", "whitespace"]


# -- Japanese: needs the optional analyser (uv sync --group japanese) ----------

# vumichien/wav2vec2-large-xlsr-japanese-hiragana's alphabet: 82 hiragana plus
# the long-vowel mark. A reading outside this set cannot be aligned at all.
_HIRAGANA_VOCAB = set(
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのは"
    "ばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんゔー"
)


@pytest.fixture(scope="module")
def segment():
    pytest.importorskip("fugashi", reason="uv sync --group japanese")
    pytest.importorskip("unidic_lite", reason="uv sync --group japanese")
    return _japanese_segmenter()


def test_japanese_segmenter_finds_word_boundaries(segment):
    """Whitespace splitting would return one word here, which leaves the loader
    with no cut point and silently disables the voice-prompt curriculum."""
    segments = segment("昨夜からずっと気配を探られていたか。")
    assert len(segments) > 1


def test_japanese_surface_forms_reconstruct_the_transcript(segment):
    """dataloader.py joins these back with word_separator="", so the text the
    model trains on is a true suffix of the transcript -- or it is a bug."""
    text = "昨夜からずっと気配を探られていたか。"
    assert "".join(s for s, _, _ in segment(text)) == text


def test_japanese_readings_are_hiragana_the_checkpoint_knows(segment):
    """Kanji readings go to the trellis; anything outside the checkpoint's
    alphabet is dropped by the vocab filter and shifts every timestamp."""
    for text in ["昨夜からずっと気配を探られていたか。", "ヴァイオリンを弾く私"]:
        for surface, reading, _ in segment(text):
            assert not set(reading) - _HIRAGANA_VOCAB, (surface, reading)


def test_japanese_surface_keeps_kanji_while_the_reading_does_not(segment):
    """The two must diverge: "word" is what the model learns to speak."""
    segments = segment("気配")
    assert segments[0][0] == "気配"
    assert segments[0][1] == "けはい"


def test_japanese_readingless_tokens_stay_in_the_text(segment):
    """Digits, latin and punctuation have no UniDic reading. They must still
    reach the training text, glued onto the morpheme before them."""
    text = "ATMで3000円おろした。"
    segments = segment(text)
    assert "".join(s for s, _, _ in segments) == text
    # 3000 and the full stop are absorbed rather than left as bare segments.
    assert [s for s, _, _ in segments] != ["ATM", "で", "3000", "円", "おろし", "た", "。"]
    # Only a readingless *leading* token has nothing to glue onto; it keeps its
    # text and simply gets no timestamps.
    unalignable = [i for i, (_, r, _) in enumerate(segments) if not r]
    assert unalignable in ([], [0]), segments
