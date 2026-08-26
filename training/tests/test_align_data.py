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
# the long-vowel mark. align_data keeps only characters in the checkpoint's
# vocabulary, so the segmenter emits a reading *candidate* and this filter is
# what decides whether a word can be aligned at all.
_HIRAGANA_VOCAB = set(
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのは"
    "ばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんゔー"
)


def _alignable(reading: str) -> str:
    """What align_data.py:_tokens_for actually receives for this word."""
    return "".join(c for c in reading if c in _HIRAGANA_VOCAB)


@pytest.fixture(scope="module")
def segment():
    pytest.importorskip("fugashi", reason="uv sync --group japanese")
    pytest.importorskip("unidic_lite", reason="uv sync --group japanese")
    return _japanese_segmenter()


def test_japanese_segmenter_finds_word_boundaries(segment):
    """Whitespace splitting would return one word here, which leaves the loader
    with no cut point and silently disables the voice-prompt curriculum."""
    assert len(segment("昨夜からずっと気配を探られていたか。")) > 1


def test_japanese_surface_forms_reconstruct_the_transcript(segment):
    """dataloader.py joins these back with word_separator="", so the text the
    model trains on is a true suffix of the transcript -- or it is a bug."""
    for text in [
        "昨夜からずっと気配を探られていたか。",
        "Hello world と言った",
        "あ Hello world い",
        "ATMで3000円おろした。",
        "むぅ……龍子",
    ]:
        assert "".join(s for s, _, _ in segment(text)) == text


def test_japanese_surface_keeps_kanji_while_the_reading_does_not(segment):
    """The two must diverge: "word" is what the model learns to speak, the
    reading is what goes into the trellis."""
    ((surface, reading, _),) = segment("気配")
    assert surface == "気配" and reading == "けはい"


def test_japanese_recovers_readings_unidic_leaves_empty(segment):
    """UniDic returns no reading for small kana or the prolongation mark, but
    both are audio spelled as themselves and both are in the checkpoint's
    alphabet. Dropped, that speech falls into no span, and the loader's cut --
    the midpoint between one span's end and the next one's start -- lands in
    the middle of it."""
    for text in ["むぅ", "かぷー", "あぁー"]:
        aligned = "".join(_alignable(r) for _, r, _ in segment(text))
        assert aligned == text, (text, aligned)


def test_japanese_unreadable_tokens_align_to_nothing_but_keep_their_text(segment):
    """Digits, latin and punctuation have no derivable reading. They must stay
    in the training text, and must NOT be folded into a neighbour that has
    timestamps: a word with no span is exactly what stops dataloader.py:129
    placing a cut beside speech the trellis never labelled."""
    segments = segment("ATMで3000円おろした。")
    unreadable = {s for s, r, _ in segments if not _alignable(r)}
    assert unreadable == {"ATM", "3000", "。"}
    assert "".join(s for s, _, _ in segments) == "ATMで3000円おろした。"


def test_japanese_katakana_words_are_alignable(segment):
    """The checkpoint's vocabulary has no katakana except the prolongation
    mark, so the reading has to be translated or every katakana word is lost."""
    for _, reading, _ in segment("ヴァイオリンを弾く"):
        assert _alignable(reading) == reading, reading


def test_normalized_transcript_survives_the_round_trip(segment):
    """The invariant align_data enforces end to end: the transcript it writes
    back to the manifest is normalize()'d, and the words it writes rebuild that
    exact string when the loader joins them with word_separator "". If this
    breaks, the model trains on text no user will ever type and nothing errors.
    """
    from training.scripts.ja_text import normalize

    for raw in [
        "ＡＢＣと１２３をみた。",
        "ああ……そうか！",
        "え゛っ、まじで？",
        "Hello world と言った",
        "ＡＴＭで３０００円おろした。",
        "  よし　　いくぞ  ",
    ]:
        text = normalize(raw)
        assert "".join(s for s, _, _ in segment(text)) == text, raw
