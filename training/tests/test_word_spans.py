"""The Viterbi alignment that produces every timestamp in a manifest.

batched_word_spans is the only thing standing between an audio file and the
`start`/`end` a manifest word carries, and training/dataloader.py trusts those
numbers absolutely: it cuts the utterance at the midpoint between one word's
end and the next one's start, feeds the audio before the cut as the voice
prompt and the audio after it as the target. A span that is wrong by a few
frames therefore does not raise -- it teaches the model to speak text whose
audio it was never shown.

Nothing here needs audio, or a checkpoint. The function takes a matrix of
per-frame log-probabilities, so the alignment can be dictated exactly and the
answer known in advance. Every expectation below is derived from what CTC
forced alignment means -- consume the tokens in order, one frame each, blanks
in between -- not from reading the implementation.
"""

from itertools import pairwise

import torch

from training.scripts.align_data import batched_word_spans

# Index in this string is the token id. "_" is the blank, "|" the word
# delimiter, which is what align_data.py's _tokens_for puts between words.
_SYMBOLS = "_abcd|"
BLANK = _SYMBOLS.index("_")
DELIM = _SYMBOLS.index("|")


def _emissions(rows: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probs in which frame t of item b emits rows[b][t], near-certainly.

    Rows shorter than the longest are padded with a uniform (i.e. maximally
    uninformative) distribution, which is what a real batch looks like past an
    item's true length.
    """
    T = torch.tensor([len(r) for r in rows])
    logits = torch.zeros(len(rows), int(T.max()), len(_SYMBOLS))
    for b, row in enumerate(rows):
        for t, ch in enumerate(row):
            logits[b, t, _SYMBOLS.index(ch)] = 20.0
    return logits.log_softmax(-1), T


def _spans_and_scores(row: str, tokens: list[int], word_of: list[int]):
    """Both halves of what the aligner returns, for a batch of one."""
    emissions, T = _emissions([row])
    return batched_word_spans(emissions, T, [tokens], [word_of], BLANK)


def _spans(row: str, tokens: list[int], word_of: list[int]):
    spans, _ = _spans_and_scores(row, tokens, word_of)
    return spans[0]


def test_each_word_claims_the_frame_that_emits_it():
    """The whole point: a word's span is where its audio is."""
    assert _spans("a__b", [1, 2], [0, 1]) == [(0, 0), (3, 3)]


def test_a_multi_token_word_spans_from_its_first_token_to_its_last():
    """Words are aligned per character, so the span has to be their union --
    the manifest records one start and one end per word, not per letter."""
    assert _spans("ab__", [1, 2], [0, 0]) == [(0, 1)]


def test_the_delimiter_between_words_belongs_to_no_word():
    """_tokens_for marks it word_of -1. Counting its frames into a neighbour
    would stretch that word over the silence a speaker pauses in, and the
    loader would site its cut inside speech."""
    spans = _spans("a_|_b", [1, DELIM, 2], [0, -1, 1])
    assert spans == [(0, 0), (4, 4)]


def test_words_are_ordered_in_time_and_never_overlap():
    """Alignment is monotonic. If it were not, the loader's cut -- the midpoint
    between one word's end and the next one's start -- could run backwards."""
    spans = _spans("abcd", [1, 2, 3, 4], [0, 1, 2, 3])
    assert all(s is not None for s in spans)
    for (_, prev_end), (start, _) in pairwise(spans):
        assert prev_end < start, spans


def test_every_word_with_tokens_gets_a_span():
    """A None here reaches the manifest as start: null, which silently removes
    both of that word's boundaries from the pool of cut points."""
    assert None not in _spans("a_b_c", [1, 2, 3], [0, 1, 2])


def test_too_few_frames_to_hold_the_tokens_is_unalignable():
    """Rather than returning a squeezed alignment the caller would believe."""
    assert _spans("a", [1, 2, 3], [0, 1, 2]) is None


def test_no_tokens_at_all_is_unalignable():
    """Every character of the transcript was outside the model's alphabet."""
    assert _spans("a__b", [], []) is None


def test_an_items_alignment_does_not_depend_on_what_it_is_batched_with():
    """Batching is an optimisation and must be invisible in the result. The
    trellis runs to the longest item in the batch, so a short item spends the
    difference on padding frames whose distribution is uniform: if those leak
    into its alignment, its spans shift and only a longer neighbour reveals it.
    """
    short, long = "a__b", "a__b__c_"
    alone, _ = batched_word_spans(*_emissions([short]), [[1, 2]], [[0, 1]], BLANK)
    batched, _ = batched_word_spans(
        *_emissions([short, long]), [[1, 2], [1, 2, 3]], [[0, 1], [0, 1, 2]], BLANK
    )
    assert batched[0] == alone[0]


def test_an_unalignable_item_does_not_take_its_batch_down_with_it():
    """One bad transcript in a shard must cost that utterance, not the shard.

    Both halves drop it, and both keep their place: a scores list that skipped
    the missing item instead of holding a None for it would be one shorter than
    the spans list, and every item after it in the batch would be written out
    carrying its neighbour's score.
    """
    results, scores = batched_word_spans(
        *_emissions(["a", "a__b"]), [[1, 2, 3], [1, 2]], [[0, 1, 2], [0, 1]], BLANK
    )
    assert results[0] is None
    assert results[1] == [(0, 0), (3, 3)]
    assert scores[0] is None and scores[1] is not None


def test_a_span_marks_where_a_sound_peaks_not_how_long_it_lasts():
    """Real speech holds a sound over many frames; CTC consumes each token on
    exactly one of them. The trellis breaks the resulting tie towards the last
    frame, so a word's `start` is the peak of its first token -- roughly the
    *end* of that first sound, not its onset.

    training/dataloader.py cuts at 0.5 * (prev["end"] + cur["start"]), so this
    biases every cut late by about half of the next word's first phoneme. Pin
    it: the numbers below are what the loader is calibrated against, and a
    change of tie-break direction would move every cut in every manifest.
    """
    # "a" occupies frames 0-2 and "b" frames 6-8, with silence between.
    assert _spans("aaa___bbb", [1, 2], [0, 1]) == [(2, 2), (8, 8)]
    # A word of several tokens still runs peak-to-peak, not onset-to-offset.
    assert _spans("aaabbb", [1, 2], [0, 0]) == [(2, 5)]


# -- how well the audio supports the text -------------------------------------


def test_a_matching_transcript_scores_higher_than_a_wrong_one():
    """The signal GOL needs. Its transcripts come from one ASR pass and there is
    no second one to disagree with, so the only evidence that text matches audio
    is how well the audio supports it -- which is exactly what the trellis
    already computes and throws away."""
    _, scores = _spans_and_scores("aaa___bbb", [1, 2], [0, 1])
    _, wrong = _spans_and_scores("aaa___bbb", [2, 1], [0, 1])
    assert scores[0][0] > wrong[0][0], (scores, wrong)


def test_the_score_comes_with_what_it_has_to_be_normalized_by():
    """Raw log-prob scales with both frame count and token count, and which
    normalization the distribution supports is not knowable before measuring it.
    So the row carries the raw score and both denominators."""
    _, scores = _spans_and_scores("aaa___bbb", [1, 2], [0, 1])
    score, frames, tokens = scores[0]
    assert frames == 9 and tokens == 2, (frames, tokens)
    assert score < 0, score


def test_an_unalignable_utterance_has_no_score():
    """None, not a sentinel number: a score that looks like a very bad
    alignment would be filtered as one, and this is a different thing."""
    spans, scores = _spans_and_scores("ab", [1, 2, 1, 2, 1], [0, 1, 2, 3, 4])
    assert spans[0] is None and scores[0] is None


def test_the_denominators_are_the_items_own_not_the_batchs():
    """A score can only be compared against another one if the counts written
    beside it describe the same utterance.

    The trellis runs to the longest item in the batch and as wide as the most
    tokens in it, so Tmax and Nmax sit right next to this item's t_end and n --
    and in a batch of one all four are the same number, which hides the swap.
    align_data.py length-sorts and chunks, so in a real run they are almost
    never equal. A chunk-wide denominator would give every row in the chunk a
    divisor it did not earn: the score would then rank utterances by a quantity
    that is not about them, and the manifest would look entirely plausible.
    """
    short, long = "aaa___bbb", "aaa___bbb___ccc_"
    _, alone_s = batched_word_spans(*_emissions([short]), [[1, 2]], [[0, 1]], BLANK)
    _, alone_l = batched_word_spans(*_emissions([long]), [[1, 2, 3]], [[0, 1, 2]], BLANK)
    _, both = batched_word_spans(
        *_emissions([short, long]), [[1, 2], [1, 2, 3]], [[0, 1], [0, 1, 2]], BLANK
    )
    assert both[0] == alone_s[0], (both, alone_s)
    assert both[1] == alone_l[0], (both, alone_l)


def test_a_transcript_with_a_tail_the_audio_never_says_scores_worse():
    """The score is the probability of the whole path, not of the best prefix.

    A single ASR pass inventing a few words at the end of an utterance is the
    exact failure this number exists to catch. A score that stopped at the last
    token the audio actually supports would rate that utterance as highly as the
    clean one it was hallucinated onto, and the filter would keep both.
    """
    _, good = _spans_and_scores("aaa___bbb", [1, 2], [0, 1])
    _, tail = _spans_and_scores("aaa___bbb", [1, 2, 3, 3, 3], [0, 1, 2, 3, 4])
    assert tail[0][0] < good[0][0] - 10, (good, tail)


# -- handing the spans back to the words they belong to ------------------------
#
# batched_word_spans only sees words it can align, because align_data.py filters
# out everything whose reading has no character in the model's alphabet. The
# manifest, though, has to carry every word -- the loader reads this field back
# as the text to speak. So the spans have to be dealt back out across a longer
# list, and an off-by-one there hands each word the timestamps of a different
# one. Nothing downstream can detect that: the numbers stay plausible.


def _reading_of(w: str) -> str:
    """Stand-in for align_data's vocabulary filter: lowercase letters align."""
    return "".join(c for c in w if c.islower())


def test_every_word_reaches_the_manifest_in_order():
    """The loader re-reads "word" as the text to speak. Losing one means
    training on audio for words the text does not contain."""
    from training.scripts.align_data import _timed_words

    words = ["Ab", "3000", "cd"]
    timed = _timed_words(words, [_reading_of(w) for w in words], [(0, 0), (4, 5)], 0.02, False)
    assert [t["word"] for t in timed] == words


def test_an_unalignable_word_does_not_shift_the_words_after_it():
    """The one that matters. "3000" has no reading, so it claims no span --
    but if it consumed one anyway, "cd" would inherit "3000"'s neighbour's
    timestamps and every word after it would speak over the wrong audio."""
    from training.scripts.align_data import _timed_words

    words = ["Ab", "3000", "cd"]
    timed = _timed_words(words, [_reading_of(w) for w in words], [(0, 0), (4, 5)], 0.02, False)
    assert timed[0]["start"] == 0.0
    assert timed[1]["start"] is None and timed[1]["end"] is None
    assert timed[2]["start"] == 0.08  # frame 4, not frame 0


def test_end_is_the_far_edge_of_the_last_frame_not_its_near_edge():
    """A one-frame word with start == end would be a zero-length span, and
    dataloader.py's cut arithmetic would place two cuts at the same instant."""
    from training.scripts.align_data import _timed_words

    (timed,) = _timed_words(["ab"], ["ab"], [(3, 3)], 0.02, False)
    assert timed["start"] == 0.06 and timed["end"] == 0.08


def test_a_word_the_trellis_could_not_place_keeps_its_text():
    """spans carries None for a word the alignment gave up on. It still has to
    reach the manifest -- with no timestamps, so it blocks a cut beside it."""
    from training.scripts.align_data import _timed_words

    timed = _timed_words(["ab", "cd"], ["ab", "cd"], [None, (4, 4)], 0.02, False)
    assert [t["word"] for t in timed] == ["ab", "cd"]
    assert timed[0]["start"] is None and timed[1]["start"] == 0.08


def test_the_reading_is_recorded_only_when_it_differs_from_the_surface():
    """For Japanese the kana that went into the trellis is kept, so a wrong
    reading can be spotted by eye in the manifest; for English it would just
    be the word again."""
    from training.scripts.align_data import _timed_words

    (with_kana,) = _timed_words(["気配"], ["けはい"], [(0, 0)], 0.02, True)
    (without,) = _timed_words(["word"], ["word"], [(0, 0)], 0.02, False)
    assert with_kana["kana"] == "けはい"
    assert "kana" not in without


# -- building the token stream the trellis consumes ---------------------------


def test_words_are_separated_by_the_delimiter_and_it_belongs_to_no_word():
    """The delimiter is how the CTC model was taught to spell a word boundary;
    marking it -1 is what keeps the pause between two words out of both of
    their spans."""
    from training.scripts.align_data import _tokens_for

    vocab = {"a": 1, "b": 2, "c": 3}
    tokens, word_of = _tokens_for(["ab", "c"], vocab, DELIM)
    assert tokens == [1, 2, DELIM, 3]
    assert word_of == [0, 0, -1, 1]


def test_there_is_no_delimiter_before_the_first_word():
    """A leading one would make the alignment demand a pause the recording
    does not open with, and every span would shift."""
    from training.scripts.align_data import _tokens_for

    tokens, word_of = _tokens_for(["ab"], {"a": 1, "b": 2}, DELIM)
    assert tokens == [1, 2] and word_of == [0, 0]


def test_no_words_produces_no_tokens():
    """align_data.py treats this as unalignable rather than aligning nothing."""
    from training.scripts.align_data import _tokens_for

    assert _tokens_for([], {}, DELIM) == ([], [])
