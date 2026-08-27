"""Dataloader behaviour that silently degrades training when it breaks."""

import json

import numpy as np
import sphn

from training.dataloader import DataLoader, load_entries

SR = 24000


def _write_wav(path, seconds=6.0):
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    sphn.write_wav(str(path), (0.2 * np.sin(2 * np.pi * 140 * t)).astype(np.float32), SR)


def _manifest(tmp_path, n=8, words=True, duration=6.0):
    wav = tmp_path / "a.flac"
    _write_wav(wav, duration)
    path = tmp_path / "m.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            entry = {"path": str(wav), "duration": duration, "transcript": "one two three four"}
            if words:
                entry["words"] = [
                    {"word": w, "start": 0.8 * j, "end": 0.8 * j + 0.6}
                    for j, w in enumerate(entry["transcript"].split())
                ]
            f.write(json.dumps(entry) + "\n")
    return str(path)


def _loader(manifest, **kw):
    kw.setdefault("batch_size", 2)
    return DataLoader(
        manifest,
        lambda s: [1] * max(1, len(s) // 3),
        kw.pop("batch_size"),
        SR,
        12.5,
        kw.pop("max_duration_sec", 30.0),
        kw.pop("max_voice_prompt_sec", 3.0),
        0,
        1,
        seed=0,
        shuffle=False,
        **kw,
    )


def test_rank_sharding_partitions_entries_without_overlap(tmp_path):
    m = _manifest(tmp_path, n=8)
    shards = [load_entries(m, rank, 4) for rank in range(4)]
    assert sum(len(s) for s in shards) == 8
    assert all(len(s) == 2 for s in shards)


def test_prompt_respects_the_configured_cap(tmp_path):
    batch = next(iter(_loader(_manifest(tmp_path), max_voice_prompt_sec=1.0)))
    assert (batch.num_voice_prompt_frames.float() / 12.5).max().item() <= 1.0 + 1e-6


def test_batches_have_the_requested_size(tmp_path):
    batch = next(iter(_loader(_manifest(tmp_path, n=8), batch_size=4)))
    assert batch.audio.shape[0] == 4
    assert len(batch.text_tokens) == 4


def test_target_audio_never_exceeds_max_duration(tmp_path):
    batch = next(iter(_loader(_manifest(tmp_path, duration=30.0), max_duration_sec=5.0)))
    assert batch.audio.shape[-1] <= int(5.0 * SR) + 1


def test_unaligned_manifest_still_yields_batches(tmp_path):
    """Manifests without word alignments are a documented input: the loader
    falls back to a random window as the prompt instead of hanging."""
    loader = _loader(_manifest(tmp_path, n=8, words=False))
    batch = next(iter(loader))
    assert batch.audio.shape[0] == 2


def test_entry_start_offsets_into_a_shared_file(tmp_path):
    """Two utterances can share one audio file: each entry reads its own
    window, at `start`, out of the shared file."""
    low_hz, high_hz = 220, 880
    t = np.linspace(0, 10.0, int(10.0 * SR), endpoint=False)
    wav = np.where(
        t < 5.0, 0.2 * np.sin(2 * np.pi * low_hz * t), 0.2 * np.sin(2 * np.pi * high_hz * t)
    )
    audio_path = tmp_path / "shared.flac"
    sphn.write_wav(str(audio_path), wav.astype(np.float32), SR)

    manifest = tmp_path / "m.jsonl"
    with open(manifest, "w") as f:
        f.write(json.dumps({"path": str(audio_path), "duration": 5.0, "transcript": "low"}) + "\n")
        f.write(
            json.dumps(
                {"path": str(audio_path), "duration": 5.0, "transcript": "high", "start": 5.0}
            )
            + "\n"
        )

    loader = _loader(str(manifest), batch_size=2)
    low_entry, high_entry = loader.get_entry(0), loader.get_entry(1)
    assert low_entry.start == 0.0
    assert high_entry.start == 5.0

    low_wav, *_ = loader._sample(low_entry)
    high_wav, *_ = loader._sample(high_entry)

    def dominant_freq(x):
        spectrum = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=1 / SR)
        return freqs[np.argmax(spectrum)]

    assert abs(dominant_freq(low_wav) - low_hz) < 2
    assert abs(dominant_freq(high_wav) - high_hz) < 2


def _ja_manifest(tmp_path, words):
    """One 6s utterance whose words carry no spaces, as Japanese is written."""
    wav = tmp_path / "ja.flac"
    _write_wav(wav, 6.0)
    path = tmp_path / "ja.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "path": str(wav),
                    "duration": 6.0,
                    "transcript": "".join(words),
                    "words": [
                        {"word": w, "start": 1.5 * j, "end": 1.5 * j + 1.2}
                        for j, w in enumerate(words)
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return str(path)


def _text_seen_by_the_tokenizer(manifest, **kw):
    seen = []

    def record(text):
        seen.append(text)
        return [1]

    loader = DataLoader(manifest, record, 1, SR, 12.5, 30.0, 3.0, 0, 1, seed=0, shuffle=False, **kw)
    loader._sample(loader.get_entry(0))
    return seen


def test_word_separator_defaults_to_a_space(tmp_path):
    """Every released model was trained on space-joined text; changing the
    default would silently retire that."""
    seen = _text_seen_by_the_tokenizer(_manifest(tmp_path, n=1))
    assert seen and all(" " in t or t in ("four",) for t in seen), seen


def test_word_separator_can_join_without_spaces(tmp_path):
    """Japanese is written without spaces. If the loader joins with one anyway,
    training text and inference input come from different distributions --
    nothing errors, the model just never quite becomes intelligible."""
    words = ["こんにちは", "世界", "です"]
    seen = _text_seen_by_the_tokenizer(_ja_manifest(tmp_path, words), word_separator="")
    assert seen, "the loader never reached the aligned cut path"
    for text in seen:
        assert " " not in text, text
        # What the model is asked to speak is a true suffix of the transcript.
        assert "".join(words).endswith(text), text


def _cut_texts(manifest, tries=40, **kw):
    """The text of every cut the loader draws over `tries` samples."""
    seen = []

    def record(text):
        seen.append(text)
        return [1]

    loader = DataLoader(manifest, record, 1, SR, 12.5, 30.0, 5.0, 0, 1, seed=0, shuffle=False, **kw)
    entry = loader.get_entry(0)
    for _ in range(tries):
        loader._sample(entry)
    return set(seen)


def test_a_word_without_timestamps_removes_the_cuts_on_both_sides_of_it(tmp_path):
    """The contract align_data.py's Japanese path depends on.

    Digits, latin and punctuation have no derivable reading, so the aligner
    labels no frames for them and writes start: null. That null is the only
    thing keeping the loader from cutting beside audio nobody aligned -- the
    cut is the midpoint between one word's end and the next one's start, so
    beside an unlabelled word it would land wherever that word's speech
    happens to be, and the audio after the cut would begin mid-sound while the
    text still names the word. An earlier version of the segmenter lost these
    nulls by gluing unreadable morphemes onto their neighbour, and about 6% of
    lines trained on mismatched text and audio with nothing reporting it.
    """
    wav = tmp_path / "ja.flac"
    _write_wav(wav, 6.0)
    manifest = tmp_path / "ja.jsonl"
    spans = [(0.0, 1.0), (1.2, 2.0), (None, None), (3.0, 3.8), (4.0, 4.6)]
    words = ["あ", "い", "3000", "え", "お"]
    manifest.write_text(
        json.dumps(
            {
                "path": str(wav),
                "duration": 6.0,
                "transcript": "".join(words),
                "words": [{"word": w, "start": s, "end": e} for w, (s, e) in zip(words, spans)],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    texts = _cut_texts(str(manifest), word_separator="")
    # Cutting before "3000" would give "3000えお"; cutting after it, "えお".
    assert texts == {"い3000えお", "お"}, texts
