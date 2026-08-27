"""The per-language text settings, and the promise that they change nothing
until a config asks them to.

Every released model is served by the defaults here. If one of them drifts,
that model's output changes and no test of its weights would notice.
"""

import dataclasses
from pathlib import Path

import pytest

from pocket_tts.utils.config import load_config
from pocket_tts.utils.text_normalization import TextRules

CONFIGS = Path(__file__).resolve().parents[1] / "pocket_tts" / "config"


def test_the_defaults_are_todays_english_behaviour():
    """These five values are what tts_model.py had hardcoded. Pinning them is
    what makes "released models are unchanged" a checkable claim."""
    rules = TextRules()
    assert rules.normalizer is None
    assert rules.sentence_boundaries == ".!...?"
    assert rules.clause_boundaries == ",;:"
    assert rules.terminal_punctuation == "."
    assert rules.segment_separator == " "


def test_rules_are_frozen():
    """They are read per chunk during generation; a mutation mid-run would
    change the text partway through."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        TextRules().terminal_punctuation = "。"


def test_every_released_config_still_loads():
    """Config is StrictModel(extra="forbid"). A new field without a default
    would make every released config unreadable."""
    configs = sorted(CONFIGS.glob("*.yaml"))
    assert len(configs) >= 13, configs
    for path in configs:
        load_config(path)


def test_a_released_config_gets_the_english_defaults():
    """None of them set the new fields, so all of them must behave as before."""
    rules = TextRules.from_config(load_config(CONFIGS / "english_2026-04_24l.yaml"))
    assert rules == TextRules()


def test_a_config_can_override_each_field():
    """What a Japanese config will do."""
    config = load_config(CONFIGS / "english_2026-04_24l.yaml")
    config.text_normalizer = "japanese"
    config.sentence_boundaries = ".!...?。…"
    config.clause_boundaries = ",;:、"
    config.terminal_punctuation = ""
    config.segment_separator = ""
    rules = TextRules.from_config(config)
    assert rules.normalizer == "japanese"
    assert rules.terminal_punctuation == ""
    assert rules.segment_separator == ""


def test_the_japanese_config_says_how_japanese_is_written():
    """Every value here was measured on the 1.4M-utterance corpus; see
    docs/Japanese Model/specs/2026-08-27-japanese-text-frontend-design.md."""
    rules = TextRules.from_config(load_config(CONFIGS / "japanese_24l.yaml"))
    assert rules.normalizer == "japanese"
    assert "。" in rules.sentence_boundaries and "…" in rules.sentence_boundaries
    assert "、" in rules.clause_boundaries
    assert rules.terminal_punctuation == ""
    assert rules.segment_separator == ""


def test_the_japanese_config_keeps_the_ascii_boundaries():
    """Normalization folds fullwidth ？！ to ASCII, so the ASCII forms are what
    actually arrive: 17.6% of corpus utterances end in ? and 13.1% in !."""
    rules = TextRules.from_config(load_config(CONFIGS / "japanese_24l.yaml"))
    for char in ".!?":
        assert char in rules.sentence_boundaries


def test_the_japanese_config_matches_its_tokenizer():
    """pocket_tts/conditioners/text.py asserts n_bins == vocab size exactly.
    The Japanese corpus has 4,092 distinct characters, so the released
    n_bins: 4000 cannot be used -- sentencepiece cannot even fit it."""
    config = load_config(CONFIGS / "japanese_24l.yaml")
    assert config.flow_lm.lookup_table.n_bins == 8000
