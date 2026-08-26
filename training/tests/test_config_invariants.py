"""The shipped configs must be runnable as-is.

Every defect these catch has actually shipped: a scratch config that stopped
before the quality transition, a batch size a quarter of the floor it needs,
and a teacher path pointing at an architecture the distill step cannot load.
"""

from pathlib import Path

import pytest

from training.args import TrainArgs, _from_dict, load_args
from training.modules.builders import load_model_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCRATCH = CONFIGS / "scratch.yaml"
DISTILL = CONFIGS / "depth_distill.yaml"
JAPANESE = CONFIGS / "finetune_language_ja.yaml"

# Below 64 rows per optimizer step the acoustic-quality transition arrives late
# or not at all, and 400k steps is where expressivity settles (see README).
MIN_EFFECTIVE_BATCH = 64
MIN_SCRATCH_STEPS = 400_000


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_config_parses(path: Path):
    load_args(path)


def test_scratch_reaches_the_effective_batch_floor():
    args = load_args(SCRATCH)
    assert args.batch_size * args.grad_accum_steps >= MIN_EFFECTIVE_BATCH, (
        "scratch must reach 64 rows per step on a single GPU: "
        f"{args.batch_size} x {args.grad_accum_steps}"
    )


def test_scratch_runs_past_the_quality_transition():
    assert load_args(SCRATCH).max_steps >= MIN_SCRATCH_STEPS


def test_scratch_builds_the_reference_teacher_depth():
    args = load_args(SCRATCH)
    config = load_model_config(args.model_config, args.model_overrides)
    assert config.flow_lm.transformer.num_layers == 24


def test_distill_teacher_is_deeper_than_its_student():
    args = load_args(DISTILL)
    student = load_model_config(args.model_config, args.model_overrides)
    teacher = load_model_config(args.distill_teacher_config, args.distill_teacher_overrides)
    assert teacher.flow_lm.transformer.num_layers > student.flow_lm.transformer.num_layers
    # Depth distillation copies every non-backbone tensor, so the rest must match.
    assert teacher.flow_lm.transformer.d_model == student.flow_lm.transformer.d_model


def test_distill_teacher_weights_point_at_the_scratch_run():
    args = load_args(DISTILL)
    assert args.distill_teacher_weights, "the distill config must name a teacher checkpoint"
    # as_posix(): the configs spell the path with forward slashes, but str() on a
    # WindowsPath spells it with backslashes and the substring check never matches.
    assert load_args(SCRATCH).run_dir.as_posix() in args.distill_teacher_weights, (
        "the documented path is scratch -> distill; the teacher checkpoint should come from "
        f"{load_args(SCRATCH).run_dir}"
    )


class TestArgValidation:
    """Misconfigurations that used to run and quietly do the wrong thing."""

    def test_num_ckpt_keep_zero_is_rejected(self):
        with pytest.raises(ValueError, match="num_ckpt_keep"):
            TrainArgs(num_ckpt_keep=0)

    def test_zero_frequencies_are_rejected(self):
        for field in ("valid_freq", "ckpt_freq", "log_freq"):
            with pytest.raises(ValueError, match=field):
                TrainArgs(**{field: 0})

    def test_distillation_without_a_teacher_is_rejected(self):
        with pytest.raises(ValueError, match="teacher"):
            TrainArgs(distill_cfg_coef=1.5, start_from_pretrained=False)

    def test_teacher_config_without_weights_is_rejected(self):
        with pytest.raises(ValueError, match="distill_teacher_weights"):
            TrainArgs(distill_teacher_config="x.yaml")

    def test_unknown_keys_are_rejected(self):
        """A key the parser doesn't recognize is a setting the user thinks is applied."""
        with pytest.raises(ValueError, match="distill_seed_layers"):
            _from_dict(TrainArgs, {"distill_seed_layers": "first"})


def test_japanese_joins_words_without_a_separator():
    """Japanese is written without spaces. Training on space-joined text that no
    user will ever type is a mismatch nothing errors on."""
    assert load_args(JAPANESE).data.word_separator == ""


def test_japanese_starts_its_text_embedding_from_scratch():
    """The released rows index English sentencepiece pieces. Same shape, so a
    strict load would succeed and the run would start on nonsense."""
    args = load_args(JAPANESE)
    assert args.start_from_pretrained and args.reset_text_embedding


def test_japanese_overrides_the_vocabulary_size():
    """4092 distinct characters were measured on the corpus, and sentencepiece
    needs a slot per character at coverage 1.0, so the released 4000 cannot fit.
    n_bins is asserted against the tokenizer at build time."""
    n_bins = load_args(JAPANESE).model_overrides["flow_lm.lookup_table.n_bins"]
    assert n_bins > 4092


def test_japanese_samples_are_not_the_english_defaults():
    """Otherwise every wav written during the run is unreadable as progress."""
    args, default = load_args(JAPANESE), TrainArgs()
    assert args.sample_sentences and args.sample_sentences != default.sample_sentences
