"""The shipped configs must be runnable as-is.

Every defect these catch has actually shipped: a scratch config that stopped
before the quality transition, a batch size a quarter of the floor it needs,
and a teacher path pointing at an architecture the distill step cannot load.
"""

import subprocess
from pathlib import Path

import pytest

from training.args import TrainArgs, _from_dict, load_args
from training.modules.builders import load_model_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCRATCH = CONFIGS / "scratch.yaml"
DISTILL = CONFIGS / "depth_distill.yaml"
JAPANESE = CONFIGS / "finetune_language_ja.yaml"
JAPANESE_PHASE1 = CONFIGS / "finetune_language_ja_phase1.yaml"
# Every Japanese-specific invariant has to hold for BOTH. A phase-1 run that
# differs from production in any of them validates a pipeline nobody will run.
JAPANESE_CONFIGS = [JAPANESE, JAPANESE_PHASE1]

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


@pytest.mark.parametrize("path", JAPANESE_CONFIGS, ids=lambda p: p.name)
def test_japanese_joins_words_without_a_separator(path: Path):
    """Japanese is written without spaces. Training on space-joined text that no
    user will ever type is a mismatch nothing errors on."""
    assert load_args(path).data.word_separator == ""


@pytest.mark.parametrize("path", JAPANESE_CONFIGS, ids=lambda p: p.name)
def test_japanese_starts_its_text_embedding_from_scratch(path: Path):
    """The released rows index English sentencepiece pieces. Same shape, so a
    strict load would succeed and the run would start on nonsense."""
    args = load_args(path)
    assert args.start_from_pretrained and args.reset_text_embedding


@pytest.mark.parametrize("path", JAPANESE_CONFIGS, ids=lambda p: p.name)
def test_japanese_overrides_the_vocabulary_size(path: Path):
    """4092 distinct characters were measured on the corpus, and sentencepiece
    needs a slot per character at coverage 1.0, so the released 4000 cannot fit.
    n_bins is asserted against the tokenizer at build time."""
    n_bins = load_args(path).model_overrides["flow_lm.lookup_table.n_bins"]
    assert n_bins > 4092


@pytest.mark.parametrize("path", JAPANESE_CONFIGS, ids=lambda p: p.name)
def test_japanese_samples_are_not_the_english_defaults(path: Path):
    """Otherwise every wav written during the run is unreadable as progress."""
    args, default = load_args(path), TrainArgs()
    assert args.sample_sentences and args.sample_sentences != default.sample_sentences


# The validation run is priced in the docs at 15k steps, 2.2 h and under $10 on a
# single GPU. Production is 250k. Nothing in train.py takes a step count -- it
# accepts a config path and nothing else -- so the number in the yaml is the only
# thing standing between a half-day validation and a 36-hour bill.
MAX_PHASE1_STEPS = 20_000

# The documented tripwire is an ear judgement at 2-3k steps: if there is no
# Japanese-sounding phonology by then, the pipeline is broken and stopping costs
# $2. A sample_freq above that is a tripwire that cannot fire.
MAX_PHASE1_SAMPLE_FREQ = 1_000


def test_the_validation_config_stops_inside_its_budget():
    """The gap this closes had shipped: phase 1 pointed at a 250k-step config
    while budgeting 15k, so following the documentation started a 36-hour job."""
    assert load_args(JAPANESE_PHASE1).max_steps <= MAX_PHASE1_STEPS


def test_production_still_runs_past_the_quality_transition():
    """The cheap way to satisfy the test above would be to lower the number in
    the production config, which is where the remaining 235k steps buy acoustic
    quality after WER has plateaued."""
    assert load_args(JAPANESE).max_steps >= 250_000


def test_the_validation_config_samples_early_enough_to_be_stopped():
    """A run nobody can hear until step 10000 has already spent the budget the
    tripwire exists to save."""
    assert load_args(JAPANESE_PHASE1).sample_freq <= MAX_PHASE1_SAMPLE_FREQ


def test_the_validation_config_keeps_its_own_run_directory():
    """Sharing one would let a validation run overwrite production checkpoints,
    and `latest_checkpoint()` would then resume production from a 15k-step run."""
    assert load_args(JAPANESE_PHASE1).run_dir != load_args(JAPANESE).run_dir


@pytest.mark.parametrize("path", JAPANESE_CONFIGS, ids=lambda p: p.name)
def test_japanese_configs_reach_the_effective_batch_floor(path: Path):
    """The validation phase runs on a 24 GB card, so it reaches 64 rows through
    grad_accum rather than batch_size. Either way the floor is the same one
    scratch is held to -- below it the acoustic transition arrives late or not
    at all, and a validation run that never gets there proves nothing."""
    args = load_args(path)
    assert args.batch_size * args.grad_accum_steps >= MIN_EFFECTIVE_BATCH, (
        f"{path.name}: {args.batch_size} x {args.grad_accum_steps}"
    )


@pytest.mark.parametrize("path", JAPANESE_CONFIGS, ids=lambda p: p.name)
def test_japanese_configs_read_the_manifests_the_pipeline_writes(path: Path):
    """prepare_moespeech.py --out data/ja writes exactly these two names. A
    config pointing anywhere else fails at training start, after the eight-stage
    preparation has already succeeded."""
    data = load_args(path).data
    assert data.train_jsonl.endswith("train_aligned.jsonl")
    assert data.valid_jsonl.endswith("valid_aligned.jsonl")


def test_the_two_japanese_configs_agree_on_everything_that_is_japanese():
    """Phase 1 exists to predict phase 2. Any drift in the settings that make
    these configs Japanese -- the tokenizer, its vocabulary, the missing word
    separator -- and the validation measures a model nobody will train."""
    a, b = load_args(JAPANESE), load_args(JAPANESE_PHASE1)
    assert a.model_config == b.model_config
    assert a.model_overrides == b.model_overrides
    assert a.data.word_separator == b.data.word_separator
    assert a.data.max_duration_sec == b.data.max_duration_sec
    assert (a.start_from_pretrained, a.reset_text_embedding) == (
        b.start_from_pretrained,
        b.reset_text_embedding,
    )
    assert a.optim.lr == b.optim.lr


REPO = CONFIGS.parents[1]


@pytest.mark.parametrize("path", JAPANESE_CONFIGS, ids=lambda p: p.name)
def test_japanese_configs_name_a_tokenizer_a_fresh_clone_has(path: Path):
    """The training run reads this file at startup, after the eight-stage data
    preparation has already succeeded. Naming a path that only exists on the
    machine the tokenizer was trained on turns hours of preparation on a rented
    box into a FileNotFoundError.

    Asked of git rather than of the filesystem, and the difference is the whole
    test: `data/ja/tokenizer.model` exists on the machine that trained it and on
    no other, so `Path.exists()` answers yes here and yes in no clone. Being
    tracked is the property a rented instance actually depends on.
    """
    named = load_args(path).model_overrides["flow_lm.lookup_table.tokenizer_path"]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", named],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,  # the return code IS the assertion
    )
    assert tracked.returncode == 0, (
        f"{path.name} names {named}, which git does not track -- a fresh clone "
        f"would not have it ({tracked.stderr.strip()})"
    )


@pytest.mark.parametrize("path", JAPANESE_CONFIGS, ids=lambda p: p.name)
def test_the_tokenizer_holds_exactly_the_vocabulary_the_config_claims(path: Path):
    """n_bins sizes the text embedding. Larger than the tokenizer and the extra
    rows never receive a gradient; smaller and a real piece indexes past the end.
    The config has said "must equal the tokenizer's vocab size exactly" since it
    was written, and nothing checked it."""
    import sentencepiece as spm

    args = load_args(path)
    sp = spm.SentencePieceProcessor(
        model_file=str(REPO / args.model_overrides["flow_lm.lookup_table.tokenizer_path"])
    )
    assert sp.get_piece_size() == args.model_overrides["flow_lm.lookup_table.n_bins"]
