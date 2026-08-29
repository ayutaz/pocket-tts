"""The shipped configs must be runnable as-is.

Every defect these catch has actually shipped: a scratch config that stopped
before the quality transition, a batch size a quarter of the floor it needs,
and a teacher path pointing at an architecture the distill step cannot load.
"""

import re
import subprocess
from pathlib import Path

import pytest

from training.args import TrainArgs, _from_dict, load_args
from training.modules.builders import load_model_config
from training.scripts.prepare_gol import MIN_VALID_SPEAKERS

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCRATCH = CONFIGS / "scratch.yaml"
DISTILL = CONFIGS / "depth_distill.yaml"
JAPANESE = CONFIGS / "finetune_language_ja.yaml"
JAPANESE_PHASE1 = CONFIGS / "finetune_language_ja_phase1.yaml"
JAPANESE_M2A = CONFIGS / "finetune_language_ja_m2a.yaml"
# Every Japanese-specific invariant has to hold for ALL of them. A run that
# differs from production in any of them validates a pipeline nobody will run.
# Production is first, and the agreement test below compares the rest to it.
JAPANESE_CONFIGS = [JAPANESE, JAPANESE_PHASE1, JAPANESE_M2A]

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


@pytest.mark.parametrize("path", JAPANESE_CONFIGS[1:], ids=lambda p: p.name)
def test_every_japanese_config_agrees_on_what_is_japanese(path: Path):
    """Every cheaper run exists to predict production. Any drift in the settings
    that make these configs Japanese -- the tokenizer, its vocabulary, the
    missing word separator -- and the cheap run measures a model nobody will
    train.

    Compared against `JAPANESE_CONFIGS[0]`, which is production, rather than
    pairwise: production is the one these are predicting, and a fourth config
    joins by being added to that list rather than by anyone editing this.
    """
    a, b = load_args(JAPANESE_CONFIGS[0]), load_args(path)
    assert a.model_config == b.model_config
    assert a.model_overrides == b.model_overrides
    assert a.data.word_separator == b.data.word_separator
    assert a.data.max_duration_sec == b.data.max_duration_sec
    assert (a.start_from_pretrained, a.reset_text_embedding) == (
        b.start_from_pretrained,
        b.reset_text_embedding,
    )
    assert a.optim.lr == b.optim.lr


# M2a asks one question: was phase 1's overfit caused by 19.5 epochs or by 27
# speakers? It answers it by running 2,451 speakers for about 3.4 epochs and
# watching whether the validation loss turns. Every number below is what makes
# that answer readable, and each of them is a thing phase 1 got wrong.
M2A_STEPS = 40_000
M2A_VALID_FREQ = 1_000


def test_m2a_runs_far_enough_to_see_the_turn():
    """Phase 1's 15,000 steps were 19.5 epochs over 49,200 utterances. The same
    step count over M2a's 745,000 is 0.6 of one, which is nowhere near the range
    the question is about -- the run would end before either candidate cause
    could show itself, and the experiment would report nothing."""
    assert load_args(JAPANESE_M2A).max_steps >= M2A_STEPS


def test_m2a_validates_often_enough_to_see_a_turn():
    """Phase 1 validated every 2,500 steps and its minimum landed at 7,500: three
    points before the curve turned. The turn is the entire readout of this run,
    and three points cannot distinguish a turn from noise."""
    assert load_args(JAPANESE_M2A).valid_freq <= M2A_VALID_FREQ


def test_m2a_keeps_enough_checkpoints_to_find_the_best_one():
    """Phase 1 kept three and the validation minimum at step 7,500 was gone
    before anyone looked at the curve.

    Asserted as a relation between three settings rather than as `num_ckpt_keep
    >= 20`, because 20 is only the right number for this max_steps and this
    ckpt_freq: what the run actually needs is that the last checkpoint written
    is not the only one left, whatever those two become.
    """
    args = load_args(JAPANESE_M2A)
    assert args.num_ckpt_keep * args.ckpt_freq >= args.max_steps, (
        f"{args.max_steps} steps at ckpt_freq {args.ckpt_freq} writes "
        f"{args.max_steps // args.ckpt_freq} checkpoints and only "
        f"{args.num_ckpt_keep} survive"
    )


def test_m2a_keeps_its_own_run_directory():
    """`latest_checkpoint()` resumes from whatever is newest in run_dir, so a run
    sharing one with production or with phase 1 would resume from the other's
    weights -- or become the other's starting point."""
    m2a = load_args(JAPANESE_M2A).run_dir
    assert m2a != load_args(JAPANESE).run_dir
    assert m2a != load_args(JAPANESE_PHASE1).run_dir


def test_m2a_reads_the_corpus_it_was_built_to_measure():
    """M2a is the GOL run. Pointed at data/ja it would train on the 27 speakers
    phase 1 already overfitted and answer the speaker-diversity question with
    phase 1's own corpus -- a run that costs the same and settles nothing."""
    data = load_args(JAPANESE_M2A).data
    assert "ja-gol" in data.train_jsonl and "ja-gol" in data.valid_jsonl, data


def test_the_m2a_prepare_command_holds_out_enough_voices():
    """The command in this config's header is the one an operator types on the
    rented box, and it has to reach the end.

    `--valid-hours` defaults to 10, and how many voices ten hours buy is a
    property of the corpus rather than of that number: `split_by_speaker` takes
    the smallest speakers first, so it is the held-out hours divided by the size
    of the smallest selected one. At the roughly one-hour speaker floor this run
    is designed around, every selected speaker is at least an hour and ten hours
    buy about ten voices -- against a floor of twenty, so `split_across_corpora`
    refuses and the documented command stops without a manifest.

    It stops at the second of the three invocations the two deliberate stops
    force, which is after the tars are already downloaded and unpacked.
    """
    documented = JAPANESE_M2A.read_text(encoding="utf-8")
    named = re.search(r"--valid-hours\s+(\d+(?:\.\d+)?)", documented)
    assert named, "the documented prepare command does not name --valid-hours"
    assert float(named.group(1)) >= MIN_VALID_SPEAKERS, documented


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
