# Tokenizers shipped with this repo

A tokenizer is small enough to commit and load-bearing enough that it has to be.
`data/` is gitignored, so a tokenizer that lives only there exists on the machine
that trained it and nowhere else — and the run that needs it reads it at startup,
*after* the data preparation has already succeeded. On a rented instance that is
hours of work discarded for a 118 KB file.

## `japanese_8000.model`

sentencepiece, 8000 pieces, `character_coverage=1.0`, `normalization_rule=identity`.

Trained on 1.4M deduplicated GOL + MoeSpeech transcripts, normalized by
`pocket_tts.utils.text_normalization.normalize_japanese`.

**Why 8000.** The corpus has 4092 distinct characters after normalization, and at
coverage 1.0 sentencepiece spends one vocabulary slot per character and fails
outright below that count — so the released 4000 cannot be used. Everything past
4092 becomes merges. Measured compression: 1.658 chars/token at 6000, **1.805 at
8000**, 1.961 at 12000. 8000 also leaves room for the characters the remaining
MoeSpeech speakers will add.

**Do not lower `character_coverage`.** CJK invites it, and it is wrong here:
0.9995 was measured to drop 1516 characters to `<unk>`.

`flow_lm.lookup_table.n_bins` must equal 8000 exactly — it sizes the text
embedding, so a larger value leaves rows that never receive a gradient and a
smaller one lets a real piece index past the end.
`test_the_tokenizer_holds_exactly_the_vocabulary_the_config_claims` holds the two
together.

### Rebuilding it

Needs `data/ja/gol_metadata.csv` and `data/ja/moe20_metadata.csv`, which come from
two gated Hugging Face datasets (`midralab/gol-dataset-2k-ljspeech`,
`ayousanz/moe-speech-20speakers-ljspeech`).

```bash
uv sync --group japanese
uv run python -m training.scripts.prepare_ja_text data/ja/corpus.txt \
    --gol data/ja/gol_metadata.csv --ljspeech data/ja/moe20_metadata.csv
uv run python -m training.scripts.train_tokenizer data/ja/tokenizer \
    data/ja/corpus.txt --vocab-size 8000 --character-coverage 1.0 \
    --normalization-rule identity
```

That writes `data/ja/tokenizer.model`. Nothing reads it there — copy it over the
committed one, and the configs pick it up:

```bash
cp data/ja/tokenizer.model training/tokenizers/japanese_8000.model
```

sentencepiece is deterministic given the same corpus and arguments, so this
reproduces the committed file rather than merely something like it. If `git
status` reports the copy as modified, the corpus or the arguments changed —
check which before committing it, because `n_bins` is pinned to 8000 in three
configs and the vocabulary is what it sizes.

### Licence

The transcripts it was trained on come from datasets whose audio may not be
redistributed. A tokenizer is a vocabulary of subword pieces derived from text,
not a copy of that text or of any audio, and publishing it is the practice the
one existing community model follows (`vvolhejn/pocket-tts-czech` ships its
`tokenizer.model` beside its weights). The audio itself stays where it is.
