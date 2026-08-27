"""Collect Japanese transcripts into one text file for train_tokenizer.py.

    python -m training.scripts.prepare_ja_text data/ja_corpus.txt \
        --gol data/gol/metadata.csv --ljspeech data/moe20/metadata.csv

The tokenizer has to be fitted once, before any training, and then left alone:
its vocabulary size fixes `flow_lm.lookup_table.n_bins`, and refitting it makes
the text embedding meaningless, so a finetune has to start over with
`reset_text_embedding` again. Fit it on every transcript you will ever train on,
not just the subset you are validating with.

Supported sources, all optional and combinable:

--gol         GOL's `metadata.csv`: `ID|raw|normalized`, pipe-separated. The
              THIRD column is damaged -- it rewrites the ellipsis as a run of
              ideographic full stops in 44.6% of rows -- so the second is used
              and the third ignored. There is no header row.
--ljspeech    LJSpeech-style `ID|speaker|text` or `ID|text`; the last field is
              the transcript.
--moe-json    A directory of MoeSpeechPlus per-utterance .json files, each with
              `anime_whisper_transcription` and `parakeet_jp_transcription`.
              Both ASR outputs are emitted: the tokenizer should see every
              spelling that could later reach it as a manifest transcript.
--jsonl       Any manifest-shaped jsonl with a "transcript" field.

Every line is passed through ja_text.normalize, which is what the manifest
builder must apply too -- see that module for why.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import typer
from typing_extensions import Annotated

from pocket_tts.utils.text_normalization import normalize_japanese as normalize

logger = logging.getLogger("prepare_ja_text")

app = typer.Typer(pretty_exceptions_show_locals=False)


def _lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8", errors="replace") as f:
        yield from f


def gol_texts(path: Path) -> Iterator[str]:
    """Column 2 of `ID|raw|normalized`. Rows that are not 3 fields are dropped:
    a handful of GOL rows are split across two physical lines, and one contains
    the separator itself, so they cannot be parsed positionally."""
    malformed = 0
    for line in _lines(path):
        parts = line.rstrip("\n").split("|")
        if len(parts) != 3:
            malformed += 1
            continue
        yield parts[1]
    if malformed:
        logger.warning("%s: dropped %d rows that were not 3 pipe-separated fields", path, malformed)


def ljspeech_texts(path: Path) -> Iterator[str]:
    for line in _lines(path):
        parts = line.rstrip("\n").split("|")
        if len(parts) >= 2:
            yield parts[-1]


def moe_json_texts(directory: Path) -> Iterator[str]:
    for p in directory.rglob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("anime_whisper_transcription", "parakeet_jp_transcription"):
            if d.get(key):
                yield d[key]


def jsonl_texts(path: Path) -> Iterator[str]:
    for line in _lines(path):
        line = line.strip()
        if line:
            try:
                text = json.loads(line).get("transcript", "")
            except json.JSONDecodeError:
                continue
            if text:
                yield text


@app.command()
def main(
    output: Annotated[Path, typer.Argument(help="plain-text corpus, one utterance per line")],
    gol: Annotated[list[Path], typer.Option(help="GOL metadata.csv (ID|raw|normalized)")] = [],
    ljspeech: Annotated[list[Path], typer.Option(help="LJSpeech metadata.csv")] = [],
    moe_json: Annotated[list[Path], typer.Option(help="MoeSpeechPlus .json directory")] = [],
    jsonl: Annotated[list[Path], typer.Option(help="manifest jsonl with a transcript field")] = [],
    dedup: Annotated[
        bool,
        typer.Option(
            help="keep each distinct utterance once. 19% of GOL is stock phrases; leaving them "
            "in biases the merges toward them without adding any coverage"
        ),
    ] = True,
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    sources = (
        [(gol_texts, p) for p in gol]
        + [(ljspeech_texts, p) for p in ljspeech]
        + [(moe_json_texts, p) for p in moe_json]
        + [(jsonl_texts, p) for p in jsonl]
    )
    if not sources:
        raise typer.BadParameter("give at least one of --gol/--ljspeech/--moe-json/--jsonl")

    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    chars: set[str] = set()
    written = skipped = 0
    with output.open("w", encoding="utf-8", newline="\n") as out:
        for reader, path in sources:
            before = written
            for raw in reader(path):
                text = normalize(raw)
                if not text:
                    skipped += 1
                    continue
                if dedup:
                    if text in seen:
                        skipped += 1
                        continue
                    seen.add(text)
                chars.update(text)
                out.write(text + "\n")
                written += 1
            logger.info("%s: %d utterances", path, written - before)

    logger.info("wrote %s: %d utterances, %d skipped", output, written, skipped)
    # The count that decides --vocab-size: sentencepiece needs one slot per
    # distinct character at character_coverage 1.0 and fails outright below it.
    logger.info(
        "%d distinct characters -- fit with --vocab-size well above this "
        "(the surplus is what becomes merges), and set the model config's "
        "flow_lm.lookup_table.n_bins to the size you choose",
        len(chars),
    )


if __name__ == "__main__":
    app()
