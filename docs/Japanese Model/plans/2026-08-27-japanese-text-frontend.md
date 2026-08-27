# 日本語テキストフロントエンド 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 日本語モデルが完成した時に実用品質で動くよう、推論側の5つの欠陥と評価側の指標を、音声データなしで直しきる。

**Architecture:** `pocket_tts/utils/config.py` の `Config` に言語別テキスト規則を足し、既定値を英語の現行動作に一致させる。`remove_semicolons` が french/german で使われているのと同じ前例に倣う。新規設定は `TextRules` frozen dataclass にまとめ、`prepare_text_prompt` と `split_into_best_sentences` へキーワード専用引数で渡す。既存の呼び出しは引数を変えずそのまま通る。

**Tech Stack:** Python 3.12 / pydantic v2 (`StrictModel`, `extra="forbid"`) / sentencepiece / pytest / jiwer / whisper_normalizer / ruff

**Spec:** `docs/Japanese Model/specs/2026-08-27-japanese-text-frontend-design.md`

## Global Constraints

- **英語の現行動作を1ビットも変えない。** 新設定の既定値は `normalizer=None`, `sentence_boundaries=".!...?"`, `clause_boundaries=",;:"`, `terminal_punctuation="."`, `segment_separator=" "`。リリース済み config はこれらを設定しないため既定値が効く。
- **`pocket_tts/` は `training/` を import してはならない。** 現状その import は0件。逆向き（`training/` → `pocket_tts/`）のみ許される。
- **`pocket_tts/` に新しい実行時依存を増やさない。** 移設する正規化コードの依存は `re` と `unicodedata` のみ。
- **`Config` は `StrictModel`（`extra="forbid"`）。** 新フィールドには必ず既定値を与える。与えないと既存 YAML が読めなくなる。
- **テストは実装より先に書き、失敗を目で確認してから通す。** 通るだけのテストは何も証明しない。各タスクは実装を意図的に壊して落ちることを確認する手順を含む。
- **音声ファイルは一切使わない。** 全タスクがそれ無しで検証できる。
- 実行環境は Windows。`git config core.autocrlf=true` のため作業ツリーは CRLF、コミットされるのは LF。ruff の検証はインデックスの内容に対して行う（Task 0）。
- コマンドはすべて `uv run` 経由。`ruff` は `uvx ruff`（`uv run ruff` は解決できない）。

---

### Task 0: 検証用ヘルパを用意する

**目的:** 以降の全タスクで「テストが本当にバグを捕まえるか」を確認できるようにする。前回、実装を先に書いたテストが不具合のある挙動を assert したまま通り、訓練データの6%を壊していた。同じ失敗を繰り返さないための道具を先に置く。

**ゴール:** `mutate.py` が CRLF を保ったまま実装を1箇所書き換えられ、`ruff-index.sh` がコミットされる内容に対して lint と format を検証できる。どちらも1回動かして確認済み。

**Files:**
- Create: `scripts/dev/mutate.py`
- Create: `scripts/dev/ruff-index.sh`

**Interfaces:**
- Produces: `python scripts/dev/mutate.py <file> <old> <new>` — `old` がちょうど1箇所であることを確認して `new` に置換し、CRLF を保って書き戻す。0箇所または2箇所以上なら `AssertionError`。
- Produces: `bash scripts/dev/ruff-index.sh <file>...` — `git add` 済みの内容（LF）を一時ディレクトリに展開して `uvx ruff check` と `uvx ruff format --check` を実行。

- [ ] **Step 1: `mutate.py` を書く**

`sed -i` は CRLF ファイルでパターンが一致せず**無言で失敗する**（変異が適用されないのにテストが通り、「テストが弱い」と誤診する）。Python の `read_text` は改行を `\n` に正規化するのでこれを避けられる。

```python
"""Swap one exact string in a source file, preserving CRLF.

Used to check that a test actually catches the bug it describes: break the
implementation, watch the test fail, put it back. sed cannot be used for this
on Windows -- the file is CRLF, sed's pattern silently fails to match, and a
mutation that was never applied looks exactly like a test that is too weak.
"""

import pathlib
import sys

path, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text(encoding="utf-8")  # newline translation: \r\n -> \n
count = text.count(old)
assert count == 1, f"site is not unique ({count} occurrences): {old!r}"
path.write_text(text.replace(old, new), encoding="utf-8", newline="\r\n")
print(f"mutated {path}")
```

- [ ] **Step 2: `ruff-index.sh` を書く**

```bash
#!/usr/bin/env bash
# Lint and format-check the bytes git will actually commit.
#
# core.autocrlf=true means the working tree is CRLF and the index is LF. ruff
# run against the working tree reports every file as needing reformatting,
# which drowns the real findings. Check the index instead.
set -euo pipefail
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for f in "$@"; do
  git show ":$f" > "$tmp/$(basename "$f")"
done
uvx ruff check "$tmp"
uvx ruff format --check "$tmp"
```

- [ ] **Step 3: 両方を1回動かして確認する**

```bash
uv run python scripts/dev/mutate.py training/scripts/align_data.py "if w_idx < 0:" "if w_idx < 99:"
uv run pytest training/tests/test_word_spans.py -q
git checkout -- training/scripts/align_data.py
git add scripts/dev && bash scripts/dev/ruff-index.sh scripts/dev/mutate.py
```

期待: 変異でテストが1件落ち、`git checkout` 後に全件通り、ruff が `All checks passed!` と `1 file already formatted` を出す。

- [ ] **Step 4: コミット**

```bash
git add scripts/dev/mutate.py scripts/dev/ruff-index.sh
git commit -m "Add the two dev helpers this plan leans on

mutate.py swaps one string in a source file so a test can be checked
against a deliberately broken implementation. sed cannot do this here:
the working tree is CRLF, sed's pattern silently fails to match, and a
mutation that was never applied is indistinguishable from a test too
weak to catch it.

ruff-index.sh checks the bytes git will commit rather than the working
tree, which autocrlf leaves as CRLF and ruff reports wholesale as
unformatted."
```

---

### Task 1: 正規化を `pocket_tts/` へ移す

**目的:** `ja_text.py` の docstring は「コーパス・マニフェスト・推論の3つが同じ正規化を通る」と主張しながら、末尾で「推論側は未接続」と自白している。推論側を繋ぐには `pocket_tts/` から呼べる場所に置く必要があるが、`pocket_tts/` は `training/` を import できない。

**ゴール:** `normalize_japanese()` が `pocket_tts/utils/text_normalization.py` に存在し、レジストリから名前で引ける。既存の import 元4箇所が新パスを指し、既存テストが全て通る。`training/scripts/ja_text.py` は消える。

**Files:**
- Create: `pocket_tts/utils/text_normalization.py`
- Create: `tests/test_text_normalization.py`
- Delete: `training/scripts/ja_text.py`
- Modify: `training/scripts/align_data.py`（import 行と `:218` の `"japanese": ja_text.normalize`）
- Modify: `training/scripts/prepare_ja_text.py:38`
- Modify: `training/tests/test_ja_text.py:11`, `training/tests/test_align_data.py:163`

**Interfaces:**
- Produces: `normalize_japanese(text: str) -> str`
- Produces: `NORMALIZERS: dict[str, Callable[[str], str]]` — `{"japanese": normalize_japanese}`
- Produces: `resolve_normalizer(name: str | None) -> Callable[[str], str]` — `None` なら恒等関数。未知の名前は `KeyError`。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_text_normalization.py` を新規作成:

```python
"""The one normalization the corpus, the manifests and inference all share.

Three strings have to come from the same distribution or a run degrades with
nothing reporting why: the corpus the tokenizer was fitted on, the transcript
the DataLoader feeds at train time, and the text a user types at inference.
Sentencepiece encodes all three happily whatever they look like, so the only
symptom of a mismatch is a model that never quite becomes intelligible.

This module lives in pocket_tts/ rather than training/ because inference needs
it and the inference package cannot import from the training package.
"""

import pytest

from pocket_tts.utils.text_normalization import (
    NORMALIZERS,
    normalize_japanese,
    resolve_normalizer,
)


def test_fullwidth_variants_are_folded():
    """A Japanese IME emits these by default. Unfolded they are characters the
    tokenizer was never fitted on, and they encode as <unk>."""
    assert normalize_japanese("ＡＢＣ１２３？！") == "ABC123?!"


def test_the_ellipsis_survives():
    """Bare NFKC turns U+2026 into three ASCII periods: a run of stops the
    model reads aloud, and three sentence boundaries where there was one.
    15.6% of corpus utterances end in it."""
    assert normalize_japanese("ああ…") == "ああ…"


def test_the_standalone_dakuten_does_not_inject_a_space():
    """NFKC turns U+309B into a SPACE plus an orphan combining mark. Injecting
    a space into a writing system that has none is worse than leaving it."""
    assert " " not in normalize_japanese("え゛っ")


def test_the_registry_resolves_by_name():
    assert resolve_normalizer("japanese") is normalize_japanese
    assert set(NORMALIZERS) == {"japanese"}


def test_no_normalizer_is_the_identity():
    """What every released model gets: its text must reach the tokenizer
    exactly as the caller wrote it."""
    identity = resolve_normalizer(None)
    for text in ["Hello world.", "ＡＢＣ", "  spaced  out  "]:
        assert identity(text) == text


def test_an_unknown_normalizer_is_an_error():
    """A typo in a config must not silently fall back to doing nothing."""
    with pytest.raises(KeyError):
        resolve_normalizer("japanesee")
```

- [ ] **Step 2: 失敗を確認する**

Run: `uv run pytest tests/test_text_normalization.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'pocket_tts.utils.text_normalization'`

- [ ] **Step 3: モジュールを移設する**

`training/scripts/ja_text.py` の中身をそのまま `pocket_tts/utils/text_normalization.py` へ移す。関数名を `normalize` → `normalize_japanese` に変え、docstring の「推論側は未接続」の記述を削り、レジストリを足す。`_PROTECTED` / `_RESTORE` / `_CONTROL` / `_SPACE_RUN` と `normalize()` の本体は1文字も変えない。

```python
from collections.abc import Callable

# ... 既存の _PROTECTED / _RESTORE / _CONTROL / _SPACE_RUN はそのまま ...


def normalize_japanese(text: str) -> str:
    """（既存の normalize() の docstring と本体をそのまま）"""
    for char, sentinel in _PROTECTED.items():
        text = text.replace(char, sentinel)
    text = unicodedata.normalize("NFKC", text)
    for sentinel, char in _RESTORE.items():
        text = text.replace(sentinel, char)
    text = _CONTROL.sub("", text)
    return _SPACE_RUN.sub(" ", text).strip()


def _identity(text: str) -> str:
    return text


NORMALIZERS: dict[str, Callable[[str], str]] = {"japanese": normalize_japanese}


def resolve_normalizer(name: str | None) -> Callable[[str], str]:
    """The normalizer a config names. None means leave the text alone.

    An unknown name raises rather than falling back to the identity: a typo in
    a config would otherwise disable normalization silently, and the only
    symptom is a model that never quite becomes intelligible.
    """
    return _identity if name is None else NORMALIZERS[name]
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run pytest tests/test_text_normalization.py -q`
Expected: 6 passed

- [ ] **Step 5: 既存の import 元を4箇所更新し、旧ファイルを消す**

```bash
git rm training/scripts/ja_text.py
```

- `training/scripts/align_data.py`: `from training.scripts import ja_text` → `from pocket_tts.utils.text_normalization import normalize_japanese`、`"japanese": ja_text.normalize` → `"japanese": normalize_japanese`
- `training/scripts/prepare_ja_text.py:38`: `from training.scripts.ja_text import normalize` → `from pocket_tts.utils.text_normalization import normalize_japanese as normalize`
- `training/tests/test_ja_text.py:11`: 同上
- `training/tests/test_align_data.py:163`: 同上

- [ ] **Step 6: 全テストが通ることを確認する**

Run: `uv run pytest tests/ training/tests/ -q -n 3`
Expected: 全件 passed（既存119 + 新規6）

- [ ] **Step 7: 移設先のテストが本当に効くことを確認する**

```bash
uv run python scripts/dev/mutate.py pocket_tts/utils/text_normalization.py \
  'return _identity if name is None else NORMALIZERS[name]' \
  'return NORMALIZERS.get(name, _identity)'
uv run pytest tests/test_text_normalization.py -q
git checkout -- pocket_tts/utils/text_normalization.py
```

期待: `test_an_unknown_normalizer_is_an_error` が落ちる。落ちなければテストが弱い。

- [ ] **Step 8: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh pocket_tts/utils/text_normalization.py tests/test_text_normalization.py
git commit -m "Move the Japanese normalization where inference can reach it

ja_text.py's docstring claims the corpus, the manifests and a user's
inference input all pass through one normalization, then admits at the
bottom that the inference side is not wired up. Wiring it up means
calling this from pocket_tts, and pocket_tts cannot import from
training -- the inference package must not depend on training code.

So the module moves into pocket_tts/utils/, and training imports it from
there. Its only dependencies are re and unicodedata, so nothing new lands
on users of the inference package.

resolve_normalizer raises on an unknown name rather than falling back to
the identity: a typo in a config would otherwise disable normalization
with no symptom but a model that never quite becomes intelligible."
```

---

### Task 2: `TextRules` と `Config` の新フィールド

**目的:** 言語別のテキスト規則を持てるようにする。ただしリリース済みモデルの出力を1ビットも変えてはならない。

**ゴール:** `TextRules()` を引数なしで作ると英語の現行動作と厳密に一致する値が入る。`Config` が5つの新フィールドを持ち、既存の全 YAML がそのまま読める。`Config` から `TextRules` を作る経路が存在する。

**Files:**
- Modify: `pocket_tts/utils/text_normalization.py`（`TextRules` を追加）
- Modify: `pocket_tts/utils/config.py:116-124`（`Config` にフィールド追加）
- Create: `tests/test_text_rules.py`

**Interfaces:**
- Consumes: Task 1 の `pocket_tts/utils/text_normalization.py`
- Produces: `TextRules` — frozen dataclass。`normalizer: str | None = None`, `sentence_boundaries: str = ".!...?"`, `clause_boundaries: str = ",;:"`, `terminal_punctuation: str = "."`, `segment_separator: str = " "`
- Produces: `TextRules.from_config(config) -> TextRules` — classmethod

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_text_rules.py`:

```python
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
```

- [ ] **Step 2: 失敗を確認する**

Run: `uv run pytest tests/test_text_rules.py -q`
Expected: FAIL — `ImportError: cannot import name 'TextRules'`

- [ ] **Step 3: `TextRules` を実装する**

`pocket_tts/utils/text_normalization.py` に追記:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TextRules:
    """How one language's text is normalized, split and punctuated.

    Every default is exactly what tts_model.py did before these settings
    existed, so a config that sets none of them -- which is every released
    config -- produces the same audio it always did.
    """

    normalizer: str | None = None
    sentence_boundaries: str = ".!...?"
    clause_boundaries: str = ",;:"
    terminal_punctuation: str = "."
    segment_separator: str = " "

    @classmethod
    def from_config(cls, config) -> "TextRules":
        return cls(
            normalizer=config.text_normalizer,
            sentence_boundaries=config.sentence_boundaries,
            clause_boundaries=config.clause_boundaries,
            terminal_punctuation=config.terminal_punctuation,
            segment_separator=config.segment_separator,
        )
```

- [ ] **Step 4: `Config` にフィールドを足す**

`pocket_tts/utils/config.py` の `Config`、`remove_semicolons` の次の行に:

```python
    # Per-language text handling. The defaults are what tts_model.py hardcoded
    # before these existed, so every released config -- none of which set them
    # -- keeps its exact behaviour. See pocket_tts/utils/text_normalization.py.
    text_normalizer: str | None = None
    sentence_boundaries: str = ".!...?"
    clause_boundaries: str = ",;:"
    terminal_punctuation: str = "."
    segment_separator: str = " "
```

- [ ] **Step 5: テストが通ることを確認する**

Run: `uv run pytest tests/test_text_rules.py -q`
Expected: 5 passed

- [ ] **Step 6: 既定値のテストが本当に効くことを確認する**

```bash
uv run python scripts/dev/mutate.py pocket_tts/utils/text_normalization.py \
  'terminal_punctuation: str = "."' 'terminal_punctuation: str = "。"'
uv run pytest tests/test_text_rules.py -q
git checkout -- pocket_tts/utils/text_normalization.py
```

期待: `test_the_defaults_are_todays_english_behaviour` と `test_a_released_config_gets_the_english_defaults` が落ちる。

- [ ] **Step 7: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh pocket_tts/utils/text_normalization.py pocket_tts/utils/config.py tests/test_text_rules.py
git commit -m "Give a config somewhere to say how its language writes

Five settings, defaulting to exactly what tts_model.py hardcoded, so the
released configs -- none of which set them -- keep producing the audio
they always did. That claim is now checkable: a test loads every config
in the directory and asserts it resolves to the default rules.

This follows the precedent already in Config rather than inventing a
mechanism: pad_with_spaces_for_short_inputs and remove_semicolons are
the same idea, and french_24l and german already use the latter.

Config is StrictModel(extra=forbid), so every new field carries a
default -- without one, every released config would stop loading."
```

---

### Task 3: `prepare_text_prompt` に規則を通す

**目的:** 学習データの49.4%が無句読点で終わるのに、推論時は `text[-1].isalnum()` で ASCII ピリオドが付く。かな・漢字は `isalnum() == True` なので、日本語入力のほぼ半分が学習時に見たことのない形になる。

**ゴール:** `terminal_punctuation=""` のとき句読点が足されず、既定では従来通り `.` が足される。正規化が `normalizer` の指定通りに走る。既存の呼び出し側とテストは1文字も変えずに通る。

**Files:**
- Modify: `pocket_tts/models/tts_model.py:962-992`
- Create: `tests/test_prepare_text_prompt.py`

**Interfaces:**
- Consumes: Task 2 の `TextRules`、Task 1 の `resolve_normalizer`
- Produces: `prepare_text_prompt(text, pad_with_spaces_for_short_inputs, remove_semicolons, *, rules=TextRules()) -> tuple[str, int]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_prepare_text_prompt.py`:

```python
"""What the model is actually asked to speak, after the frontend has had it.

The text that reaches the tokenizer has to look like the text the model was
trained on. Nothing checks that -- sentencepiece encodes whatever it is given
-- so a mismatch here surfaces only as a model that never quite becomes
intelligible.
"""

import pytest

from pocket_tts.models.tts_model import prepare_text_prompt
from pocket_tts.utils.text_normalization import TextRules

JA = TextRules(
    normalizer="japanese",
    sentence_boundaries=".!...?。…",
    clause_boundaries=",;:、",
    terminal_punctuation="",
    segment_separator="",
)


def test_english_still_gets_its_period():
    """The behaviour every released model depends on."""
    text, _ = prepare_text_prompt("Hello world", False, False)
    assert text == "Hello world."


def test_japanese_gets_no_period():
    """49.4% of training utterances end with no terminal punctuation at all,
    and only 1.12% end in the full stop. Appending anything -- the ASCII
    period or the Japanese one -- puts the text outside that distribution."""
    text, _ = prepare_text_prompt("こんにちは", False, False, rules=JA)
    assert text == "こんにちは"


def test_japanese_keeps_the_punctuation_it_already_has():
    """Fullwidth ? folds to ASCII, which is the form 17.6% of corpus
    utterances actually end in."""
    text, _ = prepare_text_prompt("元気ですか？", False, False, rules=JA)
    assert text == "元気ですか?"


def test_the_normalizer_runs():
    """A Japanese IME emits fullwidth by default; unfolded it is <unk>."""
    text, _ = prepare_text_prompt("ＡＢＣです", False, False, rules=JA)
    assert text == "ABCです"


def test_no_normalizer_leaves_text_alone():
    """Released models must receive their text exactly as written."""
    text, _ = prepare_text_prompt("ＡＢＣ", False, False)
    assert text.startswith("ＡＢＣ")


def test_empty_text_is_still_an_error():
    with pytest.raises(ValueError):
        prepare_text_prompt("", False, False, rules=JA)


def test_text_that_normalizes_to_empty_is_an_error():
    """Fullwidth spaces fold to ASCII ones and strip to nothing. If the empty
    check runs before normalization, text[0] raises IndexError instead."""
    with pytest.raises(ValueError):
        prepare_text_prompt("　　", False, False, rules=JA)
```

- [ ] **Step 2: 失敗を確認する**

Run: `uv run pytest tests/test_prepare_text_prompt.py -q`
Expected: FAIL — `TypeError: prepare_text_prompt() got an unexpected keyword argument 'rules'`

- [ ] **Step 3: `prepare_text_prompt` を書き換える**

`pocket_tts/models/tts_model.py` の import に `from pocket_tts.utils.text_normalization import TextRules, resolve_normalizer` を足し、関数を:

```python
def prepare_text_prompt(
    text: str,
    pad_with_spaces_for_short_inputs: bool,
    remove_semicolons: bool,
    *,
    rules: TextRules = TextRules(),
) -> tuple[str, int]:
    text = resolve_normalizer(rules.normalizer)(text).strip()
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    if remove_semicolons:
        text = text.replace(";", ",")
    number_of_words = len(text.split())
    if number_of_words <= 4:
        frames_after_eos_guess = 3
    else:
        frames_after_eos_guess = 1

    # Make sure it starts with an uppercase letter
    if not text[0].isupper():
        text = text[0].upper() + text[1:]

    # Let's make sure it ends with some kind of punctuation.
    # Kana and kanji are isalnum() too, so a language that does not write a
    # terminal stop sets rules.terminal_punctuation to "" and gets none.
    if text[-1].isalnum():
        text = text + rules.terminal_punctuation

    # The model does not perform well when there are very few tokens, so
    # we can add empty spaces at the beginning to increase the token count.
    if pad_with_spaces_for_short_inputs and len(text.split()) < 5:
        text = " " * 8 + text

    return text, frames_after_eos_guess
```

正規化は空文字判定の**前**に走らせること。全角空白のみの入力は NFKC で半角空白になり `strip()` で空になる。順序を誤ると `text[0]` が `IndexError` を投げる。

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run pytest tests/test_prepare_text_prompt.py -q`
Expected: 7 passed

- [ ] **Step 5: 既存テストが無変更で通ることを確認する**

Run: `uv run pytest tests/ -q -n 3`
Expected: 全件 passed。`tests/test_split_sentences.py` と `tests/test_generation_regressions.py` を1文字も変えずに通ること。

- [ ] **Step 6: テストが本当に効くことを確認する**

```bash
uv run python scripts/dev/mutate.py pocket_tts/models/tts_model.py \
  'text = text + rules.terminal_punctuation' 'text = text + "."'
uv run pytest tests/test_prepare_text_prompt.py -q
git checkout -- pocket_tts/models/tts_model.py
```

期待: `test_japanese_gets_no_period` が落ちる。

- [ ] **Step 7: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh pocket_tts/models/tts_model.py tests/test_prepare_text_prompt.py
git commit -m "Stop appending a full stop to text whose language has none

Kana and kanji are isalnum() in Python, so the rule that ends an
unpunctuated English sentence with a period fires on almost half of all
Japanese input: 49.4% of corpus utterances end with no terminal
punctuation at all, while only 1.12% end in the Japanese full stop. The
fix is to append nothing, not to append the Japanese one -- the training
distribution has no terminal stop to imitate.

The normalizer runs here too, which is the first time a user's inference
text passes through the same normalization as the corpus and the
manifests. Fullwidth ？ and ！ -- what a Japanese IME emits by default --
fold to the ASCII forms the tokenizer was actually fitted on.

It runs before the empty check, not after: fullwidth spaces normalize to
ASCII ones and strip to nothing, and the other order reaches text[0] with
an empty string and raises IndexError instead of the documented error."
```

---

### Task 4: `split_into_best_sentences` に規則を通す

**目的:** 日本語の文境界 `。` `…` がどこにも登録されておらず、87文字・6文の段落が51トークン1チャンクのまま生成され語が脱落する。さらに、分割を有効にすると詰め直しのループが**学習テキストに存在しない空白を注入する**（欠陥5、この修正自身が作り出す）。

**ゴール:** 同じ段落が複数チャンクに割れ、詰め直しで空白が1つも入らず、チャンクを繋ぐと入力が再構成される。英語の分割結果は現行と一致する。

**Files:**
- Modify: `pocket_tts/models/tts_model.py:1027-1093`, `:85-100`（`__init__`）, `:131`付近（`_from_pydantic_config`）, `:651-663`（呼び出し）
- Modify: `tests/test_split_sentences.py`（末尾に追記、冒頭に `from pathlib import Path`）
- Modify: `tests/test_generation_regressions.py:15`, `:29`付近

**Interfaces:**
- Consumes: Task 3 の `prepare_text_prompt(..., rules=...)`
- Produces: `split_into_best_sentences(tokenizer, text_to_generate, max_tokens, pad_with_spaces_for_short_inputs, remove_semicolons, *, rules=TextRules()) -> list[str]`
- Produces: `TTSModel.text_rules: TextRules`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_split_sentences.py` の末尾に追記:

```python
# -- Japanese: written without spaces, and with its own sentence boundaries ----

_JA_PARAGRAPH = (
    "昨夜からずっと気配を探られていた。だが相手の姿は見えない。"
    "こちらから動けば的になる。ならば待つしかないのだろうか。"
)


def _ja_rules():
    from pocket_tts.utils.text_normalization import TextRules

    return TextRules(
        normalizer="japanese",
        sentence_boundaries=".!...?。…",
        clause_boundaries=",;:、",
        terminal_punctuation="",
        segment_separator="",
    )


@pytest.fixture(scope="module")
def ja_tokenizer():
    """The Japanese tokenizer, wrapped in the two surfaces
    split_into_best_sentences uses: a call returning .tokens, and .sp."""
    import sentencepiece as spm
    import torch

    path = Path(__file__).resolve().parents[1] / "data" / "ja" / "tokenizer.model"
    if not path.exists():
        pytest.skip("data/ja/tokenizer.model not built (see docs/Japanese Model/)")
    sp = spm.SentencePieceProcessor(model_file=str(path))

    class _Tok:
        def __init__(self):
            self.sp = sp

        def __call__(self, text):
            return SimpleNamespace(tokens=torch.tensor([sp.encode(text)]))

    return _Tok()


def test_japanese_paragraph_splits_on_the_full_stop(ja_tokenizer):
    """Without 。 in the boundary set this is one 51-token chunk, and
    generation past roughly 90 characters drops words."""
    chunks = split_into_best_sentences(
        ja_tokenizer,
        _JA_PARAGRAPH,
        16,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
        rules=_ja_rules(),
    )
    assert len(chunks) > 1, chunks


def test_japanese_chunks_carry_no_injected_spaces(ja_tokenizer):
    """The defect this fix creates if left alone: the packing loop joins
    segments with " ", and Japanese is written without spaces. Same problem
    the DataLoader's data.word_separator: "" solves on the training side."""
    chunks = split_into_best_sentences(
        ja_tokenizer,
        _JA_PARAGRAPH,
        32,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
        rules=_ja_rules(),
    )
    for chunk in chunks:
        assert " " not in chunk, chunk


def test_japanese_chunks_reconstruct_the_input(ja_tokenizer):
    """Nothing may be dropped or invented on the way through."""
    chunks = split_into_best_sentences(
        ja_tokenizer,
        _JA_PARAGRAPH,
        16,
        pad_with_spaces_for_short_inputs=False,
        remove_semicolons=False,
        rules=_ja_rules(),
    )
    assert "".join(chunks) == _JA_PARAGRAPH
```

冒頭の import に `from pathlib import Path` と `from types import SimpleNamespace` を足すこと。

- [ ] **Step 2: 失敗を確認する**

Run: `uv run pytest tests/test_split_sentences.py -q -k japanese`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'rules'`

- [ ] **Step 3: `split_into_best_sentences` を書き換える**

署名:

```python
def split_into_best_sentences(
    tokenizer,
    text_to_generate: str,
    max_tokens: int,
    pad_with_spaces_for_short_inputs: bool,
    remove_semicolons: bool,
    *,
    rules: TextRules = TextRules(),
) -> list[str]:
    text_to_generate, _ = prepare_text_prompt(
        text_to_generate, pad_with_spaces_for_short_inputs, remove_semicolons, rules=rules
    )
```

境界トークンの2箇所:

```python
    _, *end_of_sentence_tokens = tokenizer(rules.sentence_boundaries).tokens[0].tolist()
    ...
    _, *fallback_tokens = tokenizer(rules.clause_boundaries).tokens[0].tolist()
```

詰め直しの結合:

```python
        else:
            current_chunk += rules.segment_separator + sentence
            current_nb_of_tokens_in_chunk += nb_tokens
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run pytest tests/test_split_sentences.py -q`
Expected: 全件 passed（既存の英語ケースを含む）

- [ ] **Step 5: 呼び出し側と回帰テストを直す**

`TTSModel.__init__` に `text_rules: TextRules = TextRules()` を kwarg で足し `self.text_rules = text_rules` を置く。`_from_pydantic_config` の `cls(...)` に `text_rules=TextRules.from_config(config)` を渡す。`tts_model.py:651` の `split_into_best_sentences(...)` と `:660` の `prepare_text_prompt(...)` に `rules=self.text_rules` を足す。

`tests/test_generation_regressions.py` の偽関数は位置引数の署名を固定しているので `**_` を足す:

```python
    def fake_split_into_best_sentences(
        tokenizer,
        text_to_generate,
        max_tokens,
        pad_with_spaces_for_short_inputs,
        remove_semicolons,
        **_,
    ):
```

同ファイルの `SimpleNamespace(...)` に `text_rules=TextRules()` を足し、`from pocket_tts.utils.text_normalization import TextRules` を import する。

- [ ] **Step 6: 全テストが通ることを確認する**

Run: `uv run pytest tests/ training/tests/ -q -n 3`
Expected: 全件 passed

- [ ] **Step 7: 2つのテストがそれぞれ別の変異を捕まえることを確認する**

```bash
uv run python scripts/dev/mutate.py pocket_tts/models/tts_model.py \
  'current_chunk += rules.segment_separator + sentence' 'current_chunk += " " + sentence'
uv run pytest tests/test_split_sentences.py -q -k japanese
git checkout -- pocket_tts/models/tts_model.py

uv run python scripts/dev/mutate.py pocket_tts/models/tts_model.py \
  'tokenizer(rules.sentence_boundaries)' 'tokenizer(".!...?")'
uv run pytest tests/test_split_sentences.py -q -k japanese
git checkout -- pocket_tts/models/tts_model.py
```

期待: 1つ目で `test_japanese_chunks_carry_no_injected_spaces` が落ち、2つ目で `test_japanese_paragraph_splits_on_the_full_stop` が落ちる。片方の変異で両方落ちるなら、テストが分離できていない。

- [ ] **Step 8: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh pocket_tts/models/tts_model.py tests/test_split_sentences.py tests/test_generation_regressions.py
git commit -m "Split Japanese on its own sentence boundaries, and rejoin without spaces

The boundary set was tokenizer('.!...?') and the fallback tokenizer(',;:').
Japanese has none of those: an 87-character, six-sentence paragraph came
back as a single 51-token chunk, and generation past roughly that length
drops words. With 。 and … added it becomes six segments of at most 11
tokens. 。 is its own tokenizer piece, not folded into larger ones like
です。, which is what makes this work at all.

Fixing that creates a second defect, so both land together: the loop that
packs segments back into chunks joined them with a space, and Japanese is
written without spaces. Until now that path never ran for Japanese
because Japanese never split. It is the same problem the DataLoader's
data.word_separator: '' solves on the training side, and it gets the same
shape of fix.

One existing test changes: the fake in test_generation_regressions pins
the positional signature of split_into_best_sentences, so it gains **_."
```

---

### Task 5: 日本語の推論 config

**目的:** ここまでの5つの設定を、日本語モデルが実際に使う1つのファイルに固定する。

**ゴール:** `pocket_tts/config/japanese_24l.yaml` が読め、そこから作った `TextRules` が意図した5つの値を持ち、`n_bins` がトークナイザの語彙数 8000 と一致する。

**Files:**
- Create: `pocket_tts/config/japanese_24l.yaml`
- Modify: `tests/test_text_rules.py`（末尾に追記）

**Interfaces:**
- Consumes: Task 2 の `TextRules.from_config`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_text_rules.py` の末尾に追記:

```python
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
```

- [ ] **Step 2: 失敗を確認する**

Run: `uv run pytest tests/test_text_rules.py -q -k japanese`
Expected: FAIL — config ファイルが無い

- [ ] **Step 3: config を書く**

`pocket_tts/config/italian_24l.yaml` を土台にする（24層・同じ構造）。変える点:

- 先頭の `# sig:` 行は削る（upstream の署名であり、このファイルのものではない）
- `weights_path` / `weights_path_without_voice_cloning` は日本語モデルが未完成のため、行をコメントアウトし「学習後に置き換える」と1行添える
- `flow_lm.lookup_table.tokenizer_path` を `data/ja/tokenizer.model` に
- `flow_lm.lookup_table.n_bins` を `8000` に
- 末尾に:

```yaml
# Japanese text handling. Every value is measured on the 1.4M-utterance corpus
# -- see docs/Japanese Model/specs/2026-08-27-japanese-text-frontend-design.md
text_normalizer: japanese
sentence_boundaries: ".!...?。…"   # 。 in 22.8% of lines, … ends 15.6% of them
clause_boundaries: ",;:、"          # 、 in 65.3% of lines
terminal_punctuation: ""            # 49.4% end unpunctuated; only 1.12% end in 。
segment_separator: ""               # Japanese is written without spaces
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run pytest tests/test_text_rules.py -q`
Expected: 8 passed

- [ ] **Step 5: 既存 config が全部読めることを再確認する**

Run: `uv run pytest tests/test_text_rules.py::test_every_released_config_still_loads -q`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh tests/test_text_rules.py
git commit -m "Pin what the Japanese config decided, with the measurement beside it

Each of the five settings carries the corpus number that chose it, so the
next reader can tell a measured decision from a guess.

n_bins is 8000 rather than the released 4000 because the corpus has 4,092
distinct characters -- sentencepiece cannot fit a 4000-piece vocabulary
over that alphabet at all, it fails outright.

The weights paths are commented out: there is no Japanese model yet. This
config describes how its text will be handled, and is what the training
run's output will be dropped into."
```

---

### Task 6: CER と濁点を保持する正規化器

**目的:** 日本語は分かち書きしないため、WER は1文字違いの文と全く無関係な文を**どちらも 1.0** と採点する。さらに `EnglishTextNormalizer` は濁点を除去する（が U+304C → か U+304B）。この2つにより、チェコ語モデルの WER 数値を日本語の go/no-go 基準に流用できない。

**ゴール:** 評価が WER と CER を両方出し、`--text-normalizer basic` で濁点が保持され、正規化器を変えると出力ディレクトリ名が変わる。

**Files:**
- Modify: `training/eval/librispeech.py` — `EvalResults`(`:57`), `eval_dir_name`(`:67-89`), `score_items` の正規化器(`:228-237`), 採点(`:487`), 引数定義
- Modify: `training/tests/test_eval_naming.py`（`args` ヘルパに `text_normalizer` の既定値を足す）
- Create: `training/tests/test_cer.py`

**Interfaces:**
- Produces: `build_normalizer(name: str)` — `"english"` / `"basic"`。未知は `ValueError`
- Produces: `EvalResults.cer: float`
- Produces: `--text-normalizer {english,basic}`、既定 `english`

- [ ] **Step 1: 失敗するテストを書く**

`training/tests/test_cer.py`:

```python
"""The metric that can tell a good Japanese generation from a bad one.

Word error rate cannot: Japanese is written without spaces, so an utterance is
one word, and every imperfect generation scores exactly 1.0. Judging the
validation run on WER would mean judging it on a number that is 1.0 whether
the model said almost the right thing or nothing like it.
"""

from types import SimpleNamespace

import jiwer

from training.eval.librispeech import DEFAULT_ASR, build_normalizer, eval_dir_name

REF = "今日はいい天気ですね"
CLOSE = "今日はいい天気ですわ"  # one character differs
UNRELATED = "全然関係のない文章がここにある"


def _args(normalizer: str) -> SimpleNamespace:
    return SimpleNamespace(
        temp=0.7,
        cfg=1.0,
        use_ema=True,
        num_items=None,
        seed=0,
        asr=DEFAULT_ASR,
        prompt_root=None,
        text_normalizer=normalizer,
    )


def test_wer_cannot_tell_these_apart():
    """The reason this task exists. Both score 1.0."""
    assert jiwer.wer(REF, CLOSE) == jiwer.wer(REF, UNRELATED) == 1.0


def test_cer_can():
    assert jiwer.cer(REF, CLOSE) < 0.2
    assert jiwer.cer(REF, UNRELATED) > 1.0


def test_the_english_normalizer_destroys_dakuten():
    """が (U+304C) and か (U+304B) are different sounds. Scoring Japanese
    through this normalizer forgives every voicing error in the language."""
    assert build_normalizer("english")("が") == "か"


def test_the_basic_normalizer_keeps_them():
    assert build_normalizer("basic")("が") == "が"


def test_an_unknown_normalizer_is_an_error():
    import pytest

    with pytest.raises(ValueError):
        build_normalizer("japanese")


def test_the_normalizer_is_in_the_output_directory_name():
    """eval_dir_name's own docstring: anything that changes the numbers belongs
    in the name, or two evals of one checkpoint overwrite each other."""
    assert eval_dir_name(_args("english"), 1000) != eval_dir_name(_args("basic"), 1000)


def test_the_default_name_is_unchanged():
    """Existing eval directories must keep resolving to the same path."""
    assert eval_dir_name(_args("english"), 1000) == "libri_eval_step1000_t0.7_cfg1.0"
```

- [ ] **Step 2: 失敗を確認する**

Run: `uv run pytest training/tests/test_cer.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_normalizer'`

- [ ] **Step 3: 実装する**

`training/eval/librispeech.py` にモジュール階層で:

```python
def build_normalizer(name: str):
    """The text normalizer applied to both sides of the metric.

    'english' is whisper's: it lowercases, expands numbers, and strips
    diacritics -- including the Japanese voiced-sound marks, so が becomes か
    and every voicing error in the language is forgiven. 'basic' keeps them.
    """
    if name == "english":
        from whisper_normalizer.english import EnglishTextNormalizer

        return EnglishTextNormalizer()
    if name == "basic":
        from whisper_normalizer.basic import BasicTextNormalizer

        return BasicTextNormalizer()  # remove_diacritics=False by default
    raise ValueError(f"unknown text normalizer: {name}")
```

`score_items` 内の `normalize = EnglishTextNormalizer()` を `normalize = build_normalizer(args.text_normalizer)` に置き換え、関数冒頭の `from whisper_normalizer.english import EnglishTextNormalizer` を削除。

`EvalResults` に `cer: float` を追加し、採点箇所を:

```python
        wer=jiwer.wer(refs, hyps),
        cer=jiwer.cer(refs, hyps),
```

`eval_dir_name` の `args.seed` の分岐の直後に:

```python
    if args.text_normalizer != "english":
        name += f"_{args.text_normalizer}"
```

引数定義に `--text-normalizer` を `choices=["english", "basic"]`, `default="english"`, help に「日本語など、濁点や声調記号を落としてはいけない言語は basic」と添えて追加。

- [ ] **Step 4: テストが通ることを確認する**

Run: `uv run pytest training/tests/test_cer.py -q`
Expected: 7 passed

- [ ] **Step 5: 既存の eval 命名テストを通す**

Run: `uv run pytest training/tests/test_eval_naming.py -q`

落ちる場合、そのテストの `make_args` ヘルパが作る名前空間に `text_normalizer` が無いため。既定値 `"english"` を足す。

- [ ] **Step 6: テストが本当に効くことを確認する**

```bash
uv run python scripts/dev/mutate.py training/eval/librispeech.py \
  'return BasicTextNormalizer()  # remove_diacritics=False by default' \
  'return BasicTextNormalizer(remove_diacritics=True)'
uv run pytest training/tests/test_cer.py -q
git checkout -- training/eval/librispeech.py
```

期待: `test_the_basic_normalizer_keeps_them` が落ちる。

- [ ] **Step 7: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh training/eval/librispeech.py training/tests/test_cer.py
git commit -m "Score with a metric that can tell Japanese generations apart

Japanese is written without spaces, so an utterance is one word and word
error rate is 1.0 for every imperfect generation: a one-character slip
and a completely unrelated sentence score identically. Character error
rate separates them, 0.10 against 1.40 on the pair in the test.

The normalizer mattered as much as the metric. Whisper's English one
strips diacritics, and in Japanese that means が becomes か -- it forgives
every voicing error in the language. The basic normalizer keeps them.

Both WER and CER are now always reported, so the number comparable with
the Czech run in PR #254 survives alongside the one that is meaningful
here. The normalizer goes into the eval directory name, per that
function's own rule that anything changing the numbers belongs there.

This is what makes the validation run judgeable. Until now the plan's
go/no-go criterion was a number that could not distinguish a working
model from a broken one."
```

---

### Task 7: 文書を実態に合わせる

**目的:** `training-strategy.md` の「未対応の課題（推論側・評価側）」は「モデルが完成しても、これらを直すまでは実用品質になりません」と書いている。直したなら、その記述は嘘になる。

**ゴール:** 文書が現状を正しく述べ、フェーズ1の判定基準が WER ではなく CER になっている。spec と plan への導線がある。

**Files:**
- Modify: `docs/Japanese Model/training-strategy.md` — 「未対応の課題」の節と「フェーズ1」の節
- Modify: `docs/Japanese Model/index.md`

- [ ] **Step 1: 「未対応の課題」の節を書き換える**

節名を「対応済み（推論側・評価側）」に改める。5項目それぞれに、何を測って何を決めたかを1行、対応コミットのタイトルを1行残す。欠陥5（空白注入）は「欠陥2の修正が作り出したもの」と明記する。範囲外として残るもの（日本語評価セットの構築、ASR モデルの選定）を別項に分けて明示する。

- [ ] **Step 2: フェーズ1の判定基準を直す**

「判定できること」に CER を加える。チェコ語の WER 29.5% をトリップワイヤとして参照している箇所を「WER は日本語では機能しないため CER で判定する。チェコ語の数値とは直接比較できない」に改める。2〜3k step の耳による判定はそのまま残す（これは指標に依存しない）。

- [ ] **Step 3: `index.md` に spec と plan へのリンクを足す**

- [ ] **Step 4: リンク切れが無いか確認する**

```bash
grep -ohE '\]\(([^)]+\.md)[^)]*\)' "docs/Japanese Model"/*.md | sed -E 's/\]\(//; s/[)#].*//' | sort -u
```

出力された各パスが実在することを確認する。

- [ ] **Step 5: コミット**

```bash
git add -A
git commit -m "Say that the inference side is fixed, because it is

training-strategy.md warned that a finished model would not be usable
until four defects on the inference and eval side were fixed. They are,
plus a fifth that fixing the second one created, so the warning is now
false and has to go.

Phase 1's go/no-go criterion changes with it. It named the Czech WER from
PR #254 as the thing to compare against; WER cannot measure Japanese at
all, so the criterion is CER and the Czech number is no longer directly
comparable. The 2-3k step trip-wire stays as it was -- judging by ear
whether Japanese phonology has appeared does not depend on the metric."
```

---

## 完了の定義

- [ ] `uv run pytest tests/ training/tests/ -q -n 3` が全件通る
- [ ] 変更した全ファイルで `bash scripts/dev/ruff-index.sh <files>` が通る
- [ ] 各タスクの変異確認手順を実行し、期待したテストが落ちることを確認した
- [ ] `grep -rn "^from training\|^import training" pocket_tts/` が0件
- [ ] `pocket_tts/config/*.yaml` の全ファイルが `load_config` で読める
- [ ] `training/scripts/ja_text.py` が存在しない
