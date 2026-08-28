# MoeSpeech マニフェスト構築 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MoeSpeech の zip から `data/ja/train_aligned.jsonl` と `valid_aligned.jsonl` を作る、再開可能な1本のスクリプトを完成させる。これが無いと検証ラン（フェーズ1）は開始できない。

**Architecture:** `training/scripts/prepare_moespeech.py` を `prepare_data.py` と同じ規約で作る — typer CLI、各ステージは出力が存在すればスキップ、部分出力は `.partial` に書いて完了時に rename。アライメントは `prepare_data.py` の `align()` をそのまま呼ぶ。ステージは8つで、前半5つは音声を触らず（または展開のみで）、後半3つが実データを生む。

**Tech Stack:** Python 3.12 / typer / pydantic v2 / huggingface_hub 1.28 / sphn（音声読み書き）/ jiwer（相互CER）/ pytest / ruff

**Spec:** `docs/Japanese Model/specs/2026-08-28-moespeech-manifest-design.md`

## Global Constraints

- **実行環境は vast.ai。** インスタンスは preemption と destroy で消える。**どのステージも、途中で死んだ後に同じコマンドを再実行すれば続きから進まなければならない。** 「最初からやり直し」が選択肢に入る設計は不可。
- **24 kHz 変換は行わない。** `training/dataloader.py:59` の `_load_window` が読み込み時にリサンプルする。44.1 kHz のまま保持する。
- **キャラを跨いで連結してはならない。** 連結ファイル内のどこで切っても同一話者であることが、voice prompt とターゲットの話者一致の前提。
- **閾値を推測で書かない。** 転写フィルタの閾値は Task 4 の実測から決める。それ以前のタスクで数値を仮置きしてはいけない。
- **`prepare_data.py` の `align()`（`:72`）を再利用する。** アライメントを実装し直さない。
- マニフェストの1行は `{"path", "duration", "transcript", "start"}` を持つ（`training/dataloader.py:29-34` の `Entry`）。`words` は Task 8 のアライナが足す。
- テストは実装より先に書き、失敗を目で確認してから通す。各タスクは実装を意図的に壊して落ちることを確認する手順を含む。
- **テストは音声ファイルを必要としない形で書く。** 合成した小さな WAV は可、実データセットの取得は不可。
- Python は `uv run` 経由。ruff は `uvx ruff`。`bash scripts/dev/ruff-index.sh <files>` がコミットされる内容を検査する。
- `sed -i` はこのリポジトリで無言で失敗する（作業ツリーが CRLF）。`uv run python scripts/dev/mutate.py <file> <old> <new>` を使う。
- 変異検証の前に必ず `git add <file>` を打つ。打たないと `git checkout --` が HEAD まで戻して未コミットの実装を破壊する。
- ブランチは `japanese-model-training`。切り替えない。

---

!!! success "この計画は完了しています（2026-08-28）"
    8タスクすべてが実装・レビュー済みで、`training/tests/test_prepare_moespeech.py` に
    **約100件**のテストがあります（この計画が予測した28件ではありません — レビューが
    見つけた穴の分だけ増えています）。全体で **300 passed**。

    **ただし実データの MoeSpeech には一度も触れていません。** 下の「この計画が終わっても
    残ること」は、書いた時のまま今も有効です。

    ### 実行中に、この計画を上書きした裁定が7件あります

    実行前に計画をタスク対ごと・タスク単体ごとに走査して矛盾を洗い、以下を裁定しました。
    **本文は当時の記録なので書き換えていません。以下が正です。**

    | | 計画の記述 | 裁定 |
    |---|---|---|
    | R1 | Task 5 の既定値は Task 4 の実測から決める | **既定値は作らない。** 実データがこの開発機から到達不能で `probe.json` が存在しないため、実測から決めることが物理的にできない。`--max-cer` / `--min-mos` は `None` 既定で、`main` は probe を書いて停止する。「測る前に閾値を書かない」という計画自身の規律に従うと、これしかない |
    | R2 | Task 2 は `shutil.copyfile` で取得先に直接コピー | **`.partial` に書いて `os.replace`。** 計画の実装例は Task 2 自身のゴール（「中断で生まれた不完全なファイルを取得済みと誤認しない」）と矛盾していた。コピー中に殺されると、まさにゴールが禁じたファイルが残る |
    | R3 | `concatenate(utterances, out_wav: Path, target_sec)` | `out_wav` は**1本目の名前そのもの**、2本目以降は `<stem>_001.wav`。Task 6 の2つのテストが、片方は `joined.wav` を名前で読み、もう片方は複数パスを要求するため、両者が同時に成立する形はこれだけ |
    | R4 | 「キャラを跨いで連結してはならない」 | **どのテストもそれを主張していなかった。** `concatenate` は話者が複数混ざったリストに対して例外を投げ、それをテストする |
    | R5 | Task 6 の出力は `path` / `start` / `duration` | `id` / `speaker` / `transcript` も要る。Task 6 自身のテストと Task 7 の全体がそれを消費する |
    | R6 | （記述なし） | `select_utterances` は `{"id","speaker","wav","duration","transcript","cer","mos"}` を返す。走査は `glob` ではなく **`rglob`** — テストは JSON を平置きするが、実データは1階層下に入れ子の可能性がある |
    | R7 | 「`prepare_data.py` の `align()` をそのまま呼ぶ」 | 呼べなかった。`align()` に `--segmenter` を渡す口が無く、そのままでは**日本語を whitespace セグメンタでアライメント**する。`segmenter: str = "whitespace"` を足して両分岐で転送した。これは実装のやり直しではなく通し口である |

    加えて、**各タスクの想定テスト数は下限であって一致させるべき数ではありません。** レビューが
    見つけた穴の分だけ増えています。

    ### レビューが見つけた、計画にもテストにも無かったもの

    - **`start == 0.0` をアライナが「ファイル全体」と読んでいた。** 連結ファイルの先頭発話が必ず該当し、約20件に1件が他人の発話ごとアライメントされるところだった。英語パイプラインにもあった上流由来のバグ
    - **話者ラベルが JSON の親ディレクトリ名だった。** クリップが入れ子だと全キャラが同じラベルになり、キャラ跨ぎ連結のガードが素通りする
    - **valid が空でも Task 7 の2つの保証は真になる。** `set() & set() == set()` と空 Counter に対する `all([])`
    - **フレーム数ゼロの wav が全ガードをすり抜ける。** `cursor` が進まないので次のクリップが同じ `start` を持つ
    - **probe の残存率表が過大に約束していた。** 正規化で空になる転写を「残る」と数えていた


### Task 1: キャラ選定（音声を落とさずに規模を決める）

**目的:** `info.csv` は 473 キャラ全ての `num_files` / `total_duration_min` / `f0_mean` を 12.5 KB で持つ。何を落とすかは、音声を1バイトも取得せずに決まる。ここを最初に固めると、以降のステージが扱う集合が確定する。

**ゴール:** `--hours 124` を指定すると、その時間に届くキャラの一覧が `characters.json` に書かれる。同じ引数での再実行は同じ結果を返し（決定的）、既存ファイルがあればスキップする。ネットワークは `info.csv` の取得にしか使わない。

**Files:**
- Create: `training/scripts/prepare_moespeech.py`
- Create: `training/tests/test_prepare_moespeech.py`

**Interfaces:**
- Produces: `select_characters(info_csv: Path, hours: float, order: str) -> list[dict]` — 各要素は `{"name", "num_files", "total_duration_min", "f0_mean"}`。`order` は `"largest"`（既定）または `"random"`。
- Produces: `characters.json` — `{"hours_requested": float, "hours_selected": float, "order": str, "characters": [...]}`

- [x] **Step 1: 失敗するテストを書く**

`training/tests/test_prepare_moespeech.py`:

```python
"""Building a training manifest out of MoeSpeech.

Everything here guards one property: the manifest must describe the audio
truthfully. A wrong `start` or `duration` does not raise -- it trains the model
on speech that does not match its text, and the only symptom is a model that
never quite becomes intelligible.
"""

import csv
import json

from training.scripts.prepare_moespeech import select_characters


def _info_csv(tmp_path, rows):
    """A stand-in for the dataset's info.csv: name, num_files, minutes, f0."""
    path = tmp_path / "info.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "num_files", "total_duration_min", "f0_mean"])
        w.writerows(rows)
    return path


def test_largest_first_reaches_the_target_with_fewest_characters():
    """The measurement that drove this: taking characters largest-first reaches
    124 hours in 28 characters where random order needs about 160. Fewer zips
    is less to download and unpack, which is the phase's real bottleneck."""
    pass  # replaced below -- see the real body


def test_selection_stops_once_the_target_is_reached(tmp_path):
    info = _info_csv(tmp_path, [("a", 100, 60.0, 300.0), ("b", 100, 60.0, 300.0),
                                ("c", 100, 60.0, 300.0)])
    chosen = select_characters(info, hours=1.0, order="largest")
    assert len(chosen) == 1, chosen


def test_largest_first_takes_the_biggest(tmp_path):
    info = _info_csv(tmp_path, [("small", 10, 6.0, 300.0), ("big", 100, 600.0, 300.0)])
    (chosen,) = select_characters(info, hours=1.0, order="largest")
    assert chosen["name"] == "big"


def test_selection_is_deterministic(tmp_path):
    """A re-run after an interrupted download must ask for the same zips."""
    rows = [(f"c{i}", 100, 30.0, 300.0) for i in range(20)]
    info = _info_csv(tmp_path, rows)
    first = select_characters(info, hours=5.0, order="largest")
    second = select_characters(info, hours=5.0, order="largest")
    assert [c["name"] for c in first] == [c["name"] for c in second]


def test_random_order_is_also_deterministic(tmp_path):
    """Reproducibility does not depend on which ordering was chosen."""
    rows = [(f"c{i}", 100, 30.0, 300.0) for i in range(20)]
    info = _info_csv(tmp_path, rows)
    a = select_characters(info, hours=5.0, order="random")
    b = select_characters(info, hours=5.0, order="random")
    assert [c["name"] for c in a] == [c["name"] for c in b]


def test_asking_for_more_than_exists_returns_everything(tmp_path):
    """Rather than raising: the caller asked for an upper bound, not a promise."""
    info = _info_csv(tmp_path, [("a", 10, 6.0, 300.0)])
    assert len(select_characters(info, hours=1000.0, order="largest")) == 1
```

最初のテスト関数（`pass` のもの）は実装しない。実データを要するので、書かずに削除すること。

- [x] **Step 2: 失敗を確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.scripts.prepare_moespeech'`

- [x] **Step 3: 最小限の実装**

```python
"""Turn MoeSpeech's per-character zips into an aligned training manifest.

    python -m training.scripts.prepare_moespeech --hours 124 --out data/ja

MoeSpeech is 621 hours of 44.1 kHz Japanese character speech, published as one
zip per speaker with an ASR transcript, a duration and a speechMOS score beside
each clip. This script selects speakers, fetches only their zips, filters the
utterances on transcript agreement, concatenates each speaker's clips into
pseudo-long recordings, and runs forced alignment over the result.

Every stage skips work whose output already exists, and writes partial output
to a .partial file that is renamed on completion. The script is meant to run on
a preemptible cloud instance: being killed and re-run must always be safe.
"""

import csv
import json
import logging
import random
from pathlib import Path

import typer
from typing_extensions import Annotated

logger = logging.getLogger("prepare_moespeech")
app = typer.Typer(pretty_exceptions_show_locals=False)

SELECTION_SEED = 0  # so --order random is still reproducible across re-runs


def select_characters(info_csv: Path, hours: float, order: str = "largest") -> list[dict]:
    """The speakers whose combined duration first reaches `hours`.

    Reads the dataset's info.csv, which carries num_files, total_duration_min
    and f0_mean for every character -- 12.5 KB that answers "what should I
    download" without fetching any audio.

    Largest-first is the default because it reaches a given number of hours in
    far fewer zips: 124 hours takes 28 characters this way against roughly 160
    at random, and unpacking is this phase's bottleneck rather than GPU time.
    Pass "random" when speaker diversity matters more than download size.
    """
    with open(info_csv, encoding="utf-8") as f:
        rows = [
            {
                "name": r["name"],
                "num_files": int(r["num_files"]),
                "total_duration_min": float(r["total_duration_min"]),
                "f0_mean": float(r["f0_mean"]),
            }
            for r in csv.DictReader(f)
        ]
    if order == "largest":
        rows.sort(key=lambda r: (-r["total_duration_min"], r["name"]))
    elif order == "random":
        rows.sort(key=lambda r: r["name"])  # a stable base for the shuffle
        random.Random(SELECTION_SEED).shuffle(rows)
    else:
        raise typer.BadParameter(f"--order must be 'largest' or 'random', got {order!r}")

    chosen, minutes = [], 0.0
    for row in rows:
        if minutes >= hours * 60:
            break
        chosen.append(row)
        minutes += row["total_duration_min"]
    return chosen
```

- [x] **Step 4: テストが通ることを確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q`
Expected: 5 passed

- [x] **Step 5: テストが本当に効くことを確認する**

```bash
git add training/scripts/prepare_moespeech.py
uv run python scripts/dev/mutate.py training/scripts/prepare_moespeech.py \
  'rows.sort(key=lambda r: (-r["total_duration_min"], r["name"]))' \
  'rows.sort(key=lambda r: (r["total_duration_min"], r["name"]))'
uv run pytest training/tests/test_prepare_moespeech.py -q
git checkout -- training/scripts/prepare_moespeech.py
git diff --stat   # 空であること
```

期待: `test_largest_first_takes_the_biggest` が落ちる。

- [x] **Step 6: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh training/scripts/prepare_moespeech.py training/tests/test_prepare_moespeech.py
git commit -m "Choose which MoeSpeech speakers to fetch, without fetching any

The dataset publishes one zip per character and an info.csv listing every
character's file count, total duration and mean f0 -- 12.5 KB that answers
the only question the download stage needs answered.

Largest-first is the default ordering because it reaches 124 hours in 28
characters where a random draw needs roughly 160. Unpacking is this
phase's bottleneck, not GPU time, so a sixth of the zips is most of the
saving available here. Random ordering stays available for when speaker
diversity matters more, and is seeded so a re-run after an interrupted
download asks for the same set."
```

---

### Task 2: 選んだ zip だけ取得する

**目的:** vast.ai のインスタンスは途中で消える。473 zip / 151 GB のうち必要なのは 28 zip なので、**何が既に手元にあるかを正しく判定できること**が、再実行のたびに全部落とし直さないための条件になる。

**ゴール:** `characters.json` にあるキャラの zip だけを取得し、既に完全な形で存在するものはスキップする。中断で生まれた不完全なファイルを「取得済み」と誤認しない。

**Files:**
- Modify: `training/scripts/prepare_moespeech.py`
- Modify: `training/tests/test_prepare_moespeech.py`

**Interfaces:**
- Consumes: Task 1 の `select_characters`
- Produces: `download_characters(names: list[str], dest: Path, repo: str) -> list[Path]`

- [x] **Step 1: 失敗するテストを書く**

```python
def test_download_skips_what_is_already_complete(tmp_path, monkeypatch):
    """Re-running after an interrupt must not re-fetch 5 GB it already has."""
    from training.scripts import prepare_moespeech as m

    calls = []

    def fake_fetch(repo_id, filename, **kw):
        calls.append(filename)
        p = tmp_path / filename
        p.write_bytes(b"PK\x03\x04fake")
        return str(p)

    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    (tmp_path / "aaa.zip").write_bytes(b"PK\x03\x04already here")

    m.download_characters(["aaa", "bbb"], tmp_path, repo="fake/repo")
    assert calls == ["bbb.zip"], calls


def test_download_returns_a_path_for_every_requested_character(tmp_path, monkeypatch):
    """A caller that gets fewer paths than it asked for would silently train on
    a smaller corpus than intended."""
    from training.scripts import prepare_moespeech as m

    def fake_fetch(repo_id, filename, **kw):
        p = tmp_path / filename
        p.write_bytes(b"PK\x03\x04fake")
        return str(p)

    monkeypatch.setattr(m, "hf_hub_download", fake_fetch)
    paths = m.download_characters(["aaa", "bbb", "ccc"], tmp_path, repo="fake/repo")
    assert [p.name for p in paths] == ["aaa.zip", "bbb.zip", "ccc.zip"]
```

- [x] **Step 2: 失敗を確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q -k download`
Expected: FAIL — `AttributeError: module has no attribute 'download_characters'`

- [x] **Step 3: 実装**

```python
from huggingface_hub import hf_hub_download


def download_characters(names: list[str], dest: Path, repo: str = DATASET_REPO) -> list[Path]:
    """Fetch one zip per character, skipping those already present.

    huggingface_hub downloads to a cache and only writes the final path once
    the transfer completes, so a file existing here means it arrived whole --
    an interrupted download leaves an incomplete file in the cache, not here.
    """
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        local = dest / f"{name}.zip"
        if local.exists():
            logger.info(f"{local.name} already present, skipping")
        else:
            fetched = hf_hub_download(repo, f"{name}.zip", repo_type="dataset")
            shutil.copyfile(fetched, local)
        paths.append(local)
    return paths
```

`DATASET_REPO = "ayousanz/moe-speech-plus"` をモジュール先頭に定義すること。

- [x] **Step 4: テストが通ることを確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q`
Expected: 7 passed

- [x] **Step 5: 変異で確認する**

```bash
git add training/scripts/prepare_moespeech.py
uv run python scripts/dev/mutate.py training/scripts/prepare_moespeech.py \
  '        if local.exists():' '        if False:'
uv run pytest training/tests/test_prepare_moespeech.py -q -k download
git checkout -- training/scripts/prepare_moespeech.py
git diff --stat
```

期待: `test_download_skips_what_is_already_complete` が落ちる。

- [x] **Step 6: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh training/scripts/prepare_moespeech.py training/tests/test_prepare_moespeech.py
git commit -m "Fetch only the chosen speakers' zips, and only once

The instance this runs on can be reclaimed at any moment, so the download
stage is re-entered often. Twenty-eight zips is roughly 30 GB; re-fetching
them because the stage could not tell what it already had would cost more
than everything else in this phase combined.

A zip in the destination directory means it arrived whole:
huggingface_hub assembles its download in a cache and only produces a
final path once the transfer completes, so a kill mid-transfer leaves an
incomplete file there rather than here."
```

---

### Task 3: 展開

**目的:** 30 GB の展開は数十分かかる。ここで中断されたとき、どのキャラが完了していてどれが途中だったかを区別できないと、全部やり直しになる。

**ゴール:** キャラごとに展開し、完了したキャラは印を持つ。中断して再実行すると、完了済みはスキップし、途中だったものは作り直す。

**Files:**
- Modify: `training/scripts/prepare_moespeech.py`
- Modify: `training/tests/test_prepare_moespeech.py`

**Interfaces:**
- Produces: `extract_character(zip_path: Path, dest_root: Path) -> Path` — 展開先ディレクトリを返す

- [x] **Step 1: 失敗するテストを書く**

```python
def _make_zip(path, names):
    """A zip holding `names`, each a tiny file."""
    import zipfile

    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(n, "x")
    return path


def test_extract_writes_every_member(tmp_path):
    from training.scripts.prepare_moespeech import extract_character

    z = _make_zip(tmp_path / "spk.zip", ["a.wav", "a.json", "b.wav", "b.json"])
    out = extract_character(z, tmp_path / "extracted")
    assert sorted(p.name for p in out.iterdir()) == ["a.json", "a.wav", "b.json", "b.wav"]


def test_extract_skips_a_character_already_done(tmp_path):
    """Unpacking 30 GB is tens of minutes; re-running must not redo it."""
    from training.scripts.prepare_moespeech import extract_character

    z = _make_zip(tmp_path / "spk.zip", ["a.wav"])
    out = extract_character(z, tmp_path / "extracted")
    (out / "a.wav").write_text("edited")          # prove it is not rewritten
    extract_character(z, tmp_path / "extracted")
    assert (out / "a.wav").read_text() == "edited"


def test_an_interrupted_extraction_is_redone(tmp_path):
    """The failure this guards: a directory that exists but is incomplete must
    not be mistaken for a finished one, or the corpus silently shrinks."""
    from training.scripts.prepare_moespeech import extract_character

    z = _make_zip(tmp_path / "spk.zip", ["a.wav", "b.wav"])
    half = tmp_path / "extracted" / "spk"
    half.mkdir(parents=True)
    (half / "a.wav").write_text("partial")        # no completion marker

    out = extract_character(z, tmp_path / "extracted")
    assert sorted(p.name for p in out.iterdir() if p.suffix == ".wav") == ["a.wav", "b.wav"]
```

- [x] **Step 2: 失敗を確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q -k extract`
Expected: FAIL — `ImportError: cannot import name 'extract_character'`

- [x] **Step 3: 実装**

完了マーカー（例: 展開先に `.complete` を置く）で「完了」と「途中」を区別する。ディレクトリの存在だけで判定してはいけない — それが3つ目のテストが記述している失敗。

- [x] **Step 4: テストが通ることを確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q`
Expected: 10 passed

- [x] **Step 5: 変異で確認する**

完了マーカーの確認を「ディレクトリの存在確認」に置き換える変異を当て、`test_an_interrupted_extraction_is_redone` が落ちることを確認する。手順は Task 2 Step 5 と同じ（`git add` を先に打つ）。

- [x] **Step 6: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh training/scripts/prepare_moespeech.py training/tests/test_prepare_moespeech.py
git commit -m "Unpack per speaker, and know which ones finished

Thirty gigabytes takes tens of minutes to unpack, and the instance can go
away in the middle of it. A directory that exists is not evidence that its
zip was fully extracted -- the process may have been killed halfway -- so
completion is recorded explicitly and a directory without that record is
redone rather than trusted. Mistaking a half-extracted speaker for a
finished one would quietly shrink the corpus with nothing to show for it."
```

---

### Task 4: 実測（閾値はここで決まる）

**目的:** 転写の採否と品質フィルタの閾値は、**分布を見ないと決められない**。2系統の ASR 転写があり、相互 CER が転写品質の代理指標になる（手動転写は存在しない）。この計画がここまで守ってきた規律 — すべての決定に実測を添える — をここで適用する。

**ゴール:** 展開済みのキャラから全 JSON を読み、クリップ長・相互 CER・speechMOS の分布と、各閾値での残存率を `probe.json` に書く。**このタスクは閾値を決めない。次のタスクが決めるための材料を出す。**

**Files:**
- Modify: `training/scripts/prepare_moespeech.py`
- Modify: `training/tests/test_prepare_moespeech.py`

**Interfaces:**
- Produces: `probe_utterances(root: Path) -> dict` — `{"count", "duration": {...}, "cer": {...}, "mos": {...}, "retention": [...]}`
- Produces: `read_annotation(path: Path) -> dict | None` — 1つの JSON を読み、必要フィールドを欠くものは `None`

- [x] **Step 1: 失敗するテストを書く**

```python
def _annotation(tmp_path, name, whisper, parakeet, duration=5.0, mos=3.5):
    p = tmp_path / f"{name}.json"
    p.write_text(
        json.dumps({
            "anime_whisper_transcription": whisper,
            "parakeet_jp_transcription": parakeet,
            "duration": duration,
            "speechMOS": mos,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    return p


def test_identical_transcripts_score_zero_cer(tmp_path):
    """Two ASRs agreeing is the strongest signal available that the transcript
    is right -- there is no manual transcription to compare against."""
    from training.scripts.prepare_moespeech import read_annotation

    a = read_annotation(_annotation(tmp_path, "a", "こんにちは", "こんにちは"))
    assert a["cer"] == 0.0


def test_disagreeing_transcripts_score_high_cer(tmp_path):
    from training.scripts.prepare_moespeech import read_annotation

    a = read_annotation(_annotation(tmp_path, "a", "こんにちは", "全然違う文章です"))
    assert a["cer"] > 0.5


def test_an_annotation_missing_a_field_is_dropped(tmp_path):
    """Rather than defaulting: a missing duration would become a wrong
    manifest entry, and the loader would read a window that is not there."""
    from training.scripts.prepare_moespeech import read_annotation

    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"anime_whisper_transcription": "あ"}), encoding="utf-8")
    assert read_annotation(p) is None


def test_probe_reports_retention_at_several_thresholds(tmp_path):
    """The output that decides the next task's defaults."""
    from training.scripts.prepare_moespeech import probe_utterances

    for i in range(10):
        _annotation(tmp_path, f"u{i}", "こんにちは", "こんにちは" if i < 7 else "違う")
    stats = probe_utterances(tmp_path)
    assert stats["count"] == 10
    assert any(r["kept"] == 7 for r in stats["retention"]), stats["retention"]


def test_probe_survives_a_corrupt_json(tmp_path):
    """One bad file in 400,000 must not end a 40-minute pass."""
    from training.scripts.prepare_moespeech import probe_utterances

    _annotation(tmp_path, "good", "あ", "あ")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert probe_utterances(tmp_path)["count"] == 1
```

- [x] **Step 2: 失敗を確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q -k "annotation or probe or cer"`
Expected: FAIL — `ImportError: cannot import name 'read_annotation'`

- [x] **Step 3: 実装**

`jiwer.cer` で相互 CER を取る（`training/eval/librispeech.py` が同じライブラリを使っている）。`probe.json` には分布（min / 中央値 / 各パーセンタイル / max）と、複数の閾値それぞれでの残存数を書く。

- [x] **Step 4: テストが通ることを確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q`
Expected: 15 passed

- [x] **Step 5: 変異で確認する**

`read_annotation` の必須フィールド検査を外す変異を当て、`test_an_annotation_missing_a_field_is_dropped` が落ちることを確認する。

- [x] **Step 6: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh training/scripts/prepare_moespeech.py training/tests/test_prepare_moespeech.py
git commit -m "Measure the corpus before deciding what to keep of it

Every clip carries two independent ASR transcriptions and a speechMOS
score, and no manual transcription exists to check either against. What
the two ASRs disagree about is therefore the best available proxy for
where the transcript is wrong, and this stage reports that disagreement's
distribution rather than assuming a threshold for it.

It decides nothing. It exists so the next stage's cutoffs come from the
corpus instead of from a guess, which is the rule the rest of this
project has been held to.

An annotation missing a field is dropped rather than defaulted: a missing
duration would reach the manifest as a window the audio does not contain,
and the loader would read silence and train on it as speech."
```

---

### Task 5: 発話の採否

**目的:** Task 4 の実測を閾値に変換する。ここで初めて「どの転写を使い、何を捨てるか」が決まる。

**ゴール:** `probe.json` の実測に基づいた既定値を持ち、`--max-cer` と `--min-mos` で上書きできる。採用された発話が `utterances.jsonl` に書かれ、**なぜその既定値なのかがコード中の1行で説明されている**。

**Files:**
- Modify: `training/scripts/prepare_moespeech.py`
- Modify: `training/tests/test_prepare_moespeech.py`

**Interfaces:**
- Consumes: Task 4 の `read_annotation`
- Produces: `select_utterances(root: Path, max_cer: float, min_mos: float) -> Iterator[dict]`

- [x] **Step 1: 失敗するテストを書く**

```python
def test_utterances_over_the_cer_limit_are_dropped(tmp_path):
    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "agree", "こんにちは", "こんにちは")
    _annotation(tmp_path, "differ", "こんにちは", "全然違う文章です")
    kept = list(select_utterances(tmp_path, max_cer=0.2, min_mos=0.0))
    assert [u["id"] for u in kept] == ["agree"]


def test_utterances_below_the_mos_floor_are_dropped(tmp_path):
    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "clean", "あ", "あ", mos=4.0)
    _annotation(tmp_path, "noisy", "あ", "あ", mos=1.0)
    kept = list(select_utterances(tmp_path, max_cer=1.0, min_mos=3.0))
    assert [u["id"] for u in kept] == ["clean"]


def test_the_kept_transcript_is_normalized(tmp_path):
    """The manifest transcript, the tokenizer corpus and a user's inference
    input have to be the same distribution. align_data normalizes what it
    writes back; this must match, or the two disagree from the start."""
    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "wide", "ＡＢＣです", "ＡＢＣです")
    (u,) = list(select_utterances(tmp_path, max_cer=1.0, min_mos=0.0))
    assert u["transcript"] == "ABCです"


def test_an_empty_transcript_is_dropped(tmp_path):
    """A zero-length transcript aligns to nothing and trains on nothing."""
    from training.scripts.prepare_moespeech import select_utterances

    _annotation(tmp_path, "empty", "", "")
    assert list(select_utterances(tmp_path, max_cer=1.0, min_mos=0.0)) == []
```

- [x] **Step 2: 失敗を確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q -k "cer_limit or mos_floor or normalized or empty_transcript"`
Expected: FAIL — `ImportError: cannot import name 'select_utterances'`

- [x] **Step 3: 実装**

`pocket_tts.utils.text_normalization.normalize_japanese` を使って転写を正規化する（`align_data.py` の `NORMALIZERS["japanese"]` と同じ関数）。

既定値は Task 4 の `probe.json` の実測から決め、**その根拠を1行のコメントで残す**。実測前にこの値を書いてはいけない。

- [x] **Step 4: テストが通ることを確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q`
Expected: 19 passed

- [x] **Step 5: 変異で確認する**

正規化の呼び出しを外す変異を当て、`test_the_kept_transcript_is_normalized` が落ちることを確認する。

- [x] **Step 6: コミット**

コミットメッセージには、**実測した閾値と、それを選んだ理由**を書くこと。

---

### Task 6: 連結（この計画で最も壊れやすい部分）

**目的:** キャラ別の平均クリップ長は中央値 5.8 秒で、`MIN_CUT_SEC=1.0` を両側に引くとターゲットが 4.8 秒未満しか残らない。同一キャラのクリップを連結して擬似長尺ファイルを作る。**ここで `start`/`duration` を1つ間違えると、モデルは音声と一致しないテキストで学習し、何もエラーを出さない。**

**ゴール:** 同一キャラのクリップだけが1本の WAV に連結され、各発話の `start`/`duration` がその WAV の中の正しい位置を指す。キャラを跨ぐ連結は起こり得ない。

**Files:**
- Modify: `training/scripts/prepare_moespeech.py`
- Modify: `training/tests/test_prepare_moespeech.py`

**Interfaces:**
- Produces: `concatenate(utterances: list[dict], out_wav: Path, target_sec: float) -> list[dict]` — 各要素に `path` / `start` / `duration` が入る

- [x] **Step 1: 失敗するテストを書く**

```python
def _wav(path, seconds, hz, sr=44100):
    """A pure tone, so a window can be identified by its frequency."""
    import numpy as np
    import sphn

    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sphn.write_wav(str(path), (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32), sr)
    return path


def test_each_utterance_points_at_its_own_audio(tmp_path):
    """The property everything else rests on. Each source clip is a distinct
    tone, so reading back a window and taking its dominant frequency proves
    whether start/duration point where the manifest claims."""
    import numpy as np
    import sphn

    from training.scripts.prepare_moespeech import concatenate

    tones = [220.0, 440.0, 880.0]
    utts = [
        {"id": f"u{i}", "wav": _wav(tmp_path / f"u{i}.wav", 2.0, hz), "duration": 2.0,
         "transcript": "あ", "speaker": "spk"}
        for i, hz in enumerate(tones)
    ]
    out = concatenate(utts, tmp_path / "joined.wav", target_sec=60.0)
    assert len(out) == 3

    for entry, hz in zip(out, tones):
        wav, sr = sphn.read(entry["path"], start_sec=entry["start"],
                            duration_sec=entry["duration"])
        mono = wav.mean(axis=0)
        freqs = np.fft.rfftfreq(len(mono), d=1 / sr)
        dominant = freqs[np.argmax(np.abs(np.fft.rfft(mono)))]
        assert abs(dominant - hz) < 5, (entry, dominant, hz)


def test_durations_sum_to_the_file_length(tmp_path):
    """A gap nobody accounted for would put every later start off by it."""
    import sphn

    from training.scripts.prepare_moespeech import concatenate

    utts = [
        {"id": f"u{i}", "wav": _wav(tmp_path / f"u{i}.wav", 1.5, 440.0), "duration": 1.5,
         "transcript": "あ", "speaker": "spk"}
        for i in range(4)
    ]
    out = concatenate(utts, tmp_path / "joined.wav", target_sec=60.0)
    wav, sr = sphn.read(str(tmp_path / "joined.wav"))
    assert abs(len(wav[0]) / sr - (out[-1]["start"] + out[-1]["duration"])) < 0.05


def test_a_long_run_is_split_into_several_files(tmp_path):
    """target_sec bounds each file, or one speaker becomes one enormous wav."""
    from training.scripts.prepare_moespeech import concatenate

    utts = [
        {"id": f"u{i}", "wav": _wav(tmp_path / f"u{i}.wav", 2.0, 440.0), "duration": 2.0,
         "transcript": "あ", "speaker": "spk"}
        for i in range(10)
    ]
    out = concatenate(utts, tmp_path / "joined.wav", target_sec=5.0)
    assert len({e["path"] for e in out}) > 1


def test_every_utterance_survives_the_concatenation(tmp_path):
    """Losing one is losing training data with nothing to report it."""
    from training.scripts.prepare_moespeech import concatenate

    utts = [
        {"id": f"u{i}", "wav": _wav(tmp_path / f"u{i}.wav", 1.0, 440.0), "duration": 1.0,
         "transcript": "あ", "speaker": "spk"}
        for i in range(7)
    ]
    out = concatenate(utts, tmp_path / "joined.wav", target_sec=3.0)
    assert sorted(e["id"] for e in out) == sorted(u["id"] for u in utts)
```

- [x] **Step 2: 失敗を確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q -k concat`
Expected: FAIL — `ImportError: cannot import name 'concatenate'`

- [x] **Step 3: 実装**

44.1 kHz のまま書く。無音を挟まない（挟むなら、その分を `start` に必ず反映すること — 2つ目のテストがそれを捕まえる）。

- [x] **Step 4: テストが通ることを確認する**

Run: `uv run pytest training/tests/test_prepare_moespeech.py -q`
Expected: 23 passed

- [x] **Step 5: 変異で確認する（このタスクは2つ当てる）**

```bash
git add training/scripts/prepare_moespeech.py
# (a) start を1つずらす変異 -> test_each_utterance_points_at_its_own_audio が落ちること
# (b) target_sec の分割条件を外す変異 -> test_a_long_run_is_split_into_several_files が落ちること
```

**2つの変異は別々のテストを落とさなければならない。** 片方の変異で両方落ちるなら、テストが分離できていない。

- [x] **Step 6: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh training/scripts/prepare_moespeech.py training/tests/test_prepare_moespeech.py
git commit -m "Join each speaker's clips into pseudo-long recordings

The median character's mean clip is 5.8 seconds and the loader keeps a
second of audio on each side of its cut, so a target of under five seconds
is what training would otherwise see. Only 27 of 473 characters average
eight seconds or more, so there is no subset of long clips to use instead.

This is the most dangerous code in the pipeline: a start or duration that
is wrong by half a second does not raise, it trains the model on speech
that does not match its text, and the only symptom is a model that never
quite becomes intelligible. So the test that matters gives every source
clip a distinct tone and reads each window back to check its frequency --
the manifest's claim about where a clip lives is verified against the
audio itself rather than against the arithmetic that produced it.

Clips are never joined across speakers: everything downstream assumes
that a cut anywhere inside one of these files leaves the voice prompt and
the target in the same voice."
```

---

### Task 7: マニフェスト

**目的:** ここまでの成果を、loader が読める形にする。評価プロトコルが「ある発話で声をクローンし別の発話を合成する」形なので、**valid の話者には複数の発話が要る** — 発話単位で無作為分割すると、その条件を満たさない組が生まれる。

**ゴール:** `train.jsonl` と `valid.jsonl` が書かれ、valid は話者単位で held-out されている。同じ話者が両方に現れない。

**Files:**
- Modify: `training/scripts/prepare_moespeech.py`
- Modify: `training/tests/test_prepare_moespeech.py`

**Interfaces:**
- Produces: `split_by_speaker(entries: list[dict], valid_hours: float) -> tuple[list, list]`
- Produces: `write_manifest(entries: list[dict], path: Path) -> int`

- [x] **Step 1: 失敗するテストを書く**

```python
def test_no_speaker_appears_in_both_splits(tmp_path):
    """A speaker in both makes the valid loss optimistic, and the number that
    is supposed to say 'stop training' stops meaning anything."""
    from training.scripts.prepare_moespeech import split_by_speaker

    entries = [{"speaker": f"s{i%5}", "duration": 600.0} for i in range(50)]
    train, valid = split_by_speaker(entries, valid_hours=1.0)
    assert {e["speaker"] for e in train} & {e["speaker"] for e in valid} == set()


def test_valid_speakers_have_more_than_one_utterance(tmp_path):
    """The eval protocol clones a voice from one utterance and synthesizes
    another, so a speaker with a single entry cannot be scored at all."""
    from training.scripts.prepare_moespeech import split_by_speaker

    entries = [{"speaker": f"s{i%5}", "duration": 600.0} for i in range(50)]
    _, valid = split_by_speaker(entries, valid_hours=1.0)
    from collections import Counter
    counts = Counter(e["speaker"] for e in valid)
    assert all(c > 1 for c in counts.values()), counts


def test_the_split_is_deterministic(tmp_path):
    from training.scripts.prepare_moespeech import split_by_speaker

    entries = [{"speaker": f"s{i%9}", "duration": 300.0} for i in range(90)]
    a, _ = split_by_speaker(entries, valid_hours=1.0)
    b, _ = split_by_speaker(entries, valid_hours=1.0)
    assert [e["speaker"] for e in a] == [e["speaker"] for e in b]


def test_the_manifest_is_utf8_and_reloadable(tmp_path):
    """Windows defaults to cp932; a manifest written through it is unreadable
    and the failure appears far from here."""
    from training.scripts.prepare_moespeech import write_manifest

    path = tmp_path / "m.jsonl"
    write_manifest([{"path": "a.wav", "duration": 1.0, "transcript": "こんにちは",
                     "start": 0.0}], path)
    with open(path, encoding="utf-8") as f:
        assert json.loads(f.readline())["transcript"] == "こんにちは"


def test_the_manifest_carries_what_the_loader_requires(tmp_path):
    """training/dataloader.py's Entry: path, duration, transcript, start."""
    from training.scripts.prepare_moespeech import write_manifest

    path = tmp_path / "m.jsonl"
    write_manifest([{"path": "a.wav", "duration": 1.0, "transcript": "あ",
                     "start": 2.5}], path)
    with open(path, encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert {"path", "duration", "transcript", "start"} <= set(row)
```

- [x] **Step 2〜4:** 失敗を確認 → 実装 → 通ることを確認（28 passed）

- [x] **Step 5: 変異で確認する**

話者単位の分割を発話単位の分割に置き換える変異を当て、`test_no_speaker_appears_in_both_splits` が落ちることを確認する。

- [x] **Step 6: コミット**

```bash
git add -A
bash scripts/dev/ruff-index.sh training/scripts/prepare_moespeech.py training/tests/test_prepare_moespeech.py
git commit -m "Write the manifests, holding out whole speakers

Splitting by utterance would put the same voice on both sides, which
makes the validation loss optimistic about exactly the thing it is there
to measure. It would also break the eval protocol outright: cloning a
voice from one utterance to synthesize another needs a speaker to have
more than one, and a random utterance-level split produces speakers that
have exactly one.

Manifests are written as UTF-8 explicitly. This machine's default is
cp932, and a manifest written through it fails at read time, far from
where it was produced."
```

---

### Task 8: アライメントと端から端の実行

**目的:** ここまでの8ステージを1つのコマンドに束ね、実際に動く状態にする。アライメントは `prepare_data.py` の `align()` を再利用する。

**ゴール:** `python -m training.scripts.prepare_moespeech --hours 124 --out data/ja` が、DL からアライメント済みマニフェストまでを通しで実行する。どのステージで中断しても、同じコマンドの再実行で続きから進む。

**Files:**
- Modify: `training/scripts/prepare_moespeech.py`
- Modify: `training/tests/test_prepare_moespeech.py`

**Interfaces:**
- Consumes: `training.scripts.prepare_data.align`（`:72`）
- Produces: `main(...)` — typer コマンド

- [x] **Step 1: 失敗するテストを書く**

```python
def test_the_pipeline_skips_stages_whose_output_exists(tmp_path, monkeypatch):
    """The property the whole script is designed around: re-running after a
    preemption must not redo finished work."""
    # 各ステージを記録する fake に差し替え、2回実行して
    # 2回目に呼ばれないことを確認する


def test_alignment_uses_the_japanese_segmenter_and_a_kana_model(monkeypatch):
    """align_data refuses a model without hiragana in its vocabulary, but only
    at run time on the instance -- catching it here costs nothing."""
    # align() に渡される引数を記録し、--segmenter japanese と
    # かなを持つチェックポイントであることを確認する
```

実装時に本文を書くこと。「記録する fake」の形は `tests/test_generation_regressions.py` の `monkeypatch.setattr` が手本になる。

- [x] **Step 2〜4:** 失敗を確認 → 実装 → 通ることを確認

- [x] **Step 5: 全体を通す**

```bash
uv run pytest training/tests/test_prepare_moespeech.py -q
uv run pytest tests/ training/tests/ -q -n 3
bash scripts/dev/ruff-index.sh training/scripts/prepare_moespeech.py training/tests/test_prepare_moespeech.py
```

- [x] **Step 6: 使い方を文書化する**

`docs/Japanese Model/training-strategy.md` のフェーズ1に、実際のコマンドと各ステージの所要時間の目安を追記する。**推測の数値を書かない** — 実行して測るまでは「未計測」と書く。

- [x] **Step 7: コミット**

```bash
git add -A
git commit -m "Run the whole thing with one command, and survive being killed

Eight stages from a dataset name to an aligned manifest, each skipping
work whose output already exists. The instance this runs on is
preemptible, so the design constraint throughout was that being killed
and re-run is always safe and never expensive.

Alignment reuses prepare_data.py's align() rather than growing a second
implementation: it already streams into a .partial that --resume picks up,
and only produces the real output once a pass completes."
```

---

## 完了の定義

- [x] `uv run pytest tests/ training/tests/ -q -n 3` が全件通る
- [x] 変更した全ファイルで `bash scripts/dev/ruff-index.sh` が通る
- [x] 各タスクの変異確認を実行し、期待したテストが落ちることを確認した
- [x] Task 6 の2つの変異が**別々の**テストを落とす
- [x] スクリプトがどのステージで中断されても、再実行で続きから進む
- [x] 転写フィルタの閾値が Task 4 の実測に基づき、その根拠がコードに書かれている

## この計画が終わっても残ること

vast.ai 上で実際に実行するのはユーザーであり、私はインスタンスに触れない。したがって
**この計画の成果物は「実データで一度も動いていないスクリプト」である**。テストは
音声なしで書ける範囲を全て覆うが、実データ特有の失敗（想定外の JSON フィールド、
壊れた WAV、キャラごとのディレクトリ構造の揺れ）は初回実行で初めて出る。

初回実行の出力（特に Task 4 の `probe.json` と、アライメントの ok/skipped 比）を
持ち帰れば、そこから次の判断ができる。
