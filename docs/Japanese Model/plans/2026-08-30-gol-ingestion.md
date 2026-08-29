# GOL 取り込みとフェーズ2a 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `midralab/gol-dataset` を取り込み、MoeSpeech と合流した学習マニフェストを作る。あわせて、相互 CER の代わりになる品質指標（アライメントスコア）を実装する。

**Architecture:** `training/scripts/prepare_gol.py` を `prepare_moespeech.py` と同じ規約で作る。連結と分割は `prepare_moespeech` の関数を再利用し、実装し直さない。品質フィルタは相互 CER ではなく**アライナが既に計算している対数確率**を使い、そのためステージ順が変わる（アライメント後に絞る）。

**Tech Stack:** Python 3.12 / typer / huggingface_hub / sphn / torch（アライナ）/ pytest / ruff

**Spec:** `docs/Japanese Model/specs/2026-08-30-gol-and-phase2-design.md`

## Global Constraints

- **実行は全て vast.ai 上。** どのステージも、途中で死んだ後に同じコマンドを再実行すれば続きから進まなければならない。
- **`prepare_moespeech.py` の `concatenate` と `split_by_speaker` を再利用する。** どちらもコーパスに依存しない。実装し直さない。
- **閾値を推測で書かない。** スコアの閾値はステージ9の実測から決める。それ以前のタスクで数値を仮置きしてはいけない。
- **MoeSpeech の既存の挙動を壊さない。** `prepare_moespeech.py` への変更は追加のみで、324件の既存テストが通り続けること。
- **`align_data.py` の英語パスを壊さない。** スコアの出力は追加フィールドであり、既存の `words` の内容を変えてはならない。
- GOL は **48 kHz 32bit mono**、MoeSpeech は 44.1 kHz。連結は話者内で閉じるので1ファイル内でレートは混ざらない。`_load_window` が読み込み時にリサンプルする。
- マニフェストの1行は `{"path", "duration", "transcript", "start"}` を持つ（`training/dataloader.py:29-34`）。**`path` は posix 区切りで書く**（Windows で書いて Linux で読めない事故が既に1度ある）。
- テストは実装より先に書き、失敗を目で確認してから通す。各タスクは変異を当てて落ちることを確認する手順を含む。
- テストは音声ファイルを必要としない形で書く。合成した小さな WAV は可、実データの取得は不可。
- Python は `uv run`。lint は `bash scripts/dev/ruff-index.sh <files>`（`git add` 後）。
- `sed -i` は無言で失敗する。`uv run python scripts/dev/mutate.py <file> <old> <new>` を使う。
- 変異検証の前に必ず `git add <file>` を打つ。
- ブランチは `japanese-model-training`。切り替えない。

---

### Task 1: アライナがスコアを出す

**目的:** GOL には ASR が1系統しかなく、フェーズ1を支えた相互 CER が使えない。一方 `align_data.py` の Viterbi は `trellis[b, t_end, n]` に「音声がそのテキストを支持する対数確率」を既に持っており、**`-inf` かどうかの判定にだけ使って捨てている**（`:100`）。これを出せば、転写と音声の一致度が追加計算ゼロで全発話について手に入る。相互 CER が「2つの ASR の食い違い」という間接指標だったのに対し、これは直接指標である。

**ゴール:** `align_data` の出力 jsonl が、各行に `score`（生の対数確率）・`frames`・`tokens` を持つ。**正規化はここでは決めない** — 分布を見てから決めるため、素材だけを出す。既存の `words` の内容は1バイトも変わらない。

**Files:**
- Modify: `training/scripts/align_data.py`
- Modify: `training/tests/test_word_spans.py`

**Interfaces:**
- Produces: `batched_word_spans(...) -> tuple[list[list[tuple[int,int]] | None], list[tuple[float,int,int] | None]]` — 2つ目が `(score, frames, tokens)`

- [ ] **Step 1: 失敗するテストを書く**

```python
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
```

`_spans_and_scores` は既存の `_spans` ヘルパを2値返しに広げたもの。

- [ ] **Step 2: 失敗を確認する**

Run: `uv run pytest training/tests/test_word_spans.py -q`
Expected: FAIL — `ValueError: too many values to unpack`

- [ ] **Step 3: 実装**

`batched_word_spans` の返り値を2つ組にする。`results.append(None)` の分岐では
`scores.append(None)` を、成功分岐では `scores.append((trellis_cpu[b, t_end, n].item(), t_end, n))` を積む。
`main` の書き出しで `entry["score"] / ["frames"] / ["tokens"]` を足す（`words` は触らない）。

- [ ] **Step 4: 既存19件が通ることを確認する**

Run: `uv run pytest training/tests/ tests/ -q -n 3`
Expected: 全件通過。**`words` の値が変わっていないことが条件。**

- [ ] **Step 5: 変異で確認する**

`scores.append((...))` を `scores.append((0.0, t_end, n))` に変え、
`test_a_matching_transcript_scores_higher_than_a_wrong_one` が落ちることを確認する。

- [ ] **Step 6: コミット**

```bash
git add -A && bash scripts/dev/ruff-index.sh training/scripts/align_data.py training/tests/test_word_spans.py
git commit -m "Emit the alignment score the trellis already computes

GOL's transcripts come from a single ASR pass, so the mutual-CER filter
that carried phase 1 has no counterpart there. But the aligner already
computes what that filter was approximating: the log-probability that the
audio supports the text. It reads it once to decide whether an utterance
is alignable at all, and then discards it.

The row now carries the raw score and both of its denominators, frames
and tokens, because which normalization the distribution supports is not
knowable before measuring it. Deciding that here would be the guess this
project keeps refusing to make."
```

---

### Task 2: `metadata.tsv` から tar を選ぶ

**目的:** GOL は 7 TB あり、`metadata.tsv`（1.68 GB）が音声を1バイトも落とさずに選定を決められる。MoeSpeech の `info.csv` と同じ役割である。ただし単位が違う —— MoeSpeech は「zip = 1キャラ」だったが、GOL は「tar = 1作品 = 中央値35話者」なので、選定は二段になる。

**ゴール:** `--hours 1000` で、その時間に届く `game_id` の一覧が `games.json` に書かれる。決定的で、既存ファイルがあればスキップする。ネットワークは `metadata.tsv` の取得にしか使わない。

**Files:**
- Create: `training/scripts/prepare_gol.py`
- Create: `training/tests/test_prepare_gol.py`

**Interfaces:**
- Produces: `select_games(metadata_tsv: Path, hours: float) -> list[dict]` — 各要素は `{"game_id", "hours", "speakers", "utterances"}`

- [ ] **Step 1: 失敗するテストを書く**

```python
"""Building a training manifest out of GOL.

GOL is five times MoeSpeech and shaped differently: the download unit is a
whole work rather than one character, and its transcripts come from a single
ASR pass, so the mutual-CER filter that carried MoeSpeech has no counterpart.
"""

import csv
import json


def _metadata(tmp_path, rows):
    """A stand-in for GOL's metadata.tsv: game_id, speaker, text, path, duration."""
    p = tmp_path / "metadata.tsv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["game_id", "speaker", "text", "file_path", "duration"])
        for g, s, d in rows:
            w.writerow([g, s, "あ", f"{g}/{s}/x.wav", d])
    return p


def test_largest_first_reaches_the_target_with_fewest_tars(tmp_path):
    """Measured on the real metadata: 1,000 hours is 18 tars taken largest-first.
    A tar is 11 GB on average, so the ordering decides hours of download."""
    md = _metadata(tmp_path, [("big", "s1", 7200.0), ("big", "s2", 7200.0),
                              ("small", "s3", 60.0)])
    (chosen,) = select_games(md, hours=1.0)
    assert chosen["game_id"] == "big"


def test_selection_stops_once_the_target_is_reached(tmp_path):
    md = _metadata(tmp_path, [(f"g{i}", "s", 3600.0) for i in range(5)])
    assert len(select_games(md, hours=1.0)) == 1


def test_a_game_reports_its_speakers_and_utterances(tmp_path):
    """Both decide what the next stage can use: the speaker filter needs the
    counts, and a tar of one speaker is worth less than a tar of thirty."""
    md = _metadata(tmp_path, [("g", "a", 60.0), ("g", "a", 60.0), ("g", "b", 60.0)])
    (g,) = select_games(md, hours=100.0)
    assert (g["speakers"], g["utterances"]) == (2, 3)


def test_selection_is_deterministic(tmp_path):
    md = _metadata(tmp_path, [(f"g{i}", "s", 3600.0) for i in range(20)])
    a = [g["game_id"] for g in select_games(md, hours=5.0)]
    b = [g["game_id"] for g in select_games(md, hours=5.0)]
    assert a == b


def test_the_hours_column_is_seconds(tmp_path):
    """metadata.tsv's duration is in seconds and --hours is in hours. The same
    unit confusion cost a test in the MoeSpeech pipeline."""
    md = _metadata(tmp_path, [("g", "s", 3600.0)])
    assert len(select_games(md, hours=0.5)) == 1
    assert select_games(md, hours=100.0)[0]["hours"] == 1.0
```

- [ ] **Step 2〜4:** 失敗を確認 → 実装 → 通ることを確認

- [ ] **Step 5: 変異で確認する**

並び順を昇順にする変異を当て、`test_largest_first_reaches_the_target_with_fewest_tars` が落ちることを確認する。

- [ ] **Step 6: コミット**

---

### Task 3: tar の取得と展開

**目的:** MoeSpeech の `download_characters` / `extract_character` は `.zip` を前提にしている。GOL は `.tar` で、しかも1本が中央値 11 GB と桁が違う。中断からの再開が MoeSpeech 以上に効く。

**ゴール:** 選んだ tar だけを取得・展開し、完全なものはスキップする。中断で生まれた不完全なものを完了と誤認しない。

**Files:**
- Modify: `training/scripts/prepare_gol.py`
- Modify: `training/tests/test_prepare_gol.py`

**Interfaces:**
- Produces: `download_games(ids: list[str], dest: Path, repo: str) -> list[Path]`
- Produces: `extract_game(tar_path: Path, dest_root: Path) -> Path`

**実装の指示:** `prepare_moespeech.download_characters` / `extract_character` と**同じ形**にする —— `.partial` に落として `os.replace`、完了マーカーはディレクトリの外、マーカー無しのディレクトリは消して展開し直す。拡張子だけが違う。**共通化するか複製するかは実装者の判断に委ねるが、複製するなら「なぜ共通化しないか」をコメントに残すこと。**

- [ ] **Step 1〜6:** MoeSpeech の Task 2・3 と同じ4つのテスト（スキップ・不完全の誤認・完了マーカー・残骸の除去）を tar 版で書く。変異も同じ。

---

### Task 4: 話者で絞る

**目的:** GOL の話者は 19,349人いるが、**1話者あたり時間の中央値は0.6分**で、1発話しかない話者が3,922人いる。1時間以上持つのは 2,095人だけで、その2,095人が全体の89%を持つ。額面の話者数で設計すると、評価にも連結にも使えない話者を大量に抱えることになる。

**ゴール:** 発話数と合計時間の下限で話者を絞り、`speakers.json` に書く。**下限値は実測から決める** — このタスクは分布を出し、既定値は次のタスクが決める。

**Files:**
- Modify: `training/scripts/prepare_gol.py`
- Modify: `training/tests/test_prepare_gol.py`

**Interfaces:**
- Produces: `probe_speakers(metadata_tsv: Path, game_ids: list[str]) -> dict` — 分布と、各下限での残存表
- Produces: `select_speakers(metadata_tsv: Path, game_ids: list[str], min_utterances: int, min_seconds: float) -> list[str]`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_a_speaker_with_one_utterance_is_never_selected(tmp_path):
    """One utterance cannot be evaluated -- the protocol clones a voice from one
    and synthesizes another -- and cannot be concatenated with itself either.
    3,922 of GOL's 19,349 speakers are in this state."""
    md = _metadata(tmp_path, [("g", "solo", 600.0), ("g", "pair", 300.0), ("g", "pair", 300.0)])
    assert select_speakers(md, ["g"], min_utterances=2, min_seconds=0.0) == ["pair"]


def test_the_retention_table_reports_what_each_floor_keeps(tmp_path):
    """The output that decides the next task's defaults. Measured on the real
    metadata: a 1-hour floor keeps 2,095 of 19,349 speakers and 89% of the audio."""
    rows = [("g", f"s{i}", 3600.0) for i in range(10)] + [("g", f"t{i}", 10.0) for i in range(90)]
    md = _metadata(tmp_path, rows + rows)  # two utterances each
    stats = probe_speakers(md, ["g"])
    assert stats["speakers"] == 100
    assert any(r["min_seconds"] == 3600 and r["kept"] == 10 for r in stats["retention"]), stats


def test_selection_is_bounded_to_the_games_that_were_taken(tmp_path):
    """The extract root accumulates. Asking for fewer games later has to mean
    fewer speakers, or the flag does nothing -- the same failure the MoeSpeech
    pipeline had to be corrected for."""
    md = _metadata(tmp_path, [("kept", "a", 60.0), ("kept", "a", 60.0),
                              ("dropped", "b", 60.0), ("dropped", "b", 60.0)])
    assert select_speakers(md, ["kept"], min_utterances=2, min_seconds=0.0) == ["a"]
```

- [ ] **Step 2〜6:** 失敗を確認 → 実装 → 通ることを確認 → 変異（`min_utterances` の比較を外す）→ コミット

---

### Task 5: 発話を集めて連結する

**目的:** ここまでで「どの tar のどの話者を使うか」が決まる。あとは MoeSpeech と同じ形に落とし込み、**`prepare_moespeech.concatenate` をそのまま呼ぶ**。GOL のクリップ長は中央値4.55秒で MoeSpeech（5.46秒）より短く、連結の必要性はこちらの方が高い。

**ゴール:** 選ばれた話者の発話が `utterances.jsonl` に書かれ、`concatenate` が受け取れる形（`id` / `speaker` / `wav` / `duration` / `transcript`）になっている。連結は既存関数が行う。

**Files:**
- Modify: `training/scripts/prepare_gol.py`
- Modify: `training/tests/test_prepare_gol.py`

**Interfaces:**
- Consumes: `prepare_moespeech.concatenate`
- Produces: `gol_utterances(metadata_tsv, extract_root, speakers, game_ids) -> Iterator[dict]`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_an_utterance_carries_what_concatenate_needs(tmp_path):
    """concatenate is reused as-is, so the keys have to be the ones it reads."""
    md = _metadata(tmp_path, [("g", "spk", 1.5)])
    root = tmp_path / "extracted"
    (root / "g" / "g" / "spk").mkdir(parents=True)
    _wav(root / "g" / "g" / "spk" / "x.wav", 1.5, 440.0)
    (u,) = list(gol_utterances(md, root, ["spk"], ["g"]))
    assert set(u) >= {"id", "speaker", "wav", "duration", "transcript"}
    assert u["speaker"] == "spk" and u["wav"].exists()


def test_the_transcript_is_normalized(tmp_path):
    """The manifest transcript, the tokenizer corpus and a user's input have to
    be the same distribution, exactly as on the MoeSpeech path."""
    md = _metadata(tmp_path, [("g", "spk", 1.0)], text="ＡＢＣです")
    ...
    assert u["transcript"] == "ABCです"


def test_an_utterance_whose_wav_is_missing_is_dropped(tmp_path):
    """metadata.tsv describes 7.4M files; the tars that were actually taken hold
    a subset. A row without its audio is not a training example."""
```

- [ ] **Step 2〜6:** 同上。変異は「正規化の呼び出しを外す」「wav の存在確認を外す」の2つ。

---

### Task 6: マニフェストと、MoeSpeech との合流

**目的:** 2つのコーパスが合流する唯一の点。話者単位の held-out は**両方をまたいで**行う必要があり、`prepare_moespeech.split_by_speaker` がそのまま使える（コーパスに依存しない）。フェーズ1では valid が1話者しかなく停止判断ができなかったので、ここで複数話者を確保する。

**ゴール:** GOL と MoeSpeech の `entries/*.jsonl` を結合し、話者単位で分割して `train.jsonl` / `valid.jsonl` を書く。**valid は20話者以上**。同じ話者が両方に現れない。

**Files:**
- Modify: `training/scripts/prepare_gol.py`
- Modify: `training/tests/test_prepare_gol.py`

**Interfaces:**
- Consumes: `prepare_moespeech.split_by_speaker`, `write_manifest`
- Produces: `merge_entries(dirs: list[Path]) -> list[dict]`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_the_two_corpora_are_split_as_one(tmp_path):
    """A speaker held out of GOL but present in MoeSpeech would be in both
    splits, and neither corpus's own split can see the other."""


def test_the_valid_split_has_many_speakers(tmp_path):
    """Phase 1's validation set was one speaker, and one speaker cannot say when
    to stop -- its loss bottomed at 7,500 while the samples kept improving,
    because the samples use a training voice and the valid set does not.
    For a model whose point is voice cloning, the unseen number is the one that
    matters, so there have to be enough of them to mean something."""
    entries = [{"speaker": f"s{i}", "duration": 600.0} for i in range(200) for _ in range(3)]
    _, valid = split_by_speaker(entries, valid_hours=10.0)
    assert len({e["speaker"] for e in valid}) >= 20
```

- [ ] **Step 2〜6:** 同上

---

### Task 7: スコアで絞る（新しいステージ）

**目的:** ステージ順がフェーズ1と変わる理由がここにある。スコアはアライメント後にしか得られないので、**アライメント済み jsonl を後から絞る**。捨てる分にもアライメントが走るが、2つ目の ASR を回すより桁で安い。

**ゴール:** `*_aligned.jsonl` のスコア分布を `score_probe.json` に書き、閾値で絞った学習用マニフェストを出す。**このタスクは閾値を決めない。** 正規化の取り方（フレーム毎・トークン毎・生）も分布を見てから決める。

**Files:**
- Modify: `training/scripts/prepare_gol.py`
- Modify: `training/tests/test_prepare_gol.py`

**Interfaces:**
- Produces: `probe_scores(aligned_jsonl: Path) -> dict` — 3つの正規化それぞれの分布と残存表
- Produces: `filter_by_score(aligned: Path, out: Path, min_score_per_frame: float | None) -> int`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_the_probe_reports_all_three_normalizations(tmp_path):
    """Raw log-prob scales with both frames and tokens, and which one the
    distribution supports is not knowable before measuring it. Reporting one
    would be choosing it."""
    stats = probe_scores(_aligned(tmp_path, [(-100.0, 200, 10), (-50.0, 100, 5)]))
    assert {"raw", "per_frame", "per_token"} <= set(stats)


def test_a_row_without_a_score_is_kept_and_counted(tmp_path):
    """align_data emits no score for an utterance it could not align, and those
    are already absent from the aligned manifest. A row that somehow has none is
    a different problem and must not be silently filtered as a bad alignment."""


def test_no_default_threshold(tmp_path):
    """The corpus has never been measured. A number written here now would
    afterwards be indistinguishable from a measured one -- which is the failure
    ruling R1 of the MoeSpeech plan existed to prevent, and which the speechMOS
    measurement later vindicated."""
    with pytest.raises(typer.BadParameter):
        filter_by_score(..., min_score_per_frame=None)
```

- [ ] **Step 2〜6:** 同上

---

### Task 8: 通し実行と M2a の設定

**目的:** 9ステージを1コマンドに束ね、中間検証（M2a）を回せる状態にする。

**ゴール:** `python -m training.scripts.prepare_gol --hours 1000 --out data/ja-gol` が通しで実行でき、どのステージで中断しても再実行で続きから進む。`finetune_language_ja_m2a.yaml` が M2a の設定を持つ。

**Files:**
- Modify: `training/scripts/prepare_gol.py`
- Modify: `training/tests/test_prepare_gol.py`
- Create: `training/configs/finetune_language_ja_m2a.yaml`
- Modify: `training/tests/test_config_invariants.py`

**M2a の設定で、フェーズ1と変える点:**

| | フェーズ1 | M2a | 理由 |
|---|---|---|---|
| `max_steps` | 15,000 | **40,000** | valid が turn するかを見る。745k発話なら3.4エポックで、フェーズ1が壊れた19.5には遠い |
| `valid_freq` | 2,500 | **1,000** | turn を捉える解像度 |
| `num_ckpt_keep` | 3 | **20** | フェーズ1では valid 最良の step が記録を見る前に消えていた |
| `--valid-hours` | 1.0 | **10.0** | 20話者以上を確保する |

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_m2a_keeps_enough_checkpoints_to_find_the_best_one(tmp_path):
    """Phase 1 kept three and the validation minimum at step 7,500 was gone
    before anyone looked at the curve."""
    assert load_args(M2A).num_ckpt_keep * load_args(M2A).ckpt_freq >= load_args(M2A).max_steps


def test_m2a_validates_often_enough_to_see_a_turn():
    assert load_args(M2A).valid_freq <= 1000


def test_every_japanese_config_agrees_on_what_is_japanese():
    """M2a joins the two configs the invariant already covers."""
```

- [ ] **Step 2〜7:** 失敗を確認 → 実装 → 通ることを確認 → 全体を通す → 文書化 → コミット

---

## 完了の定義

- [ ] `uv run pytest tests/ training/tests/ -q -n 3` が全件通る
- [ ] 変更した全ファイルで `bash scripts/dev/ruff-index.sh` が通る
- [ ] 各タスクの変異確認を実行し、期待したテストが落ちることを確認した
- [ ] **`align_data` の `words` の値が1バイトも変わっていない**（英語パスの回帰が無い）
- [ ] スクリプトがどのステージで中断されても、再実行で続きから進む
- [ ] スコアの閾値も話者の下限も、実測から決められており、根拠がコードに書かれている

## この計画が終わっても残ること

M2a を実際に回すのはユーザーであり、私はインスタンスに触れない。**この計画の成果物は
「実 GOL データで一度も動いていないスクリプト」である。** フェーズ1では、実データが
`.bak.json` の双子と3階層の入れ子という2つの欠陥を暴いた。GOL でも同種のものが出ると
考えるべきである。

そして M2a が答えるべき問いは1つに絞られている —— **フェーズ1の過学習は、19.5エポックの
せいだったのか、27話者のせいだったのか。** 2,451話者・3.4エポックで valid が turn しなければ
話者数が原因であり、1,000時間で足りる。turn するならエポック数が原因であり、
M2b は 5,000時間以上を要する。**この1点で M2b の規模と費用が決まる。**
