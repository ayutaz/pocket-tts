# 学習戦略とコスト

調査日: 2026-08-27

## 事前学習 vs 継続学習

**結論: 継続学習（finetune）を採用します。** 到達品質は事前学習（scratch）と変わらず、コストとリスクだけが下がるためです。

### 根拠1: 到達品質は変わらない

[PR #254](https://github.com/kyutai-labs/pocket-tts/pull/254) のチェコ語 976h での実測値です。

| step | from scratch | finetune (lr 2e-4) |
|---|---|---|
| 2k | 326% | **29.5%** |
| 10k | 45.7% | **12.0%** |
| 15k | 17.1% | 11.3% |
| 25k | 16.2% | 10.7% |

最終的に**どちらも WER 10〜12% に収束**し、finetune は約2.5倍速く到達するだけです。つまり scratch を選ぶ理由は「品質」ではありません。

### 根拠2: デバッグ可能性（最大の実利）

日本語には後述の地雷（分かち書き問題、アライナのかな変換、ASR 転写のノイズ）があります。

- **finetune なら 10〜15k step（1×H100 で1.5〜2.2時間、$5〜9）で「日本語が喋れているか」を判定できます。**
- scratch は 10k step 時点で WER 45.7% なので、**「パイプラインが壊れている」のか「まだ学習途中」なのかを区別できません**。45k step（約7時間）以上回さないと判定不能です。

パイプラインのバグを $5 で検出できるか、$50 かかるかの差です。

### 根拠3: コスト

| | scratch | finetune |
|---|---|---|
| teacher step 数 | 400k | 250k（WER 飽和は約15k） |
| 1×H100 実時間 | 58h | **36h** |
| 費用（$2〜4/GPU-h） | $120〜230 | **$75〜145** |
| 初期重み | Mimi + 新規 FlowLM | + 英語24L重み（commit pin 済みの公開重み） |
| ライセンス | CC-BY-4.0 | CC-BY-4.0 |

**ライセンス上の差はありません。** scratch でも Mimi コーデックとアーキテクチャは kyutai 由来なので、CC-BY-4.0 の帰属表示は同じく必要です。

### 根拠4: 「英語が邪魔をする」懸念は構造的に小さい

`reset_text_embedding: true` により**テキスト埋め込みは新規初期化**されます。転移するのは backbone と flow head、すなわち**言語非依存の「音声を作る能力」**です。

さらに今回のデータはギャルゲ／アニメのキャラクター演技という偏ったドメインなので、英語重みが持つ汎用的な韻律・音響の事前分布は**むしろ有利に働く**と見ています。

!!! warning "lr は 2e-4 を維持する"
    PR #254 の実測では lr 2e-5 は全区間で遅く、プラトーでも改善しません。text embedding がランダムから始まるため、backbone がそれに合わせて動く必要があるからです。**scratch と同じ 2e-4 が正解**です。

### scratch が正解になる場合

- 日本語音素体系に最適化した表現を一から作る研究目的
- finetune 後に英語アクセント／韻律の残存が**実測で**問題になった場合の対抗手段
- kyutai 重みを使えない事情がある場合（今回は該当なし）

→ **まず finetune、駄目だった時の第二案が scratch** という順序が合理的です。

---

## コスト試算

### GPU（学習本体）

`training/README.md` の**実測** steps/s（24層 teacher、実効バッチ64）を基準にした計算です。

| 構成 | steps/s | finetune 250k | scratch 400k | 蒸留 200k |
|---|---|---|---|---|
| 1×H100-80GB | 1.91 | 36h / 36 GPU-h | 58h / 58 GPU-h | 約29h |
| 2×H100 | 3.35 | 21h / 41 GPU-h | 33h / 66 GPU-h | — |
| 4×H100 | 5.25 | 13h / 53 GPU-h | 21h / 85 GPU-h | — |
| 8×H100 | 6.85 | 10h / 81 GPU-h | 16h / 130 GPU-h | 3〜8h ※ |
| 1×L40S-46GB | 0.77 | 90h / 90 GPU-h | 144h / 144 GPU-h | 約70h |
| 1×L4-23GB | 0.35 | 198h | 315h | 現実的でない |
| RTX 4090/5090（推定） | 0.5〜0.7 | 100〜140h | 160〜220h | — |

※ README は「8×H100 で蒸留 +約3h」と記載していますが、config の `max_steps: 200000` を同 steps/s で回すと 8.1h 相当です。「WER は 50k step でパリティ」との記述があるので、3h は早期打ち切り時の値と読むのが妥当です。予算は 3〜8h で見てください。

!!! tip "GPU-h は1台が最安、8台は速いだけ"
    バッチが分割されて per-GPU バッチが縮むためスケーリング効率が落ちます。急がないなら **1×H100 が最も費用効率が良い**です。

**単価（2026-08 時点の相場）**

| プロバイダ | 構成 | $/GPU-h |
|---|---|---|
| RunPod Community | H100 PCIe | **1.99** |
| RunPod Secure | H100 PCIe / SXM / NVL | 2.89 / 2.99 / 3.19 |
| Lambda | H100 SXM | 3.99〜4.29 |

### シナリオ別の総額

| シナリオ | GPU-h | 費用($) | 費用(円/150円換算) | 実時間 |
|---|---|---|---|---|
| **① 検証**（finetune 15k step、24L のまま蒸留なし） | 2〜3 | **$5〜12** | 約1,000〜2,000円 | **半日** |
| **② 実用版**（finetune 100k + 蒸留 50k、1×H100） | 22 | **$45〜90** | 約7,000〜13,000円 | 1日 |
| **③ 公式パリティ**（finetune フル + 蒸留、1×H100） | 62〜66 | **$125〜265** | 約2〜4万円 | 3日 |
| ③' 同じ内容を 8×H100 で急ぐ | 105〜145 | $210〜580 | 約3〜9万円 | 13〜18h |
| ④ scratch フル（参考） | 84〜88 | $170〜350 | 約2.5〜5万円 | 4日 |

### GPU 以外

| 項目 | コスト |
|---|---|
| ダウンロード | 472 GB（MoeSpeech 152 + GOL 320）。HF egress 無料、100 MB/s で約80分 |
| ディスク | 素のまま約1 TB → 24 kHz mp3/opus 変換で 80〜150 GB。永続ボリューム $0.05〜0.10/GB/月 |
| CPU 前処理 | リサンプル・連結・マニフェスト化。16コアで数時間 |
| 強制アライメント | 2,640h で **8〜15 GPU-h**（$20〜60） |
| トークナイザ学習 | CPU 数分 |

**GPU 以外は $30〜100 程度**で、支配的なのは学習本体です。

---

## 既知のリスク

学習を回す前に潰しておかないと、1ランぶん（$130〜265）を捨てることになります。

### リスク1: 分かち書きとアライメント（**対応済み**）

当初は `training/dataloader.py` の `" ".join(...)` だけの問題と考えていましたが、実際は**アライナ側と対**でした。`align_data.py` の `re.split(r"\s+", ...)` は日本語で全文を1単語にするため、カット点が生成されず、dataloader が「同一クリップから voice prompt を取る」リーク経路に落ちます。

対応（`--segmenter japanese` + `data.word_separator: ""`）:

| 問題 | 対応 | 実コーパス20万行での検証 |
|---|---|---|
| 全文が1単語になる | MeCab/UniDic 形態素 + 文節マージ | カット点が生成される |
| vocab がかな・`word` は漢字必要 | セグメントを `(表層形, 読み, 文節開始)` の3つ組に | `kana` フィールドで監査可能 |
| 小書きかな・`ー` に UniDic の読みが無い | 読みが空なら表層形にフォールバックし、vocab フィルタに委ねる | **読み落ち 0件**（修正前 約6%） |
| MeCab が空白を捨てる | `m.white_space` を表層形に復元 | **再構成失敗 0件**（修正前 0.16%） |
| 数字・ラテン（読み導出不能） | セグメントを結合せず独立させ、`start=None` のまま残す | `dataloader.py:129` が隣接カットを拒否 |

!!! danger "数字・ラテンを隣接語に結合してはいけない"
    読みの無い形態素を前の語に glue すると、その語の `end` は「最後に読めた文字」のままになり、**未ラベルの音声が span の外に落ちます**。`dataloader.py:131` のカットは `0.5 * (prev.end + cur.start)` なので、その隙間の中央＝発話の途中で切れます。独立させておけば `start=None` により両隣のカットが拒否されます。

### リスク2: ASR 転写の精度

`training/README.md` は「音響は良いのに transcript に従わない場合は転写が不正確」「手動転写を優先せよ」と明示しています。

MoeSpeech は ASR 2系統あるので、**2つの転写の CER 一致度 + speechMOS で選別**するのが定石です。20〜40% 落ちて**実効 400〜500h** になる想定。GOL は1系統かつ [`。。。。。。` 問題](datasets.md)があるため、生テキスト列の使用が必須です。

### リスク3: クリップが短い

`training/dataloader.py` の挙動:

- `MIN_CUT_SEC = 1.0` — カット点の前後に1秒以上を要求
- 前半を voice prompt、後半をターゲットとして使う
- **アライメントが無い／カット不能な場合のフォールバックは、同一クリップから voice prompt を取る = リーク**

MoeSpeech 平均5.7秒、GOL 平均4.4秒なので、ターゲットが2〜4秒しか残りません。

対処: **同一話者のクリップを数本連結して擬似長尺ファイル化**する（英語版が章まるごとの mp3 に `start` / `duration` で窓を張っているのと同じ形）。

- MoeSpeech はキャラ別フォルダがあるのでそのまま連結できる
- GOL は話者ラベルが無いので、短尺を捨てるか、話者埋め込みでクラスタリングしてから連結する（追加で数 GPU-h）

前処理の工数は増えますが、GPU コストは増えません。

### リスク4: ライセンスとドメイン

- **MoeSpeech は再配布禁止**。チェコ語モデルのように `voices/` へサンプル音声を同梱してはいけません。サンプルボイスは自前または許諾済み音声で用意します。
- 感情ラベルに `Sexual1` / `Sexual2` が含まれます。公開モデルにするなら該当クリップの除外を検討。
- ドメインがアニメ／ギャルゲのキャラクター演技に限定されるため、**ニュース読み上げのような中立トーンは苦手**になります。汎用性が必要なら Emilia-JA / YODAS-ja / Common Voice ja / ReazonSpeech の追加を検討（学習ステップ数は変わらないので、増分コストは前処理のみ）。

---

## 対応済み（推論側・評価側）

いずれも `pocket_tts/` 側の変更で、学習側の作業とは別枠で解消しました。基盤になっているのは、文の切り方と正規化の要不要を言語ごとに宣言する `TextRules`（frozen dataclass）と、`Config` の5フィールド（`text_normalizer` / `sentence_boundaries` / `clause_boundaries` / `terminal_punctuation` / `segment_separator`）です（`b21f5c9 Give a config somewhere to say how its language writes`）。既定値は今日の英語の挙動のままなので、他言語への影響はありません。

### 欠陥1: 推論時の正規化が未接続（**対応済み**）

全角の `！` や `ＡＢＣ` をそのまま渡すと `<unk>` になり round-trip に失敗していました（正規化後は unk 0 で完全一致）。日本語 IME は既定で全角の `？！` を出すため実害は大きく、コーパスの 17.6% が ASCII `?`、13.1% が `!` で終わっており、いずれも全角から折り畳まれたものでした。

対応: `training/scripts/ja_text.py` にあった `normalize()` を `pocket_tts/utils/text_normalization.py` の `normalize_japanese` として移設しました。`pocket_tts/` は `training/` をインポートできないため、この移設が推論側からの呼び出しを可能にしています。`prepare_text_prompt` は `rules` を受け取り、`resolve_normalizer(rules.normalizer)` でこの正規化を実行します。

対応コミット: `118aad4 Move the Japanese normalization where inference can reach it`, `9e672d2 Stop appending a full stop to text whose language has none`

### 欠陥2: 日本語が文分割されない（**対応済み**）

`tts_model.py` の `split_into_best_sentences` は境界トークンに `。！？` を含んでいませんでした。実測: 66トークンの日本語段落が `max_tokens=32` でも**1チャンクのまま**返り、`Chunk has 66 tokens (max 32), generation may skip words` が出ていました。既定の `MAX_TOKEN_PER_CHUNK=50` と実測 1.805 字/token から、約90文字（3〜4文）を超える日本語入力は語が脱落する計算でした。

対応: `split_into_best_sentences` が `rules` を受け取り、`sentence_boundaries` と `clause_boundaries` で境界を判定するようにしました。

対応コミット: `a254dbb Split Japanese on its own sentence boundaries, and rejoin without spaces`（`71f8eb0`・`e68ac33` でトークナイザの fixture を `tests/fixtures/ja_tokenizer.model` としてコミットし、日本語のテストが CI で skip されず実行されるようにしています）

### 欠陥3: 半数の入力に ASCII ピリオドが付く（**対応済み**）

`prepare_text_prompt` は英数字で終わるテキストに `.` を付けていました。学習発話の 49.4% が英数字（かな・漢字を含む）で終わる一方、`.` で終わる学習発話は 0.001%（139.8万中13件）しかなく、EOS の較正が効く位置でモデルがほぼ見たことのないトークンを要求されていました。

対応: 固定の `.` の代わりに `rules.terminal_punctuation` を付けるようにしました。日本語ではこれが空文字列なので、何も付きません。

対応コミット: `9e672d2 Stop appending a full stop to text whose language has none`

### 欠陥4: WER が日本語で機能しない（**対応済み** — 測定手段を用意した、という意味で）

`training/eval/librispeech.py` の単語 WER は、分かち書きの無い日本語では実質的に文単位の誤り率になります。実測: 1文字だけ違う仮説が WER 1.0 で、全く無関係な仮説と同じスコアです。さらに `EnglishTextNormalizer` が濁点を除去します（`が`→`か`、`ご`→`こ`）。この性質自体は直しようがなく変わっていません。

対応: `build_normalizer` と `--text-normalizer {english,basic}` を追加し、`EvalResults`（`results.json`）に `cer` を常時出力するようにしました。`basic` は濁点・半濁点を保持するので、日本語の評価では `basic` を使います。

対応コミット: `694b384 Score with a metric that can tell Japanese generations apart`

### 欠陥5: 分割した文が空白で繋ぎ直される（**対応済み** — 欠陥2の修正が作り出したもの）

これは元の4件の一覧には無かった欠陥です。`split_into_best_sentences` は分割したセグメントをリテラルの `" "` で繋ぎ直しており、日本語は空白なしで書くのでこれは壊れています。ただしこの欠陥はこれまで一度も発火していませんでした。日本語は欠陥2のせいでそもそも分割されなかったからです。**欠陥2を直したことが、この欠陥5を初めて表面化させました。** 実測: テストの段落（57文字・空白ゼロ）を分割させると、2箇所に空白が注入されました。学習側で `data.word_separator: ""` が解決しているのと同じ問題です。

対応: セグメントの結合に使う区切り文字を `rules.segment_separator` にし、日本語では空文字列にしました。欠陥2と同じコミットで一緒に直っています。

対応コミット: `a254dbb Split Japanese on its own sentence boundaries, and rejoin without spaces`

### 範囲外として残るもの

- **日本語評価セットの構築** — 欠陥4への対応で CER を測る仕組みは用意しましたが、実際に評価に使う日本語の参照テキストと音声のペアはまだありません。
- **ASR モデルの選定** — `training/README.md` が言うとおり、日本語に対応する ASR を選ぶ必要があります（上記「リスク2: ASR 転写の精度」参照）。

どちらも音声データの取得を伴うため、データ収集フェーズの範囲です。

---

## 段階的な進め方

### フェーズ0: トークナイザは最初に確定する

**トークナイザは検証を始める前に、両データセットのテキストで1回だけ学習してください。**

理由: `reset_text_embedding: true` は「トークナイザが変わったから text embedding を捨てる」処理です。**後から GOL を足してトークナイザを作り直すと embedding をもう一度捨てることになり、検証ランの重みを引き継げません**（実質やり直し）。

幸い **音声を落とさずにテキストだけ取得できます** — GOL の `metadata.csv` は 227 MB（165万行）、`ayousanz/moe-speech-20speakers-ljspeech` の `metadata.csv` は 6.3 MB（6万行）で、いずれも音声本体とは別ファイルです。

**実施済み**（`training/scripts/prepare_ja_text.py` → `train_tokenizer.py`）:

| 項目 | 値 | 根拠（すべて実測） |
|---|---|---|
| コーパス | 139.8万発話・96 MB | GOL 134.0万 + MoeSpeech 5.8万、正規化後に重複除去 |
| 異なり字 | **4,092字** | GOL 単体 4,238字、MoeSpeech が足すのは**わずか2字**（`U+576A` `U+9DC8`） |
| `--vocab-size` | **8000** | 下表参照 |
| `--character-coverage` | **1.0** | 0.9995 は 1,516 字を `<unk>` にする。`<unk>` は `tts_model.py` でチャンク境界トークンにもなるため実害が大きい |
| `--normalization-rule` | **identity** | 既定の `nmt_nfkc` は tokenizer 内部で `…`→`...` に書き換える（GOL 3列目の破損の再現） |
| `n_bins` | **8000** | vocab size と厳密一致。`pocket_tts/conditioners/text.py:30` のアサートで検証済み |

!!! danger "4000 では学習が失敗します"
    異なり字が 4,092 あるため、`character_coverage 1.0` では sentencepiece が `Vocabulary size is smaller than required_chars` で**即座に落ちます**。リリース済み英語モデルの `n_bins: 4000` はそのままでは使えません。

vocab_size の実測（held-out 2万発話）:

| vocab | 圧縮率 | 単字ピース | マージ数 | 埋め込み |
|---|---|---|---|---|
| 6000 | 1.658 字/token | 4,061 | 1,939 | 6.1M |
| **8000** | **1.805** | 4,118 | 3,882 | **8.2M** |
| 12000 | 1.961 | 4,236 | 7,764 | 12.3M |

最終トークナイザは全コーパス 230万トークンに対し **`<unk>` 0件**。

```bash
uv run python -m training.scripts.prepare_ja_text data/ja/corpus.txt \
    --gol data/ja/gol_metadata.csv --ljspeech data/ja/moe20_metadata.csv
uv run python -m training.scripts.train_tokenizer data/ja/tokenizer \
    data/ja/corpus.txt --vocab-size 8000 --character-coverage 1.0 \
    --normalization-rule identity
```

`normalize()` はコーパス構築と `align_data.py --segmenter japanese`（マニフェストの `transcript` を書き戻す）の両方から呼ばれます。NFKC から除外しているのは2つ:

- **`…` `‥`** — NFKC は ASCII ピリオドの羅列にする（GOL 3列目の破損の再現）
- **`゛` `゜`** — NFKC は **空白 + 孤立した結合文字**にする。「え゛っ」のような強調表記はこのドメインで頻出で、空白の無いテキストに空白を注入するのは有害

センチネルには私用領域ではなく**非文字**（`U+FDD0`〜）を使っています。私用領域にはキャリア絵文字が存在し、GOL には実際に `U+E63E` が405件含まれているためです。

### フェーズ1: 検証（MoeSpeech のみ・$10・半日）

**目的は「日本語が喋れるか」ではなく「パイプラインが正しいか」の切り分け**です。変数は少ないほど価値が上がるため、**MoeSpeech 単独**で行います。

MoeSpeech 単独を選ぶ理由:

| | MoeSpeech | GOL |
|---|---|---|
| サンプルレート | **44.1 kHz**（Mimi の 24 kHz に余裕） | 22.05 kHz |
| 話者ラベル | **あり** | なし |
| 転写 | **2系統**（相互 CER で判定可） | 1系統・要クリーニング |
| 容量 | 152 GB（**部分 DL 可**） | 320 GB |

- **22.05 kHz を混ぜると、出音がこもった時に「サンプルレートのせい」か「バグ」か区別できません。** 44.1 kHz 単独なら、悪ければバグです。
- **話者ラベルが無いとリスク3の対処自体を検証できません。** MoeSpeech なら連結処理の正しさも同時に検証できます。

さらに zip がキャラ単位なので、**30 GB / 約160キャラ / 約123h だけ落とせば足ります**（README の最低ラインは100h）。検証フェーズのボトルネックは GPU ではなく前処理の待ち時間なので、ここを削るのが最も効きます。

**実行コマンド**

zip の取得からアライメント済みマニフェストまでは1コマンドで通ります。

```bash
python -m training.scripts.prepare_moespeech --hours 124 --out data/ja
```

閾値には既定値がありません（この corpus を測ったものが存在しないため）。初回はステージ4の
probe を書いた直後に停止するので、`data/ja/probe.json` の retention 表を読んで `--max-cer` と
`--min-mos` を決め、**同じコマンドに2つを足して再実行**します。ステージ5から続きます。

```bash
python -m training.scripts.prepare_moespeech --hours 124 --out data/ja \
  --max-cer <probe.json から読む> --min-mos <probe.json から読む>
```

| # | ステージ | 出力（`--out` 以下） | 再実行でスキップする条件 | 所要時間 |
|---|---|---|---|---|
| 1 | キャラ選定 | `characters.json` | ファイルが在る | 未計測 |
| 2 | ダウンロード | `zips/<name>.zip` | zip が在る（`.partial` は無視する） | 未計測 |
| 3 | 展開 | `extracted/<name>/` | `extracted/<name>.complete` が在る | 未計測 |
| 4 | probe | `probe.json` | ファイルが在る | 未計測 |
| 5 | 発話選定 | `utterances.jsonl` | ファイルが在り、`extracted/*.complete` より新しい | 未計測 |
| 6 | 連結 | `audio/<speaker>.wav`・`entries/<speaker>.jsonl` | その話者の `entries/<speaker>.jsonl` が在り、`utterances.jsonl` より新しい | 未計測 |
| 7 | マニフェスト | `train.jsonl`・`valid.jsonl` | 両方が在り、`entries/*.jsonl` より新しい | 未計測 |
| 8 | アライメント | `train_aligned.jsonl`・`valid_aligned.jsonl` | 出力が在り、元のマニフェストより新しい（中断時の `.partial` は `--resume` が拾う） | 未計測 |

スキップの判定は**出力の有無と更新時刻**で、オプションは見ていません。閾値や `--target-sec` を
変えて実行し直したい場合は、その段の出力を消してから打ち直してください。**消すのは1つで足ります。**
各段は自分の入力より新しい出力しか再利用しないので、`utterances.jsonl` を消せば `entries/`・
`audio/`・`train.jsonl`・`valid.jsonl`・`*_aligned.jsonl` まで一緒に作り直されます。これが無いと、
厳しい `--min-mos` で打ち直しても `utterances.jsonl` だけが書き換わり、`train.jsonl` は却下した
はずの選定を指したまま「完了」と表示されます。`--hours` を増やして新しいキャラを展開した場合も
同じ理由で発話選定からやり直しになります。

`probe.json` だけは有無のみで判定します。ここは何も決めない計測で、後段はこのファイルを読まない
（閾値を選ぶ人間が読む）ので、測り直したいときは消してください。

**所要時間は全て未計測です。** vast.ai 上でまだ一度も実行していないので実測値がありません。
下の見積り表は着手前の試算であり、初回実行後にこの列を実測で置き換えてください。

インスタンスは preemption で消えるので、**落ちたら同じコマンドを打ち直す**のが正しい復帰手順
です。完了済みのステージは上の条件で飛ばされ、途中で死んだ出力は `.partial` に残るだけなので
「完成済み」と誤認されることはありません。

アライメントは日本語に固定されています（`--segmenter japanese`、`--align-model` の既定は
`vumichien/wav2vec2-large-xlsr-japanese-hiragana`）。かなを語彙に持たないモデルを渡すと
`align_data.py` が起動時に拒否します。分かち書きの無い日本語を whitespace で分割すると
1発話が1単語になり、カット点が消えて dataloader の voice prompt 機構が黙って無効になります
（リスク1）。

**見積り**

| 工程 | コスト |
|---|---|
| DL（30 GB） | 約5分 |
| 展開・24 kHz 変換・同一キャラ連結・マニフェスト化 | 16コアで1〜2時間 |
| 強制アライメント（123h） | 約1 GPU-h / $2〜4 |
| finetune 15k step（24層のまま、蒸留なし） | 1×H100 で 2.2h / $4〜9 |
| **合計** | **$10前後・半日** |

**判定できること** — トークナイザ、アライメント、`" ".join` 修正、`n_bins` 一致、日本語として意味が取れるか、EOS で停止するか、CER（`--text-normalizer basic`）。

**判定できないこと** — 最終的な WER / UTMOS の絶対値、中立トーンの汎用性。

!!! note "キャラ演技調になるのは仕様"
    ドメインがアニメ／ギャルゲなので、検証モデルはキャラクター演技調に喋ります。これを失敗と読み違えないでください。

!!! danger "WER は日本語の go/no-go 基準にできません"
    分かち書きの無い日本語では WER は実質的に文単位の誤り率になるため、PR #254 のチェコ語 WER（2k step で 29.5% など）を日本語の go/no-go 基準として流用できません。**判定基準は CER**です。ツール自体は用意できましたが、チェコ語の数値とは指標が違うので直接比較はできません。

!!! tip "早期トリップワイヤ"
    チェコ語は 2k step で WER 29.5%（PR #254、既に言語として成立と読める水準）でした。日本語は指標が異なるため同じ数字を目標にはできませんが、**2〜3k step（約25分・$1〜2）で「日本語らしい音韻」すら出ないならパイプラインのバグ**と判断してよいのは変わりません。これは指標ではなく耳で聞いた判断なので、CER に差し替えても成立します。ここで止めれば損失は $2 です。

### フェーズ2以降

| フェーズ | データ | 内容 |
|---|---|---|
| **2. 本番 finetune** | MoeSpeech + GOL（約2,640h） | `finetune_language.yaml` ベース、24層、250k step |
| **3. 仕上げ** | **MoeSpeech のみ**・低 lr | 44.1 kHz 由来で音響品質の上限を引き上げる |
| **4. 蒸留** | 同上 | `depth_distill.yaml` で 24層 → 6層、CFG を焼き込む |
| **5. 公開** | — | `model.safetensors` + `tokenizer.model` + `japanese.yaml` + 自前サンプルボイス |

24層モデルのままでも pocket-tts は `--language italian_24l` のように動作するので、**蒸留は最後に一度だけ**でよく、それまでは 24L で評価を回せます。

公開時の成果物の型は [チェコ語モデルの構成](community-survey.md)を参照してください。

---

## 参照

- `training/README.md` — steps/s 実測表、データ要件、ハイパーパラメータの注記
- `training/configs/scratch.yaml` / `depth_distill.yaml`
- [PR #254](https://github.com/kyutai-labs/pocket-tts/pull/254) — `finetune_language.yaml`（マージ済み: `8c98c9b`）
- [H100 Rental Prices Compared (IntuitionLabs)](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
