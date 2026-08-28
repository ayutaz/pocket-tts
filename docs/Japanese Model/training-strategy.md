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
| ディスク | 素のまま約1 TB。**前処理は変換しません**（`concatenate()` は 44.1 kHz のまま書き、リサンプルは dataloader が読み込み時に行う）ので、この形のまま持つ前提で見積もってください。mp3/opus 化すれば 80〜150 GB に落とせますが、それをする段はまだ実装されていません。永続ボリューム $0.05〜0.10/GB/月。フェーズ1（124h）の実数は下の「ディスク」参照 |
| CPU 前処理 | 展開・連結・マニフェスト化。**全て単一プロセスです**（`prepare_moespeech.py` に並列化は入っていません）。コアを増やしても速くなりません。効くのはシングルコア性能とディスクのスループットだけです。数時間というのは**単一スレッド前提の未計測の当て推量**です |
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

さらに zip がキャラ単位なので、**落とすのは約30 GB / 約124h で足ります**（README の最低ラインは100h）。既定の `--order largest` なら約28個の zip で 124h に届きます（`--order random` だと約160キャラ必要で、落とす量は同じです）。検証フェーズのボトルネックは GPU ではなく前処理の待ち時間なので、ここを削るのが最も効きます。

**ただしダウンロード量とディスク所要量は別物です。** 実行中はこの 30 GB が同時に4つ分に膨らみます。インスタンスを借りる前に下の「ディスク」を読んでください。

**実行コマンド**

先に依存です。**`uv sync` だけでは足りません。** ステージ8のアライメントは MeCab（fugashi +
unidic-lite）を必要とし、これは `pyproject.toml` の `[dependency-groups]` にある `japanese`
グループです。extra ではないので `--extra` ではなく `--group` を使います。

```bash
uv sync --group japanese
```

（`dev` は既定で入るのでこれ1本で足ります。逆に素の `uv sync` は fugashi と unidic-lite を
**アンインストールします**。）

入れ忘れは高くつきます。`align_data._japanese_segmenter` は fugashi を遅延 import するため、
ダウンロード・展開・probe・選定・連結・両マニフェストまで全部通り切ってから最後に落ち、しかも
アライナは別プロセスなので表に出るのは `CalledProcessError` の終了コードであって、足りない
パッケージの名前ではありません。**そのため確認は2箇所に入っています。**

**1つ目はステージ0**（ステージ1の前、まだ1バイトも落としていない位置）です。ここで止めるのは
「この実行はアライナに到達する」と判断したときだけで、その判断式は
`require_japanese_segmenter(will_align=...)` に渡される次の式です ——
**`--max-cer` と `--min-mos` の両方が与えられ、かつ「両アライメントが在り、かつ `cutoffs.json`
が今回渡した対とちょうど同じ」ではないとき**。場合分けするとこうなります。

| 実行 | ステージ0 |
|---|---|
| 閾値が片方でも無い（probe で止まる実行） | **警告のみ。** アライナに到達しないので必要ありません（ダウンロードを待つ間に入れられます） |
| 閾値が揃い、両アライメントが在り、`cutoffs.json` も同じ対 | **警告のみ。** 作り直すものが何も無く、`align()` は出力の有無だけでスキップします。preemption 後に同じコマンドを打ち直すのがこれです（`test_a_re_run_with_nothing_left_to_align_does_not_need_the_segmenter`） |
| 閾値が揃い、アライメントが片方でも欠けている | **`exit 1`**（`test_a_missing_alignment_still_refuses_a_run_without_the_segmenter`） |
| 閾値が揃い、両アライメントは在るが `cutoffs.json` の対が違う（無い・読めない場合も含む） | **`exit 1`。** 閾値を変えるとステージ5が選定を、ステージ7が両マニフェストを書き直し、ステージ8は両アライメントを古いものとして捨てます。つまり「在る」ことは「残る」ことを意味しません（`test_changed_cutoffs_over_a_finished_tree_are_refused_again`） |

ステージ0の答えは**まだ予測**です。ステージ5と7がマニフェストに何をするかを上から言い当てる
ことはできないので、外すときは拒否側に外します —— 要らなかった拒否の代償は
`uv sync --group japanese` の30秒、見逃しの代償は有料インスタンスの1日だからです。

**2つ目はステージ8**、2回の `align()` のそれぞれ1行前です（`require_segmenter_to_align()`）。
ここではもう予測ではありません。マニフェストは全て書かれ、古いアライメントは捨てられた後なので、
`already_aligned()`（= 出力が在るか）が答えの全部です。出力が在ればそのまま戻り、無ければ
セグメンタを実際に組んでみて、駄目ならパッケージ名を出して `exit 1` します。ステージ0が構造上
見られないのは**手で消された中間成果物**です —— `utterances.jsonl` を消せばステージ0の時点では
両アライメントも `cutoffs.json` も揃っているのに、ステージ5〜7が全部書き直し、ステージ8が両方を
捨てて張り直します。それを拾うのがこの2つ目で、`CalledProcessError` の終了コードが数時間後に
出る代わりに、その場でパッケージ名の出る停止になります
（`test_the_aligner_is_not_reached_when_only_the_late_guard_can_tell`）。上の段は全てディスクに
残るので、入れて打ち直せばそこから続きます。

どちらも `_japanese_segmenter_error()` を通り、`align_data.SEGMENTERS["japanese"]` を**実際に
組んでみて**判定します（辞書も wrapper も `japanese` グループなので、import できるかだけを見ると
足りない側を見逃します）。2箇所が別々の条件を持たないのはこのためです。

zip の取得からアライメント済みマニフェストまでは1コマンドで通ります。

```bash
uv run python -m training.scripts.prepare_moespeech --hours 124 --out data/ja
```

閾値には既定値がありません（この corpus を測ったものが存在しないため）。初回はステージ4の
probe を書いた直後に停止するので、`data/ja/probe.json` の retention 表を読んで `--max-cer` と
`--min-mos` を決め、**同じコマンドに2つを足して再実行**します。ステージ5から続きます。

```bash
uv run python -m training.scripts.prepare_moespeech --hours 124 --out data/ja \
  --max-cer <probe.json から読む> --min-mos <probe.json から読む>
```

!!! note "retention 表の `kept` はステージ5までの約束です"
    表の件数はステージ5の選定が実際に残す件数と一致します（probe と選定が同じ走査を共有して
    いるため。`test_selection_keeps_exactly_as_many_clips_as_the_table_promised`）。ただし
    **マニフェストに載る件数はそこからさらに減ることがあります**。ステージ6の `concatenate()` が
    飛ばすクリップがあるからです —— wav が開けないもの（欠損・切り詰め・フレーム数0）と、
    **その話者で最初に開けたクリップに固定したサンプルレート／チャンネル数と食い違う**ものです。
    飛ばした数は
    `<話者>: N of M clips could not be joined and are not in the manifest` という警告に、
    固定した値とその固定元のクリップは続く1行に出ます（固定と食い違って飛ばした数が繋げた数を
    上回るときは、この行も警告に上がります —— 固定したクリップの側が外れだった徴候です）。
    **表と `train.jsonl` の行数が合わないときの差はここなので、件数はこのログで確かめて
    ください。**

`--out` 以下は0段目を除いて8つです（0段目は上のセグメンタ確認で、ディスクには何も残しません）。

| # | ステージ | 出力（`--out` 以下） | 再実行でスキップする条件 | 所要時間 |
|---|---|---|---|---|
| 1 | キャラ選定 | `characters.json` | ファイルが在る（`--hours` が違っても選び直さず、警告して再利用する） | 未計測 |
| 2 | ダウンロード | `zips/<name>.zip` | zip が在る（`.partial` は無視する） | 未計測 |
| 3 | 展開 | `extracted/<name>/`・`extracted/<name>.complete` | `extracted/<name>.complete` **と** `extracted/<name>/` の**両方**が在る（マーカーだけ在ってディレクトリが消えていれば展開し直す。ディレクトリだけ在ってマーカーが無ければ中断とみなし、消してから展開し直す）（`test_a_marker_without_its_directory_is_not_trusted`） | 未計測 |
| 4 | probe | `probe.json` | 在り、かつ**選定話者の** `extracted/<name>.complete` と `characters.json` のどれよりも新しい | 未計測 |
| 5 | 発話選定 | `utterances.jsonl`・`cutoffs.json` | 在り、かつ**上と同じ入力 + `cutoffs.json`** のどれよりも新しい（`cutoffs.json` は前回と違う対のときだけ書き直すので、同じ対なら更新時刻は動きません） | 未計測 |
| 6 | 連結 | `audio/<speaker>.wav`・`entries/<speaker>.jsonl` | その話者の `entries/<speaker>.jsonl` が在り、`utterances.jsonl` より新しい | 未計測 |
| 7 | マニフェスト | `train.jsonl`・`valid.jsonl` | 両方が在り、`utterances.jsonl` と `entries/*.jsonl` のどれよりも新しい | 未計測 |
| 8 | アライメント | `train_aligned.jsonl`・`valid_aligned.jsonl`・`*_aligned.jsonl.shards`（`align()` が使った分割数の記録。valid 側は常に `1`） | 出力が元のマニフェストより新しい（古ければ `.partial`・`.shard*` ごと捨てて張り直す。中断時の `.partial` は `align()` の `--resume` が拾う） | 未計測 |

ステージ6は書くだけでなく**消します**。選定から外れた話者の `entries/<speaker>.jsonl` と、そこに
書かれていた `audio/` の wav を消し、`--target-sec` を伸ばして必要ファイル数が減ったときも余った
`<speaker>_NNN.wav` を消します。`audio/` は 124h・44.1 kHz で数十 GB あり、preemptible
インスタンスのディスクは固定なので、放置は後続の段を殺します。

**スキップの判定は出力の有無と更新時刻で、オプションは基本的に見ていません。例外は `--max-cer`・
`--min-mos`・`--align-shards` の3つです。** ステージ5は前の2つを `cutoffs.json` に書き（前回と違うときだけ
書きます）、それを発話選定の入力に数えます。したがって**閾値を変えて同じコマンドを打ち直すだけで、
選定・オフセット・`audio/`・両マニフェスト・両アライメントまで自動で作り直されます。何も消す必要は
ありません**（`test_new_cutoffs_rebuild_the_selection_they_decided` が固定しています）。これが無い
場合に何が起きるかが、この仕組みの理由です —— 厳しい `--min-mos` で打ち直しても
`utterances.jsonl` だけが書き換わり、`train.jsonl` は却下したはずの選定を指したまま「完了」と
表示されます。

例外の3つ目、`--align-shards` は効き方が違います。まず**掛かるのは学習側のマニフェストだけ**
です —— valid 側は `align(valid_manifest, valid_aligned, 1, ...)` と分割数 `1` が直接書かれて
いて（`prepare_moespeech.py:1341`）、`--align-shards` をいくつにしても変わりません。ステージ8
全体を分割するオプションではなく、`train_aligned.jsonl` を分割するオプションです
（`valid_aligned.jsonl.shards` は常に `1` が書かれます）。そのうえで、**効くのは中断で残った
途中結果に対してだけ**です。
`align()` は分割数を出力の隣（`train_aligned.jsonl.shards`）に記録し、前回と違う数で打ち直すと
`.partial` と `.shard*`（マニフェスト側の分割ぶんも）を捨ててアライメントを最初からやり直します
（`prepare_data.py:113-123`）。各ワーカーの `--resume` は (path, start) で再開位置を決めるため、
4分割ぶんの途中結果を2分割で拾うと同じ行を二重にアライメントし、マージ時に**静かに重複する**
からです。完成した `*_aligned.jsonl` が在るときは分割数に関係なくスキップされるので、
**やり直しの対象になるのは未完のアライメントだけ**で、その上の段には波及しません。

それ以外のオプション（`--target-sec`・`--valid-hours`・`--order`）は再利用の判定に使われないので、
**変えても何も再実行されません。その段の出力を消してから打ち直してください。消すのは1つで
足ります。** 各段は自分の入力より新しい出力しか再利用しないので、`utterances.jsonl` を消せば
`entries/`・`audio/`・`train.jsonl`・`valid.jsonl`・`*_aligned.jsonl` まで一緒に作り直されます。

ここまでで触れたのは main() の12個のオプションのうち7個です。残る
`--out`・`--zips`・`--repo`・`--align-model`・`--verbose` は再利用の判定に一切現れませんが、
**このうち `--align-model` だけは黙って効かない**ので、先にそれを書きます。

- **`--align-model`** —— **どこにも記録されず、どことも照合されません。** ステージ8のスキップは
  `align()` が出力の有無だけで決めるので（`already_aligned()` は `out.exists()` そのものです）、
  **完成した木に対して CTC モデルだけ差し替えて打ち直しても、何一つ再実行されません。警告すら
  出ません**（`--order` は少なくとも `characters.json` に記録だけはされますが、これは記録も
  ありません）。張り直したいときは `train_aligned.jsonl` と `valid_aligned.jsonl` を消して
  ください（残っていれば `.partial`・`.shard*`・`*.shards` も一緒に）。かなを語彙に持たない
  モデルを `align_data.py` が起動時に拒否するのは**実際に起動したときの話**で、スキップされた
  実行では起動しません。
- **`--out`** —— 再利用の判定は全てこの下のファイルの有無と更新時刻なので、別の `--out` は別の木
  です。何も共有しません（共有するのは HF キャッシュだけなので、ダウンロードは再取得ではなく
  キャッシュからの複製で済みます）。
- **`--zips`** —— 既定は `<out>/zips`。ステージ2が見るのは `<zips>/<name>.zip` だけなので、別の
  ディレクトリを指せば zip はそこへもう一度書かれます（キャッシュが在れば複製で、約30 GB）。
  展開の判定は `<out>/extracted` 側のマーカーなので、展開はやり直しになりません。逆に**複数の
  `--out` で同じ `--zips` を指せば、大きい `--hours` の実行が小さい実行の zip をそのまま使えます**
  —— それがこのオプションの目的です。
- **`--repo`** —— 記録も照合もされません。`info.csv` と zip の取得先を決めるだけなので、
  `characters.json` と zip が既に在る木では**何も起きません**。別のリポジトリから取り直すときは
  `characters.json` と `zips/` を消してください。
- **`--verbose`** —— ログの詳細度だけです。ディスク上のものは何も変わりません。

`--order` だけは「**記録はされるが、比較されない**」ことに注意してください。選定時の値は
`characters.json` に `"order"` として書かれています（`prepare_moespeech.py` の選定書き出し）。
ただし再利用時に照合されるのは同じファイルの `hours_requested` だけで、`--order` は読まれません。
つまり **`--hours` を変えれば少なくとも警告が出るのに対し、`--order` を変えて打ち直すと警告すら
出ずに黙って無視されます。** 並び順を変えたいときは `characters.json` を消してください。

`--hours` はさらに別の扱いです。**`characters.json` が在る限り話者は選び直されません。**`--hours`
を増やしても減らしても、「この `characters.json` は `--hours` X の選定で、`--hours` Y に再利用して
いる。選び直すなら消せ」と警告が出るだけで、キャラは増えも減りもしません。増減させたいときは
`characters.json` を消してください。消せば選定が書き直され、その更新時刻によって probe から下が
全部作り直されます（`test_a_speaker_a_wider_selection_adds_reaches_the_manifests`）。

`probe.json` も他と同じ規則で、有無だけでは判定しません。選定話者の `.complete` か
`characters.json` より古ければ測り直します —— 後から展開された話者を含まない retention 表や、
もう使わない話者を含んだままの retention 表で閾値を決めてしまわないためです。単に測り直したい
ときは消してください。

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

**ディスク: 150 GB 用意してください（30 GB では足りません）**

124h の実行は完了時点で**同じ音声を4つ持ちます**。どれも自動では消えません。

| 置き場所 | 中身 | 概算 |
|---|---|---|
| `$HF_HOME`（既定 `~/.cache/huggingface`） | `hf_hub_download` のキャッシュ。zip の1つ目のコピー | 約30 GB |
| `<out>/zips/` | そこから `zips/` へ複製した2つ目 | 約30 GB |
| `<out>/extracted/` | 展開した wav + json | 約37 GB |
| `<out>/audio/` | 連結した擬似長尺 wav（44.1 kHz のまま） | 最大約39 GB |
| **ピーク合計** | | **約136 GB** |

算数です。[datasets.md](datasets.md) の実測（zip 151.6 GB、展開後約184 GB、623h）からの按分で、
**すべて概算**です。キャラごとの圧縮率のばらつきは見ていません。

分母の 623h は datasets.md の実測値で、同じ調査の 395,000発話 × 平均5.68秒 ≈ 623h と整合します。
按分の分子（151.6 GB と 184 GB）を測ったのと同じ出所なので、ここはこの値を使います。
なお `prepare_moespeech.py` の docstring と設計文書は 621h と書いています。これは同じ corpus を
`info.csv` の `total_duration_min` を473キャラぶん合計して測った別の値で、差は 0.3% ——
この見積り自体の精度よりはるかに小さいので、どちらを入れても下の4式の答えは変わりません。

```
zip          : 151.6 GB ÷ 623h × 124h ≈ 30 GB
展開後       : 184   GB ÷ 623h × 124h ≈ 37 GB
連結 wav     : 124h × 3600秒 × 44100 Hz × 2 byte ≈ 39 GB（16bit mono 無圧縮）
HF キャッシュ: zip と同じ            ≈ 30 GB
```

`--hours` を変えるときは4本の式の 124 を差し替えてください。

`audio/` に入るのは選定を通った発話だけなので 39 GB は上限です。相互 CER と speechMOS で
20〜40% 落ちる想定（リスク2）なら実際は 24〜31 GB でしょう。ただし**閾値を決めるより先に
ディスクを確保するなら上限で見てください。**

24 kHz への変換は**しません**。`concatenate()` は 44.1 kHz のまま書き、リサンプルは dataloader が
読み込み時に行います。

**消してよいもの、消したときの代償**

| 消せるもの | いつから | 打ち直したときの代償 |
|---|---|---|
| HF キャッシュ | `zips/` に全 zip が揃った後 | **なし。** ステージ2は `zips/<name>.zip` の有無だけを見るので、キャッシュが無くても再取得しません。**まずこれを消してください（約30 GB）** |
| `<out>/zips/` | `extracted/<name>.complete` が全話者ぶん揃った後 | **約30 GB の再ダウンロード。** main は毎回ダウンロード段を通り、zip が無ければ展開済みでも落とし直します（そのときキャッシュぶんの30 GB も一時的に要ります） |
| `<out>/extracted/` | ステージ8まで終わり、**このコマンドをもう打たないと決めた**後 | **ほぼ全部。** 打ち直すと再展開され、`.complete` の更新時刻が新しくなるので probe・選定・連結・マニフェスト・アライメントまで作り直しになります |
| `<out>/audio/`・`train_aligned.jsonl`・`valid_aligned.jsonl` | — | **消さないでください。学習が読むのはこれだけです。** `finetune_language_ja.yaml` の `train_jsonl`/`valid_jsonl` がこの2つのマニフェストを指し、その各行の `path` が `audio/` の wav を指します |
| `entries/`・`utterances.jsonl`・`probe.json`・`train.jsonl`・`valid.jsonl` | いつでも | **消せます。代償は下の段を作り直すことだけです。** `utterances.jsonl` を消せば `entries/`・`audio/`・両マニフェスト・両アライメントまで作り直し（`--target-sec` を変える手順がこれです。上の「それ以外のオプション」参照）、`train.jsonl`・`valid.jsonl` を消せば分割とアライメントのやり直し、`probe.json` は測り直すだけです。ただし `entries/<speaker>.jsonl` は**選定から外れた話者の `audio/` を消すための唯一の手掛かり**でもあるので、先に消すとその wav は以後誰にも消されず残ります |
| `cutoffs.json` | — | **消しても得はありません。** ステージ5の出力で、`--max-cer`/`--min-mos` を記録している唯一の場所です。消すと次の実行がその場で書き直し、**その新しい更新時刻が `utterances.jsonl` を古くする**ので、閾値は同じままなのに選定・オフセット・`audio/`・両マニフェスト・両アライメントまで作り直しになります |
| `*_aligned.jsonl.shards` | アライメントが完成した後 | **完成後は無害**（`align()` は出力が在れば分割数を見る前にスキップします）。**未完のアライメントが残っている状態で消すと危険です**: 前回の `--align-shards` を記録している唯一の場所なので、消してから違う分割数で打ち直すと `.partial`・`.shard*` が捨てられずに拾われ、同じ行を二重にアライメントして**マージ時に静かに重複します** |

**見積り**

| 工程 | コスト |
|---|---|
| DL（30 GB、ディスクには60 GB 書かれる） | 約5分 |
| 展開・同一キャラ連結（44.1 kHz のまま）・マニフェスト化 | **単一プロセスで1〜2時間（未計測の当て推量）**。下の注を読んでからインスタンスを選んでください |
| 強制アライメント（123h） | 約1 GPU-h / $2〜4 |
| finetune 15k step（24層のまま、蒸留なし） | 1×H100 で 2.2h / $4〜9 |
| **合計** | **$10前後・半日** |

!!! warning "CPU 段は単一プロセスです。コアを買っても速くなりません"
    `prepare_moespeech.py` に並列化は入っていません —— `multiprocessing` も
    `concurrent.futures` もプールも、どこにもありません。ステージ3は `zipfile` の
    `extractall` そのまま、ステージ4・5はアノテーションを1件ずつ歩き、ステージ6は
    `sphn.read` と `np.concatenate` を1プロセスで回します。複数プロセスを使うのは
    ステージ8のアライメントだけで、それも CPU コアではなく **GPU 単位**の分割です
    （`--align-shards`）。したがって**コア数を増やしても CPU 段は速くなりません**。
    効くのはシングルコア性能とディスクのスループットだけです。
    **上の「1〜2時間」は実測ではなく単一スレッド前提の当て推量**で、vast.ai では
    まだ一度も計測していません（前掲の「所要時間は全て未計測です」と同じ扱いです）。

**判定できること** — トークナイザ、アライメント、`" ".join` 修正、`n_bins` 一致、日本語として意味が取れるか、EOS で停止するか、CER（`--text-normalizer basic`）。

**判定できないこと** — 最終的な WER / UTMOS の絶対値、中立トーンの汎用性。

!!! note "キャラ演技調になるのは仕様"
    ドメインがアニメ／ギャルゲなので、検証モデルはキャラクター演技調に喋ります。これを失敗と読み違えないでください。

!!! danger "WER は日本語の go/no-go 基準にできません"
    分かち書きの無い日本語では WER は実質的に文単位の誤り率になるため、PR #254 のチェコ語 WER（2k step で 29.5% など）を日本語の go/no-go 基準として流用できません。**判定基準は CER**です。ツール自体は用意できましたが、チェコ語の数値とは指標が違うので直接比較はできません。

!!! tip "早期トリップワイヤ"
    チェコ語は 2k step で WER 29.5%（PR #254、既に言語として成立と読める水準）でした。日本語は指標が異なるため同じ数字を目標にはできませんが、**2〜3k step（約25分・$1〜2）で「日本語らしい音韻」すら出ないならパイプラインのバグ**と判断してよいのは変わりません。これは指標ではなく耳で聞いた判断なので、CER に差し替えても成立します。ここで止めれば損失は $2 です。

!!! danger "GPU を借りる前に — 学習側に3つ穴があります"
    マニフェスト構築を終えたあと、フェーズ1を「明日インスタンスを借りて実行する人」の目で
    読み直して出たものです。**どれもマニフェスト側ではなく学習側**で、設定ファイルは意図的に
    変更していません（どう直すかは判断が要るため）。借りる前に片付けてください。

    **1. `data/ja/` はリポジトリに入っていません。** `.gitignore:95` の `/data*/` が
    ディレクトリごと除外していて、`git ls-files data/ja/` は空を返します。新しいインスタンスに
    clone しただけでは `tokenizer.model` も `corpus.txt` も `gol_metadata.csv` も
    `moe20_metadata.csv` もありません。前処理8段を全部終えたあと、**学習開始時に
    `SentencePieceTokenizer.__init__` がファイル無しで落ちます**。上のフェーズ0は「実施済み」と
    書かれていますが、それはこの開発機の話です。`data/ja/tokenizer.model` をインスタンスへ
    転送し、`vocab_size` が 8000 であることを確かめてから GPU を借りてください。作り直す場合は
    gated repo 2つ（`midralab/gol-dataset-2k-ljspeech`、
    `ayousanz/moe-speech-20speakers-ljspeech`）から metadata.csv を取ったうえで、
    `finetune_language_ja.yaml` 冒頭のコマンド列をその順に実行します。

    **2. `max_steps` は 250000 です。** `finetune_language_ja.yaml` は
    `finetune_language.yaml` を継承しており、そこが 250k step（`:36`）になっています。この
    フェーズの見積りは「15k step・2.2h・$4〜9」ですが、そのまま起動すると**約36時間・$75〜145**
    のジョブが始まります。`training/train.py` は設定ファイルのパスしか受け取らない
    （`assert len(sys.argv) == 2`）ので、**yaml を書き換える以外に止める手段はありません**。
    15k step で止めるなら起動前に `max_steps` を書き換えてください。`ckpt_freq: 2500` なので
    最終チェックポイントは `checkpoint_00015000.pt` になります。

    **3. すぐ上の「早期トリップワイヤ」は現状では発火できません。** `sample_freq` の既定は
    10000（`training/args.py:116`）で、`finetune_language_ja.yaml` は上書きしていません。
    `train.py` は `(step + 1) % sample_freq == 0` で合成するので、**最初の wav が出るのは
    10k step ≈ 1.45時間 ≈ $3〜6** です。「2〜3k step・$1〜2 で耳で判断して止める」を実際に
    行うには、ja 設定に `sample_freq: 500` 程度を足す必要があります。

!!! note "ruff の4件は元からです"
    `bash scripts/dev/ruff-index.sh` をこのフェーズで触った5ファイルにかけると4件出ます
    ——`align_data.py:41` と `prepare_data.py:40` の UP035（`typing_extensions` からの
    `Annotated`）、`prepare_data.py:64` と `test_prepare_data.py:16` の FURB122。**4件とも
    `0eeb407` 時点のファイルで再現する既存の指摘**で、この作業が触った行ではありません。
    uvx が引く ruff が新しくなって既定ルールが増えたので見えているだけです。壊したわけでは
    ないので、無関係なコードを書き換えて黙らせないでください。

    5件目を作らないための注記: `prepare_moespeech.py` の `_japanese_segmenter_error()` は
    `except Exception` を**返します**（送出しないので BLE001 が付きます）。あの広さは意図した
    ものなので、理由を添えた `# noqa: BLE001` を1行だけ置いてあります。件数が5になっていたら
    この作業で増えたということなので、既存4件と混ぜずに直してください。

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
