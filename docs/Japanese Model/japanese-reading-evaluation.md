# 日本語読み上げ・EOSの比較評価

> **2026-10-15 最終状態:** 日本語モデル開発は終了し、全6モデルのpublic/manual gateでのHF保存公開と専用Vast.ai環境の削除を完了しました。本文は当時の設計・計画・測定の履歴です。現在の状態は[終了・保存公開サマリ](index.md)を参照してください。

2026-10-07にVast.aiで220 stepモデルの144条件、その後GOL単独M2aの2k/3k/5k/10k/20k初期216条件と5k/10k/20kの拡張3,024条件を評価しました。[M2aの結果と回収](research/2026-10-07-gol-only-m2a-training-results.md)、[旧220 stepの結果](research/2026-10-07-japanese-reading-evaluation-results.md)、[最新計画](plans/2026-10-07-next-training-and-release.md)を参照してください。任意のrun・checkpoint・train/valid manifest・参照音声・評価文を明示指定する一般化は実装・実評価済みです。現在は人手試聴・teacher選定待ちで、専用インスタンスはデータを保持して停止中です。計算ジョブは稼働していません。

## M2aの固定評価

計算は専用Vast.aiインスタンスで行います。監査済み`train_bounded.jsonl`・`valid_bounded.jsonl`から参照を選び、`runs/m2a-control/m2a-prompts-full.json`に20のvalid話者IDグループと1つのtrain参照を固定します。5〜10秒、完全な単語alignment、未知tokenなし、通常の文章という条件を満たす候補から、固定seedのハッシュ順と作品間の巡回で選びます。合成結果やCERは選定に使いません。IDグループの分離は実声優の同一性を保証しません。

初期2k/3k診断の`m2a-prompts-early.json`は、そのうち同じ2つのvalid参照とtrain参照です。既存6文・12生成条件で216件。全21参照と`japanese_m2a_cases.json`の12文による評価は3,024件です。保存点同士は同じ参照、文章・seed、生成設定、ASR snapshot・設定で比較します。旧220 step評価とは参照もASRも異なるため、CERを直接比較して改善率を出しません。

M2aではASRを`Systran/faster-whisper-large-v3`、revision `edaa852ec7e145841d8ffdb056a99866b5f0a478`に固定しています。CUDA float16、chunk length 15、beam size 5、temperature 0、前の文からの条件付け・VAD・initial promptなしで実行し、実音声の対照も同じASRで転写します。CUDA 12用のASRライブラリはメインのCUDA 13版Torch環境と別に配置し、ASRの起動時だけライブラリのパスを指定します。

以下は実行済み2k初期評価のコマンド例です。現在は停止中で、再実行を指示するものではありません。追加評価が必要になった場合は、学習の正常終了とcheckpoint・optimizerの実更新・保存状態を確認し、既存結果を変更しない専用出力先を用意します。`inference.local.yaml`は日本語24層設定のコピーであり、Mimiは学習と同じ固定英語teacher bundleを読み、日本語FlowLMは明示したcheckpointから読みます。既存CUDA環境ではCPU向けTorch indexを使う`uv sync`を再実行せず、記録済み環境を`--no-sync`で使います。

```bash
cd /workspace/pocket-tts
export UV_PYTHON=3.12 UV_CACHE_DIR=/workspace/uv-cache
# runs/finetune_language_ja_m2a/inference.local.yaml と固定参照JSONを用意済みの場合
uv run --no-sync python -m training.eval.japanese generate \
  --artifacts-root /workspace/pocket-tts \
  --out runs/japanese-m2a-eval-00002000-early/shard-0 --device cuda \
  --run-dir runs/finetune_language_ja_m2a \
  --checkpoint runs/finetune_language_ja_m2a/checkpoint_00002000.pt \
  --train-manifest data/ja-gol-only-m2a/train_bounded.jsonl \
  --valid-manifest data/ja-gol-only-m2a/valid_bounded.jsonl \
  --prompts-file runs/m2a-control/m2a-prompts-early.json \
  --cases-file training/eval/japanese_cases.json

LD_LIBRARY_PATH=/workspace/asr-cuda12/nvidia/cublas/lib:/workspace/asr-cuda12/nvidia/cudnn/lib:/workspace/asr-cuda12/nvidia/cuda_nvrtc/lib \
uv run --no-sync python -m training.eval.japanese score \
  --out runs/japanese-m2a-eval-00002000-early/shard-0 \
  --device cuda --asr-compute-type float16 --asr-threads 4 \
  --asr-path /root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478

uv run --no-sync python -m training.eval.japanese_merge \
  runs/japanese-m2a-eval-00002000-early --shards 1
```

2k/3k/5k/10k/20kの初期評価と5k/10k/20kの全参照・12文評価は正常終了しました。待機・生成・ASRジョブは終了済みです。同じ出力先に別ジョブを起動しないでください。追加の保存点を評価する場合もcheckpointと新しい出力先を対応させ、参照と評価条件を維持します。全参照・12文の評価では参照JSONと文章ファイルをfull用に切り替え、初期診断と共通する216条件は別途照合します。

M2a試聴ページは保存点ごとに作成・回収済みです。10k/20kの全量フォルダは全3,067ファイルの転送後ハッシュが一致し、Vast.ai Linux Chromiumでは各36音声の再生開始とメモ識別対応を検証しました。Windows再生・人手試聴は未確認です。[回収済みページと試聴手順](research/2026-10-07-gol-m2a-stage3-completion-audit.md#試聴の具体的な手順)を使います。メモは音声の識別情報に紐づきますが、保存だけで人手評価全体の完了とは扱いません。

各実行済み保存点では、解決済み学習設定、progress、実更新・保存状態の検証結果、参照選定と評価ソースの識別情報を`input_snapshot/`と`evaluation_context.json`に保存しました。後続の再開で`args.yaml`が上書きされても、その保存点の条件を照合できます。今後も次の区間を開始する前にcontextを保存・照合します。

Vast.ai上の`runs/m2a-control/compare_evaluations.py`は、この保存済みcontextとWAVハッシュを確認し、文章・seed・参照PCM・生成条件・ASR条件・評価ソース・学習設定が一致する保存点だけを比較します。学習設定で異なってよいのは段階停止の`max_steps`だけです。参照・文章の行列が異なるearly/full評価はこのツールでは直接比較しません。条件・参照ごとに文字数で重み付けしたCERと置換・削除・挿入数、1秒未満の出力、音声長・終了結果、validation履歴を併記します。CER差だけでteacherを自動選定しません。比較の6テスト、設定保存の2テストは人工的なfixtureでの制御検証であり、実際のM2aの改善を示すものではありません。

```bash
# Vast.ai。各評価のcontextを保存済みの場合
uv run --no-sync python runs/m2a-control/compare_evaluations.py \
  runs/japanese-m2a-eval-00002000-early \
  runs/japanese-m2a-eval-00003000-early \
  --out runs/m2a-control/comparison-2000-to3000.json
```

制御スクリプトと実行証拠はこのインスタンスのGit対象外の`runs/m2a-control/`に配置し、回収対象に含めます。これらのパスが新規cloneにも存在することは前提にしません。

## 220 step評価の記録

この評価は、保存済みモデルが指定した日本語を読めるか、途中で終了しないか、上限まで終了しないかを調べるための診断である。未知話者一般への品質保証、自然さ、話者類似度の本格的なベンチマークではない。

## 固定するもの

- 評価文は[`training/eval/japanese_cases.json`](https://huggingface.co/tts-dataset/pocket-tts-ja-experiments/blob/9cf5427bda9ccf18a9dadb736fa686a4bc990785/source/stage4-source.tar)。日常文2文、日時、地名、数字、長文の計6文。各文のseedは42〜47で固定する。
- 2026-10-06の試運転220 step checkpointのrawとEMAを比較する。元のcheckpointとEMA exportのSHA-256も記録する。
- 学習済み話者ラベルとheld-out話者ラベルから、参照音声を1つずつ選ぶ。5秒の波形を保存し、各条件で同じ波形を使う。ラベル間の実声優の同一性は未監査である。
- temperature 0.3、decode 1 step、モデルfloat32。完了した144条件はVast.aiのCUDAで生成した。各生成の直前にseedを設定し直す。重みを変更したときは音声のKV stateを作り直す。
- 学習側はraw/EMA × CFG1/2 × EOS −4/−1の8条件。通常APIはCFG1でraw/EMA × EOS −4/−1の4条件。2声 × 6文 × 12条件 = 144件。

## 生成経路の違い

| 項目 | 学習側 `TrainableTTS.generate` | 通常 `TTSModel.generate_audio` |
|---|---|---|
| CFG | 1 / 2を指定可能 | CFG1のみ |
| EOSを認める最初のframe | 0 | 6 |
| EOS後に出すframe | 今回のcountdown 1では、EOSを出したframeを追加しない | 自動選択。通常の短い日本語入力では5 frame |
| 長さ上限 | 375 frame = 30秒 | tokenizerのtoken数から推定、text chunkごと |
| 長文 | 今回は全文を1回で生成 | 通常の50 token上限で分割される場合がある |
| Mimi decode | latent全体をdecode | ストリーミングdecodeと最初のfade-in |

同じ経路内でEOS、CFG、raw/EMAを比較する。学習側と通常APIの音声長差から、EOS閾値だけが原因だと結論しない。各経路のEOS head出力と終了条件を記録し、通常APIの生成処理は置き換えず観測だけを追加する。

乱数を消費する順序も経路間で異なる。通常APIはテキストの条件付け時にもFlowLMを呼び、未使用のlatent用noiseを生成してから音声の生成へ進む。学習側はprefixと最初のlatentをまとめて処理する。同じseedは各経路内の比較を固定するが、経路間で同じlatent noise列を使った比較とはならない。

CPUとCUDAでも、同じseedが同一の乱数列を意味するわけではない。中断したローカルCPUの音声とVast.aiのCUDA音声を、同じnoiseによる対として比較しない。

## CERと読み落としの扱い

日本語ASRは[`kotoba-tech/kotoba-whisper-v2.0-faster`](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster)の固定snapshotを使用する。[faster-whisper](https://github.com/SYSTRAN/faster-whisper)をVast.aiのCUDA float16で実行し、language=`ja`、beam size 5、temperature 0、前の文からの条件付けなし、VADなし、正解文を渡すinitial promptなしで転写する。device、compute type、model.binと評価コードのハッシュ、実際のパッケージversionも保存する。中断したCPU int8の途中結果と、CUDA float16での結果は混ぜない。

評価前に、学習・検証コーパスの実音声を3本ずつ転写する。同じASRが日本語を認識できるか、転写・表記差による誤差がどの程度あるかを確認する。

CERはWhisper BasicTextNormalizerで濁点を保持し、空白を除去して算出する。句読点を除いた参照文字数で重み付けしたcorpus CERと、置換・削除・挿入数を報告する。空の転写も削除として含める。単純な文ごとのCER平均ではない。

CERは挿入が多い場合に100%を超える。反復や無関係な語の転写を上限100%へ切り詰めない。

これは表記に対するCERである。漢字/かな、カタカナ/ひらがな、数字の表記違いも誤りとして入る。ASRの削除数を、モデルが音声として読み落とした文字数と同一視しない。試聴で確認する必要がある。1秒未満の音声は診断用に数えるが、読み落としが確定したラベルにはしない。

終了結果は次を区別する。

- `eos`: EOSを受け入れ、その後の必要frameまで出して終了。
- `max_length_without_eos`: EOSを受け入れず上限に到達。
- `max_length_after_eos`: EOSは受け入れたが、その後のframeを出し切る前に上限に到達。
- `incomplete`: 上記のどれでもない未完了。

## 実行

**2026-10-07のユーザー指定により、生成・ASR・集計・テストはすべてVast.ai上で実行する。ローカルPCでは接続・転送・結果確認だけを行い、評価ジョブを動かさない。**

回収済みの試運転成果物をVast.aiへ転送して使う。`inference.local.yaml`はVast.aiインスタンス内の重みとtokenizerを指す必要がある。ここでのファイル名の`local`はインスタンス内のパスを意味する。評価は学習を行わない。以下のコマンドはVast.aiのSSHセッションで実行する。

```bash
# 記録済みCUDA環境を使用する
uv run --no-sync python -m training.eval.japanese generate --device cuda \
  --artifacts-root runs/vast-smoke-artifacts/final \
  --out runs/japanese-eval
uv run --no-sync python -m training.eval.japanese score --device cuda --asr-compute-type float16 \
  --out runs/japanese-eval \
  --asr-path <固定したCTranslate2のローカルsnapshotディレクトリ>
uv run --no-sync python -m training.eval.japanese_review runs/japanese-eval
```

上記はVast.aiのSSHセッションで実行する。faster-whisperはdev dependencyとしてuvで管理している。CLIのdevice既定値はCUDA。Vast.aiのCPUで実行する場合に限り`--device cpu`、ASRには`--asr-compute-type int8`を明示する。

並列化する場合は、独立したプロセスで`--shards 4 --shard 0`〜`3`を指定し、出力先を`runs/japanese-eval/shard-0`〜`shard-3`へ分ける。それぞれgenerate、scoreを完了した後、統合する。

```bash
uv run --no-sync python -m training.eval.japanese_merge runs/japanese-eval --shards 4
```

統合時には、入力・設定の一致、全144条件の一意性と欠落、WAVのハッシュ、参照音声のPCM一致を検証する。FLOAT WAVには書き込み時刻を含むPEAK headerがあるため、参照音声は各ファイルのcontainer hashを個別検証したうえで、sample rateと波形sampleを比較する。

同じ出力先で再開する場合は、checkpoint、tokenizer、音声入力、評価文、設定、評価スクリプトのハッシュが一致することを要求する。完了済みWAVもハッシュを確認する。異なる入力やコードでは別の出力先を使う。

## 成果物と人手評価

今回の実行記録は[2026-10-07の評価結果](research/2026-10-07-japanese-reading-evaluation-results.md)にまとめる。

`generation.json`、`scores.json`、`identity.json`、`asr_identity.json`に機械評価の証拠を保存する。各WAVの個別JSONにはEOS headのtraceも入る。実音声とモデル重みはGit対象外。

`review.html`は、文・声・生成経路を選んで各条件を試聴するローカルページである。ASR転写、CER、音声長、終了結果を表示する。全文の読み落とし、発音、自然さ、反復、終了位置は人が確認し、メモをJSONへ保存する。機械評価完了と人手試聴完了は別々に記録する。

この220 stepモデルから推論設定を選んでも、本番の最適設定とは言えない。十分に学習したteacherでも同じ評価を行い、未知話者を増やして再検証する。

2026-10-06の試運転trainは34分・3話者ラベルで、全trainマニフェストの全文をtokenizeしても8,000語彙中1,859種類しか現れない。評価文には「天」「駅」「茶」など、train全文にない文字も含まれる。これは実際に切り出された学習batchでのtoken出現回数ではなく、全文に対する包含状況である。未学習の文字・表現が多い小規模モデルから、推論設定だけで日本語品質を確立できるとは考えない。
