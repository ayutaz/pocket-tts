# Vast.ai日本語GPU試運転の結果と次の計画

> **2026-10-15 最終状態:** 日本語モデル開発は終了し、全6モデルのpublic/manual gateでのHF保存公開と専用Vast.ai環境の削除を完了しました。本文は当時の設計・計画・測定の履歴です。現在の状態は[終了・保存公開サマリ](../index.md)を参照してください。

確認日: 2026-10-06。対象ソースは`ce123ec119f56cfb4b01b3cb92c269c5d57fd133`に、この試運転の未コミット変更を重ねたもの。転送した126ファイルのSHA-256を確認した。

## 結論

Vast.ai認証・再現可能な環境準備と、小規模な実コーパスでの一連のGPU試運転を完了した。前処理、日本語CTCアライメント、24層teacherの200 step学習、保存、optimizerを含む220 stepまでの再開、現行`TTSModel`によるcheckpoint/EMA exportのロードと音声生成、成果物の回収、専用GPUの削除まで確認した。

日本語の実用品質は未確認。通常APIの生成音声は0.88〜0.96秒と短く、EOS設定だけを変えた比較でも上限まで終了しない例がある。学習の機械的成立と、全文を正しく自然に読むことは別の判定である。

## 環境と手順

| 項目 | 確認結果 |
|---|---|
| OS / Python | Linux、managed Python 3.12.14 |
| GPU / driver | NVIDIA A100-SXM4-40GB、580.178.04 |
| Torch / CUDA | 2.14.0+cu130 / 13.0、実際のCUDA演算成功 |
| イメージ | `vastai/base-image@sha256:384c89f43928ce4c9e96fb0a1127a7d6ec04aae6571c5cb25c98e3b395e25937` |
| 初期重み | `kyutai/pocket-tts`、revision `492522650173a0653b7575cdc25ae09810e5d741`、英語24層 |
| 日本語CTC | `vumichien/wav2vec2-large-xlsr-japanese-hiragana`、snapshot `017225bb128a6b1c6de9d58391f891908d69bc7b` |
| テキスト | 日本語SentencePiece 8,000語彙、embedding再初期化、`word_separator: ""` |
| 学習 | 約320.1M trainable parameters、LSD、micro-batch 16 × grad accumulation 4、lr 2e-4、warmup 20 |
| その他 | EMA 0.999、latent事前計算、`compile: true`、loader process 1、bucket pool 1 |

`.env`の`VASTAI_API`をCLI標準名`VAST_API_KEY`へ渡すヘルパーを追加した。認証確認自体はGPUを作成しない。キー、`.env`、SSH秘密鍵はソース転送物とGitに含めていない。手順は[GPU試運転runbook](../vastai-gpu-smoke.md)を参照。

最終確認は新しいrun directoryで最終版スクリプトを最初から実行したもの。途中の手動解除やソース変更なしで、jobとpipelineの終了コードはともに0。HF・取得データ・latent・torch compileのキャッシュは先行する診断runから再利用している。

## 実データ

GOL、MoeSpeechから小さいアーカイブを各2本取得し、既存の展開・収集・連結・話者分割処理を使用した。

| データ | 固定revision | 取得後の使用可能クリップ / 秒 |
|---|---|---|
| `midralab/gol-dataset` | `3548e28b35705caf2b9b8abd8f5b791ed040cfed` | 241 / 1,099.7586 |
| `ayousanz/moe-speech-plus` | `b63eb1c7353deaf83dcb506ea212b695280af1c9` | 197 / 1,458.0560 |

| 分割 | アライメント入力 | 成功 / skip | フィルタ後 | 音声時間 | 話者ラベル | loaderで切り出し可能 |
|---|---:|---:|---:|---:|---:|---:|
| train | 332 | 330 / 2 | 313 | 34.06分 | 3 | 275 / 313 (87.86%) |
| valid | 106 | 105 / 1 | 90 | 6.33分 | 1 | 69 / 90 (76.67%) |

train/validの話者ラベルは重複していない。ただし別ラベルが同じ声優を指すかという実人物の同一性は未監査。GOLの最低発話条件、MoeSpeechのCER p90/MOS p10、学習側アライメントper-frame score p5は、この小規模試運転だけの条件である。本番への閾値推薦ではない。M2aの未知話者最低20ラベルという条件は変更していない。

## 学習・再開・性能

- GPU環境の選択テストは **101 passed**。設定、前計算、loaderの終了・signal、実更新の検証を含む。
- 実更新区間は **0→200、200→220**。更新のない再起動は最終runでは0回。
- optimizerに記録されたstepも **220**。checkpointとEMA exportを、それぞれ現行公開Python APIでロードできた。
- ログに記録された学習・検証metricと勾配は有限値。
- 最終一連の処理は **514秒（8分34秒）**。キャッシュを再利用した条件なので、初回の全取得時間ではない。
- 通常更新の中央値は **1.3317 step/s**、範囲0.9225〜1.5135。step 20未満、試聴・検証・保存を含む110、再開コンパイルを含む201を集計から除外した。
- CUDA allocatorのピークは最初の学習 **11.9 GiB**、再開 **11.6 GiB**。GPU全体の使用量を測った値ではない。
- 最初の更新は初回・再開とも約20秒。compile cacheを再利用している。

`torch.compile`はattentionの`.item()`によるgraph breakやrecompile limitの警告を出した。一部のストリーミング推論はeagerへ戻る。全処理のコンパイル成功や1,000時間での同じ速度は主張しない。

| step | train loss | valid loss |
|---|---:|---:|
| 100 | 0.137653 | 0.812581 |
| 200 | 0.066123 | 1.028732 |
| 220 | 0.027969 | 未実行 |

train loss低下に対してvalid lossは上昇した。未知話者1ラベル、検証2 batchという小さな条件から、日本語品質や汎化を判断しない。

## 音声生成とEOSの問題

最終EMA、CFG1、temperature 0.3、同じheld-out音声5秒、seed 42〜44で3文を生成した。WAVは24 kHz、有限値、非空、非無音。通常APIのEOS閾値は−4。学習時試聴の既定値は−1で、さらにraw重み・CFG2・参照音声も異なるため、学習時WAVとの直接比較はできない。

EOSだけを−4から−1へ変えた追加比較を実施した。他の条件は同一。

| 文 | EOS −4 | EOS −1 |
|---|---:|---:|
| こんにちは、今日はいい天気ですね。 | 0.96秒 | 1.12秒 |
| 今日は十月六日、午後三時です。 | 0.88秒 | 6.00秒 |
| 東京駅の近くで、温かいお茶を飲みました。 | 0.96秒 | 7.04秒 |

比較runは「Maximum generation length reached without EOS」を2件記録した。長くなったことは、全文を正しく読み、適切に終了した証拠にならない。人手試聴、日本語ASRによるCER、話者再現性は未評価である。

## 成果物・容量・費用・終了

ローカルの`runs/vast-smoke-artifacts/final/`に400ファイル、合計6,730,776,347 bytesを回収した。全400件のサイズとSHA-256、ソース126件のハッシュ、JUnitの101件成功、WAV10本の有限値と非無音を確認済み。220 step checkpoint、optimizer、EMA exportを含む。EOS比較tarも別途サイズ・SHA-256を確認して回収した。

回収後、Windows上でもEMA exportを現行`TTSModel`でCPUへロードできた。ローカルの重み・tokenizerを絶対パスで指定する`runs/vast-smoke-artifacts/final/runs/finetune_language_ja_smoke/inference.local.yaml`を作成した。これはロード確認のみで、Windowsでの生成速度や音声品質の確認ではない。

| 主要成果物 | SHA-256 |
|---|---|
| `checkpoint_00000220.pt` | `f777fb50f9b94b123012b94e712d2b362b867c43795abacecd619f48c5dacff9` |
| `optim_00000220.pt` | `5a6a372d0e30f8822f4bc9f68673acbc41bfa95de47b983ad11f98e4373484c4` |
| `model.safetensors` | `ecc71eb63341c812b96d5ea5996ca8976d71f821a1c960e64a4e3cd71b03eaac` |

クラウドでの使用量は、試運転データ0.93 GB、最終run 9.05 GB、venv 6.46 GB、HF cache 5.83 GB。`df`は48.02 GB使用だったが、先行する診断runと古い大きな回収archiveも含む。単独実行のピーク容量ではない。M2aの1.6 TB運用、取得archive削除後の再開、latentと約100 GiBのcheckpoint保持は未実測。

| 試行 | 結果 | レンタル壁時計 | API取得時点の課金額 (USD) |
|---|---|---:|---:|
| 1 | 旧イメージのAPT mirrorで起動が停滞、学習前に削除 | 0.3914時間 | 0.023 |
| 2 | PyPI・SSH転送が極端に遅いホスト、学習前に削除 | 0.2773時間 | 0.108 |
| 3 | 最終試運転成功、回収後に削除 | 0.9055時間 | 0.679 |
| 合計 | | | **0.810** |

費用は2026-10-06 11:52 UTC頃に、各専用instance labelで絞ったAPI課金明細のamountを合算したもの。削除直後のため明細が後から更新される可能性がある。アカウント残高の差額ではない。試行3の選択時見積りは150 GB込み$0.722311/h。

今回作成した3インスタンスはすべてdestroy成功かつAPI一覧からの消失を確認した。ほかのインスタンスは操作していない。期限監視プロセスも削除後に自然終了した。

## 試運転で修正した問題

1. ソースtarだけではGit管理されたtokenizerを確認する3テストが失敗した。実HEADのGit bundleを復元して、未コミット変更を重ねる手順へ修正した。
2. Linuxで`file_system`共有の`torch_shm_manager`がresource trackerのpipeを保持し、学習終了後にtrainerが停止した。実tensor IPCを行う失敗回帰テストで再現し、利用可能な環境で`file_descriptor`共有を選ぶよう修正した。Linux回帰テストとsignalテストは修正後に通った。
3. 診断中に実行中のシェルスクリプトを書き換え、200 stepの更新なし再起動が一度生じた。実更新区間と無更新起動を区別する検証を追加した。その診断runは最終結果から分離し、最終スクリプトを新規runで最後まで実行し直した。

機械的検証の補足: ローカルの対象既存93テスト、認証3テスト、実更新の回帰6テストが通った。loader終了テストはWindowsでskip、Linuxで実行している。リポジトリ指定Ruff 0.12.7のcheck/import順序、変更Pythonのformat、Linux向けty（Python 3.10/3.14を含む）も成功。実際のGitHub CI実行・公開・pushは行っていない。

## 現在の進捗と計画（2026-10-07追記）

この報告は10月6日の機能試運転の記録であり、実用品質の合格ではない。その後、[10月7日の読み上げ比較](2026-10-07-japanese-reading-evaluation-results.md)で固定6文・2参照音声・144条件の生成とASR評価をVast.ai上で完了した。推論設定だけでは正確な日本語の全文読み上げを確認できず、人手試聴は未完了である。後続の144評価は、本報告の初期3本のWAVとは別の生成結果である。

| 段階 | 現在の状態 |
|---|---|
| 固定条件による機械評価 | 完了。raw/EMA・CFG・EOS・生成経路別に記録。CER 95.94〜122.19% |
| 人手試聴とM2a向け評価の一般化 | 未完了。現在の評価スクリプトは試運転のrun/data/2参照を前提にする |
| GOL単独、約1,000時間を目標とした準備 | 10月7日のユーザー指定を反映。未実行。話者・転写・重複・長尺行・容量・整理後の再開を監査する |
| 英語24層からの新規M2a | 未実行。早期診断を経て上限40k、1,000ごとに保存・検証。今回の220 stepを再開元にはしない |
| 6層蒸留・CPU実用性・公開 | 未実行。teacherの品質を確認してから判断する |

実行順・完了条件・費用の扱いは[最新計画](../plans/2026-10-07-next-training-and-release.md)を基準にする。今後の前処理・学習・推論・ASR・評価・負荷の高いテストはすべてVast.aiで行い、CPU性能測定もVast.ai上で実施する。Windowsネイティブ動作は別の未検証項目であり、Linux CPUの結果から保証しない。ローカルPCは編集・接続・転送・結果の閲覧に限定する。本報告のローカル試験は方針指定前の歴史的記録である。

大規模M2aの総費用は未確定。小規模試運転や8月の料金表から断定せず、取得・アライメント・latent事前計算・容量・保持checkpoint・評価・回収を含めて見積り直す。

機械可読の確認結果は[証拠JSON](2026-10-06-vastai-gpu-smoke-evidence.json)に保存する。実データと重みはGit対象外。
