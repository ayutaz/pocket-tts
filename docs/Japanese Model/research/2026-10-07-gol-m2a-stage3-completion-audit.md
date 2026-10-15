# ステップ1〜3の要件照合と残る試聴

> **2026-10-15 最終状態:** 日本語モデル開発は終了し、全6モデルのpublic/manual gateでのHF保存公開と専用Vast.ai環境の削除を完了しました。本文は当時の設計・計画・測定の履歴です。現在の状態は[終了・保存公開サマリ](../index.md)を参照してください。

本ページは「3まで」の実行後に作成した照合履歴である。その後の指示は「4まで、公開はまだ行わない」で、会話の4には6層蒸留とCPU評価を含む。現在の範囲・インスタンス状態・蒸留進捗は[最新計画](../plans/2026-10-07-next-training-and-release.md)と[蒸留・CPU結果](2026-10-07-japanese-distillation-and-cpu-results.md)を参照する。以下のGPU停止と蒸留未実行は当時の状態であり、人手試聴・teacherの品質選定が未完了である点は引き続き有効である。

更新日: 2026-10-07。対象は[最新計画](../plans/2026-10-07-next-training-and-release.md)の1〜3と「GPUが決まったら3まで進める」という指示。4の追加学習・蒸留、5のCPU比較・公開を完了範囲に含めない。ステップ3はまだ完了していない。

前の実行では20kの全量結果・試聴フォルダ回収、継続判断、所有GPU停止まで進めた。本照合では回収ファイルと証拠JSONを読み、APIで停止状態を再確認した。ML処理・テストはローカルで再実行していない。

## 要件と証拠

| 計画の要件 | 現在の証拠と範囲 | 判定 |
|---|---|---|
| 大容量ディスク・高速回線のGPUを選び、実環境を確認 | [選定結果](../plans/2026-10-07-next-training-and-release.md#gpuの選定結果2026-10-07更新)。RTX PRO 6000 96GB、4,000GB指定、CUDA認識。回線は表示値であり回線全体の実効速度ではない | 実行環境を確保・使用済み |
| 処理をVast.aiに限定し、ソース・設定・revision・環境を固定 | [ソース保存の照合](2026-10-07-gol-source-archive-verification.json)、各保存点のevaluation context。135学習ファイル・136評価ファイルと環境を保存。PCは編集・制御・転送・結果閲覧・回収照合 | 実行記録あり |
| 1: 任意のM2a checkpointを明示して評価 | [学習・評価記録](2026-10-07-gol-only-m2a-training-results.md)。2k/3k/5k/10k/20kを明示して生成・ASR・比較。既存6文と拡張12文を固定 | 実データで実行済み |
| 1: 最低20 valid話者ラベル、全条件の充足と参照一致 | [20k全量診断](2026-10-07-gol-m2a-20000-full-diagnostics.json)。20 valid IDグループ＋1 train参照、12文、12 profile＝3,024条件。実声優の人数・同一性を保証しない | 固定matrix検証済み |
| 1: raw/EMA・生成経路・CFG・EOS・seed・ASR等を識別 | [10k対20k全量比較](2026-10-07-gol-m2a-10000-vs-20000-full.json)、[共通216条件](2026-10-07-gol-m2a-20000-early-vs-full-common.json)。固定条件一致、共通216件のPCM・転写・score一致 | 比較条件を検証済み |
| 1: 個別音声・ログ・試聴メモの対応 | [20k全量試聴UI検証](2026-10-07-gol-m2a-20000-full-offline-review-verification.json)。Vast Linux Chromiumで3参照×12文＝36件の再生開始とメモ識別を確認。全件の人手試聴やWindows再生ではない | 機械的な対応は検証済み、人手記録未受領 |
| 1: 旧220 step音声の発音・全文・反復・終了を聴き、メモを保存 | [旧144条件の評価](2026-10-07-japanese-reading-evaluation-results.md)、回収済み`runs/japanese-eval-20261007-vast/review.html`の存在を確認 | 人手試聴未完了 |
| 2: GOLのみ約1,000時間を取得、MoeSpeechを混ぜない | [準備記録](2026-10-07-gol-only-m2a-preparation.md)。固定GOL revision、18作品・1,003.736時間、21アーカイブ。既存SP8000のGOL＋MoeSpeech由来は別途開示 | 音声データ準備済み |
| 2: floor/cutoff・採否理由、重複・ラベル分離・長尺・語彙の監査 | [最終監査](2026-10-07-gol-final-data-audit.json)。train 723.507h・447,090行、valid 22.329h・12,951行。元PCM・ラベルID分割、窓と転写、30秒以下を確認。近似音声・声優同一性は監査対象外 | 定義した機械監査を完了 |
| 2: Mimi latentsの事前計算・識別・再利用 | [latents実測](2026-10-07-gol-latents-resources.json)。全447,090行のヘッダーとmanifest/Mimi identityを検証。validationは音声manifest | 完了 |
| 2: 本番batch・長さ分布のVRAM/速度・保存再開・容量/費用見積り | [負荷試験](2026-10-07-gol-m2a-training-benchmark.json)、[準備資源測定](2026-10-07-gol-preparation-resources.json)、準備記録の最長64行試験。本番16×4、0→30→35と実optimizer counter、最長batchの有限値を確認。観測VRAM・ディスクと選定単価を保存。請求総額や全工程の瞬間ピークは未確定 | 段階別実測と見積りあり、請求額とは区別 |
| 3: 英語24層から新規SP8000 M2a、本番設定を維持 | 学習・評価記録、[20k保存状態](2026-10-07-gol-m2a-20000-checkpoint-verification.json)。toy/負荷試験重みを開始元にせず、実効64・constant lr2e-4、warmup1000。optimizer実counter20,000とraw/EMA/optimizer有限値を直接確認 | 20kまで実行・検証済み |
| 3: 保存点を比較し、問題があれば診断。40kは条件付き上限 | 2k/3kの初期診断、5k/10k/20kの全3,024条件。[20k後の判断](2026-10-07-gol-m2a-20000-review-gate.json)で主要条件と長文の悪化、他10 profileの改善を併記。30k/40kは未実行 | 段階実行済み。試聴前に次区間を開始しない |
| 3: 必要保存点と再開状態の回収・照合 | [10k組](2026-10-07-gol-m2a-10000-local-recovery.json)、[20k組](2026-10-07-gol-m2a-20000-local-recovery.json)、両全量archive・試聴フォルダreceipt。回収したCP/optimizerのサイズも再確認。2kの組・5k CPも存在。3k/5kの古いoptimizerは削除済みで復元を主張しない | 比較候補10k/20kの保存・再開状態は確保 |
| 3: 複数参照の全文・反復・発音等を試聴しteacherを選定 | [20k後の判断](2026-10-07-gol-m2a-20000-review-gate.json)で`human_listening_completed=false`、`teacher_selected=false`。10kは機械評価上の優先候補に留まる | 未完了 |
| 所有インスタンスを回収照合後にdestroy。他プロジェクトは操作しない | [停止記録](2026-10-07-gol-m2a-20000-instance-stopped.json)。2026-10-07 01:35:33 UTCのAPI再照会でも所有54521683は`cur_state/intended_status=stopped`。削除しておらずデータ・ストレージ課金は残る | GPU停止済み。最終destroy未完了 |

## 試聴の具体的な手順

回収済みの10k全量ページ（ローカル成果物: `runs/vast-m2a-control/offline-review/japanese-m2a-eval-00010000-full/review.html`）と20k全量ページ（ローカル成果物: `runs/vast-m2a-control/offline-review/japanese-m2a-eval-00020000-full/review.html`）を開く。ページと音声の相対配置は回収済みで、インスタンスやSSHトンネルを起動する必要はない。

1. 両ページの生成経路を`trainer`にし、カード`ema / CFG 2 / EOS -4`を比較する。同じ文章・声・条件同士を聴く。
2. 最初に`held_00`・`held_09`・`held_16`の`long`と`long_sequence`を最後まで比較する。末尾の反復、途中の抜け、途中終了、声の崩れを記録する。この小セットは最初の診断であり、それだけでteacherを決定しない。
3. `date`・`digits`・`proper_name`・`units`も同じ声で比較し、数字や固有名詞の実際の発音を確認する。さらに`held_01`・`held_02`と通常文・短文へ広げ、複数参照で全文・自然さ・終了位置を確認する。
4. 必要に応じてEMAの他条件、raw、通常APIのEMAも同じ経路内で比較する。trainer CFG2の結果を通常APIの結果として扱わない。
5. 各音声の欄に「末尾の○○を2回繰り返す」「第2文が抜ける」「全文を読める」等の観察を入れ、ページごとの「試聴メモを保存」でJSONを保存する。10k/20kの保存ファイル名を区別する。checkpoint・音声SHA・文章・条件はJSONに付く。メモ保存の自動フラグは`notes_only`であり、人手確認の全範囲が完了した証拠にはしない。

旧220 stepも計画1の確認対象に残る。旧ページ（ローカル成果物: `runs/japanese-eval-20261007-vast/review.html`）の同じ6文について発音・全文・反復・終了のメモを保存する。旧モデルとM2aではデータ・参照・matrixが違うため、そのCER差を同一条件の改善率にしない。

## この時点で確定しないこと

teacher品質、実用品質、話者類似度、Windows再生、CPU性能、公開可否は未証明。CERやEOSで人手所見を作らず、未回答を試聴済み・承認済みと解釈しない。試聴所見を受けてteacherの選定または次の限定区間を判断し、必要な成果物を揃えて所有インスタンスを最終削除する。現時点ではこの人手所見が、品質判断を先へ進めるために欠けている入力である。
