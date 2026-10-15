# Vast.aiでの日本語GPU試運転

> **2026-10-15 最終状態:** 日本語モデル開発は終了し、全6モデルのpublic/manual gateでのHF保存公開と専用Vast.ai環境の削除を完了しました。本文は当時の設計・計画・測定の履歴です。現在の状態は[終了・保存公開サマリ](index.md)を参照してください。

目的は、少量の実コーパスから前処理・日本語アライメント・24層学習・保存・再開・公開APIによる音声生成まで通ることを確認することです。日本語の実用品質、M2aの未知話者評価、1,000時間規模の容量と費用は別の検証です。

2026-10-06に一連の試運転を完了しました。[結果・制約・次の計画](research/2026-10-06-vastai-gpu-smoke-results.md)を参照してください。

2026-10-07には[144条件の読み上げ機械評価](research/2026-10-07-japanese-reading-evaluation-results.md)もVast.aiで完了しました。人手試聴は未完了です。現在の実行順は[最新計画](plans/2026-10-07-next-training-and-release.md)を参照してください。今後の計算処理はすべてVast.ai、ローカルPCは編集・接続・転送・結果の閲覧だけに限定します。

その後、96GBのRTX PRO 6000・4,000GB指定の専用インスタンスでGOL単独の全量準備と[M2aの20kまでの学習・機械評価・結果回収](research/2026-10-07-gol-only-m2a-training-results.md)を完了しました。現在はデータを保持してGPUを停止し、10k/20kの人手比較を待っています。以下の150GBや試運転用の設定は完了済みの小規模試運転の記録であり、次のM2a実行手順ではありません。

## ローカル認証

リポジトリ直下のGit対象外の`.env`に`VASTAI_API`を設定します。標準名`VAST_API_KEY`も読み取れます。`.env`の値を既存プロセスの古いキーより優先し、Vast.ai CLIには子プロセスの環境変数`VAST_API_KEY`として渡します。CLIの既存キーファイルは書き換えません。

既存の制御用Python環境を使う場合のPowerShell例です。ローカルで日本語ML環境の同期や学習処理は実行しません。

```powershell
.\.venv\Scripts\python.exe -m training.scripts.vast_auth --check
```

この確認コマンドはGPUを作成しません。APIが認証エラーを返した場合は、CLI自身の終了コードが0でも失敗と判定します。

## Linux/NVIDIA環境

専用インスタンスを使い、選択時のGPU・ディスク・転送料金と終了期限を記録します。Vast.aiの[公式CLI手順](https://docs.vast.ai/cli/hello-world)に従ってSSH公開キーを登録し、料金を確認してレンタルします。今回の試運転は150 GBのディスクを確保します。M2aの容量要件として150 GBを使うことはできません。

ローカルの変更を含むソースを転送します。Git cloneまたは実際のHEADから作ったGit bundleで履歴・indexも用意し、その上に変更したファイルを重ねます。設定の回帰テストはtokenizerがGit管理されていることも確認するため、ソースだけのtar展開では3件が失敗します。転送した各ファイルのSHA-256をローカルと照合してから実行します。`.env`、Vast.aiのAPIキー、SSH秘密鍵を転送する必要はありません。GOLとMoeSpeechの取得には、利用承認済みのHugging Face認証をGPU環境に設定します。

```bash
cd /workspace/pocket-tts
uv sync --frozen --python 3.12 --group japanese
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
bash training/scripts/run_ja_smoke.sh
```

PyTorchのCUDA buildとホストdriverの互換性を確認します。今回の試験は`torch.compile`を有効にするため、CUDAを認識するだけでなくコンパイルと最初の更新まで確認します。

最初からの試験には新しいrun directoryを使ってください。既存checkpointがあると自動再開になるため、0→200→220の試験になりません。実行中のスクリプトは変更せず、変更後の試験は別runで行います。Linuxではtensor共有に利用可能な`file_descriptor`を選び、`torch_shm_manager`がresource trackerのpipeを保持して終了を妨げる問題の回帰テストも実行します。

## 試験内容と合格条件

1. GPU環境で設定・小型モデル・キャッシュの既存テストを実行します。
2. GOLとMoeSpeechから、それぞれ小さいアーカイブ2本を取得します。メタデータ・アーカイブを同じHub revisionに固定し、既存の展開・収集・連結・話者分割関数を使います。
3. 日本語CTCモデルもsnapshotを固定します。学習用・検証用の音声にGPUでアライメントを行い、学習側のスコア分布を測定してフィルタします。
4. `finetune_language_ja_smoke.yaml`で英語24層モデルから日本語へ200ステップ追加学習します。日本語SentencePiece 8,000語彙、micro-batch 16、勾配蓄積4、実効バッチ64、EMA、latent事前計算、compileを使用します。
5. 同じrun directoryで上限を220ステップに変更し、200ステップから再開します。optimizer内のstepも220に到達することを確認し、重みだけの再開と区別します。
6. 実チェックポイントとEMA exportを、それぞれ現行の`TTSModel`でロードします。EMA・通常推論のCFG1で、日常文・日付時刻・地名の3文を生成し、WAVの非空・非無音・有限値を確認します。
7. ログ、設定、データrevision、件数・時間・話者数、WAV、重み・optimizer、処理速度、GPUメモリ、容量を回収します。ローカル回収物のハッシュを照合後、今回作成したインスタンスをdestroyし、一覧から消えたことを確認します。

合格には、有限のlossと勾配、話者の重複がないtrain/valid、実際の学習・保存・optimizerを含む再開、両形式のロード、音声ファイル、回収・終了の証拠が必要です。単なるテスト成功やプロセス起動では合格にしません。

## 本番との違い

- 小さなアーカイブを使うため、試運転の話者・音声時間・スコア分布を本番へ一般化できません。
- GOLの話者floorは非singletonを残す試験用設定です。MoeSpeechのcutoffはこの小規模subsetのCER p90/MOS p10、アライメントcutoffは学習側per-frame score p5を使います。本番閾値の推薦ではありません。
- 試運転では最低1つのvalid話者ラベルを要求します。M2aの最低20ラベルという条件は変更しません。ラベル分離は実声優の分離を保証しません。
- 少量データ用にloader processとbucket poolを縮小し、warmupも20ステップにしています。
- 学習中の試聴サンプルはraw/CFG2/EOS −1、公開APIによる確認音声はEMA/CFG1/EOS −4です。参照音声も異なるため直接比較できません。最終runではEOSだけを変えた追加比較も行い、終了上限の警告を確認しました。
- 220ステップでのWAV生成だけでは日本語の正確さや自然さの証拠になりません。その後の旧144条件ASR評価でも正確な読み上げは未確認でした。現在はM2aの20kまでの機械評価を完了していますが、人手試聴・teacher選定・実用品質は未確認です。
- 試運転の処理時間・ピーク容量は、コンパイルの初期費用や小アーカイブの条件を含みます。1,000時間の費用へ単純比例させません。

主な結果は`runs/finetune_language_ja_smoke/verification.json`、`environment.json`、`progress.jsonl`、`data/ja-smoke/preparation.json`、`aligned_summary.json`に残ります。実データとモデルはGit対象外です。
