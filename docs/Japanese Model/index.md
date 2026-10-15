# 日本語モデル：開発終了・成果物保存済み

最終更新: 2026-10-15。調査開始: 2026-08-27。作業ブランチ: `japanese-model-training`。

## 現在の状態

[`takuma104/pocket-tts-ja`](https://huggingface.co/takuma104/pocket-tts-ja)の公開を受け、ユーザー指定により日本語モデル開発を終了しました。追加学習・蒸留・最終モデルの品質選定は予定していません。その後に許可された既存成果物の保存公開を完了しています。

保存先は **[`tts-dataset/pocket-tts-ja-experiments`](https://huggingface.co/tts-dataset/pocket-tts-ja-experiments)** です。モデルページはpublic、ファイル取得はmanual gateです。全6モデルを含む31ファイル・12,062,580,944 bytesを転送し、サイズ・LFS SHA256・添付文書の内容を照合しました。最終revisionは`9cf5427bda9ccf18a9dadb736fa686a4bc990785`です。[保存公開の記録](research/2026-10-15-huggingface-archive-publication.md)と[照合receipt](research/2026-10-15-huggingface-experiments-publication.json)を参照してください。

**研究実験のアーカイブであり、実用品質の合格モデルではありません。** 人手試聴・teacher選定は未完了のまま終了しました。開発終了と、学習・保存・公開の技術的な完了を区別します。

## 保存したモデル

| モデル | HF内の保存先 | 状態 |
|---|---|---|
| GOL teacher 2k | `gol/teacher/2000/checkpoint_00002000.pt` | 24層、raw/EMA checkpoint |
| GOL teacher 5k | `gol/teacher/5000/checkpoint_00005000.pt` | 24層、raw/EMA checkpoint |
| GOL teacher 10k | `gol/teacher/10000/checkpoint_00010000.pt` | 24層、暫定蒸留teacher |
| GOL teacher 20k | `gol/teacher/20000/checkpoint_00020000.pt` | 24層、比較候補 |
| GOL student 5k | `gol/student/5000/model.safetensors` | 6層、検証済みmerged EMA export |
| 旧220-step検証モデル | `legacy/smoke220/model.safetensors` | 小規模パイプライン検証。M2aの初期値には不使用 |

SP8000、学習・推論設定、固定ソース、評価記録も同梱しています。実験時の日本語実装はHFの`source/stage4-source.tar`に保存しています。teacherの`.pt`は学習checkpointで、単体の推論用safetensorsではありません。異なる版の公式wheelとの互換性は新たに検証していません。

8月の検証用 **[`ayousanz/pocket-tts-ja-phase1`](https://huggingface.co/ayousanz/pocket-tts-ja-phase1)** は別モデルとして変更せずに残しています。Phase 1、10月の220-step検証、GOL M2aを混同しません。

## 完了した作業と未達の品質

| 項目 | 終了時点の結果 |
|---|---|
| 日本語前処理・SP8000・alignment・評価の実装 | 実装し実データで実行済み。固定ソースをHFに保存 |
| GOL単独データ準備 | train 723.507h / 447,090行 / 327 IDグループ、valid 22.329h / 12,951行 / 106グループ |
| 日本語24層teacher学習 | 20,000 optimizer updatesまで実行。5k/10k/20kを全3,024条件で評価 |
| 日本語24層→6層蒸留 | 暫定teacher10k EMAから新規studentを5,000 updatesまで学習 |
| teacher10k対student5k | 主要条件CER **12.24% → 51.22%**。studentの読み・長文・終了に大きな品質差 |
| teacher10k対teacher20k | 主要条件CER **12.24% → 15.01%**。他profileの改善もあり、一律の品質悪化とは断定しない |
| CPU評価 | Vast.ai Linux CPUでFP32/INT8、固定12文×2回×4構成＝96リクエスト。異なるguidance・品質の速度値を同品質の比較にしない |
| export・保存再開 | checkpointなしexportロードと1通常文のPCM一致、raw/EMA/optimizerの復元と5000→5005の実更新を確認 |
| 人手試聴 | `held_00 / daily`のteacher/student一組のみ。teacherは入力文と一致、studentの「っすね。きっ」は実発話と確認 |
| 最終品質選定 | 長文・反復・自然さ・終了位置のレビューとteacher選定は未完了。追加作業をせず終了 |
| 保存公開 | public/manual gateで全6モデル・31ファイルを保存し照合済み |

主要条件は20 valid参照×12文、trainer EMA、EOS−4、teacher CFG2／student CFG1（CFG2を蒸留）です。studentは235/240本がEOS終了し、5本は生成上限へ到達しました。長文CERはteacher4.29%／student45.36%、手順長文は7.56%／65.06%でした。ASRによるCERと、人手による自然さ・声質の判断は別です。[蒸留・CPU評価の詳細](research/2026-10-07-japanese-distillation-and-cpu-results.md)を参照してください。

## 環境とデータの終了処理

専用Vast.aiインスタンス`54521683`は、成果物回収後の2026-10-15に**削除済み**です。APIの一覧から消えたことを確認しました。GPU停止・ストレージ課金継続の段階を終えており、学習・評価・アップロードの待機ジョブも終了しています。他プロジェクトのインスタンスは操作していません。

大規模GOLの生音声・派生latentsは外部回収せず、専用環境とともに削除されました。HFには生音声・参照録音・latents・認証情報を含めていません。optimizerとstudent/legacyの重複するtraining stateはローカルに保持しています。再実験にはデータの再取得・再処理が必要ですが、現在は再開予定はありません。

実行時の方針は、学習・推論・ASR・データ処理・集計・評価・CPU測定・負荷の高いテストをVast.aiへ限定し、ローカルPCは編集・クラウド操作・転送・結果閲覧だけにする、というものでした。この保存公開ではモデル変換や再推論は行っていません。

GOL単独とは10月のM2a/studentの**学習音声**を指します。SP8000の学習テキストは過去のGOL＋MoeSpeech由来です。詳細は[データセット](datasets.md)に記録しています。

## 記録の読み方

- [開発終了と先行モデルの確認](research/2026-10-15-project-end-and-released-model-check.md)：終了判断と先行配布物の確認範囲。
- [HF保存公開の完了記録](research/2026-10-15-huggingface-archive-publication.md)：組織への移管、保存対象、照合結果。
- [データセット](datasets.md)・[学習戦略](training-strategy.md)：最終データと実行履歴。
- [M2a準備](research/2026-10-07-gol-only-m2a-preparation.md)・[学習結果](research/2026-10-07-gol-only-m2a-training-results.md)：実行時の測定。
- [蒸留とCPU評価](research/2026-10-07-japanese-distillation-and-cpu-results.md)・[人手所見](research/2026-10-08-japanese-teacher-student-listening-notes.json)：品質差と確認範囲。
- [旧継続計画](plans/2026-10-07-next-training-and-release.md)・[コミュニティ調査](community-survey.md)：当時の計画・調査。現在の追加実行指示ではありません。

8月・10月6日・10月7日の研究報告とJSON/XMLは実行当時の記録を保持しています。「未公開」「停止・データ保持」「次の試聴」などは当時の状態です。現在の状態は本ページと10月15日の保存公開receiptを基準にします。公開用コピーでは他プロジェクトの情報・ホスト接続情報・個人環境の絶対パスを省略または正規化し、該当JSONに`publication_redactions`を付記しています。元の記録はローカルに保持しています。
