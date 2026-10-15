# 終了済み日本語モデル実験のHF保存公開

2026-10-15。開発終了後、ユーザーは「ローカルに残っているモデルもpublic manual gateで上げてください」と指定した。これは既存成果物の保存公開を認める追加指示であり、学習や品質選定の再開ではない。

現在の保存先は、ユーザーの後続指定による[`tts-dataset/pocket-tts-ja-experiments`](https://huggingface.co/tts-dataset/pocket-tts-ja-experiments)。初回の個人namespace `ayousanz/pocket-tts-ja-experiments`から組織へ移管した。公開ページ（`private=false`）と手動承認（`gated=manual`）をAPIで確認した。初回保存先で匿名のファイル取得はHTTP401・`GatedRepo`となることも確認した。既存の[`ayousanz/pocket-tts-ja-phase1`](https://huggingface.co/ayousanz/pocket-tts-ja-phase1)は別リポジトリとしてそのまま残す。

## 保存対象

- GOL teacherの2k／5k／10k／20k checkpoint（raw/EMA）。
- GOL student5kのmerged EMA safetensors・元の推論設定・HFの配布先を参照する設定。
- 旧220-step日本語検証モデルのsafetensors。GOL M2aの開始元とは区別する。
- SP8000、固定ソース、学習設定とresolved args、評価・export・再開検証・通常文の人手所見。

元の英語initializer、生音声、参照録音、latents、認証情報はアップロード対象に含めない。モデルと再開状態は既存ファイルを読み出して転送し、student exportは既存の非圧縮回収archiveから該当ファイルだけを取り出した。ローカルでモデル変換・学習・推論・ASR・評価は行わない。転送のチェックサム計算を再利用して以前の照合値と比較し、サーバー側LFSのサイズ・SHA256と小さな添付ファイルの内容を照合する。

## 品質と開発の状態

これは終了した研究実験のアーカイブで、合格モデルのリリースではない。主要条件のteacher10k CER12.24%、student5k CER51.22%、teacher20k CER15.01%と、数字・長文・生成終了の問題をモデルカードに記載する。通常文1組の人手所見のみ受領済みで、長文等の試聴・teacherの最終選定は未完了のまま保持する。

重みは元Pocket TTS由来のCC BY 4.0、同梱の固定ソースはMITとして帰属と変更点を表示する。manual gateは品質の合格や追加のライセンス許諾を意味しない。

## 作業状態

**完了：組織namespaceで全6モデルを含む31ファイル・12,062,580,944 bytesを保存公開し、照合を完了した。** 最終revisionは`9cf5427bda9ccf18a9dadb736fa686a4bc990785`。[最終publication receipt](2026-10-15-huggingface-experiments-publication.json)に全ファイルの照合結果を保存した。LFSはサイズ・SHA256、小さな添付ファイルは認証付き取得後の内容を照合した。public/manual設定と、匿名での重み取得がHTTP401になることも最終確認した。モデルカードを全件転送完了の表示へ更新し、旧待機ジョブと今回の転送処理は終了している。以下は途中経過の記録である。

初回はoptimizer・重複する再開用checkpointを含む38ファイル・約26.2GBを予定したが、teacher20kの転送前にHF APIがHTTP403「public storage space exceeded」を返した。モデルを優先する31ファイル・約12.06GBの構成へ変更した。optimizerとstudent／legacyの重複するtraining checkpointはローカルのみで保持する。モデルの変換や追加生成は行っていない。

今回作成したリポジトリ内で、既に転送していた任意の再開状態4ファイル・6,501,979,745 bytesを削除し、今回の一時的なアップロード履歴を整理した。保持する3モデルのLFS SHA256が変わらないことを確認した。公式OpenAPIのLFS一覧でも削除対象は既に残っていなかったが、APIの`used_storage`は約12.08GBのままで、現在ファイルの合計約5.58GBと一致していない。履歴整理後の容量反映には[公式案内で最大36時間](https://huggingface.co/docs/hub/storage-limits#super-squash-your-repository-using-the-api)かかる。

個人namespaceで**student5k EMA export・teacher2k・teacher10kの3モデルをアップロード**した。teacher5k・teacher20k・legacy220 exportの3モデルは容量反映待ちとなった。公開ページ・manual gate、匿名取得のHTTP401、転送済みモデルのサイズ・SHA256、既存Phase1のrevision不変を[当時の部分公開の照合記録](2026-10-15-huggingface-partial-publication-verification.json)に保存した。全件完了は[最終publication receipt](2026-10-15-huggingface-experiments-publication.json)が実際に生成されてから判定する。

容量が反映されたら残りの転送を再開するため、実行ツール管理の停止可能な待機ジョブ（session 70058）を起動した。常駐プロセスの起動は自動承認レビューで`blocked by policy`と拒否されたため、独立した常駐サービスを作らず、通常の実行ツールの管理下で動かす方法に変更した。待機中は1分ごとのクラウドmetadata確認だけを行った。

その後、ユーザーが`https://huggingface.co/tts-dataset`を保存先に指定したため、実際のPID・cmdlineで所有を確認して**旧待機ジョブを停止**した。認証アカウントの組織admin権限と、移管先に同名モデルが存在しないことを確認してHFの`move_repo` APIで移管した。移管前後で28ファイルの存在・サイズとLFS SHA256が変わらず、public/manual設定とPhase1のrevisionも保持されている。[組織移管の照合記録](2026-10-15-huggingface-organization-transfer.json)を参照。以降の転送とHF用configの参照先は組織namespaceへ変更し、残り3モデルの転送を再開した（実行session 61135）。

[待機・転送の状態](2026-10-15-huggingface-publication-status.json)は`publication_verified`、未転送モデルは0件となった。`runs/vast-m2a-control/hf-publication-state-20261015.json`にも転送履歴を保持する。開発終了と専用Vast.aiインスタンス削除は変更しない。optimizerと重複するtraining stateは引き続きローカル保持とし、既存Phase1・他プロジェクトのHFリポジトリは操作していない。
