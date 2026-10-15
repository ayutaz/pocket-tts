# 日本語モデル開発の終了と公開モデルの確認

2026-10-15、ユーザーは先行モデルの公開を受けて日本語モデル開発を終了しました。続く保存公開の指示により、既存のローカルモデルを[tts-dataset/pocket-tts-ja-experiments](https://huggingface.co/tts-dataset/pocket-tts-ja-experiments)へpublic/manual gateで保存し、全6モデル・31ファイルの転送と照合を完了しました。[保存公開の完了記録](2026-10-15-huggingface-archive-publication.md)を参照してください。

追加学習・蒸留・最終品質選定を再開する計画はありません。未達の品質を合格にはせず、既存Phase1も変更せずに残します。以下は終了判断に使った配布物の固定revisionの確認と終了処理の記録です。先行モデルの最新mainで再測定した値ではありません。

## 公開モデルの確認

Hugging Faceの公開API、モデルカード、設定ファイル、GitHub READMEを確認した。確認対象は[`takuma104/pocket-tts-ja`](https://huggingface.co/takuma104/pocket-tts-ja)、モデルカードのrevisionは`119685fd7a51356429c6d25e37d570b423c17663`、API上の最終更新は2026-10-14 14:26:35 UTC。設定内の重み・tokenizerは`562a82cba331499b75a050b7e200f4ee3241b9e9`に固定されており、そのrevisionで実ファイルへのHTTP HEADがすべて200となることも確認した。重みの全量ダウンロード、推論、試聴は行っていない。

| 配布物 | 確認結果 |
|---|---|
| `japanese.yaml` / `japanese/model.safetensors` | 6層FlowLM、SP4000、24 kHz、温度0.3。重み219,029,196 bytes、Mimi込み約110M parametersと作者が説明 |
| `japanese_24l.yaml` / `japanese_24l/model.safetensors` | 24層FlowLM。重み1,304,206,614 bytes、Mimi込み約336M parametersと作者が説明 |
| `tokenizer.model` | 60,589 bytes。両設定から同じ固定revisionを参照 |
| 推論方法 | 公式`pocket-tts` 3.3.0以降に設定を渡すCPU推論、日本語の漢字かな入力、数秒の参照音声によるzero-shot voice cloningを作者が説明 |
| ライセンス | 重みはCC BY-NC 4.0（非商用）、[GitHubのコード](https://github.com/takuma104/pocket-tts-ja)はMIT |

配布サイズと設定はAPIと公開ファイルから確認した。推論互換性、音質、速度、学習内容は作者の記載であり、こちらでの実動作検証とは区別する。[固定revisionのモデルカード](https://huggingface.co/takuma104/pocket-tts-ja/blob/119685fd7a51356429c6d25e37d570b423c17663/README.md)によれば、24層teacherは700k step、Emilia系日本語1,531時間とgain変更227時間、読みを補う合成音声828時間を使用し、CFG2を24層・6層の重みに蒸留した。現在の配布では日本語で調整したMimi decoderも含む。

作者の最新HFカードの6層モデル評価は、JKYB-Parakeet全13,536文の対象語読み正解率94.75%（4参照声を交代）／94.84%（jvs001）、公式パッケージCPUでJSUT Kana-CER 2.15%、JVS zero-shot Kana-CER 1.70%。GitHub READMEには元のMimi decoderを使った初回版の数値が残っており、最新版HFカードと混ぜない。数値は作者のASR評価であり、人手評価でも独立追試でもない。こちらの固定12文・文字CERとは文章、正規化、参照、条件が異なるため直接比較しない。

記載された限界は、珍しい読みの誤り、6層で約0.1〜0.3%のJKYB文における生成上限到達、参照声による品質差など。数字・助数詞・ラテン文字の独立評価は未実施とされる。数値だけからあらゆる用途で実用品質とは判定しない。

日本語、CPU、声の複製、6層軽量化という本プロジェクトの目標と重なる配布物が存在することを確認した。非商用という条件は利用上の差だが、これを理由にユーザーの終了指示を変更したり、商用対応の新規計画を立てたりしない。

## 残す成果物と終了時の状態

既存の[`ayousanz/pocket-tts-ja-phase1`](https://huggingface.co/ayousanz/pocket-tts-ja-phase1)は変更・削除しない。公開APIで存在とrevision `2c3c1a7dab2ff066b5069c30e7951a980ff6e914`、manual gateを確認し、片づけ前後でrevisionが同じであることを確認した。これは8月のPhase 1検証モデルであり、10月のGOL teacher10k／20kやstudent5kを新たにHFへアップロードしたという意味ではない。

GOL teacher10k／20kのcheckpoint・optimizer、student5kのcheckpoint・optimizer・export、評価音声・CPU結果・固定ソース・設定・ログは回収済みのローカル成果物を保持する。今回、主要checkpoint・optimizerと4つの回収archiveの存在・サイズを再確認した。ハッシュは以前の転送時に照合済みの記録を参照し、大容量ファイルのローカル再ハッシュやML処理は行っていない。[終了処理の記録](2026-10-15-japanese-project-closeout.json)に確認対象とクラウド処理の結果を保存する。

終了時点はteacherの20k学習、暫定teacher10kからstudent5kへの蒸留、全3,024条件評価、CPU96回測定、export・保存再開・回収検証まで。主要条件CERはteacher10k 12.24%、student5k 51.22%。通常文1組の人手所見を受領したが、長文等の人手品質確認と最終teacher選定は未完了のまま終了する。[実験結果](2026-10-07-japanese-distillation-and-cpu-results.md)と[試聴所見](2026-10-08-japanese-teacher-student-listening-notes.json)はそのまま残す。

所有Vast.aiインスタンス54521683だけを削除し、独立したAPI再照会で一覧から消えたことを確認した。最初のCLI呼び出しは確認フラグ不足で削除されず、実状態とインストール済みCLIのhelpを確認して`--yes`を付けた。CLIはexit 0でもJSONを返さなかったため、応答parserの成功ではなく再照会を削除の証拠とする。他プロジェクトは操作していない。削除後もHFの既存revisionは同じだった。専用インスタンスのGPU・保存領域を残さず終了する。

元GOL音声・派生latentsは外部回収に含まれておらず、インスタンス削除に伴い失われた。再実験には再取得・再処理が必要となる。5005 stepの技術的な再開試験の重みは品質候補にせず、回収済み証拠・ログを残す。
