# 日本語対応の既存実装レビュー（2026-10-06）

> **2026-10-15 最終状態:** 日本語モデル開発は終了し、全6モデルのpublic/manual gateでのHF保存公開と専用Vast.ai環境の削除を完了しました。本文は当時の設計・計画・測定の履歴です。現在の状態は[終了・保存公開サマリ](../index.md)を参照してください。

レビュー完了時の状態: コードレビューと修正後の回帰検証は完了。重要な不具合を発見・修正した。その時点では実コーパスの再処理、日本語音声の品質、GPU学習・蒸留は未検証だった。

2026-10-07追記: 後続の[小規模GPU試運転](2026-10-06-vastai-gpu-smoke-results.md)で実コーパス処理・学習・再開・推論ロードを確認し、[144条件の機械評価](2026-10-07-japanese-reading-evaluation-results.md)も完了した。さらに[GOL単独の全量準備とM2aの20kまでの学習・機械評価](2026-10-07-gol-only-m2a-training-results.md)を完了し、10k/20kの再開状態と全量試聴音声も回収・照合した。主要条件のCER悪化から、試聴前に30kは開始しない。人手試聴・teacher選定・蒸留・実用品質は未確認であり、実運用で問題がないとは判断しない。現在の実行順は[最新計画](../plans/2026-10-07-next-training-and-release.md)に従う。本文のWindows検証は当時の記録で、今後の計算処理はすべてVast.aiで行う。

## 対象と判断の範囲

`japanese-model-training` の `5073bba` と作業ツリーを対象に、8月の日本語テキスト処理、MoeSpeech取り込み、GOL取り込み、学習・推論・評価の接続を確認する。ベースmainは `41cbc84af539ea78a804ffca5f9c6edc1a22ce44`。

4エージェントがデータ取り込み、形態素分割・アライメント、推論・評価、学習設定・loader・再開を独立にレビューし、親エージェントが指摘の再現、横断検査、検証範囲の監査を担当する。

今回の環境はWindows、Python 3.12.13、PyTorch 2.14.0+cpu。公式学習環境はLinux/NVIDIA GPUであり、本レビューのWindows CPU結果をGPU学習完走の証拠とは扱わない。新たなGPU契約、GOL全量取得、学習、モデル公開は実施しない。

8月29日のPhase 1は過去の学習成果である。今回のテスト結果と区別する。GOLの9ステージはコード・合成データでの実装検証と、実コーパスでの品質・スケール検証を分ける。

## 初期検査の実出力

| 検査 | 結果 | 範囲 |
|---|---|---|
| `pytest -n 3 -q tests training/tests`（下記除外付き） | `507 passed, 12 warnings in 81.46s` | モデル取得を伴う5ファイル、Linux signalの1ファイルを除外 |
| `tests/test_python_api.py` | `3 passed in 11.57s` | 公開APIの形。実音声生成の証明ではない |
| Ruff 0.12.7、全リポジトリ | 585件、全てANN | 新mainの型注釈必須ルールに既存日本語コード・テストが未対応 |
| `ty check`、Windows | 85件 | 型引数、オプショナル値、fixture型、プラットフォーム依存APIなど |
| `ty check --python-platform linux`（小修正後） | 77件 | Windows固有のsignal/affinityを除いてもCI上の型検査は未達 |

初期pytestの除外は `tests/test_python_api.py`、`tests/test_cli_generate.py`、`tests/test_documentation_examples.py`、`tests/test_quantization.py`、`tests/test_streaming_cancellation.py`、`training/tests/test_loader_signals.py`。APIテストは後から全3件を実行した。量子化backend検出、TextRules、CERの追加試験は `17 passed in 14.01s`（一部初期検査との重複あり）。

静的検査の診断数は初期状態の記録であり、最終状態の成功を表す数ではない。

## 仕様との対応

参照仕様: [日本語フロントエンド](../specs/2026-08-27-japanese-text-frontend-design.md)、[MoeSpeech](../specs/2026-08-28-moespeech-manifest-design.md)、[GOLとPhase 2](../specs/2026-08-30-gol-and-phase2-design.md)。

| 要件 | 必要な証拠 | 今回の判定 |
|---|---|---|
| 日本語正規化の一元化・省略符と濁点の保持 | tokenizer学習、aligner、推論の実呼び出しとUnicode試験 | shared normalizerの配線と`test_text_normalization`/`test_ja_text`で確認 |
| 日本語の文・節分割、句読点補完、空白なし結合 | 長短入力の実tokenizer試験と既定英語回帰 | `test_text_rules`/`test_dataloader`で確認。句読点なし長文の上限保証はない |
| CERと濁点保持、評価出力名の識別 | 実metric計算、CLI配線、結果の衝突試験 | `test_cer`で計算・normalizer・出力名を確認。日本語音声評価は未実施 |
| MoeSpeech 8ステージと中断再開 | 各成果物、原子的書き込み、設定変更時の無効化、通し試験 | 114試験で合成音声・実stage関数・外部処理のfakeを使い確認。snapshot identityとalignment freshnessを接続 |
| GOL 9ステージ、作品・話者の識別、曖昧なパスの拒否 | tar構造、複合話者キー、選定・合流・分割の再現 | 116試験で確認。実音声除外後のfloor、metadata/archive revision固定、score/filterの再生成も試験 |
| 話者を跨がない連結、start/duration、loader可読 | 合成音声と実loader、コーパス別話者排他性 | 両取り込みテストと`test_dataloader`で合成音声の連結・窓・読み込みを確認 |
| 日本語CTCアライメント・スコア | 表記復元、読み変換、時間窓、出力scoreの計算試験 | `test_align_data`/`test_word_spans`と独立全経路計算で確認。実音声の時刻精度は未再測定 |
| 実データのアライメント成功率・カット可能率 | 現行コードで実音声を再処理したログ | 今回未再測定。8月の記録のみ |
| GOLスコア閾値の妥当性 | 低スコア音声の聴取と転写照合 | 未検証。コード試験では代替できない |
| 初期embeddingのリセット、8000語彙、学習用日本語設定 | 実config合成、実tokenizer構築、checkpointとの形状・ロード確認 | 3学習configの実tokenizerと実English24Lからの構築、合成Flow checkpointと実Mimiロードを確認 |
| M2aの40k step、実効batch64、valid20話者以上、checkpoint保持 | config、splitガード、実GPU学習曲線 | 設定とsplitガードを試験。保存を1,000 step・40本へ修正。学習曲線は未取得 |
| mainとの互換性、既存英語API・音声生成、CI | 回帰テスト、実CLI、全Ruff/ty | 実英語生成・streaming・量子化を含む回帰通過。全Ruff/import order、変更Pythonのformat、Linux向けty 3.10/3.14通過。実GitHub jobは未実行 |
| 6層日本語モデル・配布用重み/声/推論設定 | 蒸留run、実ロード・生成、公開artifact | 後段の未実装・未実行項目。現状の完成範囲には含めない |

## 指摘と修正

以下は独立レビューで再現された指摘。状態は修正後の再検証に合わせて更新する。

| ID | 重要度 | 再現された問題 | 状態 |
|---|---|---|---|
| R1 | High | 英語base configのtokenizer backendが`tokenizers`へ変更され、日本語SentencePiece binaryをJSONとして開く | 3つの学習configで`tokenizer: sentencepiece`を明示。実configからの構築を含む57試験が通過 |
| R2 | High | CTC trellisが連続同一ラベルのblank必須条件を満たさず、2フレームの`aa`を`a,a`として整合済みにする | expanded CTC trellisへ修正。25条件の全経路oracleと回帰試験通過 |
| R3 | High | alignmentの完成判定がファイル存在、partial再開が`path,start`だけで、model/segmenter/transcript変更後に古い結果を使う | manifest SHA-256、model識別子、segmenter、shard数、phrase merge、algorithm versionをfingerprint化。wrapper/directの実装を一元化し両取り込み処理へ接続。stale完了出力の破棄とmatching partialの保持を試験。欠落manifestはfreshとせず再生成へ進む |
| R4 | High | partial JSONL末尾が途中で切れると、resumeがJSON parseで失敗する | 改行なし不完全末尾を切り戻し、中間または改行付き破損行は拒否。JSONとUTF-8の回帰試験通過 |
| R5 | High | precomputeがMimi hashと行indexだけでlatentsを再利用。同indexの音声path変更後も古い音声のlatentsを新transcriptに対応付ける | manifest SHA-256とMimi hashをnamespaceに追加。全行の順序・file名・file存在・row countも起動時に検査。回帰試験通過 |
| R6 | High | 日本語推論configにMimi sourceが無い。学習checkpointはMimiを保存しないため`--checkpoint`でもロード不可 | 学習時と同じEnglish24Lのpinをcodec sourceに指定。full bundleのprefix抽出・旧名変換を修正。実Mimi strictロードがmissing/unexpectedとも0、合成Flow checkpointとの通しロードと回帰試験通過 |
| R7 | Medium | 一部だけCTC語彙にある読みでも、全surfaceに時刻を付ける | 不完全な読みには時刻を付けない。回帰試験通過 |
| R8 | Medium | metadataのrevision/contentが成果物に記録されず、更新後も古い話者選定・download cacheを再利用する | metadata SHA-256・repo・snapshot revisionを記録しarchive downloadを同revisionへ固定。不一致やidentity不明の旧cacheは新しい出力先を要求して停止 |
| R9 | Medium | GOLの話者floorが空text・欠落音声・曖昧パスの除外前の行数/時間で成立し、実収集後は0件にもなる | 正規化後の空textを除外し、収集後の実発話数・時間でfloorを再検査。回帰試験通過 |
| R10 | Medium | repo外cwdでは日本語推論configのtokenizer相対pathが見つからない | cwd優先を維持しconfig相対/repo相対の解決を追加。回帰試験通過。wheelはtokenizerを同梱しないため配布は未完成 |
| R11 | Medium | 日本語のdefault textが未登録で、入力省略時やserve UIが英語文を使う | 日本語文を登録しCLIで実効configの言語を参照。回帰試験通過 |
| R12 | Medium | Phase 1/M2a sampleはCFG2、公開TTSModelの通常推論はCFG1。学習中の試聴品質を通常推論品質と同一視できない | 制約として明記。CFGの焼き込みは未実施の蒸留工程 |
| R13 | Medium | M2aが反復回数・時間・話者数を同時に変えるのに、validの折返しから「量か多様性か」を原因断定する | live config/設計/計画/戦略/進捗の因果断定を訂正 |
| R14 | Medium | mainで追加されたRuff ANN/tyのCIを日本語追加コード・テストが満たさない | production・fixture・record・fake境界の型を補強。全RuffとLinux向けtyを通過。Windowsの3件は非対応SIGUSR1のみ |
| R15 | Medium | M2aで検証1,000 step・保存2,500 stepのため、全checkpointを保持しても最良検証点が未保存になりうる | 先に回帰試験の失敗を確認。保存1,000・保持40へ変更し、config全58試験通過。保存領域約100 GiBの概算を記録 |
| R16 | Medium | CIは`dev`のみをinstallし、実fugashi/UniDicを使う日本語試験が`importorskip`で実行されない | ty/pytestのPython 3.10・3.14 jobに`japanese` groupを追加。YAMLのparseと設定配線を確認。GitHub上の実jobは未実行 |
| R17 | Medium | latentの存在だけでfreshと判断し、ゼロバイト・切断・別keyのfileをloaderへ渡す。再計算でも既存fileを再利用する | shared header検証で`latents` key・2D非空shape・safetensorsの有効性をreadinessと再計算判定の双方で確認。データ本体はロードしない。回帰試験通過 |
| R18 | Medium | Windowsで`Path.rename()`は既存のlatent/annotated manifestを置換できず、cache更新が失敗する | `os.replace()`へ修正し、既存file上書きの回帰試験通過。Windowsでの学習全体の対応を意味しない |

句読点のない日本語長文では、実SentencePieceで112 tokensの入力が`max_tokens=50`に対して1チャンクのまま返ることも再現した。文字列の欠落はこの分割試験では無いが、生成側は「generation may skip words」と警告する。日本語文・節境界の対応だけで、任意の長文が50 tokens以下になるとは保証できない。入力の制約または安全な追加分割が必要。

R2を修正してscore/spansの定義が変わる場合、旧alignmentの再利用と旧score閾値の流用はできない。GOLはまだ実コーパスで閾値を決めていないが、過去成果物を持つ利用者への無効化規則を検証する。

修正したCTC scoreは、親エージェントが独立に1〜5フレーム・3ラベルの全経路を列挙し、CTCの「隣接重複を畳んでからblankを除く」規則で求めた最大log probabilityと比較した。単一音・連続同音・異音の5ターゲット、計25条件が一致した。これはトレリスの数学的な確認であり、外部ASRモデルによる実音声の時刻精度確認とは異なる。

このCTC修正は日本語・英語の共通経路に作用する。8月のGOL計画にある「score追加時は既存wordsを変更しない」という制約は当時の追加実装の範囲であり、今回の不正なCTC経路の修正後に全入力で旧時刻と同一とは主張しない。peak-frameによるword span表現は保持し、経路とscoreの妥当性を優先する。実データでの再alignmentとcut品質の比較は残る。

学習担当はキャッシュ済みの実English 24L重みを用いて日本語`build_models()`も実行した。24層、実SentencePiece語彙8000、embedding `[8001, 1024]`、embeddingリセットを含む構築は成立した。これは日本語学習済み重みのロード・音声品質の検証とは異なる。

### サポート範囲と評価の未完成部分

- `training/README.md`は学習をLinux/NVIDIAに限定している。Windowsには`SIGUSR1`がなく、trainer/loaderはそのまま起動できない。compile無効化とUTF-8指定だけでWindows学習が動くと読めた3設定の注記を修正した。Windowsでの学習を今回の検証済み機能とはしない。
- `librispeech.py`にはCER/basic正規化があるが、既定はLibriSpeechと英語ASRである。日本語の評価セット、ASR選定、実音声によるCER・speaker similarity・主観品質評価は別途必要である。
- 公開用日本語重み、6層蒸留結果、権利を満たす公開用参照声、tokenizerを含む配布artifactは未完成。Phase 1のgated成果物が存在することを、pipインストールから日本語の公開版が動く証拠とはしない。
- latent cacheのfingerprintはmanifest bytesとMimi encode-path weightsを対象にする。同じmanifest bytes・同じpathのまま音声だけを差し替える操作は検知しない。音声更新時にはdataset revision等でmanifest bytesを変えるか、該当latent namespaceを削除する。readinessは全fileのheaderとshapeを確認するが、tensor値のchecksumや有限性を毎回検査するものではない。
- alignmentのfingerprintはmodel識別子を記録し、同名remote checkpointの内容変更や、同じmanifestのpath上での音声差し替えは検知しない。変更時はmodel識別子・manifest bytesを更新するかalignment artifactsを破棄する。実行deviceもfingerprintには含めず、CPU/CUDA間でscoreがbit単位に一致する保証はしない。
- 分散latent事前計算は共有filesystemとtorchrun supervisorを前提にする。rank 0の内部待機loopには独自timeoutがなく、外部worker停止時の挙動はsupervisorに依存する。通常のtorchrun起動で独立した不具合の再現は確認しておらず、分散GPUでの障害注入は未実施。

### 既存成果物を再利用する場合

fingerprintのない旧alignmentは作り直す。新CTCのscoreを使って閾値も再校正する。metadata/archive identityのない旧archive cacheは自動的に採用せず、別の出力先で取得・構築するよう停止する。完成した旧データを変更せず保存し、新しい処理結果との時刻・cut率・音声品質の比較に使うことはできる。

## 最終検証と残る制約

今回確認したのは実装の接続と回帰である。テスト通過をもってGOL本番取り込み、未知話者品質、GPU学習、蒸留後CPU性能が確認できたとは判断しない。実Phase 1 checkpointの現行mainでのロード・日本語生成も今回の試験には含めず、合成Flow checkpointと実Mimiの接続試験と区別する。

| 最終検査 | 実出力・結果 | 証拠 |
|---|---|---|
| core（取り込み2ファイル、doc examples、Linux signalを別扱い） | `333 passed, 118 warnings in 91.44s` | [JUnit](2026-10-06-core-results.xml) |
| MoeSpeech/GOL + freshness | `233 passed, 4 warnings in 61.43s` | [JUnit](2026-10-06-ingestion-results.xml) |
| README/quickstart実行例（repo外cwd） | `10 passed in 64.97s` | [JUnit](2026-10-06-examples-results.xml) |
| alignment readiness・中断resumeの統合 | `19 passed in 17.84s`。実wrapperと実resume parserで両取り込みのpartial保持・未完了行のみ追記を確認 | [JUnit](2026-10-06-alignment-readiness-results.xml) |
| Ruff 0.12.7 + import order | `All checks passed!` | 全リポジトリ |
| Ruff format | `25 files already formatted` | 変更/追加したPythonファイル |
| ty 0.0.81、Linux platform 3.10/3.14 | `All checks passed!` | 型検査のtarget指定。Linux上の実pytestやGitHub jobではない |
| ty 0.0.81、Windows | `Found 3 diagnostics` | `subproc.py`、`test_loader_signals.py`、`train.py`の`signal.SIGUSR1` |

テスト集合には一部重複があるため単純加算せず、JUnitのclass/nameで重複を除くと**576試験が通過、failure/error/skipは0**。件数と実行条件は[証拠JSON](2026-10-06-implementation-evidence.json)に保存した。Linux専用の`test_loader_signals.py`はWindowsでは実行対象から外し、その他の日本語試験はfugashi/UniDicをinstallした状態で実行した。warningsは主に量子化APIとtar extractionの廃止予告。

英語の実CLI生成、custom voice、連続生成、streamingの中断、量子化を含む13試験は、公開アセット取得後に `13 passed, 110 warnings in 150.83s`。warningsは主にPyTorchの旧量子化API廃止予告。

README/quickstartの実モデルロード、生成、streaming、voice prompt、batch連結の例は、修正後も出力先を一時ディレクトリにして `10 passed in 64.97s`。[実行結果](2026-10-06-examples-results.xml)。英語モデルの確認であり、日本語モデル品質の証明ではない。

最初のoffline試行は新しい声アセットのcache不足で `10 failed, 3 passed`。その失敗を実装バグとして数えず、ネットワークを許可して取得後に同じ13試験を再実行した。記録: [初回offline結果](2026-10-06-integration-offline-results.xml)、[取得後の実行結果](2026-10-06-integration-results.xml)。

別の再試行では、共有`.venv`がPython 3.10へ作り直されている途中に10例が`torch.Tensor`欠落で失敗した。[その失敗記録](2026-10-06-examples-env-retry-results.xml)も保存した。環境の更新を止め、Python 3.12.13・dev+japanese groupを復元した後の最終結果が上表である。失敗を日本語実装の品質評価へ混ぜない。
