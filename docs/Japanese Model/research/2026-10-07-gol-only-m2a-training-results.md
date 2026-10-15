# GOL単独M2aの学習と保存点評価

> **2026-10-15 最終状態:** 日本語モデル開発は終了し、全6モデルのpublic/manual gateでのHF保存公開と専用Vast.ai環境の削除を完了しました。本文は当時の設計・計画・測定の履歴です。現在の状態は[終了・保存公開サマリ](../index.md)を参照してください。

更新日: 2026-10-07。RTX PRO 6000 Blackwell・4,000 GB指定の専用Vast.aiインスタンス54521683で実行した記録。20kまでの学習・機械評価・回収は完了し、[実行計画](../plans/2026-10-07-next-training-and-release.md)のステップ3は人手試聴・teacher選定待ちで未完了。GPUはデータを保持して停止中である。

## 最新の結果と判断

同一の20 valid参照・12文における`trainer / EMA / CFG2 / EOS−4`のCERは5k 26.37%、10k 12.24%、20k 15.01%。10k→20kでは長文と手順の長文も悪化した。一方、12 profile中10のCERは改善しており、全条件が悪化したわけではない。[固定条件の全量比較](2026-10-07-gol-m2a-10000-vs-20000-full.json)と個別診断を根拠に、[試聴前に30kを開始しない判断](2026-10-07-gol-m2a-20000-review-gate.json)を保存した。10kは機械評価上の優先候補で、teacherの最終選定ではない。

10k/20kのcheckpoint・optimizer、全3,024本の音声を含むアーカイブと試聴フォルダ、固定ソース・設定・データ識別情報・共有重みを回収し、サイズ・ハッシュを照合した。[要件照合と試聴手順](2026-10-07-gol-m2a-stage3-completion-audit.md)に残件を整理した。Windows再生、人手試聴、teacher選定、インスタンスの最終destroyは未完了。停止中もストレージ課金は続く。以下は各区間の実行履歴であり、古い起動時記録を現在のジョブ状態とは扱わない。

## 学習の開始条件

[全量準備](2026-10-07-gol-only-m2a-preparation.md)を経たGOL単独のtrain 447,090行・723.507時間、valid 12,951行・22.329時間を使用する。Mimi latentsの全件整合性、本番batchの0→30→35 stepとoptimizer再開、最長の実64行の16×4処理を先に検証した。負荷試験のrunとM2aは分離している。

英語24層のrevision `492522650173a0653b7575cdc25ae09810e5d741`から日本語8,000語彙のembeddingを初期化し直して開始した。8月Phase 1、220 step、短期負荷試験のcheckpointは開始元に使わない。実効batch 64、lr 2e-4、warmup 1,000、validation/checkpoint 1,000ごと、通常サンプル500ごと。初回の停止点のみ2,000とし、40kは以後の品質診断に基づく上限候補として維持する。

## 初回0→2,000 step

2026-10-07 05:41:46〜05:48:40 JSTのrunで2,000 stepまで実更新し、正常終了した。起動・事前計算等を含む全作業時間ではない。

| 保存点 | validation loss | flow loss | EOS loss |
|---|---:|---:|---:|
| 1,000 | 0.479474 | 0.476229 | 0.032454 |
| 2,000 | 0.366997 | 0.364137 | 0.028601 |

validationは設定された50 batchの指標であり、valid全行の平均や音声品質の合格値ではない。学習ログに記録されたloss・gradientの有限値に加え、実際の2k checkpointのraw/EMAとoptimizer全体の有限値、optimizer counterがすべて2,000、embeddingが8,001×1,024であることを確認した。

- checkpoint SHA-256: `296f388855a63bdc20b0a59d9b10fd0161b57d1c06208cabf0bab0b435951432`
- optimizer SHA-256: `ad3b639f793ced264385be32ffb4d26990e1028ece7d29af81cb0250be70924e`
- 解決済みargs SHA-256: `5c2bd73a6790f9346cceb8345cf32e4f5f461480e6a76279348c0d6e153e50f1`

[実更新検証JSON](2026-10-07-gol-m2a-2000-checkpoint-verification.json)を回収した。通常サンプルは固定比較用の評価とは区別する。学習のログにあるpeak GPU memoryは16.2 GiBで、事前計算や隔離した最長batch負荷試験のピークを含まない。

## 固定評価と次の判断

20のvalid話者IDグループと1つのtrain参照を、合成結果を使わず固定seedと品質条件から選定済み。初期診断はそのうち2つのvalid参照と同じtrain参照、既存6文、raw/EMA・CFG・EOS・trainer/public APIの216条件を使う。最終比較は全21参照と拡張12文の3,024条件を使い、初期216条件とは別のmatrixとして扱う。

2k学習の終了後に、検証済み試聴UIを適用し、現行の評価・集計・UIの15テストが成功した。評価ソース136ファイルを固定して216条件の生成・ASR・集計を完了し、全条件、WAVハッシュ、参照PCMとASR設定の一致を検証した。再開で上書きされるargs・progress・実更新検証・参照識別情報・source manifestは評価内の`input_snapshot`に保存済みである。

[2kの機械診断JSON](2026-10-07-gol-m2a-2000-diagnosis.json)に、参照文字数で重み付けした集計を保存した。各行は3参照×6文の18本、参照文字数480。生成設定ごとのCERは76.04〜107.08%で、teacherの品質合格ではない。

| 経路 | 重み | CFG | EOS threshold | CER | 上限到達 / 18本 |
|---|---|---:|---:|---:|---:|
| trainer | raw | 2 | −4 | 76.04% | 0 |
| trainer | raw | 2 | −1 | 78.13% | 0 |
| trainer | raw | 1 | −4 | 85.63% | 0 |
| trainer | raw | 1 | −1 | 90.83% | 0 |
| trainer | EMA | 1 | −4 | 94.17% | 0 |
| trainer | EMA | 1 | −1 | 99.38% | 3 |
| trainer | EMA | 2 | −4 | 94.58% | 0 |
| trainer | EMA | 2 | −1 | 102.29% | 2 |
| public API | raw | 1 | −4 | 89.17% | 7 |
| public API | raw | 1 | −1 | 90.42% | 9 |
| public API | EMA | 1 | −4 | 105.00% | 5 |
| public API | EMA | 1 | −1 | 107.08% | 16 |

1秒未満の出力は216本中0本。上限到達は42本（public API 37/72、trainer 5/144）で、EOS後の必要frameを出し切れない場合も含む。public APIの文章長に応じた上限とtrainerの30秒上限は異なるため、経路を混ぜて終了挙動の優劣を決めない。日付・数字・長文等には誤りが多く、設定の変更だけで全文を正しく読む条件は確認できない。

同じ固定ASRで実音声6本を確認したCERは14.17%（参照120文字、置換12・削除4・挿入1）。これは小標本であり、合成CERから差し引く補正値にはしない。漢字/かなの表記とASR誤りを含む指標なので、合成の読み落とし・発音・自然さは試聴で確認する。旧220 stepは参照音声等が異なり、この2kとの正式な同条件比較ではない。

Vast.ai上のChromiumで、実際の2kページの3参照のWAV再生、音声長、checkpoint表示、空メモJSONとscores/統合identityの一致を確認した。無音設定でのheadless再生はUIとデコードの確認であり、人手による音質評価ではない。試聴ページはVast.aiのlocalhost静的配信とSSHトンネルで閲覧できる。ローカルで生成・ASRは実行しない。人手試聴は未完了、teacherは未選定である。

## 2k→3kの再開

2kの機械診断と評価設定の保存を確認後、凍結source全ファイルとargsの一致、最新checkpointが2kであることを検証し、停止stepだけを3,000へ変更して再開した。optimizer込みの2k checkpointから実更新2,001以降を確認し、3,000 stepまで正常終了した。[保存状態の検証JSON](2026-10-07-gol-m2a-3000-checkpoint-verification.json)ではraw/EMA/optimizerは有限値で、optimizer counterは3,000と一致した。validation lossは0.288313、flow loss 0.285614、EOS loss 0.026995。

固定216条件の生成・ASR・集計・入力設定の保存も完了した。[2k→3k比較JSON](2026-10-07-gol-m2a-2000-vs-3000.json)で、文章・seed・参照/実音声PCM・生成/ASR設定・source・学習設定（停止step以外）の一致を確認した。[3kの診断JSON](2026-10-07-gol-m2a-3000-diagnosis.json)の主な比較は以下。

| 同一の生成条件（各18本） | 2k CER | 3k CER |
|---|---:|---:|
| trainer / raw / CFG2 / EOS−4 | 76.04% | 45.00% |
| trainer / raw / CFG2 / EOS−1 | 78.13% | 45.83% |
| trainer / raw / CFG1 / EOS−4 | 85.63% | 57.29% |
| trainer / EMA / CFG2 / EOS−4 | 94.58% | 60.42% |
| public API / raw / CFG1 / EOS−4 | 89.17% | 53.33% |
| public API / EMA / CFG1 / EOS−4 | 105.00% | 73.96% |

全条件の改善ではない。trainer / EMA / CFG1 / EOS−1は99.38%→107.50%へ悪化した。全profileを含む文別集計では依頼文70.97%→23.61%、地名の文89.51%→58.80%は改善した一方、数字の文107.52%→113.43%は挿入誤りが増えた。表記の影響と発音の誤りは試聴で切り分ける。実音声6本のASR転写とscoreは2kと一致した。3kも1秒未満は0件で、teacherの品質合格は未確認である。

## 3k→5kの再開と固定評価

3kの正常終了と固定評価・比較、凍結source、最新checkpointとargsのハッシュ一致を確認後、停止stepのみ5,000に変更してoptimizer込みで再開した。3,001〜5,000 stepを実更新し、学習と同じ216条件の生成・ASR・集計・context保存・3k比較まで正常終了した。[5k保存状態の検証](2026-10-07-gol-m2a-5000-checkpoint-verification.json)ではraw/EMA/optimizerは有限値で、optimizer counterは5,000。4kのvalidation lossは0.240192、5kは0.216086（flow 0.213575、EOS 0.025114）だった。

[3k→5k比較](2026-10-07-gol-m2a-3000-vs-5000.json)では固定文章、seed、参照/実音声PCM、生成/ASR設定、source、学習設定（停止step以外）の一致を確認した。[5k診断](2026-10-07-gol-m2a-5000-diagnosis.json)の各行は同じ18本・480文字である。

| 同一の生成条件 | 3k CER | 5k CER |
|---|---:|---:|
| trainer / EMA / CFG2 / EOS−4 | 60.42% | 18.75% |
| trainer / EMA / CFG2 / EOS−1 | 69.58% | 33.13% |
| trainer / EMA / CFG1 / EOS−4 | 75.63% | 34.58% |
| trainer / EMA / CFG1 / EOS−1 | 107.50% | 33.33% |
| trainer / raw / CFG2 / EOS−4 | 45.00% | 36.25% |
| trainer / raw / CFG2 / EOS−1 | 45.83% | 82.50% |
| trainer / raw / CFG1 / EOS−4 | 57.29% | 46.25% |
| trainer / raw / CFG1 / EOS−1 | 75.42% | 48.54% |
| public API / EMA / CFG1 / EOS−4 | 73.96% | 45.83% |
| public API / EMA / CFG1 / EOS−1 | 75.42% | 45.21% |
| public API / raw / CFG1 / EOS−4 | 53.33% | 50.83% |
| public API / raw / CFG1 / EOS−1 | 54.58% | 50.42% |

trainer / EMA / CFG2 / EOS−4では、valid参照held_00は55.00%→25.63%、held_01は64.38%→19.38%、train参照seenは61.88%→11.25%だった。一方、raw / CFG2 / EOS−1では挿入誤りが増えて悪化した。数字の文の全profile集計は113.43%→73.26%へ改善したが、表記差以外の誤りも残る。5kの1秒未満は0/216本、上限到達は17本（public API 15/72、trainer 2/144）。実音声6本のASR結果は3kと同一でCER 14.17%だった。この値を合成CERから差し引かず、自然さや正しい発音の証明にも使わない。

## 5kの全21参照・12文の拡張評価

固定した20のvalid IDグループと1つのtrain参照・全12文・12生成profileの**3,024条件**について、生成・ASR・全条件/WAV照合・context保存・診断を完了し、ジョブはexit 0で正常終了した。[全参照の集計](2026-10-07-gol-m2a-5000-full-diagnosis.json)と[valid/train参照別・参照別・文章別の診断](2026-10-07-gol-m2a-5000-full-diagnostics.json)を回収した。各profileのvalid参照は240本・参照7,220文字、train参照は12本・361文字である。以下の上限到達はEOS後に必要frameを出し切れない場合も含む。

| 生成条件 | 全21参照 CER | valid参照のみ CER | train参照 CER | valid上限到達 / 240本 |
|---|---:|---:|---:|---:|
| trainer / EMA / CFG2 / EOS−4 | 26.00% | 26.37% | 18.56% | 0 |
| trainer / raw / CFG2 / EOS−4 | 32.23% | 31.87% | 39.34% | 0 |
| public API / EMA / CFG1 / EOS−1 | 37.92% | 37.69% | 42.66% | 29 |
| public API / EMA / CFG1 / EOS−4 | 39.32% | 39.21% | 41.55% | 20 |

trainer / EMA / CFG2 / EOS−4のvalid参照は置換1,141・削除266・挿入497、EOS検出240/240本。参照ごとのCERは15.24〜39.06%で、参照による差がある。このprofileのvalid参照の文章別CERは、依頼6.00%、地名10.28%、日常10.67%、長文18.21%に対し、日付49.62%、桁を読む文48.85%、数字42.50%、固有名詞38.20%、単位34.83%だった。全profile・3,024本中の1秒未満58本はすべて`short_question`で、それ以外の文は0本。短文の短い出力やEOS検出をそのまま品質合格とはしない。実音声6本のASRはCER 14.17%で、初期評価と同じ転写・編集誤りだった。合成CERから差し引かない。

[同じ5k checkpointの初期評価と拡張評価の共通216条件](2026-10-07-gol-m2a-5000-early-vs-full-common.json)も照合した。入力・参照PCM・生成/ASR条件・凍結source・保存済みargsを検証し、216/216の生成PCM、ASR転写、scoreが完全一致した。共通条件での再現性の確認であり、checkpoint間の改善比較ではない。初期216条件と拡張3,024条件はmatrixが異なるので、18.75%と26.00%を学習品質の変化として比較しない。

[拡張評価の資源測定](2026-10-07-gol-m2a-5000-full-resources.json)では、開始・終了markerに基づく実行時間は3,641秒（60分41秒）、5秒間隔728サンプルのNVIDIA表示メモリの最大は5,054 MiBだった。選定時のGPU＋ストレージ単価で時間費用は約2.12 USDに相当する。controller内の事前確認・生成・ASR・統合・context保存等を含み、後から実行した参照別診断、圧縮・転送、以前の学習費用は含まない。メモリはdevice全体の観測値で、必要VRAMの最低値やPyTorchのallocated memoryではない。請求総額でもない。

## 5k→10kの再開と残る判断

5kの拡張評価・context保存・診断の正常終了と上記の再現性を確認後、10kまでの再開を開始した。凍結source全136ファイル、最新5k checkpoint・args、保存済み初期評価contextを検証し、停止stepのみ変更した。optimizer込みで5,001〜10,000 stepを実更新し、学習は正常終了した。[10k保存状態の検証](2026-10-07-gol-m2a-10000-checkpoint-verification.json)ではraw/EMA/optimizerは有限値、optimizer counterは10,000、embeddingは8,001×1,024と一致した。

| 保存点 | validation loss | flow loss | EOS loss |
|---|---:|---:|---:|
| 6,000 | 0.206451 | 0.203951 | 0.025000 |
| 7,000 | 0.198402 | 0.195819 | 0.025826 |
| 8,000 | 0.199299 | 0.196652 | 0.026471 |
| 9,000 | 0.197436 | 0.194928 | 0.025079 |
| 10,000 | 0.182652 | 0.180124 | 0.025280 |

validationは50 batchの指標である。10k checkpoint SHA-256は`51f538a3fe5231e4b404f46c47ae49dc4f39a20b6a07ce2bb1cbec03976e17d1`。

### 10kの初期216条件と5k比較

初期216条件の生成・ASR・照合・context保存・5k比較までexit 0で完了した。[同一条件の比較JSON](2026-10-07-gol-m2a-5000-vs-10000.json)で、文章・seed・参照/実音声PCM・生成/ASR条件・凍結source・学習設定（停止step以外）の一致を確認した。[10k診断](2026-10-07-gol-m2a-10000-diagnosis.json)の各profileは同じ3参照×6文の18本・480文字である。

| 同一の生成条件 | 5k CER | 10k CER |
|---|---:|---:|
| trainer / EMA / CFG2 / EOS−4 | 18.75% | 8.33% |
| trainer / EMA / CFG2 / EOS−1 | 33.13% | 19.17% |
| trainer / EMA / CFG1 / EOS−4 | 34.58% | 15.21% |
| trainer / EMA / CFG1 / EOS−1 | 33.33% | 16.46% |
| trainer / raw / CFG2 / EOS−4 | 36.25% | 21.04% |
| trainer / raw / CFG2 / EOS−1 | 82.50% | 28.54% |
| trainer / raw / CFG1 / EOS−4 | 46.25% | 32.08% |
| trainer / raw / CFG1 / EOS−1 | 48.54% | 41.04% |
| public API / EMA / CFG1 / EOS−4 | 45.83% | 23.96% |
| public API / EMA / CFG1 / EOS−1 | 45.21% | 23.96% |
| public API / raw / CFG1 / EOS−4 | 50.83% | 41.46% |
| public API / raw / CFG1 / EOS−1 | 50.42% | 40.83% |

trainer / EMA / CFG2 / EOS−4のvalid参照held_00は25.63%→9.38%、held_01は19.38%→6.88%、train参照seenは11.25%→8.75%だった。10k同profileは置換28・削除4・挿入8、18/18でEOS検出、上限到達0本。全profileのCERは改善したが、上限到達は17→21本へ増加した（trainer 2→4、public API 15→17）。10kのtrainer側の4本はraw / CFG1 / EOS−1である。1秒未満は両保存点とも0/216本。全profileの文章別CERでは日付67.31%→47.22%、数字73.26%→52.66%で、これらの誤りは残る。実音声6本のASR転写・scoreは同一でCER 14.17%。合成CERから補正として差し引かない。

初期評価はvalid参照2つとtrain参照1つだけである。全20 valid IDグループ・拡張12文の結果と人手試聴を確認するまでteacherに選定しない。10kの初期試聴ページはVast.aiとSSHトンネル経由のlocalhostの双方でHTTP 200を確認した。URLは`http://127.0.0.1:18725/japanese-m2a-eval-00010000-early/review.html`。

10kの学習・初期評価controller（実PID 90109）と監視controller（90799）の正常終了を確認した。監視側が全量評価の実PID 94967を起動し、そのcommandと生成workerの実PID 94986〜94989、各workerの生成ログを確認した。20のvalid IDグループ＋1 train参照・12文・3,024条件の事前確認、checkpoint/args・初期context・凍結source全136ファイルの一致も通過した。4つの生成processでshard評価し、ASRは各shardを順番に処理する。凍結評価コード・条件・seedは変更しない。初期診断と全量評価の共通216条件も後から照合する。並列化による所要時間短縮はまだ測定していない。監視は20kへの再開を行わない。

10k全量評価の実PID 94967を監視する別controller（実PID 95338）も起動・command確認した。正常終了とreadyを必要とし、共通216条件の同checkpoint再現性確認、controller全体の資源測定、3,024条件の音声・試聴HTML・入力snapshot・診断のアーカイブ作成/照合まで実行した。これらの完了・回収結果は次節に記録する。監視processの稼働だけを完了の証拠にしない。

10k評価開始時には、以後の20k/30k/40kを結果から選ぶ方針とした。その後20kまで進め、現在は試聴前に30kを開始しない判断である。上限40kまでの無条件継続ではない。人手試聴・teacher選定・選定保存点の回収は残っており、初期診断や5k拡張評価の機械的完了だけをステップ3の完了とはしない。5k初期診断の試聴ページはHTTPで到達確認済みだが、人手試聴の結果はまだない。

2k checkpointとoptimizerをローカルへ回収し、双方のSHA-256が実更新検証JSONと一致することを確認した。ローカルは転送・回収物の確認のみで、学習・生成・ASR・集計・テストはVast.aiで実行している。試聴ページもVast.aiの静的配信をSSHで転送する閲覧経路である。

### 10kの全21参照・12文の拡張評価

全20 valid IDグループと1 train参照・12文・12 profileの**3,024条件**を生成・ASR評価し、全条件/WAV・context照合、5k比較、参照別診断までexit 0で完了した。[比較JSON](2026-10-07-gol-m2a-5000-vs-10000-full.json)は固定文章・seed・参照/control PCM・生成/ASR条件・凍結source・学習設定（停止step以外）の一致を確認している。[全参照の集計](2026-10-07-gol-m2a-10000-full-diagnosis.json)と[参照・文章別の診断](2026-10-07-gol-m2a-10000-full-diagnostics.json)を回収した。各profileのvalid参照は240本・7,220文字であり、初期216条件のCERとはmatrixが異なる。

| 同一の生成条件 | 5k valid CER | 10k valid CER | valid上限到達（5k→10k） |
|---|---:|---:|---:|
| trainer / EMA / CFG2 / EOS−4 | 26.37% | 12.24% | 0→0 |
| trainer / EMA / CFG2 / EOS−1 | 31.81% | 17.65% | 0→0 |
| trainer / EMA / CFG1 / EOS−4 | 38.35% | 19.70% | 0→0 |
| trainer / EMA / CFG1 / EOS−1 | 37.22% | 22.65% | 2→2 |
| trainer / raw / CFG2 / EOS−4 | 31.87% | 22.70% | 0→0 |
| trainer / raw / CFG2 / EOS−1 | 42.16% | 29.86% | 5→1 |
| trainer / raw / CFG1 / EOS−4 | 57.88% | 40.93% | 0→0 |
| trainer / raw / CFG1 / EOS−1 | 58.31% | 47.44% | 40→28 |
| public API / EMA / CFG1 / EOS−4 | 39.21% | 17.78% | 20→20 |
| public API / EMA / CFG1 / EOS−1 | 37.69% | 17.98% | 29→22 |
| public API / raw / CFG1 / EOS−4 | 49.20% | 38.17% | 23→46 |
| public API / raw / CFG1 / EOS−1 | 48.38% | 38.13% | 93→87 |

全12 profileのvalid参照CERは改善したが、上限到達の変化は一様ではない。trainer / EMA / CFG2 / EOS−4のvalid参照は置換515・削除183・挿入186、EOS検出240/240本、上限到達0本。参照別CERは7.20%（held_03）〜20.50%（held_16）。同profileのtrain参照1つはCER 18.56%→15.79%で、validと標本数が異なるため両者の順位を一般化しない。1秒未満は全profileで58→59本、すべて`short_question`で、それ以外の文章は0本。best profileの短文13本が1秒未満だったことも、品質失敗や合格の判定にはしない。

同profileのvalid参照の文章別CERは、日付49.62%→29.62%、桁を読む文48.85%→31.35%、数字42.50%→23.54%、固有名詞38.20%→28.80%、単位34.83%→23.62%、長文18.21%→4.29%、手順の長文29.26%→7.56%。日常・依頼・短い質問のCERは10kで0%だったが、地名は10.28%→10.83%とわずかに悪化した。表記差・ASR誤りを含む機械指標であり、発音・自然さや実用品質の証明ではない。実音声6本のASRは同じCER 14.17%で、合成CERから差し引かない。

[同じ10k checkpointの初期・拡張評価の共通216条件](2026-10-07-gol-m2a-10000-early-vs-full-common.json)では、入力・設定・参照PCMを照合し、216/216の生成PCM・ASR転写・scoreが完全一致した。[拡張評価の実行資源](2026-10-07-gol-m2a-10000-full-resources.json)は3,136秒（52分16秒）、5秒間隔628サンプルのNVIDIA表示メモリ最大17,918 MiB、選定時のGPU＋ストレージ単価による時間費用の推定約1.83 USDだった。5kとはcheckpointと生成音声長が異なるため、この時間差だけで4分割の高速化率を主張しない。後から行う圧縮・転送等と請求総額は別である。

[全量回収アーカイブ](2026-10-07-gol-m2a-10000-full-recovery-archive.json)は3,069ファイル・2,038,776,930 bytesとしてVast.aiで作成し、tar内の内容ハッシュを照合した。SHA-256は`3d972766c8564b5d0914752054cb9d0867d3901f4c0d9dca47e3fcb9d65cd252`。[インスタンス外への転送は正常終了し、全体のサイズ・SHA-256が一致した](2026-10-07-gol-m2a-10000-full-local-recovery.json)。監視controller（95338）もexit 0とreadyで終了した。

[全量版の試聴UI確認](2026-10-07-gol-m2a-10000-full-offline-review-verification.json)では、Vast.ai上に全量の試聴用フォルダを3,067ファイル・2,236,464,814 bytesとしてコピー・照合した後、Linux Chromiumの`file://`で3参照×12文の36本のハッシュ・デコード・再生開始・メモJSONの対応を確認した。[全量版のローカル転送も正常終了し、全3,067ファイルのサイズ・SHA-256が一致した](2026-10-07-gol-m2a-10000-full-offline-local-recovery.json)。`runs/vast-m2a-control/offline-review/japanese-m2a-eval-00010000-full/review.html`で全21参照・12文を閲覧できる構成であり、転送で条件を削減していない。このUI確認は人手試聴やWindowsネイティブ再生の検証ではない。トンネル稼働中は`http://127.0.0.1:18725/japanese-m2a-eval-00010000-full/review.html`でも閲覧でき、Vast.aiからのGETとローカルからのHEADの双方でHTTP 200を確認した。

validation lossの低下、全12 profileの同一条件CERの改善、最良profileの上限到達0本を根拠に、[20kまでの学習区間への継続判断](2026-10-07-gol-m2a-10000-continue-to20000.json)を保存し、10kから再開した。全量評価アーカイブの回収と[再開直前の組の事前照合](2026-10-07-gol-m2a-10000-before20000-resume-preflight.json)を完了し、実PID 145707の`m2a_stage.sh 10000 20000`の稼働を確認した。設定の変更は停止点のみで、学習率scheduleはconstant、実効batch 64を維持する。数字等の誤りと条件別の上限到達を次の同一matrixで比較し、40kまでの無条件継続やteacherの品質承認にはしない。

20k起動時には、監視controller（実PID 146155）が学習controllerの終了とexit 0/readyを確認し、20kの組をcleanup対象外へ保護してから全21参照・12文・3,024条件の評価を開始する構成とした。観測失敗で学習を再起動せず、全量評価後の共通216条件・資源測定・回収アーカイブ作成までを行い、次の学習区間は開始しない。これらの機械作業は後述のとおり正常終了した。人手試聴とteacher選定は残っている。

### 20kの学習・保存状態

20kまでの学習は2026-10-07 00:19:32 UTCに終了した。[保存状態の直接検証](2026-10-07-gol-m2a-20000-checkpoint-verification.json)では、optimizerの実更新counterがすべて20,000であり、raw/EMAの重みとoptimizer tensorが有限値であることをVast.aiで確認した。埋め込みは8001×1024、学習率scheduleはconstantのままである。checkpoint SHA-256は`499c037ab2cc63f506592d4203e53d72a68914a4f5eb0025b2efc418c93e417d`、optimizerは`abee657cebac6a22b67616345294b345fd101e2db8b56e06a364e443097883b0`。

20kのvalidation lossは0.1620995305（flow 0.1593205126、EOS 0.0277901722）で、10kの0.1826521873から低下した。19kは0.1613640787であり、毎回単調に低下したという意味ではない。validationは固定50 batchの測定で、全valid音声の品質や日本語の自然さを証明しない。学習時のpeak GPU memory表示は16.4 GiBで、データ準備・評価を含めた必要VRAMの上限ではない。

初期評価後にcleanup対象外へ[20kの組をhardlinkしてハッシュを照合した](2026-10-07-gol-m2a-20000-restart-state-preserved.json)。[checkpointとoptimizerのインスタンス外への転送も正常終了し、双方のサイズ・SHA-256が一致した](2026-10-07-gol-m2a-20000-local-recovery.json)。それぞれ2,561,110,827 bytesと2,561,133,510 bytesである。[元ファイル・保護したファイル・回収receiptの事前照合](2026-10-07-gol-m2a-20000-preserved-resume-preflight.json)もVast.aiで通過した。これは再開状態の識別であり、拡張評価の完了やteacherの品質承認ではない。次の学習区間は開始していない。

### 20kの初期216条件と拡張評価の開始

学習・初期評価controller（145707）はexit 0/readyで正常終了した。[10k→20kの比較](2026-10-07-gol-m2a-10000-vs-20000.json)は固定文章・seed・参照/control PCM・生成/ASR条件・凍結source・停止点以外の学習設定が一致することを確認した。[初期診断](2026-10-07-gol-m2a-20000-diagnosis.json)は3参照×6文×12 profileの216条件で、各profileは18本・480文字である。

| 同一の生成条件 | 10k CER | 20k CER |
|---|---:|---:|
| trainer / EMA / CFG2 / EOS−4 | 8.33% | 11.25% |
| trainer / EMA / CFG2 / EOS−1 | 19.17% | 11.25% |
| trainer / EMA / CFG1 / EOS−4 | 15.21% | 10.63% |
| trainer / EMA / CFG1 / EOS−1 | 16.46% | 13.13% |
| trainer / raw / CFG2 / EOS−4 | 21.04% | 11.88% |
| trainer / raw / CFG2 / EOS−1 | 28.54% | 16.46% |
| trainer / raw / CFG1 / EOS−4 | 32.08% | 24.17% |
| trainer / raw / CFG1 / EOS−1 | 41.04% | 44.58% |
| public API / EMA / CFG1 / EOS−4 | 23.96% | 19.79% |
| public API / EMA / CFG1 / EOS−1 | 23.96% | 19.79% |
| public API / raw / CFG1 / EOS−4 | 41.46% | 23.54% |
| public API / raw / CFG1 / EOS−1 | 40.83% | 23.13% |

12 profileのうち10はCERが改善したが、10kの最良profile（trainer / EMA / CFG2 / EOS−4）は悪化した。同profileのheld_00は9.38%→19.38%（挿入3→22）、held_01は6.88%のまま、seenは8.75%→7.50%だった。20kの置換19・削除3・挿入32、18/18でEOS、上限到達0本。20k全profileの上限到達は21→15本（trainer 4→2、public API 17→13）、1秒未満は両保存点とも0/216本。実音声6本のCER 14.17%は変わらず、合成CERから補正として差し引かない。validation lossの低下だけではこの品質変化を説明できない。

[悪化した2 profileの文章別診断](2026-10-07-gol-m2a-10000-vs-20000-early-regressions.json)では、両保存点の全216条件/contextと固定条件を照合し、対象profileの全18本ずつを比較した。trainer / EMA / CFG2 / EOS−4のheld_00長文はCER 7.14%→27.14%、置換5→2・挿入0→17、音声長12.64→17.44秒で、ASR転写に末尾の句の繰り返しが現れた。双方EOSで終了しているため、EOS到達だけでは反復のない全文読み上げを証明しない。同profileのheld_01日付は0%→30.77%だが、転写の「十月六日午後三時」と「10月6日午後3時」の表記差を含み、日付の誤読とは断定しない。trainer / raw / CFG1 / EOS−1のheld_00数字では挿入2→63、20kは30秒の上限到達で、ASR転写に長い同一文字の連続が現れた。これらは機械転写上の所見であり、音声での反復・発音の確認は人手試聴に残す。

初期216条件の結果は241ファイル・132,032,429 bytesの[アーカイブ](2026-10-07-gol-m2a-20000-early-recovery-archive.json)としてVast.aiで全内容と同梱manifestを照合し、[インスタンス外への転送後の全体ハッシュも一致した](2026-10-07-gol-m2a-20000-early-local-recovery.json)。SHA-256は`c5c2afac150774345fcbcd56f9a1b7f4cbf8e66553665846d2c62400d2d04bab`。

[20k初期評価の試聴UI](2026-10-07-gol-m2a-20000-early-review-verification.json)もVast.aiのLinux Chromiumで確認した。3参照×6文の18本についてWAVハッシュ・デコード・再生開始・音声長とメモJSONの識別対応が通過した。無音設定のheadless検証であり、人手試聴ではない。トンネル経由の`http://127.0.0.1:18725/japanese-m2a-eval-00020000-early/review.html`はローカルからHEADでHTTP 200を確認した。10k初期ページと同じ参照・文章・生成条件で比較できる。

20k初期評価の試聴HTML・全216本の生成音声・参照/control・score/identity・入力snapshotも、Vast.aiで240ファイル・147,022,362 bytesとして試聴用フォルダにコピーし、元結果との全件ハッシュ一致を確認した。[ローカルへのフォルダ転送が正常終了し、全240ファイルのサイズ・SHA-256も一致した](2026-10-07-gol-m2a-20000-offline-local-recovery.json)。`runs/vast-m2a-control/offline-review/japanese-m2a-eval-00020000-early/review.html`は、トンネルとインスタンスの終了後も結果を閲覧できる構成である。ローカルは結果の転送・識別確認だけで、生成・ASR・音声評価は行っていない。Windowsネイティブでの再生と人手試聴は未検証である。

多数の小ファイルの回収用に、既存SSHトンネル経由で最大8ファイルを並行HTTP転送し、完了後に全件を順番にハッシュ照合する`get_offline_review_parallel.py`を別途用意した。[Vast.ai内のloopbackでの実ファイル確認](2026-10-07-gol-parallel-review-transfer-proof.json)では、同じ20k初期結果の全240ファイル・147,022,362 bytesの転送と一致を確認した。その後、20k全量のPCへの実転送と照合も完了した。loopbackの所要時間はPCとの転送性能ではなく、実転送もcheckpoint・音声長・同時転送条件の異なる10k SCPとの差を因果的な高速化率にはしない。学習・評価の凍結sourceは変更していない。

監視controller（146155）は20kの保存状態を保護後、全20 valid IDグループ＋1 train参照・拡張12文・12 profileの3,024条件を実PID 150888で開始した。4つの生成workerの実commandを確認し、初期context・凍結source・checkpointと固定入力の事前照合も通過した。開始時には初期結果だけで継続を決めず、拡張評価を待った。その後の正常終了と全量結果は次節に記録する。人手試聴は引き続き未完了である。

### 20kの全21参照・12文の拡張評価

全20 valid IDグループ＋1 train参照・12文・12 profileの3,024条件の生成、ASR、完全性/context照合、[10kとの固定条件比較](2026-10-07-gol-m2a-10000-vs-20000-full.json)、[参照・文章別の診断](2026-10-07-gol-m2a-20000-full-diagnostics.json)をexit 0で完了した。生成4分割とASR4分割はすべてexit 0。監視controller（146155）も回収アーカイブ作成までexit 0/readyで終了した。[全体の数値診断](2026-10-07-gol-m2a-20000-full-diagnosis.json)と初期216条件のmatrixは異なり、直接比較しない。

| 同一の生成条件 | 10k valid CER | 20k valid CER | valid上限到達（10k→20k） |
|---|---:|---:|---:|
| trainer / EMA / CFG2 / EOS−4 | 12.24% | 15.01% | 0→0 |
| trainer / EMA / CFG2 / EOS−1 | 17.65% | 18.05% | 0→0 |
| trainer / EMA / CFG1 / EOS−4 | 19.70% | 18.84% | 0→0 |
| trainer / EMA / CFG1 / EOS−1 | 22.65% | 17.96% | 2→1 |
| trainer / raw / CFG2 / EOS−4 | 22.70% | 19.00% | 0→0 |
| trainer / raw / CFG2 / EOS−1 | 29.86% | 21.90% | 1→4 |
| trainer / raw / CFG1 / EOS−4 | 40.93% | 36.93% | 0→4 |
| trainer / raw / CFG1 / EOS−1 | 47.44% | 43.45% | 28→31 |
| public API / EMA / CFG1 / EOS−4 | 17.78% | 17.19% | 20→18 |
| public API / EMA / CFG1 / EOS−1 | 17.98% | 17.16% | 22→18 |
| public API / raw / CFG1 / EOS−4 | 38.17% | 28.34% | 46→28 |
| public API / raw / CFG1 / EOS−1 | 38.13% | 27.83% | 87→40 |

12 profile中10はvalid参照CERが改善したが、10kの最良条件（trainer / EMA / CFG2 / EOS−4）は12.24%→15.01%に悪化した。各profileはvalid参照240本・7,220文字である。同profileは置換515→507・削除183→243・挿入186→334、両保存点ともEOS検出240/240本・上限到達0本。20kの参照別CERは8.59%（held_02）〜29.92%（held_09）。train参照1つは15.79%→9.14%だが、12本の結果だけで過学習や実声優への汎化を断定しない。全profileの1秒未満は59→79本で、すべてshort_question、それ以外の文章は0本。実音声6本のCER 14.17%は変わらず、合成CERから差し引かない。

主要条件のvalid参照の文章別CERは、長文4.29%→12.50%（挿入24→132、削除0→6）、手順の長文7.56%→14.60%（挿入36→60、削除48→98）に悪化した。日付29.62%→31.54%、数字23.54%→25.83%、固有名詞28.80%→33.80%も悪化した。一方、桁を読む文31.35%→28.08%、地名10.83%→0.83%、外来語3.21%→0.54%、単位23.62%→21.38%は改善した。EOS到達やvalidation lossの低下だけでは、反復・全文の読み上げ・自然さの改善を保証しない。人手試聴を合わせて判断する。

[同じ20k checkpointの初期・拡張評価の共通216条件](2026-10-07-gol-m2a-20000-early-vs-full-common.json)は、入力・設定・参照PCMを照合し、216/216の生成PCM・ASR転写・scoreが完全一致した。[拡張評価の資源測定](2026-10-07-gol-m2a-20000-full-resources.json)は2,990秒（49分50秒）、5秒間隔598サンプルのNVIDIA表示メモリ最大18,566 MiB、選定時のGPU＋ストレージ単価による時間費用の推定約1.74 USDだった。回収・転送と請求総額は別で、checkpointや生成音声長の異なる10kとの時間差を高速化率として扱わない。

[20k全量アーカイブ](2026-10-07-gol-m2a-20000-full-recovery-archive.json)は3,069ファイル・1,923,740,800 bytesとしてVast.aiで作成・全内容照合し、[インスタンス外への転送後のサイズ・SHA-256も一致した](2026-10-07-gol-m2a-20000-full-local-recovery.json)。SHA-256は`0c69b1b649783bf021399822ca60a733f6587d8b3f30eaccebe249d568031958`。全量試聴用フォルダも3,067ファイル・2,105,793,078 bytesとしてコピー・全件ハッシュ照合した。[Vast.aiのLinux ChromiumでのUI確認](2026-10-07-gol-m2a-20000-full-offline-review-verification.json)は3参照×12文の36本のデコード・再生開始・音声長・メモJSONの識別対応を通過した。[PCへの全量フォルダ転送も正常終了し、全3,067ファイルのサイズ・SHA-256が一致した](2026-10-07-gol-m2a-20000-full-offline-local-recovery.json)。最大8ファイルの並行転送と逐次照合の実所要時間は696.844秒。ローカルは転送・識別確認のみで、生成・ASR・集計はVast.aiで行った。Windows再生・人手試聴・teacher選定は未完了である。

### 20k後の継続判断と比較試聴

[30kを試聴前に開始しない判断](2026-10-07-gol-m2a-20000-review-gate.json)をVast.aiで保存した。固定条件の比較、全3,024条件の完全性、10k/20kのcheckpoint・optimizerと全量結果の回収記録、終了済みcontrollerのexit 0/readyを照合した上で、主要条件のvalid CERと長文2種の悪化を根拠に追加区間を見合わせる。12 profile中10の改善も記録し、20k全体が悪化したとは扱わない。10kを機械評価上の優先候補、20kを比較候補として保持する。これはteacherの最終選定でもM2a全体の完了でもない。

試聴は回収済みの10k全量ページ（ローカル成果物: `runs/vast-m2a-control/offline-review/japanese-m2a-eval-00010000-full/review.html`）と20k全量ページ（ローカル成果物: `runs/vast-m2a-control/offline-review/japanese-m2a-eval-00020000-full/review.html`）で行える。`trainer / EMA / CFG2 / EOS−4`を揃え、複数valid参照の`long`・`long_sequence`を中心に、全文、反復、読み落とし、発音、自然さ、終了位置を比較する。数字・日付等も確認する。ASR転写の表記差を誤読の証拠にしない。試聴結果からteacher選定または次の限定区間を判断する。

[所有インスタンス54521683の停止を独立したAPI照会で確認した](2026-10-07-gol-m2a-20000-instance-stopped.json)。`cur_state=stopped`・`intended_status=stopped`で、APIの`actual_status=exited`はこの停止操作後の値である。停止CLIはexit 0だったが出力がJSONでなく、制御helperのparserが失敗したため、操作を繰り返さず状態照会で確認した。インスタンスは削除しておらずデータは保持される。[Vast.aiの課金仕様](https://github.com/vast-ai/docs/blob/main/guides/reference/billing.mdx)では停止中もストレージ料金が発生する。選定時単価のストレージ推定は約0.2963 USD/hで、請求総額ではない。再開はGPU空き次第となる。試聴後に追加区間または最終削除を判断し、他プロジェクトのインスタンスは操作しない。

### 固定した共通モデル資産の保持

[英語24層の初期重みと日本語tokenizerの保持記録](2026-10-07-gol-shared-model-assets-recovery.json)では、Vast.aiで使用したrevision・SHA-256・実体サイズと、ローカルに存在する同版の資産との一致を確認した。初期重みは1,304,206,614 bytes・SHA-256 `8fecefe7206ccfb02fe61a1bd1496d3238f534056f19e5dc0421049cbfbfa4df`で、キャッシュのシンボリックリンク先の実体から回収フォルダへ別途保持した。日本語tokenizerは`training/tokenizers/japanese_8000.model`にあり、SHA-256 `4549ee1be7a484cbfd0cc38e3277e7fc91141f42a897152c81238eedb692d25f`が一致した。trainerのcheckpointと、Mimiを含む初期資産は区別する。ローカルでモデルやtokenizerを実行した確認ではなく、ファイル識別の確認である。tokenizerの過去のGOL＋MoeSpeech由来は変わらず、今回のGOL単独音声学習とは区別する。

### optimizerの保持方式と再開用保存

現行`training/checkpointing.py`はraw/EMA checkpointを最大40本残す一方、optimizer sidecarは最新1本だけを保持する。5kのoptimizerは6k保存時に削除されていた。3k/5k保存時の実更新・有限値の検証と、それぞれ次の区間へのoptimizer込み再開は当時確認済みだが、現在の3k/5kの重みだけから当時のoptimizer状態は復元できない。2kの組は回収済みで残っている。5k checkpointも回収・SHA-256照合を完了した。5k optimizerの回収完了を意味しない。

容量について「40組のcheckpoint＋optimizerを自動保持する」という説明を訂正した。40 checkpoint＋最新1 optimizerは約105.0 GB（97.8 GiB）で、別保存したoptimizerは追加容量となる。次の学習区間へ進む前に、停止点の初期評価・保存済みcontext・checkpoint/optimizerハッシュを確認し、`runs/m2a-control/preserved-restart-state/<step>/`へ組を別保存する。実行中のモデル/評価sourceは変更しない。[10kでこの別保存を実行・確認した](2026-10-07-gol-m2a-10000-restart-state-preserved.json)。checkpointとoptimizerを同じファイルシステム内でhardlinkし、双方のハッシュが実更新検証JSONと一致した。trainerの元pathへのunlink後も別pathは残る。[10kの組をインスタンス外へ回収し、双方のサイズとSHA-256を照合した](2026-10-07-gol-m2a-10000-local-recovery.json)。Vast.aiでの実更新・有限値・contextの検証と、ローカルでの転送物の識別確認を区別する。

[10kの再開ファイルの事前照合](2026-10-07-gol-m2a-10000-preserved-resume-preflight.json)では、現在のrun内のcheckpoint/optimizer、cleanup対象外の退避ファイル、回収済みファイルのreceiptについて、保存点・実更新counter・サイズ・ハッシュの対応を確認した。現行loaderはoptimizerが欠けると新規optimizerへ進みうるため、再開前に`runs/m2a-control/preflight_preserved_resume.py`で欠落や改変を拒否する。Vast.aiで実際の10kの組を照合し、小さな実ファイルを使った正常系、optimizer欠落、同サイズの改変、未回収の4テストも成功した。この確認は再開状態の識別であり、拡張評価の完了や品質判断、teacher選定を代替しない。

[20k区間への再開後の確認](2026-10-07-gol-m2a-10000-post-cleanup-preservation.json)では、元の10k optimizerが消え、12kのoptimizerが保存された実際の状態を確認した。12k sidecarをVast.aiで直接読み、optimizerの実更新counterはすべて12,000だった。cleanup対象外の10k optimizerのSHA-256は`2dfb82fb507467335262897e31c3557d4d38331873f34fc9a1385cee37aecc64`のままで、保護時・回収時の値と一致した。12kの検査範囲はcounterであり、その全tensorの有限値は確認していない。20kの最終保存状態・有限値は別途直接検証済みである。

実行ソースの回収では、[アーカイブ内容の検証JSON](2026-10-07-gol-source-archive-verification.json)に従い、学習・準備時の`stage4-source.tar`内の凍結135ファイル、評価時の`stage7-evaluation-source.tar`内の凍結136ファイルの全件ハッシュをVast.aiで照合した。各tarにはmanifest対象外の静的HTML・README・smoke shellの4ファイルも含まれる。tarを展開せずに内容を照合し、両アーカイブとmanifest・パッケージ一覧を回収して転送後のハッシュも一致した。Git HEADだけでは再現できない未コミット変更を含む実行状態の保存であり、公開用の成果物一式の回収完了とは区別する。

最終データの来歴も、train/valid bounded manifest、latent manifestとmetadata、取得・前処理の設定/監査、除外記録、連結音声のハッシュ、評価参照・モデルrevisionを含む504ファイルとして保存した。[データ回収の記録](2026-10-07-gol-data-recovery-archive.json)の約383.8 MBの圧縮アーカイブは、Vast.aiで全内容のハッシュを照合し、最終監査と5k評価の入力に一致することを確認した。回収後のアーカイブSHA-256も一致した。元音声・連結音声・latent tensor・重みを含まないため、再学習には音声の再取得/再構築とlatents計算が別途必要である。

[初期評価の回収記録](2026-10-07-gol-early-evaluations-recovery-archive.json)では、2k・3k・5kの各216条件、計648本の生成音声と参照/実音声control、試聴HTML、score・identity、保存済みargs/progress等を721ファイル・約450.9 MBの圧縮アーカイブに保存した。Vast.aiで各評価の完全性・保存済みcontext・固定条件一致とtar内の全ファイルハッシュを確認し、回収後のアーカイブSHA-256も一致した。重複する`shard-0`は含めず、照合済みの統合結果を保存する。5kの拡張3,024条件と選定するteacherの重み・optimizerは、これとは別に回収が必要である。

[10kの初期216条件も別アーカイブに回収した](2026-10-07-gol-m2a-10000-early-recovery-archive.json)。241ファイル・140,452,442 bytesで、実際のWAV・参照/control・試聴HTML・score/identity・入力snapshotと診断を保存した。Vast.aiで全条件/context・5kとの比較可能性、tar内の全内容と同梱manifestを照合し、回収後のサイズとSHA-256 `8e25b370b84f7d7f0ad04576c925123f55c721addf3fdbe6e3b1501e9b27d23b`も一致した。拡張評価の同じ文章の音声で初期結果を置き換えず、初期評価のWAV headerと識別情報をそのまま残す。

[10kのオフライン試聴ページの確認](2026-10-07-gol-m2a-10000-offline-review-verification.json)もVast.aiのLinux Chromiumで実行した。`file://`でページを開き、trainer / EMA / CFG2 / EOS−4の3参照×6文、18本についてハッシュ・デコード・再生開始・音声長を確認した。モデル表示とメモJSONのscore/identity対応、trainerの8条件とpublic APIの4条件の表示も確認した。無音設定のheadless確認であり、人手試聴・日本語品質・Windowsネイティブ動作の証拠ではない。HTMLはscore等を内包し、音声は同じdirectoryのWAVを参照する。

SSHトンネルの終了後も結果を閲覧できるよう、Vast.ai上の`runs/m2a-control/offline-review/japanese-m2a-eval-00010000-early/`に同じ初期評価の試聴HTML、WAV、score/identity、入力snapshot等を240ファイル・157,012,726 bytesとしてコピーし、全件の内容ハッシュを照合した。[ローカルの同名フォルダへの転送は正常終了し、全240ファイルのサイズとSHA-256が一致した](2026-10-07-gol-m2a-10000-offline-local-recovery.json)。`runs/vast-m2a-control/offline-review/japanese-m2a-eval-00010000-early/review.html`をブラウザで開いて閲覧できる構成である。ローカル側は結果の転送とファイル識別の確認だけを行い、生成・ASR・音声評価は再実行していない。Windowsネイティブでの再生は未検証で、人手試聴も未完了である。

5kとの同一条件の試聴比較用に、5k初期評価もVast.ai上で同様に240ファイル・157,024,555 bytesをコピー・ハッシュ照合した。[5kのローカル版も転送が正常終了し、全240ファイルのサイズとSHA-256が一致した](2026-10-07-gol-m2a-5000-offline-local-recovery.json)。`runs/vast-m2a-control/offline-review/japanese-m2a-eval-00005000-early/review.html`を同じ文・参照・生成条件で聴き比べに使える構成である。[5kコピーのページもVast.aiのLinux Chromiumの`file://`で確認し](2026-10-07-gol-m2a-5000-offline-review-verification.json)、同じ3参照×6文の18本の音声ハッシュ・デコード・再生開始・メモJSONの対応が通過した。Windowsネイティブでの再生・人手による聴き比べの完了とは区別する。

[5k拡張評価の回収アーカイブ](2026-10-07-gol-m2a-5000-full-recovery-archive.json)も、生成音声3,024本・参照/control・試聴HTML・score/identity・入力snapshot・診断等を3,069ファイル、2,038,507,722 bytesとしてVast.aiで作成し、tar内の内容ハッシュを確認した。重複shard directory、モデル重み・optimizer、元データは含まない。SHA-256は`8f17b057fbd9b3946de4df030aa06e1364470de4e439b911a00776cbd561ec0b`。転送は正常終了し、回収後のファイルサイズとアーカイブSHA-256も一致した。5kの拡張試聴ページはVast.aiとSSHトンネル経由のlocalhostの双方でHTTP 200を確認した。閲覧URLは`http://127.0.0.1:18725/japanese-m2a-eval-00005000-full/review.html`で、トンネルとremote静的配信の稼働中に利用できる。

10k拡張評価の生成shard controller・起動/監視script、5kの回収・診断・再現性・資源測定helperなど10ファイルも、`runs/vast-m2a-control/m2a-10k-execution-controls.tar`として回収した。SHA-256 `5b4f17c83b8e33e5f582845c4b67179e589a369ad99b7374761363e7a4259eb4`はVast.aiで作成した記録と回収後に一致した。このtarは制御scriptの保存で、凍結したモデル/評価ソースのアーカイブや学習結果とは別である。

[試聴回収・再開チェックの制御アーカイブ](2026-10-07-gol-offline-and-resume-controls-archive.json)も回収した。12ファイル・143,360 bytesで、転送用フォルダ作成・転送時の識別確認、再開状態の事前照合と4テスト、試聴UI確認のsource、関連receiptと転送manifestを含む。Vast.aiでtar内の全ファイルのサイズ・ハッシュと同梱manifestを照合し、回収後のアーカイブのサイズ・SHA-256 `c7d00117771b52daa9da082ec6aa23ce5ed91ef2e26a41f48a2c469e33fdf4fc`も一致した。モデル重みや音声はそれぞれ別に回収した成果物であり、この制御アーカイブには含まない。

[20k区間の実行制御アーカイブ](2026-10-07-gol-m2a-20000-execution-controls-archive.json)も回収した。起動・監視script、固定設定、継続判断、再開状態の回収記録等を20ファイル・92,160 bytesに保存し、Vast.aiでtar内の全ファイルと同梱manifestを照合した。回収後のサイズとSHA-256 `f98b5fe3ebfc5ca4e267bb4b14c092370858703ed5a41a825573a22429d142bf`も一致した。モデル重み・音声・秘密情報は含まない。実行手順の保存であり、20kの学習・評価完了の証拠とは区別する。

初期216条件は2kを、全21参照・12文の拡張3,024条件は5kを比較基準にする。保存点の比較では固定文章・seed・参照PCM・生成/ASR条件と評価ソースを揃え、validation、文字数で重み付けしたCER、編集誤り、短い出力、反復、EOS/上限到達を診断する。実音声によるASR誤りも併記する。機械指標だけでteacherに選定せず、必要な保存点の機械評価と人手試聴を継続する。
