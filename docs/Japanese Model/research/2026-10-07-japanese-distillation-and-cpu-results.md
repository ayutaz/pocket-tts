# 日本語6層蒸留・CPU評価の結果と残件

> **2026-10-15 最終状態:** 日本語モデル開発は終了し、全6モデルのpublic/manual gateでのHF保存公開と専用Vast.ai環境の削除を完了しました。本文は当時の設計・計画・測定の履歴です。現在の状態は[終了・保存公開サマリ](../index.md)を参照してください。

**2026-10-15追記：ユーザー指定により開発終了。** 以下は終了時点で保持する実験結果と未達条件の記録であり、追加学習や品質選定の継続指示ではない。既存HFモデル・回収済み成果物を残す。[公開モデルの確認と終了記録](2026-10-15-project-end-and-released-model-check.md)を参照。

2026-10-07。実行範囲は6層蒸留とCPU評価までで、公開は行わない。学習・推論・テスト・集計はすべてVast.aiの専用インスタンス54521683で実行した。

## 確認済みの結果

### 日本語studentの最初の5k区間

2026-10-07 13:29 JSTに5kの蒸留学習を完了した。[保存状態の直接検証](2026-10-07-japanese-distill-5000-verification.json)でoptimizer counter 5,000、raw/EMA/optimizerの有限値、6層構造、teacher由来head 70個の完全一致、backbone 48 tensorの初期値からの変化を確認した。5kのcheckpointは709,394,483 bytes、SHA256 `f9dee3bc786fbcc1358a74a35d1885e7084700eb9d4d5b40d3875a3d9a7d41a8`、optimizerは670,318,242 bytes、SHA256 `5334737c97b1ed61568aefbf896404042cf6e6b7f5774a62a773ce569b3b06c4`。cleanup対象外の`runs/stage4-control/distill-preserved/5000`にも同じ組を別保存した。

50 batchの蒸留validation MSEは500 stepの0.09778から5kの0.04255へ低下した。これはteacherの中間表現に対する誤差で、M2aのflow/EOS lossや音声品質とは別である。trainerが報告した学習時のpeak GPU allocationは10.8 GiBであり、4 worker生成評価などを含むインスタンス全体の最大VRAMではない。

全量評価の初回は、隔離workspaceの相対case pathを成果物ルート外の絶対パスに解決したため、4 workerとも音声生成前に停止した。初回controllerと失敗ログを保持し、case pathだけを元の固定ファイルの絶対パスに直した新controllerで、新出力先`runs/japanese-student5000-eval-full-v2`に再実行した。固定の評価ソース・文章・参照・seedは変更していない。14:00 JSTまでに全3,024条件の生成・ASRとteacher比較を完了した。条件の重複・欠落、参照PCM、入力と評価ソースのハッシュ、実音声controlの転写・score・PCM、全出力WAVのハッシュを照合した。

### 5kの全量品質評価

[24層10k対6層5kの比較](2026-10-07-teacher10000-vs-student5000-full.json)では、主要条件（trainer / EMA / EOS−4、teacher CFG2 / student CFG1、20 valid参照×12文）のCERは12.24%→51.22%。studentは235/240本がEOS終了、5本がEOSなしで上限到達した。長文CERは4.29%→45.36%、手順の長文は7.56%→65.06%。通常文は0%→15%、地名は10.83%→9.17%だった。数字・長文・固有名詞を含め全体の品質差が大きく、5k studentを実用品質の合格にはしない。raw studentでも同じCFG1/EOS−4のCERは42.37%で、EMAだけを外せばteacherに並ぶ状況ではない。

[単独exportの検証](2026-10-07-japanese-student5000-export-verification.json)でcheckpointを使わずCPUにロードし、全FlowLM tensorが5kのmerged EMAと完全一致すること、同じ通常文のPCMがcheckpoint経路とrtol 1e-5 / atol 1e-6で一致することを確認した。export SHA256は`a1b27310b7daf6a09f3a5289f0477ec2f20217d53d6438b5bea9c0bdd5ba01ae`。ロード成功は品質承認ではない。[実際の試聴ページ検証](2026-10-07-japanese-student5000-review-verification.json)ではVast Linux Chromiumで3参照×12文＝36本の主要条件のdecode・再生開始と、メモのWAV hash／文章／参照／条件の対応を確認した。音声はmuteであり、人手試聴やWindows再生の証拠ではない。

### 5kモデルのCPU固定12文評価

14:06 JSTまでに、通常文・短文・長文・数字・単位・固有名詞を含む固定12文×2 pass×4 model/precision、計96リクエストのCPU生成とASR評価を完了した。[結果](2026-10-07-japanese-cpu-full5000-summary.json)に各リクエストの速度・応答開始・メモリ・WAV hash・ASR転写・誤り・上限警告を保存した。参照は同じheld_00、Vast.ai CPU threads 1、公開API CFG1。全CPU測定が終わってから固定ASRで評価し、実音声control 6本の転写・scoreとASR identityの一致を確認した。生成上限の警告を完了リクエストへ対応させる処理はVast.aiで6件の回帰テストを通した。

| モデル | 精度 | RTF中央値 | CER | peak RSS（GiB） | 上限警告あり（24回中） |
|---|---|---:|---:|---:|---:|
| 24層 / M2a 10k | FP32 | 0.521 | 18.56% | 4.30 | 4 |
| 24層 / M2a 10k | INT8 | 0.183 | 17.45% | 4.30 | 4 |
| 6層 / student 5k | FP32 | 0.156 | 42.66% | 1.73 | 5 |
| 6層 / student 5k | INT8 | 0.069 | 37.67% | 1.73 | 1 |

ここでのCERは1参照のCPU公開API・24リクエストであり、20 valid参照のtrainer評価とは集計対象が異なる。studentはCFG2を蒸留し、teacher公開APIはCFG1のため、同じguidance／品質での速度比較ではない。RTFの分母は出力音声長であり、余分な発話や上限到達を含む出力でも値が小さくなりうる。警告なしは独立したEOS traceの証拠ではない。INT8のCERが低い1参照の結果だけで、量子化により一般的な品質が改善したとは判断しない。

CPU生成96本、参照・実音声control、固定ソース、設定・テスト・実行ログ・ASRを含む追加archive（83,804,160 bytes、SHA256 `5765bb593771a62a1dec013741f1cbf9d34faff1172c4fcbde5c6205f0af8e29`）は回収し、[308ファイルのサイズ・ハッシュを照合](2026-10-07-japanese-cpu-full5000-local-recovery.json)した。5k重み・optimizer・export・固定ソース141ファイル・全3,024条件を含む本archive（4,448,563,200 bytes、SHA256 `93cf11e71de8a6ecf34242ca469d79cc95e512e2589c9300d8b6c5c86238a296`）も[回収・照合を完了](2026-10-07-japanese-student5000-local-recovery.json)した。ローカルで行ったのは転送後のサイズ・ハッシュ照合と試聴成果物の展開であり、学習・推論・ASR・品質集計はVast.aiで実行した。

teacher／studentの試聴ページ（ローカル成果物: `runs/vast-m2a-control/offline-review/japanese-teacher10000-student5000-paired-review/review.html`）はheld_00／held_09／held_16の固定12文を左右で比較できる。teacher EMA / CFG2とstudent EMA / CFG1の実音声72本を、元の条件・WAV hash・ASR所見へ対応させた。85ファイルを回収・照合し、[Vast Linux Chromiumで全72音声のdecode・再生開始とモデル別メモ対応を確認](2026-10-07-japanese-teacher-student-paired-review-verification.json)した。2026-10-08に通常文1組の人手所見を受領した。長文等の試聴、teacher選定・studentの品質承認は未完了。

### 2026-10-08の人手試聴所見（通常文1組）

ユーザーは、teacher10000 EMA / CFG2 / EOS−4の3.68秒・CER0%と、student5000 EMA / CFG1 / EOS−4の5.12秒・CER20%の音声について「はasr通りに読めていますよ」と報告した。引用された条件・秒数・ASRからheld_00 / dailyの組へ対応づけ、[WAV hash付きの所見](2026-10-08-japanese-teacher-student-listening-notes.json)を記録した。

teacherの発話内容「こんにちは今日はいい天気ですね」は入力文と一致する。studentの「こんにちは。今日はいい天気っすね。きっ」も実際の発話として確認されたため、「ですね」から「っすね」への変化と末尾の余分な「きっ」はASRだけの誤認ではない。この通常文の内容確認から、自然さ・終了位置の適切さや他の文章の品質までは判断しない。EOS終了は既存の機械記録であり、余分な発話がないことを意味しない。次の試聴対象は同じheld_00のlongとlong_sequenceである。

### アーカイブからの隔離再生

[隔離した作業場所での再現確認](2026-10-07-japanese-student5000-isolated-archive-replay.json)では、Vast.ai上で回収用archiveからモデル重みと固定package sourceを展開し、相対パスのモデル設定・tokenizer・固定参照を用意した。Pythonを`-I`で起動し、元の学習runを含まないディレクトリの`pocket_tts`をimportした。HF_HOMEは新しい空ディレクトリ、HF_HUB_OFFLINEは1、deviceはCPUとし、checkpointを渡さずにロード・通常文の音声生成を実行した。出力5.36秒のPCMは、既存のcheckpoint経路の出力とrtol 1e-5 / atol 1e-6で一致した。

これは同じVast Linuxホスト・既存の依存環境におけるartifact再現性の証拠であり、依存関係の新規インストールやWindows動作の確認ではない。生成ではEOSなしの上限警告が出ており、一致した音声を日本語品質の合格にしない。5kの品質差がexportや元runへの偶然の依存だけで生じたとは考えにくく、今後は蒸留モデルの読み・終了を診断する必要がある。

### 5kの保存状態からの再開

[再開試験](2026-10-07-japanese-student5000-resume-verification.json)では、回収・照合済み5kの組を元runと別のrunへ用意し、更新前にraw model・EMA shadow・optimizerの全状態がcheckpointと完全一致することを確認した。その後、実際に5回更新し、optimizer counter 5,005、raw/EMA/optimizerの有限値、teacher由来head 70個の一致、backbone 48 tensorの変化とEMAの変化を直接確認した。元の5k checkpoint/optimizerのSHA256は更新後も同じだった。データloader位置・RNGはcheckpointに保存されないため、中断なし学習とのbitwise一致は主張しない。

初回試験はhelperにmain guardがなく、spawnされたloaderが初期化を再実行して停止した。親も子への送信待ちで止まったため、実際のPID・cmdline・run_start step 5,000のみの履歴を確認してその試験の親だけを終了した（controller exit 143）。学習ループには入っておらず、元runは保持した。main guardを追加した別helper・別runで再試験し、controller exit 0と実更新を確認した。失敗ログも保持した。[追加controlsの90ファイル](2026-10-07-japanese-student5000-final-controls-local-recovery.json)は回収・照合済み。5,005の重みそのものは技術試験用としてVast.aiに保持し、最終品質候補として回収した5k重みとは区別する。

### GPU停止と品質判断の残件

14:26 JSTに、[実PIDとGPU利用者の確認](2026-10-07-japanese-stage4-final-quiescence.json)で学習・評価・controllerが残っていないことを確認した。回収照合後に所有54521683へ停止要求を1回送信した。CLIはexit 0だがJSONを返さずparserでエラーになったため、要求を繰り返さず[APIで停止状態を再照会](2026-10-07-japanese-stage4-instance-stop-status.json)した。`actual_status=exited`、`cur_state=intended_status=stopped`を確認した。[停止記録](2026-10-07-japanese-stage4-instance-stop.json)には応答形式の問題と独立した状態確認を併記する。他プロジェクトのinstance状態は停止前後で一致した。GOL音声・latentsは保持しており、storage課金は残る。最終削除はまだ行わない。

| ステップ4の確認項目 | 証拠と判定 |
|---|---|
| 日本語24層→6層の実学習と保存 | 5k actual optimizer counter、有限値、6層、teacher head保持、protected pairを直接確認済み |
| 保存・再開 | raw/EMA/optimizerの完全復元と5000→5005の実更新を確認済み |
| 複数valid参照・固定条件による品質比較 | 20 valid＋1 train参照・12文・12 profile、全3,024条件を検証。主要条件ではstudent CER51.22%でteacherより悪い |
| CPU FP32／INT8の性能・品質 | 固定12文×2回×4構成の96リクエスト、速度・latency・RSS・ASR・警告・WAV hashを保存済み。Linux単一ホスト・1参照の範囲 |
| checkpointなしのartifactロード・再生成 | 元runを含まないsource・空HF cache・offline設定でロードし、1通常文のPCM一致を確認済み |
| 重み・optimizer・source・設定・環境・ログ・音声の回収 | 本archive、CPU追加archive、paired review、追加controlsを回収・照合済み |
| 人手試聴、teacher／studentの品質選定、必要なら次区間の判断 | **一部受領。10/08にheld_00 / dailyのteacher／studentのASRと実発話の一致を確認。長文等の試聴・品質選定は未完了。5k studentは機械評価でも品質差が大きく、最終候補へ採用しない** |
| 公開しない、処理をVast.aiへ限定する | 新規公開は未実行。ML・ASR・評価・テストはVast.ai。GPU停止確認済みでdataを保持 |

ステップ4の全体完了は未証明である。次に人手所見を機械診断へ対応させ、teacher選定とstudentの次の限定区間または設定変更を判断する。人手所見なしで品質承認を作らず、M2a 30kや蒸留200kへ自動継続しない。

### 35 stepの技術検証と予備測定

10kのM2aを暫定teacherとして、6層studentの0→30→35 stepを実行した。optimizerの実更新counter、raw/EMA/optimizerの有限値、6層の構造、teacher由来の70 head tensorの完全一致、30 stepからの保存再開を直接確認した。[検証記録](2026-10-07-japanese-distill-pilot-verification.json)と[隔離workspaceの確認](2026-10-07-japanese-distill-workspace-preflight.json)を参照。元の固定ソース136ファイルを変更せず、新しい作業場所から同じGOLデータ・latentsを参照した。

CPU測定の初回はFLOAT WAVをPCM専用readerへ渡して全4条件が失敗した。SoundFileで固定参照のfloat32 monoをそのまま読み込み、sample rateが違う場合は拒否するよう修正した。参照stateは初回に一度生成し、以後はアプリケーションで保持して再利用する。ファイル名に基づく内部LRUキャッシュとは区別する。回帰テストと蒸留設定テストは[10件成功](2026-10-07-japanese-cpu-and-distill-tests.xml)。Ruffのcheckとformat checkもVast.aiで成功した。

修正版CPU測定は全4条件で各6リクエストを完了し、出力WAVのハッシュを検証した。CPUはAMD Ryzen 9 9950X3D2、Torch 2.14.0+cu130、CPU threads 1。通常文・長文・短い質問の3文を2回ずつ生成した。[集計と実行条件](2026-10-07-japanese-cpu-pilot-summary.json)にcheckpoint・設定・参照・測定ソースのハッシュを含む。

| モデル | 精度 | RTF中央値 | 非無音chunk到着の中央値（秒） | 生成上限警告（6回中） |
|---|---|---:|---:|---:|
| 24層 / M2a 10k | FP32 | 0.551 | 0.167 | 2 |
| 24層 / M2a 10k | INT8 | 0.192 | 0.049 | 2 |
| 6層 / pilot 35 | FP32 | 0.164 | 0.054 | 5 |
| 6層 / pilot 35 | INT8 | 0.074 | 0.026 | 4 |

RTFは生成処理秒数/出力音声秒数。非無音chunk到着は振幅が1e-4を超えるchunkを受け取った時間であり、音声デバイスの再生開始時間ではない。初回モデルロードと参照処理は別に記録し、上表の生成時間には含めない。モデルロードを含むプロセス生涯のpeak RSSは24層約4.30 GiB、6層約1.73 GiBで、INT8でもロード途中のピークを含む。

## 解釈と残件

最初の日本語5k区間・保存状態の直接検証・全3,024条件・CPU固定12文評価・単独exportロード・5kからの再開試験・成果物の回収照合を完了した。pilotの学習状態は再利用していない。GOL manifest、latents manifest、tokenizer、teacherのハッシュと元ソース136ファイルを開始直前に再照合し、6件の設定テストをVast.aiで成功させた。残る品質判断は上表の人手試聴・teacher／student選定である。

最初の保存確認スクリプトには`shrink()`のtupleを辞書として扱う誤りがあった。実行中の学習・controller・確認スクリプトは変更せず、終了後に独立した修正版がoptimizer counter、有限値、構造、teacherのhead保持と実際のbackbone変化を確認済み。controllerの終了コード1は学習後の確認スクリプトの失敗であり、修正版による保存状態確認は別の証拠として記録する。

開始直前の[入力・ソース照合](2026-10-07-japanese-distill-5000-preflight.json)、[6件の設定テスト](2026-10-07-japanese-distill-5000-tests.xml)、[現在のジョブとログ確認](2026-10-07-japanese-stage4-current-status.json)を保存した。現在のジョブ確認は各PIDの実コマンドと存在を照会したもので、startedファイルだけによる稼働判定ではない。全量評価、CPU測定、単独exportロード、回収用アーカイブ作成は順に前段を待つ。アーカイブを作成しただけではローカルへの回収・照合完了とは扱わない。

**35 stepは実更新・再開の技術検証であり、日本語の蒸留品質を確認したモデルではない。** teacherも人手試聴前の暫定候補である。複数リクエストがEOSなしで上限到達したため、上表を実用的な全文読み上げ速度や品質の証拠にしない。studentはCFG2を蒸留する設定、teacherの公開APIはCFG1であり、同じ品質条件の比較でもない。既存HFキャッシュを利用した新規プロセスの測定で、ネットワークからの冷ダウンロードは含まない。Vast.aiのLinux CPU結果でWindowsの動作は検証できない。

次は回収済みteacher候補と5k studentの人手試聴を記録し、機械診断との対応からteacher選定とstudentの次の限定区間または設定変更を決める。最終候補を選んだ後、FP32/INT8の音声品質とCPU性能を再評価する。重み・optimizer・環境・設定・評価・音声を回収・照合するまで作業環境を保持する。公開は未実行であり、今回の範囲にも含めない。
