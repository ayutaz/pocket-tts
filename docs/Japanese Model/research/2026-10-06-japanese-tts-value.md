# Pocket TTS 日本語版の優位性とインパクトの再評価

> **2026-10-15 最終状態:** 日本語モデル開発は終了し、全6モデルのpublic/manual gateでのHF保存公開と専用Vast.ai環境の削除を完了しました。本文は当時の設計・計画・測定の履歴です。現在の状態は[終了・保存公開サマリ](../index.md)を参照してください。

> 2026-10-07追記: 市場調査の確認日は10月6日のままで、今回の文書更新で市場検索を再実行していません。その後、[小規模GPU試運転](2026-10-06-vastai-gpu-smoke-results.md)、[144条件の読み上げ機械評価](2026-10-07-japanese-reading-evaluation-results.md)、[GOL単独M2aの20kまでの学習と5k/10k/20k各3,024条件の評価](2026-10-07-gol-only-m2a-training-results.md)を完了しました。主要条件のvalid CERは10k 12.24%→20k 15.01%で、30kは試聴前に開始しない判断です。実用品質、人手試聴・teacher選定・6層蒸留・競合実測は未確認です。現在の実行順は[最新計画](../plans/2026-10-07-next-training-and-release.md)を基準にし、今後の計算処理はCPU比較も含めすべてVast.aiで行います。本文中の調査時点の「未測定」と後続結果を区別してください。

調査日: 2026-10-06。対象は、一般的なPCのCPUで動く、ローカル音声合成・声の複製・音声ストリーミングに使える日本語モデル。確認したupstream mainは `41cbc84af539ea78a804ffca5f9c6edc1a22ce44`、調査開始時の作業HEADは `5073bba`。

**結論は、実用的な日本語Pocket TTSを作って公開することに価値がある。** 公式・コミュニティの公開検索で完成した日本語版を確認できず、Pocket TTS利用者と開発者に不足する言語を届けられる。他方式のTTSより速いことや高品質であることは、この公開価値の必要条件ではない。

一方、「日本語・約100M・CPU・声の複製・ストリーミング」という機能の組合せには、すでにMOSS-TTS-Nanoという直接競合がある。Pocket TTSに日本語を追加する価値は明確だが、他方式に対する性能上の優位性はまだ立証されていない。競合比較は、公開を否定するためではなく、品質目標と特徴、学習へ投入する規模を決める材料として使う。

## 判断の要点

| 判断対象 | 評価 | 根拠と条件 |
|---|---|---|
| Pocket TTS利用者への日本語提供 | 価値がある | 公式の対応言語と公開検索で、完成した日本語版を確認できなかった。既存APIを使える日本語モデルは追加の選択肢になる |
| CPUで動く日本語TTSという新規性 | 低い | Kokoro、Supertonic 3、Piper系、VOICEVOX、Style-Bert-VITS2等がすでにある |
| CPUで日本語の声を複製する新規性 | 限定的 | MOSS-TTS-NanoがCPU・日本語・ローカルの声の複製・ストリーミングを公開。MioTTSも小型の日本語モデルを提供 |
| 日本語品質での優位性 | 未証明 | 現在のPhase 1はパイプライン検証用。日本語の読み・自然さ・短文・長文を競合と比較していない |
| OSSへの貢献 | 価値がある | 日本語モデルそのものが言語の不足を補う。前処理、学習レシピ、蒸留、評価を公開すれば再利用価値も広がる |
| 研究上の新規性 | 成果次第 | 日本語への適応だけでなく、読み精度・CPU遅延・蒸留の知見を再現可能に示すと強くなる |
| 広い市場へのインパクト | 条件付き | 機能の有無だけでは弱い。品質、応答開始時間、CPU負荷、導入容易性などの実測差が必要 |
| 継続の方針 | 日本語版公開を目標に継続 | M2aと蒸留を段階的に検証する。競合比較は目標品質と追加投資の規模を決める材料とする |

この評価は公開資料とローカル構造検証からの判断であり、市場需要や採用数の予測ではない。今回、新しい日本語音声の合成、競合の速度測定、ブラインド聴取評価は実施していない。

## 公式モデルとコミュニティの現状

公式READMEの対応言語は英語・フランス語・ドイツ語・ポルトガル語・イタリア語・スペイン語・オランダ語で、日本語は含まれない。公式の公称値は約100M、MacBook Air M4で約6倍の実時間速度、最初の音声チャンクまで約200ms、CPU 2コア。**これらは日本語版の性能保証ではない。** [公式README](https://github.com/kyutai-labs/pocket-tts)

公式HFカードのlanguageメタデータは6言語のままで、READMEの7言語と一致していない。言語対応の判断はメタデータだけに依存せず、READMEとローカル設定も照合した。HFのモデル更新日は2026-10-01。 [公式モデルカード](https://huggingface.co/kyutai/pocket-tts)

公式READMEには現在、チェコ語、ヒンディー語、韓国語、ペルシャ語、インドネシア語、エストニア語、ウェールズ語、ポーランド語、ギリシャ語の9件が掲載されている。8月の記録にあった「チェコ語1件のみ」は現状に合わない。 [コミュニティモデル一覧](https://github.com/kyutai-labs/pocket-tts#models-trained-by-the-community)

日本語の公式公開予定は今回確認した資料では見つからなかった。Issue #118には日本語への要望と、8月25日の学習コード公開案内がある。**公開予定を確認できないことは、内部計画がないことを意味しない。** 「日本語は計画外」「今後も公式は出さない」とは断言できない。 [Issue #118](https://github.com/kyutai-labs/pocket-tts/issues/118)

公開検索の範囲では、日本語Pocket TTS候補は本プロジェクトの `ayousanz/pocket-tts-ja-phase1` と `zwaiwng/maneko`。前者は未完成の検証用24層モデル。後者はPocket TTSとIrodoriを同梱するエンジンで、日本語合成をPocket TTSが担う証拠にはならない。検索は、HFの名前検索156件、`pocket-tts`フィルタ99件、GitHub公式リポジトリのJapanese検索2件を確認したもので、非公開・未タグ付けモデルの不在は証明しない。 [Phase 1カード](https://huggingface.co/ayousanz/pocket-tts-ja-phase1)、[manekoの配布説明](https://huggingface.co/zwaiwng/maneko?library=pocket-tts)、[取得メタデータ](2026-10-06-evidence.json)

## 日本語の競合比較

表のパラメータ数は各公開元の名称・公称値。LM、コーデック、テキストエンコーダの計数範囲が異なるため、同じ数値でも同じRAM使用量とは限らない。「対応」は公開元の記載であり、同条件での日本語品質を保証しない。

| モデル・方式 | 日本語 | CPUとストリーミング | 声の複製 | 公開条件と比較での位置 |
|---|---|---|---|---|
| 本プロジェクトのPocket TTS | Phase 1に加え、10月7日までに220 stepの試運転・144条件の機械評価を完了。正確な全文読み上げは未確認 | 24層のCPU実用速度は未測定。6層版は未完成 | 2参照音声で機械評価済み。実声優の分離・自然さ・話者類似度の人手確認は未完了 | コードMIT、元重みCC BY 4.0。日本語向けの学習・蒸留を作る対象 |
| [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) | 公開対応20言語に含む | 約100M、CPU・ストリーミング。公式ONNX版あり | 参照音声からローカルに対応 | Apache 2.0。最優先の直接比較対象 |
| [Supertonic 3](https://huggingface.co/Supertone/supertonic-3) | 公開対応31言語に含む | 約99M、ONNXによるローカルCPU推論 | 配布モデルはプリセット。独自voice-style作成は別サービス経由 | コードMIT、重みOpenRAIL-M。固定声での軽量日本語合成の比較対象 |
| [Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M) | 日本語プリセットあり | 軽量。文・区間単位の生成を比較する | 標準配布は既存voice tensorを使う。任意参照音声のエンコーダは標準公開なし | 重みApache 2.0。読み上げ用途の重要な基準 |
| [MioTTS 0.1B](https://huggingface.co/Aratako/MioTTS-0.1B) | 日本語・英語 | 小型LM。公開RTFはRTX 5090上の値で、CPU速度の証拠ではない | 対応 | Falcon-LLM License。小型の声の複製モデルとして比較する |
| [Irodori TTS v4.1 Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) | 日本語専用 | 量子化版あり。この調査ではCPU低遅延性能を確認していない | 対応、Voice Designもあり | MIT表記とモデルカード上の追加利用条件あり。日本語品質・表現力の基準 |
| [Qwen3 TTS 0.6B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | 公開対応10言語に含む | ストリーミングを掲げる。公開例はCUDA使用 | 対応 | Apache 2.0。品質の比較対象。公称97msをCPU測定値として扱わない |
| [piper-plus](https://github.com/ayutaz/piper-plus) | 対応 | CPU向け、複数言語バインディングとストリーミング | 通常は学習済み話者のモデルを利用 | コードMIT、重み等は個別確認。既存の自プロジェクトとの役割分担を比較する |
| [VOICEVOX](https://voicevox.hiroshiba.jp/qa/) / [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) | 対応 | CPU合成が可能 | 固定・学習済み話者中心 | VOICEVOXはソフト・話者規約、SBV2コードはAGPL/LGPL等。日本語編集・既存話者用途で強い代替 |

### MOSS TTS Nanoが差別化の基準になる

MOSSは4月10日にNanoを公開し、16日にfinetuningコード、17日に独立したONNX CPU版を公開した。公開元はONNX版について、元版比で約2倍の処理効率と、M4の1コアでの動作を報告している。参照音声を直接入力する声の複製、ブラウザでの実行経路、Androidの例もある。**これらの開発者報告をPocket TTSのM4公称値と直接比較して勝敗を決めることはできない。** [MOSS README](https://github.com/OpenMOSS/MOSS-TTS-Nano)、[学習ガイド](https://github.com/OpenMOSS/MOSS-TTS-Nano/blob/main/finetuning/README.md)

したがって、CPU・ローカル・声の複製・学習可能性・ブラウザ対応を並べても、Pocket TTS日本語版だけの優位性にはならない。MOSSの日本語品質が十分なら、MOSSの改善や統合も比較すべき開発先になる。

### SupertonicとMioTTSに対する差別化候補

Supertonic 3の公開パッケージは固定voice-styleを含み、参照音声から独自styleを作る公開案内は有償Voice Builderに誘導している。Pocket TTSで参照音声の処理も完全ローカルにできれば、この標準フローとの差を示せる。ただしMOSSにもローカル処理がある。 [Supertonicモデルカード](https://huggingface.co/Supertone/supertonic-3)

MioTTS 0.1Bの公開J-HARD評価では、Shortの平均CERが50.55%、Repetitionが22.29%、Rhymeが1.022%。通常文と短い相づち・反復で性能が大きく異なる。公開RTF 0.04〜0.05はRTX 5090＋vLLMでの測定。この数値からCPU性能を推定してはいけない。短文の安定性はPocket TTS日本語版で狙う価値のある課題だが、現状の優位性ではない。 [MioTTSモデルカード](https://huggingface.co/Aratako/MioTTS-0.1B)

これらの主要競合は2〜5月に公開されており、8月以降に突然登場したものではない。以前の「Pocket TTS系の日本語モデルが空白」という調査は、別方式を含む用途上の競争力とは区別する必要がある。

## 残っている価値とインパクト

### Pocket TTSの既存利用者に日本語を届ける

日本語を同じPython API・CLI・モデル状態管理で扱えることは、すでにPocket TTSを使っているアプリに価値がある。別エンジンの導入・声状態の管理・ストリーミング方式の統合を減らせる可能性がある。ただし単一の日本語モデルを追加しても、日英混在文や多言語の話者一貫性が自動的に得られるわけではない。

これは「日本語TTS全体で初」より、**既存エコシステムに不足する言語を補う貢献**として説明する方が根拠に合う。公式の学習案内も、新言語などのコミュニティ成果物の掲載を歓迎している。掲載や採用は保証されない。 [公式学習ガイド](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md)

### 日本語の対話に使える小型モデル

採用理由になり得るのは、「はい」「えっ」「了解です」といった短文で読み落ちない、日付・金額・単位を誤読しない、長い回答を途中から再生できる、普通のWindows PCで処理が安定する、といった使用時の差である。

想定用途はローカル音声アシスタント、GPUをほかの処理に使うゲーム/NPC、デスクトップアプリ、オフライン端末。いずれも**用途仮説**で、今回ユーザー調査や導入意思の確認はしていない。CPU対応からESP32級マイコンでの実行可能性を推定しない。

### 再現可能な日本語学習と評価

実用重みに加え、正規化・漢字の読み・音声と転写の整合性・話者分割・蒸留前後の劣化を記録すれば、別の日本語音声モデルにも使える知見になる。アーキテクチャを言語に適応しただけでは研究上の新規性は限定的だが、読み精度とCPU遅延の両立や蒸留条件を再現可能に示せれば貢献が強くなる。

同じ日本語テスト、同じCPU、固定されたモデルrevisionで比較できる評価セット自体にも価値がある。J-HARDには短文・反復・韻・継続の課題があり、Joyo Kanji Yomi Benchmarkは漢字の読みをKana-CERで評価する。 [J-HARD](https://github.com/Parakeet-Inc/J-HARD-TTS-Eval)、[Joyo Kanji Yomi Benchmark](https://github.com/sbintuitions/Joyo-Kanji-Yomi-Benchmark)

### ライセンスと配布の説明を揃える

Pocket TTSはコードMIT、元重みCC BY 4.0。CC BYは商用の共有・改変を認め、帰属・ライセンスへのリンク・変更表示を求める。ただし声やデータのほかの権利を一括して解決するものではない。 [Pocket TTSカード](https://huggingface.co/kyutai/pocket-tts)、[CC BY 4.0の説明](https://creativecommons.org/licenses/by/4.0/)

Supertonicの重みはOpenRAIL-Mで、用途制限を下流の利用者にも適用する条件がある。これとの配布運用上の差はあり得るが、Apache 2.0のMOSS・KokoroよりPocket TTSが一律に有利とはいえない。 [Supertonicライセンス](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE)

本プロジェクトの既存記録ではMoeSpeech音声は再配布不可、GOLの権利関係は確認済みとされる。本調査でその個別許諾を再審査したわけではない。公開成果物では、学習データ・モデル重み・同梱参照音声を分けて説明し、参照音声は自前または配布許諾のあるものにする。既存の許諾判断を、一般的な「どんな派生物も無条件に公開可」という主張へ拡張しない。 [既存データ記録](../datasets.md)、[GOL公開カード](https://huggingface.co/datasets/midralab/gol-dataset)

## 現在の計画をどう評価するか

### 24層で日本語が話せることとCPU実用版の完成は別

Phase 1の公開カードは、85.5時間・27話者・15k stepの検証用24層モデルで、完成品ではないと明記している。GOL取り込みとM2aはリポジトリ上で実装・設定があるが、実行完了の記録はない。 [Phase 1カード](https://huggingface.co/ayousanz/pocket-tts-ja-phase1)、[進捗記録](../index.md)

現行アーキテクチャに日本語8000語彙を設定し、重みをダウンロードせずにPyTorchのmetaデバイス上で構築して数えた結果:

| 構成 | FlowLM | Mimi | 合計 | FP32パラメータだけの容量 |
|---|---:|---:|---:|---:|
| 24層teacher | 320,076,865 | 20,054,337 | 340,131,202 | 1,297.50 MiB |
| 同構造を6層にしたstudent案 | 93,510,721 | 20,054,337 | 113,565,058 | 433.22 MiB |

この実測は**構造のパラメータ数**で、保存済みPhase 1重みの検査や、合成品質・実行時RAM・速度の測定ではない。KVキャッシュ、演算用バッファ、PyTorchのメモリは表に含まない。6層日本語モデルの完成も意味しない。 [数値の保存先](2026-10-06-evidence.json)、[構造設定](../../../pocket_tts/config/english_2026-04_24l.yaml)

公開された韓国語モデルも、英語teacherから適応した24層・約336Mのモデルで、約100M studentとは区別している。韓国語への適応事例は実現可能性の参考になるが、日本語品質や6層化の成功の証明にはならない。 [韓国語モデルカード](https://huggingface.co/seastar105/pocket-tts-korean-300m)

### 最新mainは学習費用とレシピの再評価を必要とする

最新の公式学習ガイドは、train用Mimi潜在の事前計算を使い、英語2000h・有効batch 64を1×H100で約6.3 step/s、200k stepまで約8.8hと報告している。潜在の事前計算は同条件で約50分、1000h当たり約6GB。旧計画の学習時間を、そのまま現在の所要時間と扱うべきではない。 [公式学習ガイド](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md)

日本語8000語彙、長さ分布、ASR転写、アライメント、GPUとbatchの違いがあるため、この英語速度も日本語へ直接換算できない。過去のPhase 1の$4.59は履歴上の実測、本番の$130〜265は8月の見積りであり、現在の総額や品質達成費用の保証ではない。 [既存コスト記録](../training-strategy.md)

再見積りは「取得・整形・アライメント・Mimi事前計算・teacher学習・student蒸留・評価・保存」の実時間を分ける。実験初期の1000 step程度で日本語のstep/sとメモリを測り、その時点のGPU/ディスク料金に掛ける。今回、レンタルGPUの予約や課金は行っていない。

9月系のdriftingレシピもあるが、日本語の初回比較でLSDから変更すると、データ量と学習法を同時に変えることになる。まず現在のLSD計画を基準とし、必要なら別実験で比較する。 [公式学習ガイド](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md)

### 調査中に確認した設定不整合

最新mainの英語設定は `tokenizer: tokenizers` を使う。日本語の3つの学習設定はバイナリSentencePieceのパスだけを上書きしていたため、JSON tokenizerとして読み込み、`stream did not contain valid UTF-8` で失敗した。これは前回のリベース検証が見落とした互換性問題。

3設定に `flow_lm.lookup_table.tokenizer: sentencepiece` を明示し、実際の設定から同梱トークナイザを構築して日本語をencodeする回帰テストを追加した。修正前の3失敗を確認し、修正後に設定テスト全体を再実行した。これは起動経路の修正であり、日本語学習を新たに実行したわけではない。 [学習設定](../../../training/configs/finetune_language_ja_m2a.yaml)、[回帰テスト](../../../training/tests/test_config_invariants.py)

## 続ける場合の優先順位と判定基準

以下は実験の提案。数値は既存性能から導いた達成保証ではなく、実用性の判断を具体化するための暫定目標。

### 先に競合が満たしている条件を測る

最低限、MOSS-TTS-Nano ONNX、Supertonic 3、KokoroをVast.ai上の同じx86 CPU環境で測る。固定声のSupertonic/Kokoroと、任意参照音声を使うPocket/MOSSの話者類似度は別枠で評価する。MioTTS 0.1Bを追加し、日本語品質の基準としてIrodoriまたはQwenをVast.aiのGPU上で評価する場合は、CPU群と速度順位を混ぜない。Windowsネイティブ動作は別の未検証項目として残し、Linuxでの速度をWindowsの実測として発表しない。

最初は通常文・短文・数字・難読語の小さな固定セットで比較する。候補と設定を絞った後、J-HARDの160文×5回とJoyoの正式プロトコルへ拡張する。Joyoの小さな層化サンプルを開発用に使う場合は、正式な全件結果と呼ばない。テスト文章を学習や設定調整に使い回さず、開発用と最終評価用を分ける。 [J-HARD](https://github.com/Parakeet-Inc/J-HARD-TTS-Eval)、[Joyo](https://github.com/sbintuitions/Joyo-Kanji-Yomi-Benchmark)

### M2aでteacherを選び、小さな蒸留実験を挟む

10月7日のユーザー指定により、M2aはGOL単独、約1,000hを目標とし上限40k stepで検証する。ただし、validation lossだけで学習量を決めず、複数チェックポイントの日本語読み・短文失敗・未見話者での自然さを併せて見る。データ量・話者数・コーパス構成も変わる実験なので、lossの折返しだけで過学習の原因を一意に決めない。

良いteacherを選び、6層への小規模蒸留を先に試す。teacherでは良いがstudentで読みや声が崩れるなら、100M級CPU版という目標に到達する追加費用が必要になる。250k step・数千時間への拡大は、この結果を見てから判断する。teacherとstudentの語彙・前処理・学習済み重みの対応を揃える必要がある。 [M2a設計](../specs/2026-08-30-gol-and-phase2-design.md)、[公式蒸留手順](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md)

### 性能を表す単位と比較条件を揃える

RTFは `処理秒数 / 生成音声秒数` と定義し、小さい方を速いとする。6倍実時間はRTF約0.167。一部のREADMEがRTF欄に記す「2x」「6x」は速度倍率なので、比較時に逆数へ統一する。

音声の最初のチャンクが返る時間と、最初の聞こえる発話までの時間を両方測る。無音チャンクだけ早く返っても対話体験が速いとは限らない。参照音声の初回エンコード、キャッシュ済み声状態、モデル冷起動、warm状態を分け、モデルrevision、量子化、スレッド数、入力長、音声長、seed、温度、EOS設定を保存する。

| 評価軸 | 暫定目標と判断方法 |
|---|---|
| 実時間動作 | Vast.aiのCPU型番を記録した同一環境のwarm状態で、日本語短文〜30秒音声のRTF中央値≤0.5、p95<1.0を暫定目標とする。1/2/4スレッドを別集計。一般的なWindows PCでの達成は未検証 |
| 応答開始 | 完成した入力テキストを渡してから、聞こえる音声までのwarm p95≤500ms。LLMの生成待ち時間は別測定 |
| 差別化する速度 | 同等の読み精度・自然さを満たしたMOSSに対し、RTFまたは応答開始を20%以上改善する等、事前に選んだ軸で意味のある差を示す |
| 日本語の読み | 固定された同一ASR・正規化で通常文のSentence Kana-CER≤5%を暫定目標とし、競合との差と人手の読み確認を併記 |
| 短文と反復の安定性 | J-HARDの平均・最悪値を掲載。追加の対話用セットで、欠落・不要反復・途中終了などの重大失敗を1%未満に抑える目標 |
| 自然さと声の類似度 | 未見話者・同一参照音声でのブラインド比較を行う。MOSSとの差を信頼区間付きで判断。固定声群の類似度順位は作らない |
| 量子化とRAM | FP32とINT8の品質・速度・ピークRSSを比較。パラメータ容量を実行時RAMとして発表しない |
| 導入 | Vast.aiの新規環境で日本語生成・参照音声処理・保存・連続再生まで通す。Windows対応は遠隔Windows検証環境を確認できるまで未検証。ブラウザ版は別経路として検証する |

短文ではASRの幻覚が大きく、通常の漢字表記CERだけでは「橋/箸」やかな/漢字表記の差を読みの誤りと混同する。J-HARDはWhisperとReazonSpeechを併用する方式を示している。独自に片方だけへ置き換えた結果を公式表と同じ評価と扱わない。UTMOSの単一値で日本語キャラクター声の優劣を決めず、人手評価を残す。 [J-HARD評価方法](https://github.com/Parakeet-Inc/J-HARD-TTS-Eval)、[既存のMOSとドメイン差の記録](../datasets.md)

### 継続と方針転換の条件

**日本語版公開を進める条件:** 日本語が明瞭に読めるモデルを、再現可能な設定・重み・トークナイザ・利用条件・評価結果とともに公開できること。完成した日本語版が不足している現状では、この言語追加だけでも貢献になる。競合を上回ることを公開の必須条件にしない。6層版がCPUの実時間条件を満たせば、当初目標の軽量実用版としての価値がさらに強くなる。

**性能上の優位性を主張する条件:** MOSS等に対し、速度、短文安定性、声の類似度、導入運用のいずれかで再現可能な改善を示す。この条件は、日本語版を公開する条件とは別である。

**用途を絞る条件:** 24層では品質が良いが6層化で崩れる場合は、teacherを品質重視版として扱う。キャラクター声では良いが一般朗読が弱い場合は、一般向け日本語TTSの主張を控え、用途と弱点をモデルカードへ明記する。

**追加投資を見直す条件:** 他方式の代替品が十分であっても、日本語Pocket TTSの公開価値は残る。ただし、良いteacherや実用studentが得られないまま全量取得・長時間学習を繰り返すことは別の判断となる。段階的な改善が止まった場合は、データ品質・日本語表現・蒸留条件を見直し、公開可能な成果と制約を先に残す。性能競争への追加投資と、言語追加の公開目標を混同しない。

## 周辺候補と調査の限界

NeuTTS-2Mは、日本語を含む7言語、CPU、声の複製を公式に案内している。企業向けの問い合わせ導線であり、公開NeuTTS-Nanoの言語やライセンスと同一視しない。現時点の導入条件を揃えられれば、商用代替として追加比較する。 [NeuTTS-2M](https://www.neuphonic.com/models/neutts-2m)

Sarashina2.2-TTSは日本語中心の品質比較候補だが、公開モデルはNonCommercial Licenseで商用利用は問い合わせを案内している。日本語最高品質という主張を検証するときには対象になり、軽量商用CPU版の比較とは条件が違う。 [Sarashina](https://github.com/sbintuitions/sarashina2.2-tts)

CuteTTSの公開languageメタデータは英語・中国語・フランス語・ドイツ語・スペイン語で、日本語の完成版は今回確認できなかった。別の日本語適応候補ではあるが、公開対応の直接競合と同じ扱いにはしない。 [CuteTTS](https://huggingface.co/OPPOer/CuteTTS)、[蒸留版](https://huggingface.co/OPPOer/CuteTTS-distill)

本調査では公開されていないモデル、将来の公式ロードマップ、各社の学習データ全体、顧客の支払意思、各デバイスでの品質を確定していない。スター・ダウンロード件数を日本語ユーザー数や採用の証拠に使っていない。公開元が異なる英語/日本語CER、MOS、GPU/CPUの速度を並べて順位を作っていない。

## 証拠と再確認の入口

公開モデル11件のrevision・公開日時・最終更新・言語・gating・ライセンスタグ、公式mainのSHA、検索件数と日本語候補、構造のパラメータ数、修正前の設定失敗は [JSON](2026-10-06-evidence.json) に保存した。モデル名検索とlibraryフィルタの集合は異なり、156件と99件を足してモデル総数とはしない。

取得したAPIは `https://huggingface.co/api/models/<id>`、`/api/models?search=pocket-tts&limit=1000&full=true`、`/api/models?filter=pocket-tts&limit=1000&full=true`、GitHubの `repos/kyutai-labs/pocket-tts/commits/main` と `search/issues?q=repo:kyutai-labs/pocket-tts Japanese`。READMEのコミュニティ一覧とIssue #118/#50のコメントも照合した。本文中のリンクは各主張を支える公開元またはローカル記録。

ローカル検証は `load_args` → `load_model_config` → `build_tokenizer` の実際の構築経路、およびmeta上での `FlowLMModel.from_pydantic_config` と `build_mimi` のパラメータ計数。計数時は英語24層設定の語彙を8000、tokenizerを同梱SentencePieceへ変更し、6層案は `num_layers=6` にした。推論ベンチマークや学習結果として扱わない。
