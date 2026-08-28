# 日本語モデル

Pocket TTS の日本語モデルを作るための調査記録と方針をまとめたセクションです。

調査開始: 2026-08-27 / 起点コミット: `8c98c9b`（PR #254 マージ後） / 作業ブランチ: `japanese-model-training`
最終更新: 2026-08-28（MoeSpeech マニフェスト構築まで完了）

## 現状サマリ

| 論点 | 結論 |
|---|---|
| 日本語モデルは存在するか | **存在しない**。公式・コミュニティともに0件（[調査結果](community-survey.md)） |
| 公式の対応予定はあるか | ない。Issue #118 の計画は es/fr/de/pt/it のみで、日本語は対象外 |
| 学習コードは使えるか | 使える。2026-08-25 に公開済み。「任意の言語で学習可能」と公式アナウンス |
| 手本になる先行事例 | **チェコ語モデル 1件のみ**（`vvolhejn/pocket-tts-czech`） |
| 事前学習か継続学習か | **継続学習（finetune）を採用**（[根拠](training-strategy.md)） |
| 使用データセット | `ayousanz/moe-speech-plus`（約623h）+ `midralab/gol-dataset-2k-ljspeech`（約2,020h）（[詳細](datasets.md)） |
| 想定コスト | 検証 **約$10**、本番 finetune + 蒸留 **$130〜265** |

## どこまで進んだか

| 段 | 状態 |
|---|---|
| 推論側・評価側の欠陥5件 | **対応済み**（[設計](specs/2026-08-27-japanese-text-frontend-design.md) / [計画](plans/2026-08-27-japanese-text-frontend.md)） |
| 日本語トークナイザ（8000語彙） | **学習済み・リポジトリ同梱**（`training/tokenizers/japanese_8000.model`） |
| フェーズ1の検証用設定 | **あり**（`training/configs/finetune_language_ja_phase1.yaml`、15k step・約$10） |
| MoeSpeech マニフェスト構築 | **実装済み・未実行**（[設計](specs/2026-08-28-moespeech-manifest-design.md) / [計画](plans/2026-08-28-moespeech-manifest.md)） |
| フェーズ1の検証ラン | **未実行** |

!!! danger "実装済みと、動いたことがある、は違います"
    `training/scripts/prepare_moespeech.py` は300件のテストを通っていますが、**実データの
    MoeSpeech に一度も触れていません**。テストは音声なしで書ける範囲を全て覆いますが、
    実データ特有の失敗 — 想定外の JSON フィールド、壊れた WAV、キャラごとのディレクトリ
    構造の揺れ — は初回実行で初めて出ます。

    特に**キャラのディレクトリ構造は未確認**です。クリップが `extracted/<name>/` の直下に
    あるのか、その下に入れ子になっているのかで話者ラベルの取り方が変わるため、両方で
    正しく動くよう作ってあり、想定外のラベルを見つけたら**結合を始める前に停止します**。

    GPU を借りる前に読むべき注意は[学習戦略とコスト](training-strategy.md)のフェーズ1にあります。

## ページ構成

- **[コミュニティ調査](community-survey.md)** — 公式リポジトリ、フォーク、Hugging Face、日本語コミュニティの網羅調査。なぜ空白地帯なのか。チェコ語モデルの成果物の型。
- **[データセット](datasets.md)** — 使用する2つのデータセットの実測値と、それぞれの注意点。前処理のディスクとコスト。
- **[学習戦略とコスト](training-strategy.md)** — 事前学習 vs 継続学習の判断根拠、コスト試算、既知のリスク、段階的な進め方。**フェーズ1の実行手順はここ。**
- **[日本語テキストフロントエンドの設計](specs/2026-08-27-japanese-text-frontend-design.md)** — 推論側・評価側にあった4つの欠陥（後にもう1件見つかり計5件）をどう直すかの設計。
- **[日本語テキストフロントエンドの計画](plans/2026-08-27-japanese-text-frontend.md)** — 上記設計をタスクに分割した実装計画。**完了済み。**
- **[MoeSpeech マニフェストの設計](specs/2026-08-28-moespeech-manifest-design.md)** — zip からアライメント済みマニフェストまでの8ステージ。`info.csv` の実測が設計を3つ変えた記録。
- **[MoeSpeech マニフェストの計画](plans/2026-08-28-moespeech-manifest.md)** — 上記設計をタスクに分割した実装計画。**完了済み**（実行時に7件の裁定で上書きされた箇所あり、冒頭に記録）。

## 日本語固有の作業（全体像）

`training/README.md` は非英語言語について「アライナとトークナイザの両方を差し替える必要がある」と明記しています。日本語の場合、それに加えて学習コード自体への小改修が必要です。

1. **トークナイザ** — `training/scripts/train_tokenizer.py` で sentencepiece を学習。CJK だからといって `character_coverage` を下げてはいけない — 実測で 0.9995 は 1,516 字を `<unk>` にするため、**1.0 を維持**する（[詳細](training-strategy.md)）。**学習済み**（語彙8000）。
2. **強制アライメント** — `training/scripts/align_data.py` のアライナを日本語 wav2vec2 に差し替え。日本語は分かち書きがないため単語分割が別途必要（**対応済み**、MeCab/UniDic 形態素 + 文節マージで分割）。
3. **DataLoader の単語連結** — `training/dataloader.py:152` の `word_separator.join(...)`（**対応済み**、`data.word_separator: ""` で解決）。
4. **短尺クリップの連結** — MoeSpeech のキャラ別平均クリップ長は中央値 5.8 秒で、loader がカット点の前後に1秒ずつ要求するとターゲットが 4.8 秒未満しか残らない。同一キャラのクリップを連結して擬似長尺ファイルを作る（**対応済み**、`prepare_moespeech.py` のステージ6）。
5. **アライナへの窓の受け渡し** — `align_data.py` はマニフェストの `start == 0.0` を「この行はファイル全体」と読んでいた。連結ファイルの先頭発話は必ず `start == 0.0` になるため、そのままでは20件に1件が他人の発話ごとアライメントされる（**対応済み**、`read_window()` が常に窓を読む）。**これは英語パイプラインにもあった上流由来のバグ**で、章の先頭発話が章まるごとに対してアライメントされていた。

1 は[学習戦略とコスト](training-strategy.md)のフェーズ0節、2・3 は同ページの「リスク1: 分かち書きとアライメント（対応済み）」、4・5 は [MoeSpeech マニフェストの設計](specs/2026-08-28-moespeech-manifest-design.md)に詳しい。
