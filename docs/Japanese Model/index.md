# 日本語モデル

Pocket TTS の日本語モデルを作るための調査記録と方針をまとめたセクションです。

調査日: 2026-08-27 / 対象コミット: `65534c9` / 作業ブランチ: `japanese-model-training`

## 現状サマリ

| 論点 | 結論 |
|---|---|
| 日本語モデルは存在するか | **存在しない**。公式・コミュニティともに0件（[調査結果](community-survey.md)） |
| 公式の対応予定はあるか | ない。Issue #118 の計画は es/fr/de/pt/it のみで、日本語は対象外 |
| 学習コードは使えるか | 使える。2026-08-25 に公開済み。「任意の言語で学習可能」と公式アナウンス |
| 手本になる先行事例 | **チェコ語モデル 1件のみ**（`vvolhejn/pocket-tts-czech`） |
| 事前学習か継続学習か | **継続学習（finetune）を採用**（[根拠](training-strategy.md)） |
| 使用データセット | `ayousanz/moe-speech-plus`（623h）+ `midralab/gol-dataset-2k-ljspeech`（約2,020h）（[詳細](datasets.md)） |
| 想定コスト | 検証 **約$10**、本番 finetune + 蒸留 **$130〜265** |

## ページ構成

- **[コミュニティ調査](community-survey.md)** — 公式リポジトリ、フォーク、Hugging Face、日本語コミュニティの網羅調査。なぜ空白地帯なのか。
- **[データセット](datasets.md)** — 使用する2つのデータセットの実測値と、それぞれの注意点。
- **[学習戦略とコスト](training-strategy.md)** — 事前学習 vs 継続学習の判断根拠、コスト試算、既知のリスク、段階的な進め方。

## 日本語固有の作業（全体像）

`training/README.md` は非英語言語について「アライナとトークナイザの両方を差し替える必要がある」と明記しています。日本語の場合、それに加えて学習コード自体への小改修が必要です。

1. **トークナイザ** — `training/scripts/train_tokenizer.py` で sentencepiece を学習。CJK 向けに `character_coverage` を下げる。
2. **強制アライメント** — `training/scripts/align_data.py` のアライナを日本語 wav2vec2 に差し替え。日本語は分かち書きがないため単語分割が別途必要。
3. **DataLoader の単語連結** — `training/dataloader.py:150` の `" ".join(...)` を日本語向けに修正（[詳細](training-strategy.md)）。

いずれも [学習戦略とコスト](training-strategy.md) のリスク節で詳述しています。
