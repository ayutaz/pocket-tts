# コミュニティ調査

調査日: 2026-08-27

**結論: pocket-tts（CALM/Mimi 系）の日本語モデルは、公式にもコミュニティにも存在せず、学習に着手した形跡もありません。** 記事レベルの検討はありますが実装は0件で、完全な空白地帯です。

（2026-08-29 追記: 本プロジェクトが検証用の1件を公開しました。実用モデルではありません — 本ページ末尾の「この空白に1件入りました」を参照。）

## 公式リポジトリの状況

| 項目 | 状況 |
|---|---|
| 対応言語 | en / fr / de / es / it / pt のみ（`pocket_tts/config/` にも日本語なし） |
| 公式ロードマップ | [Issue #118](https://github.com/kyutai-labs/pocket-tts/issues/118)（2026-02-10, closed, 65コメント）で Spanish/French/German/Portuguese/Italian のみを announce。**日本語は計画外** |
| 日本語の要望 | @agavrel が「韓国語・日本語を手伝えないか」と質問（👍7）→ 公式回答なし。他に中国語・ヒンディー語・トルコ語・ビルマ語の要望あり |
| 現在の open issue | 日本語の要望 issue は**存在しない**（トルコ語 #215、ビルマ語 #227 のみ）。issue/PR 全文＋コメント検索で "Japanese" のヒットは #118 と #50 の2件のみ |
| 転換点 | 2026-08-25 @manukyutai:「Training code released! You can now train in any language you like :)」→ **言語追加は事実上コミュニティに委譲された** |
| コミュニティモデル一覧 | **チェコ語（vvolhejn）の1件のみ** |

### 学習コード公開までの経緯

Issue #30（Training/fine-tuning code）での公式の立場は、当初は否定的でした。

- 2026-01-17 @gabrieldemarmiesse:「現時点でタイムラインはない。学習コードの公開は非常に手間がかかる。誰かが先に公開したいなら質問してくれれば手助けする。秘伝のタレはなく、論文とブログ記事に全て書いてある」
- 2026-01-21 @vvolhejn:「少なくとも短期的には公開予定なし。想像以上に工数がかかる上、恩恵を受けるのはユーザーのごく一部」
- 2026-01-26 @vvolhejn: 学習に必要な入力射影の重みのみを個別提供
- **2026-08-25: 方針転換して [PR #244](https://github.com/kyutai-labs/pocket-tts/pull/244) で学習コード一式を公開**

### PR #254 — 新言語への finetune レシピ（重要）

[PR #254](https://github.com/kyutai-labs/pocket-tts/pull/254)（2026-08-26 に `8c98c9b` としてマージ済み）が、日本語モデルを作る上で最も重要な変更です。

公開重みから新言語へ finetune するレシピで、チェコ語 976h（ParCzech）での実測は以下の通り。1127文で Whisper を使い WER を評価しています。

| step | from scratch | ft lr 2e-4 | ft lr 2e-5 |
|---|---|---|---|
| 2k | 326% | **29.5%** | — |
| 4k | 217% | **23.7%** | 101% |
| 10k | 45.7% | **12.0%** | 18.6% |
| 15k | 17.1% | 11.3% | 14.7% |
| 25k | 16.2% | 10.7% | 11.8% |

マージされた内容は10ファイルに及び、**PR の説明とは異なります**（説明にある「形状不一致テンソルの自動破棄」は採用されず、明示フラグ方式になりました）。

- `training/args.py` — `reset_text_embedding: bool = False` を追加（TrainArgs、行68）
- `training/modules/builders.py` — このフラグが真のとき `conditioner.embed.` で始まる重みだけを破棄して新規初期化。形状ベースの自動判定ではない
- `training/configs/finetune_language.yaml` — 新言語 finetune のレシピ（新規）
- `training/configs/finetune.yaml` — 同一言語での finetune レシピ（新規）
- **設定ファイルのリネーム** — `lsd_scratch.yaml` → `scratch.yaml`、`lsd_depth_distill.yaml` → `depth_distill.yaml`
- `training/scripts/train_tokenizer.py` — `--vocab-size` の既定を 3999 → **4000** に変更（`n_bins: 4000` を上書きせずに済むように）

!!! warning "リネームに注意"
    既存の手順書・シェル履歴・CI が `lsd_scratch.yaml` を参照していると、存在しないパスを指して静かに壊れます。

!!! note "lr を下げてはいけない"
    PR の説明によれば、lr 2e-5 は全区間で遅く、プラトーでも改善しません。「text embedding がランダムから始まるので、backbone がそれに合わせて動く必要がある」ためで、**scratch と同じ 2e-4 を維持するのが正解**です。

## フォークと GitHub 全体

- **`seastar105/pocket-tts-korean-training`**（2026-08-26 作成）— 韓国語勢は動き出しています。ただし独自コミットはまだ0（`ahead_by: 0`）で、リポジトリ名だけの状態。
- 日本人と思われるフォーク（`ayutaz/`, `kenekoba/`, `himomohi/`）はいずれも `ahead_by: 0` で無改変。
- GitHub のコード検索・リポジトリ検索で "pocket-tts japanese" 系のヒットは **0件**。

## Hugging Face

`pocket-tts` を含むモデル100件超を全件確認した結果、**日本語モデルは0件**でした。

内訳:

- **ランタイム変換**（大多数）— ONNX / CoreML / MLX / GGUF / LiteRT / ExecuTorch など、公式重みの形式変換
- **言語派生** — es（`ipsilondev`）、pt-br・fr（`marcosremar2`）、gu（`Arjun4707` の tokenizer/base）のみ

!!! warning "`zwaiwng/maneko` は日本語 pocket-tts ではない"
    `ja` タグが付いているため検索に引っかかりますが、これは Rust/Candle 製の TTS エンジン `maneko` が、pocket-tts（非日本語）と**別アーキテクチャの Irodori-TTS（日本語）を同梱している**だけです。日本語 pocket-tts モデルではありません。

なお `kyutai/pocket-tts` の HF Discussions は無効化されており、議論の場は GitHub のみです。

## 日本語コミュニティの動き

### 記事

- **[話題のPocket TTSを日本語対応にするにはどうしたらできるか。](https://note.com/pocketstudio/n/n1fbb74e3dd1a)**（2026-01-15, ぬるぽん）— 日本語対応にはトークナイザ・コーデック・生成モデルの再学習が必要という分析。結論は「個人には Style-Bert-VITS2 / VOICEVOX が現実的、新規学習は企業・研究機関向け」。**実装は伴っていません。**
- Zenn / Qiita は「Lambda に載せた」「試した」「日本語TTS比較」の記事のみで、学習系の記事はありません。

### @Aratako（日本の TTS 開発者）

最も近い位置にいますが、方向性が異なります。

| リポジトリ | 内容 | 状況 |
|---|---|---|
| [`CALM-DACVAE`](https://github.com/Aratako/CALM-DACVAE) ★19 | pocket-tts と同じ CALM 論文の再現実装。DACVAE を audio VAE として使用 | **「まだ聞き取れる音声を生成できていない」WIP**。2026-02 から更新停止 |
| [`Irodori-TTS`](https://github.com/Aratako/Irodori-TTS) ★1219 | 日本語 TTS として成功済み。RF-DiT + DACVAE 48kHz / 766M | 活発（v4.1 まで） |
| `MioTTS` ★202 | 軽量 LLM ベース日本語 TTS | — |

Irodori-TTS は日本語 TTS としては完成度が高いものの、**pocket-tts（Mimi 24kHz / 100M / CPU 動作）とは別物**です。

つまり **「日本語 × CALM/pocket-tts 系（CPU で動く 100M）」は誰も埋めていない**、というのが本調査の結論です。

!!! note "2026-08-29 追記: この空白に1件入りました"
    上記は調査日（2026-08-27）時点の網羅結果で、その意味では今も正確です。ただし本
    プロジェクトが [`ayousanz/pocket-tts-ja-phase1`](https://huggingface.co/ayousanz/pocket-tts-ja-phase1) を公開しました。

    **これは実用モデルではありません。** 85.5時間・15k step の検証ランの成果物で、
    24層のまま（蒸留なし）、話者27人のアニメ／ギャルゲ演技に限定され、valid loss は
    7,500 step 以降悪化しています。目的は「パイプラインが正しいか」を $4.59 で確かめる
    ことであって、品質ではありません。gated（manual）にしてあるのもそのためです。

    **空白が本当に埋まるのはフェーズ2以降**（2,640時間・250k step・蒸留）です。

## 手本となる先行事例: チェコ語モデル

唯一のコミュニティモデルであり、成果物の型として参照すべきものです。

**構成** — [`vvolhejn/pocket-tts-czech`](https://huggingface.co/vvolhejn/pocket-tts-czech)

```
model.safetensors     # 6層 student（24層 teacher から蒸留）
tokenizer.model       # sentencepiece
czech.yaml            # 設定ファイル（weights_path / tokenizer_path は commit pin 付き hf:// URL）
voices/*.wav          # ParCzech の held-out 話者から切り出したサンプル音声 6本
README.md             # ライセンス cs、pocket-tts タグ
```

**利用方法** — `--config` に yaml の URL を渡すだけです。

```bash
uvx pocket-tts generate \
  --config hf://vvolhejn/pocket-tts-czech/czech.yaml@7c1fbd0acba765617749dd17f3dbddc2be791cc7 \
  --voice your_voice.wav \
  --text "Dobrý den, toto je český model."
```

**yaml の要点**

```yaml
weights_path: hf://vvolhejn/pocket-tts-czech/model.safetensors@b7eead8...
default_temperature: 0.3

flow_lm:
  transformer:
    num_layers: 6          # 蒸留後の student
  lookup_table:
    n_bins: 3999           # sentencepiece の語彙数と厳密に一致させる必要がある
    tokenizer_path: hf://vvolhejn/pocket-tts-czech/tokenizer.model@b7eead8...
```

!!! danger "組み込みボイスは使えない"
    `alba`、`cosette` などの名前付きボイスは、**英語の公開重みで事前計算された conditioning state** です。コミュニティモデルでは動作しません。ユーザーは音声ファイルを `--voice` に渡すか、そのモデルで `pocket-tts export-voice` した state を使う必要があります。

    日本語モデルでも同じことが起きます。ただし `voices/` を同梱するのは**要件ではなく、チェコ語モデルがそうしているという先例**です。上の利用例の通り CLI は `--voice your_voice.wav` を受け取るので、同梱ゼロでもモデルは使えます。

    同梱する場合、**その音声を MoeSpeech から作ることはできません。** MoeSpeech の許諾は著作権法30条の4の情報解析利用に限られ、**学習した重みの公開は問題ない一方、音声ファイルそのものの再配布は不可**です。チェコ語モデルが ParCzech の held-out 話者から6本切り出した手は、そのまま真似できません。同梱するなら自分で録るか、再配布可能なライセンスの音声を使ってください（[詳細](datasets.md)）。

## 参考リンク

- [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) — [Issue #118](https://github.com/kyutai-labs/pocket-tts/issues/118) / [Issue #30](https://github.com/kyutai-labs/pocket-tts/issues/30) / [PR #244](https://github.com/kyutai-labs/pocket-tts/pull/244) / [PR #254](https://github.com/kyutai-labs/pocket-tts/pull/254)
- [vvolhejn/pocket-tts-czech](https://huggingface.co/vvolhejn/pocket-tts-czech)
- [seastar105/pocket-tts-korean-training](https://github.com/seastar105/pocket-tts-korean-training)
- [Aratako/CALM-DACVAE](https://github.com/Aratako/CALM-DACVAE) / [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- [CALM 論文 (arXiv:2509.06926)](https://arxiv.org/abs/2509.06926)
