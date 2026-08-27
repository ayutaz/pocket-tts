# 設計: 日本語のテキストフロントエンドと CER 評価

作成日: 2026-08-27 / 対象ブランチ: `japanese-model-training`

## 何のための文書か

日本語モデルの学習側（トークナイザ、アライナ、ローダ）は揃った。しかし
**モデルが完成しても、推論側と評価側に4つの欠陥がある限り実用品質にならない**。
本文書はその4件をまとめて直す設計を定める。音声データは一切不要で、
すべてユニットテストで検証できる。

## 背景: 4つの欠陥

いずれもコーパス（139.8万発話）での実測に基づく。

### 1. 推論時に正規化が走らない

`training/scripts/ja_text.py` の `normalize()` はコーパス構築と
`align_data.py` から呼ばれるが、**推論経路からは呼ばれない**。同ファイルの
docstring 自身が「推論側は未接続」と記している。

結果として、日本語 IME 既定の全角 `？` `！` `ＡＢＣ` が学習時の分布に
存在しない文字のまま入力され、`<unk>` に落ちる。

### 2. 日本語が文分割されない

`split_into_best_sentences`（`pocket_tts/models/tts_model.py:1027`）は
境界トークンを `tokenizer(".!...?")` から、再分割用のフォールバックを
`tokenizer(",;:")` から得る。日本語の `。` `、` `…` はどちらにも無い。

実測: 87文字・6文の段落は **51トークン、境界トークン0個、フォールバックも0個**。
既定の分割上限を超えても分割できず、1チャンクとして生成され語が脱落する。

### 3. 半数の入力に ASCII ピリオドが付く

`prepare_text_prompt`（同 `:962`）は `text[-1].isalnum()` なら `.` を足す。
かな・漢字は Python では `isalnum() == True` である。

実測: コーパスの **49.4%** が無句読点（かな・漢字）で終わる。一方
`。` で終わる発話は **1.12%** しかない。つまり学習データはそもそも
「文末に句点を持たない」分布であり、`.` を足すのも `。` を足すのも誤り。

### 4. WER が日本語で機能しない

`training/eval/librispeech.py` は `EnglishTextNormalizer` で正規化し
`jiwer.wer` で採点する。日本語は分かち書きしないため、文全体が1語になる。

| | 1文字違い | 無関係な文 |
|---|---|---|
| WER | 1.0 | 1.0 |
| CER | 0.100 | 1.400 |

さらに `EnglishTextNormalizer` は **濁点を除去する**（が U+304C → か U+304B）。
`BasicTextNormalizer()` は U+304C のまま保持する。

**この帰結として、PR #254 のチェコ語 WER 数値を日本語の go/no-go 基準に
流用できない。** 検証ランの判定手段として CER が要る。

### 5. 分割した文が空白で繋ぎ直される

`split_into_best_sentences` は分割したセグメントをチャンクに詰め直す際、
`current_chunk += " " + sentence`（`tts_model.py:1076`）で連結する。

**これは欠陥2の修正が作り出す欠陥である。** 現状は日本語が分割されないため
再結合も起きない。境界に `。` を足した瞬間、分割された文が学習テキストに
存在しない空白で繋ぎ直される。ローダ側で `data.word_separator: ""` として
直したのと同じ問題が、推論側に残っている。

実測: 上記の段落を `max_tokens=32` で詰め直すと、**空白が2個注入される**
（セグメント自体は綺麗にデコードされる。注入源は詰め直しのループのみ）。

## 設計方針

`pocket_tts/utils/config.py` の `Config` に**言語別のテキスト規則を持たせ、
デフォルトを英語の現行動作に一致させる**。

これは発明ではなく既存の前例に倣う。`Config` は既に
`pad_with_spaces_for_short_inputs` と `remove_semicolons` を持ち、
`french_24l.yaml` と `german.yaml` が後者を `true` にしている
（`pocket_tts/utils/config.py:121-122`）。

利点:

- リリース済み config は新フィールドを設定しないため、**既存モデルの出力は不変**
- 他言語にも使え、upstream への PR も可能な形
- `tts_model.py` の呼び出し経路を書き換えないので、リベース時の衝突が最小

## 変更1: `Config` の新フィールド

```python
# pocket_tts/utils/config.py — Config に追加。既定値は今日の英語の挙動そのもの
text_normalizer: str | None = None    # "japanese" なら正規化器を通す
sentence_boundaries: str = ".!...?"   # split_into_best_sentences の文境界
clause_boundaries: str = ",;:"        # 長すぎる文の再分割に使う節境界
terminal_punctuation: str = "."       # 無句読点終わりに補う文字。"" なら補わない
segment_separator: str = " "          # チャンク内で文をつなぐ文字列
```

`StrictModel` は `extra="forbid"` だが、**新フィールドに既定値があるため
既存 YAML はそのまま読める**。

日本語 config（`pocket_tts/config/japanese_24l.yaml`）が設定する値:

```yaml
text_normalizer: japanese
sentence_boundaries: ".!...?。…"
clause_boundaries: ",;:、"
terminal_punctuation: ""
segment_separator: ""
```

### 値の根拠（すべてコーパス実測）

| 値 | 根拠 |
|---|---|
| `。` を境界に追加 | 22.8% の行が含む。NFKC では ASCII に畳まれない |
| `…` を境界に追加 | **15.6% の行末**がこれ。この領域では文末記号として機能している |
| `、` を節境界に追加 | 65.3% の行が含む。`,` と同じ役割 |
| ASCII `.!?` を残す | 正規化が全角 `？！` を ASCII に畳むため、実際に届くのはこちら（行末 `?` 17.6%、`!` 13.1%、全角は**0件**） |
| `terminal_punctuation: ""` | 49.4% が無句読点終わり、`。` 終わりは 1.12%。**何も足さないのが分布に最も近い** |
| `segment_separator: ""` | 日本語は空白で書かない。学習側の `data.word_separator: ""` と対になる設定 |

!!! check "境界が BPE に吸収されていないことを検証済み"
    BPE は `。` を `です。` のような大きなピースに畳み込みうる。そうなっていれば
    境界文字を足しても境界トークンとして現れず、この修正は効かない。

    実際の `_find_boundary_indices` に日本語トークナイザ（`data/ja/tokenizer.model`）
    で通したところ、`。` は **id 3938 の独立ピース**であり、上記87文字の段落は

    | 境界集合 | セグメント数 | 最大トークン数 |
    |---|---|---|
    | 現行 `.!...?` | 1 | 51 |
    | 提案 `.!...?。…` | **6** | **11** |

    となる。修正は実際に効く。

## 変更2: 関数への配線

`prepare_text_prompt` は引数3個、`split_into_best_sentences` は5個。
4個ずつ足すと7個と9個になり読めなくなるため、新規分は1つの
frozen dataclass にまとめ、**キーワード専用・既定値つき**で渡す。

```python
@dataclass(frozen=True)
class TextRules:
    """言語ごとのテキストの正規化・分割・句読点の規則。既定は英語の現行動作。"""
    normalizer: str | None = None
    sentence_boundaries: str = ".!...?"
    clause_boundaries: str = ",;:"
    terminal_punctuation: str = "."
    segment_separator: str = " "   # チャンク内で文をつなぐ文字列


def prepare_text_prompt(
    text, pad_with_spaces_for_short_inputs, remove_semicolons, *, rules=TextRules()
) -> tuple[str, int]: ...

def split_into_best_sentences(
    tokenizer, text_to_generate, max_tokens,
    pad_with_spaces_for_short_inputs, remove_semicolons, *, rules=TextRules()
) -> list[str]: ...
```

既存の呼び出し・テストは**引数を変えずにそのまま通る**。

`TTSModel` は `pad_with_spaces_for_short_inputs` と同じ経路で受け取る
（`__init__` の kwarg → `self` → `from_config` が `Config` から配線、
`tts_model.py:85-135`）。

!!! note "既存2フラグを `TextRules` に移さない理由"
    `pad_with_spaces_for_short_inputs` と `remove_semicolons` は同種の設定だが、
    リリース済み YAML がトップレベルで設定しており、移すと既存 config が
    読めなくなる。一貫性より互換性を採る。

### 既存テストへの影響

`tests/test_generation_regressions.py:15` は `split_into_best_sentences` を
**位置引数の署名ごと** monkeypatch している。呼び出し側が `rules=` を
渡すようになるため、この偽関数に `**_` を足す1行の修正が要る。
これが既存テストへの唯一の変更。

## 変更3: `normalize()` を `pocket_tts/` へ移す

**`pocket_tts/` は `training/` を import できない**（現状その import は0件、
逆向きは `align_data.py` が実際に行っている）。推論パッケージが学習用コードに
依存してはならない。

- `training/scripts/ja_text.py` → `pocket_tts/utils/text_normalization.py` へ移動
- 正規化器はレジストリで引く: `NORMALIZERS = {"japanese": normalize_japanese}`
- 既存の import 元（`align_data.py`、`prepare_ja_text.py`、`training/tests/test_ja_text.py`）を新しいパスに更新

`normalize()` の依存は `re` と `unicodedata` のみ、いずれも標準ライブラリ。
**推論利用者に新しい依存は増えない。**

これにより `ja_text.py` の docstring が主張していた
「コーパス・マニフェスト・推論の3つが同じ正規化を通る」が**初めて事実になる**。

## 変更4: CER

`training/eval/librispeech.py`:

- **WER と CER を常に両方算出する。** 生成には影響せず、チェコ語と比較可能な
  数値も残るため。`EvalResults` に `cer: float` を追加
- `--text-normalizer {english,basic}` を追加。既定は `english`（現行動作）。
  日本語は `basic` — `BasicTextNormalizer()` は既定で `remove_diacritics=False`
  であり濁点を保持する
- **`--text-normalizer` を `eval_dir_name` に含める。** 同関数の docstring が
  「数値を変えるものは名前に入れる」と定めており、入れないと別条件の結果が
  同じディレクトリで衝突する（`librispeech.py:67-89`）

日本語の評価セット構築と ASR モデルの選定は**本設計の範囲外**。音声が要るため、
データ取得フェーズで決める。

## テスト戦略

TDD。実装より先にテストを書き、失敗を確認してから通す。**全件、音声不要。**

| 対象 | 何を固定するか |
|---|---|
| 英語の回帰 | 既定の `TextRules()` で `prepare_text_prompt` と `split_into_best_sentences` の出力が現行と一致すること。**既存モデルを壊していない唯一の証明** |
| 文分割 | `。` を含む段落が複数チャンクに割れること。`tests/test_split_sentences.py` に追記 |
| 句読点補完 | `terminal_punctuation=""` でかな終わりの文に何も足されないこと。既定では `.` が足されること |
| チャンク結合 | `segment_separator=""` で分割した日本語が空白なしに繋がること。既定では空白で繋がること |
| 正規化 | 全角 `？！ＡＢＣ` が畳まれること。`…` と `゛` が保存されること（既存 `test_ja_text.py` から移設） |
| CER | 1文字違いと無関係な文が別スコアになること。`BasicTextNormalizer` が濁点を保持し `EnglishTextNormalizer` が落とすこと |
| 評価名 | `--text-normalizer` を変えると `eval_dir_name` が変わること |

各テストは**実装を意図的に壊して落ちることを確認**する。通るだけのテストは
何も証明しない。

## 範囲外（測って落としたもの）

| 項目 | 理由 |
|---|---|
| 先頭文字の大文字化 | 小文字ラテン始まりの行はコーパスの **0.073%** |
| `.replace("  ", " ")` の単一パスバグ | 日本語固有でなく、`normalize()` が空白を畳むため実害なし |
| 日本語評価セット・ASR 選定 | 音声が要る。データ取得フェーズ |
| フェーズ1以降の学習 | 本設計の後段 |

## 前提

- `pocket_tts/config/japanese_24l.yaml` は本設計で新規作成する。学習用の
  `training/configs/finetune_language_ja.yaml` とは別物（前者は推論、後者は学習）
- リリース済みモデルの重みは触らない
