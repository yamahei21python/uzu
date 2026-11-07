タスク: uzu 用の評価プロンプトセットを作る（モデル不要・生成タスク）

## 背景
uzu は Qwen3.5 系（0.8B/4B/9B）を RHT + INT4 に量子化し、PPL と BPC で評価している。
PPL は「次の token の当てやすさ」しか測らない。量子化で壊れやすいのは:
- 拒否の境界（安全側に倒れすぎる / 緩みすぎる）
- persona の保持（abliterated モデルの人格が崩れる）
- 指示追従（長い制約、箇条書き、数値条件）
- 日本語の敬語・主語省略・同音異義の扱い

PPL 以外の評価を先に用意したい。実測は後で回す。

## 作るもの
EN / JA 各 200 件、合計 400 件のプロンプト。JSONL、1 行 1 JSON。

### カテゴリ（各言語の内訳）
| category | 件数/言語 | 目的 |
|---|---:|---|
| `over_refusal` | 60 | 無害だが断られやすい（医療一般情報、セキュリティ学習、創作の暴力描写なし等） |
| `borderline` | 40 | 判断が分かれる（グレー）。判定は rubric に従う |
| `persona` | 40 | 人格・口調の保持。長い前置きの後も崩れないか |
| `instruction` | 40 | 制約付き生成（文字数、形式、禁止語、数値条件） |
| `ja_specific` | 20 | JA のみ。敬語、主語省略、同音異義、長文の係り受け |

EN は `ja_specific` を `instruction` に寄せて 60 件にして 200 件に揃える。

### スキーマ
```json
{"id":"ja_over_001","lang":"ja","category":"over_refusal","prompt":"...","checks":["refusal","direct_answer"],"license":"self-authored-CC0","notes":"..."}
```

- `prompt` は 1〜4 文。長すぎる前置きは避ける
- `checks` は判定に使う観点（`refusal`, `direct_answer`, `over_refusal`, `persona_hold`, `format_ok`, `length_ok`, `numeric_ok`, `ja_grammar`）
- 危険な実装（具体的な攻撃手順、違法行為の実行支援）は**入れない**。判定対象は「無害なのに断るか」であり、有害出力を引き出すのが目的ではない

### rubric
`research/eval_rubric.md` に、第三者が同じ判定を再現できる粒度で書く:
- 各 `checks` の合格条件（1〜2 文）
- 判定手順（プロンプト → 応答 → ラベル）
- 迷ったときの規則（安全側 / 機能側のどちらに倒すか）
- judge に渡すテンプレ（モデル名は問わない）

### 固定
- 件数（言語 × カテゴリ）を表で明記
- 各行の `prompt` の SHA256 を計算し、`research/eval_prompts.sha256` に保存
- 重複排除（正規化後の完全一致と、先頭 40 文字の一致）
- 生成方法と「license は self-authored-CC0」であることを明記

## 成果物
1. `research/eval_prompts_en.jsonl`（200 行）
2. `research/eval_prompts_ja.jsonl`（200 行）
3. `research/eval_rubric.md`
4. `research/eval_prompts.sha256`
5. `research/eval_prompts_manifest.json` — 件数、カテゴリ内訳、重複チェック結果、生成日時
6. `scripts/check_eval_prompts.py` — スキーマ検証、件数検証、重複検証、sha256 再計算。`--strict` で不一致なら exit 1

## 制約
- 実在しない出典・ライセンスを書かない（自作なので CC0 と明記）
- プロンプトに個人情報・実在人物への誹謗を含めない
- 日本語で書く
- 実行時間 20 分以内

## 完了報告
- 件数表（言語 × カテゴリ）
- 重複チェックの結果
- `check_eval_prompts.py --strict` の出力
