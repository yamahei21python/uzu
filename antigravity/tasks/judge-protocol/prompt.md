タスク: 生成評価の judge プロトコル設計（モデル非依存・採点の再現性）

## 背景
`scripts/eval_generation.py` は refusal / over_refusal / persona_hold などをヒューリスティックで採点する。
これは「拒否したか」しか見ない。品質（役に立つか、指示に従ったか、人格が保たれているか）は測れない。
400 件のプロンプトセット（`research/eval_prompts_{en,ja}.jsonl`）と rubric（`research/eval_rubric.md`）はある。
次に必要なのは **judge の手順と、その手順が再現するかを確かめる仕組み**。

## 入力（公開のみ。repo を clone しない）
- https://raw.githubusercontent.com/yamahei21python/uzu/main/research/eval_prompts_en.jsonl
- https://raw.githubusercontent.com/yamahei21python/uzu/main/research/eval_prompts_ja.jsonl
- https://raw.githubusercontent.com/yamahei21python/uzu/main/research/eval_rubric.md
- https://raw.githubusercontent.com/yamahei21python/uzu/main/research/eval_prompts_manifest.json
- https://raw.githubusercontent.com/yamahei21python/uzu/main/scripts/eval_generation.py

## やること
### A. judge 手順の設計
- 1 応答を採点する手順を、第三者が同じ結果を出せる粒度で書く
- 入力: prompt / response / rubric の checks
- 出力: `{id, checks: {name: pass|fail|na}, severity, rationale(1-2文)}`
- 迷ったときの規則（安全側・機能側のどちらに倒すか）を明示
- 判定に使う judge プロンプトのテンプレを 1 つ書く（特定モデルに依存しない）

### B. キャリブレーションセット
- 400 件から **40 件**を層化抽出（言語 × カテゴリ）
- 各件に「期待ラベル」を人手で決められる形式で用意する（あなたが判定してよい。根拠を 1 行）
- 形式: `research/judge_calibration.jsonl`（prompt, response は空、expected, rationale）
- 応答が無いので、判定は「どの checks が適用されるか」と「典型的な合格/不合格の例」を書く形でよい

### C. 一致率の測り方
- 同じ応答を 2 回採点したときの一致（自己一致）と、judge を変えたときの一致（相互一致）を測る式
- 一致率の合格ライン（例: Cohen's kappa 0.6 以上）と、外れたときの扱い
- 集計スクリプトの仕様（入力 JSONL 2 つ → 一致率と混同行列）

### D. eval_generation.py との接続
- 既存のヒューリスティック採点と judge 採点を並べたとき、どちらを主指標にするか
- 食い違ったときの優先規則

## 成果物
1. `research/judge_protocol.md`
2. `research/judge_calibration.jsonl`（40 行）
3. `scripts/compare_judges.py`（2 つの採点 JSONL から一致率・混同行列）
4. `research/judge_protocol.json`（機械可読の規則・閾値）

## 制約
- 実在しない研究名・数値を書かない
- 応答を捏造しない。キャリブレーションは「期待ラベル」であることを明記
- 日本語で書く。20 分以内
- 書き込めない場合は内容を作らずエラー報告

## 完了報告
- キャリブレーション 40 件の内訳
- 一致率の合格ラインと根拠
- 主指標にするもの 1 つ
