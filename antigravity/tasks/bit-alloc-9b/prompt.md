タスク: 9B の層別ビット配分（4/6/8bit）の設計と根拠づけ

## 背景
9B は INT4 group32 で 5.19GB。M3 16GB で動くが、品質の余裕がない。全部 8bit にはできない。
層ごとに感度が違うはずなので、同じメモリ予算で品質を上げる配分を設計したい。
4B/9B の calibration はまだ取れていない。0.8B の実測 trace と理論で、実行可能な配分案まで持っていく。

## 入力
- HF dataset: https://huggingface.co/datasets/yamahei21/uzu-quant-kit
  - `configs/qwen35-9b.json`, `configs/qwen35-4b.json`, `configs/qwen35-0.8b.json`
  - `scripts/rht_int4_convert.py`, `scripts/yaqa_calibration.py`
- 0.8B の実測（main に committed）:
  - `experiments/qwen35-0.8b/clean-enonly-int4-g32/calibration.json`
    - 96 site の `input_trace` / `input_min_eig` / `input_max_eig` / `input_condition`
  - `experiments/qwen35-0.8b/clean-enonly-int4-g32/ppl_en.json`, `ppl_ja.json`
  - `research/bpc_fertility_audit.md`
- 既存の設計メモ: https://raw.githubusercontent.com/yamahei21python/uzu/main/research/9b_mixed_precision_allocation.md
- lalamo: https://github.com/trymirai/lalamo commit e401e17761f2fcd337b0cf20218d67d656e6ecba

## やること
### A. 感度の代理指標
0.8B の `calibration.json` から site ごとの指標を作る:
- `input_trace`（大きいほど感度が高い、という仮説の検証）
- `input_condition`（悪条件 = 量子化に弱い、という仮説）
- 層種別（mixer in/out, mlp up/down, embedding, lm_head）
指標ごとに順位相関を出し、**同じ順位になるか**を確認する。ならないなら、どれを主指標にするか理由を書く。

### B. 配分の探索
- ビット候補: 4 / 6 / 8（group 32 固定）
- 予算: 9B の合計を 4.5GB / 5.0GB / 5.5GB の 3 ケース
- 層種別ごとの重み（例: mlp down を厚く、mixer out を薄く）を仮定し、貪欲法で配分
- 目的関数: 量子化誤差の代理（trace × 量子化ステップ幅の二乗の和）。理論式を明記
- 各ケースで「ビット配分表」「推定サイズ」「推定誤差」を出す

### C. 4B と 0.8B への示唆
- 同じ配分方針が 4B / 0.8B でも妥当か
- 9B の untied embedding / lm_head をどう扱うか（8bit に上げる価値があるか）

### D. 検証計画
- 実測で確かめる手順を 3 つ。各項目に: 対象、コマンド、期待値、失敗時の解釈
- 特に「proxy 誤差モデルが実際の PPL 劣化と相関するか」を 0.8B で先に確かめる方法を書く

## 成果物
1. `research/9b_bit_allocation.md`
2. `research/9b_bit_allocation.json` — 配分表（層 × ビット × サイズ）
3. `scripts/bit_alloc.py` — 上記を再計算できるスクリプト（`--budget-gb`, `--model 9b`）

## 制約
- 推定と実測を明示的に区別する。9B の数値はすべて推定であることを表に書く
- **repo を clone しない。**必要なファイルは raw.githubusercontent から個別取得
- 捏造禁止。計算式と係数を全部書く
- 日本語。20 分以内
- 書き込めない場合は数値を作らずエラー報告

## 完了報告
- 予算 5.0GB のときの配分（層種別ごとのビット）
- 主指標に何を選んだか、その理由
- 0.8B で先に検証すべき 1 実験
