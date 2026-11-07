タスク: lalamo 量子化パスのホットスポット特定（CPU 実測）

## 背景
0.8B の YAQA calibration はローカルで約 10 分、変換は数分。eval は 100 chunks で 10 分超。
4B/9B に上げると比例以上に遅くなる。どこが支配的なのか分かっていない。
GPU は使わず、CPU で同じ形の合成配列を流して**1 演算あたりのコスト**を測る。

## 入力（公開のみ）
- lalamo: https://github.com/trymirai/lalamo commit e401e17761f2fcd337b0cf20218d67d656e6ecba
  - `lalamo/compressed/int.py`, `lalamo/compressed/hybrid.py`, `lalamo/model.py`
- HF dataset: https://huggingface.co/datasets/yamahei21/uzu-quant-kit
  - `scripts/rht_int4_convert.py`, `scripts/yaqa_calibration.py`, `scripts/eval_quantized.py`
  - `configs/qwen35-0.8b.json`

## やること
### A. 呼び出し経路の整理
量子化 Linear の forward で、1 回の matmul に対して何が走るかを列挙する:
- RHT（Hadamard 32）
- group ごとの scale / zero-point 適用
- INT4 unpack
- dequant + matmul
- 出力側ブロック（output_block）の有無
各段の計算量を式で書く（O(何)）。

### B. CPU 実測
numpy で同じ shape を作り、段ごとの時間を測る。条件:
- in/out: 1024×1024, 1024×8224, 2048×1024, 1024×7168, 7168×1024（0.8B の実 shape）
- group 32、bits 4
- batch: 1 トークン、128 トークン
- 反復 20 回の中央値
測るもの: RHT、pack/unpack、dequant、matmul、合計。**ボトルネックの順位**を出す。

### C. 設定で変えられる部分
以下を変えたときの時間・サイズを表にする:
- group size 32 / 64 / 128
- block size 32 / 64 / 128（RHT）
- output_block あり / なし
- bits 4 / 8

### D. 改善案
- ライブラリ改変が必要なもの / 設定で済むもの を分ける
- 期待削減量（実測比）と副作用（品質）
- 変換時と eval 時で効くものが違う点を明記

## 成果物
1. `research/lalamo_quant_hotspots.md`
2. `research/lalamo_quant_hotspots.json` — 計測値
3. `scripts/bench_quant_ops.py` — 再実行可能（`--repeats`, `--group-size`）

## 制約
- GPU 不要。numpy / jax CPU のみ
- 実測と推定を区別。中央値と試行回数を書く
- **repo を clone しない**（必要ファイルは raw URL）
- 日本語。20 分以内
- 書き込めない場合は数値を作らずエラー報告

## 完了報告
- 支配的な段 1 つと、その割合
- 設定変更で削れる最大の項目
