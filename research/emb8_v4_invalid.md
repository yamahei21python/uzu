# emb8 v4 / 4B eval v3 の結果は無効（stale script 混入）

2026-11-07 の `uzu-emb8-08b-gpu` v4 は完走したが、**評価スクリプトが古い版**だった。

## 証拠

カーネルログ（`/tmp/emb8_kaggle_out_v4/uzu-emb8-08b-gpu.log:11`）:

```
scripts: .../uzu-quant-kit/scripts/rht_int4_convert.py .../uzu-quant-kit/scripts/yaqa_calibration.py
         /kaggle/input/datasets/yamahei21python/uzu-ja-corpora/eval_quantized.py   <-- ここ
```

- 旧カーネルは `next(Path("/kaggle/input").rglob("eval_quantized.py"))` でスクリプトを探していた
- attach していた `uzu-ja-corpora` に古い `eval_quantized.py` が入っており、それが採用された
- `uzu-ja-corpora` はライセンス問題（CC0 宣言）で削除予定の dataset

## 影響

- emb8 v4: BF16 baseline が clean 実行と一致しない（EN 37.16 vs 35.38、JA 42.36 vs 43.42）。同じコーパス・同じ sha・同じ chunk 設定なので、評価系の差と判断する
- **数値を採用しない**
- 4B RTN eval v3 も同じ `rglob` を使っている。走行中の結果は同様に疑わしい

## 対応

1. 旧カーネルのスクリプト解決を kit 固定に変更（`.../uzu-quant-kit-public` のみを探す）
2. 4B RTN eval は lane で再実行する: `lanes/eval-4b-rtn.json`
3. `uzu-ja-corpora` は走行中ジョブの終了後に削除する（`research/license_actions_2026-11-07.md`）

## 教訓

`rglob` でスクリプトを探さない。dataset を跨ぐと古い版を拾う。`run_lane` は kit パスを明示的に受け取るので、この問題は起きない。
