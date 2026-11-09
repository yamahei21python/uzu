# uniform4 + emb8 の seed 1〜3（Muse の CPU 枠で並列実行）

`scripts/make_kaggle_kernel.py` の生成物。手で編集しない。

## 目的

`wide8 + emb8 + d1e0`（4 seed 取得中）と **uniform4 + emb8** を、同じ土俵で比較する。
速度差が約 25% あるので、品質が僅差なら厳密に知りたい。

## 構成

| 項目 | 値 |
|---|---|
| 8bit 配分 | **なし**（uniform 4bit） |
| embedding | 8bit（symmetric） |
| damping | 0.1 |
| parallel | replicated |
| eval | float32 / 100 chunks / 128 tokens |
| 校正 | s32 / q128 / f32 / mixed50 |

seed は**校正のサンプリングのみ**を変える（変換の量子化キーは固定）。

## 使い方

3 ディレクトリ（s1 / s2 / s3）をそれぞれ push する。並列で 1.5 時間。

```bash
for s in 1 2 3; do
  kaggle kernels push -p kernels/muse-uniform4-emb8-s$s
done
```

`kernel-metadata.json` の id は `kentasatomuse/uzu-uniform4-emb8-s{1,2,3}`。
アカウント名が違えば書き換える。

## 出力

`=== FINAL ===` に EN/JA の PPL と BPC Δ。
`result.json` は `out/<lane>/<condition>/result.json`。
