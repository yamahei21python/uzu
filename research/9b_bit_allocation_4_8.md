# 9B / 4B のビット配分（4bit / 8bit / bf16）

2026-11-08 / 6bit が使えないことが判明したため、実測サイズから再計算した。

## 前提: 6bit は不可

`lalamo/compressed/quant_params.py:70-80` は量子化パラメータの幅を **4/8/16/32** のみ許す。
6bit は圧縮できても export で落ちる（実測: `ValueError: unsupported quantization parameter width: 6`）。
`research/9b_bit_allocation.md` の「4bit 59層 + 6bit 69層」は**実装不能**。

## 実測のベースサイズ

各 INT4 モデルの safetensors を合計（実ファイル）。

| モデル | 4bit 全体 | 内訳 |
|---|---:|---|
| 4B | **2.27 GB** | 重み 1.89 / scales 0.31 / zero 0.07 |
| 9B | **4.83 GB** | 重み 4.17 / scales 0.52 / zero 0.13 |

8bit に上げると重み部分だけが 2 倍になる（scales / zero_points は同じ）。

## 層種別ごとの 8bit 追加コスト（実測パラメータ数）

### 9B（base 4.83GB）

| 層種別 | パラメータ | 8bit 追加 | 合計 |
|---|---:|---:|---:|
| up_projection | 3,221M | +1.500 GB | 6.33 GB |
| down_projection | 1,611M | +0.750 GB | 5.58 GB |
| in_proj | 1,214M | +0.565 GB | 5.40 GB |
| out_proj | 403M | +0.188 GB | **5.02 GB** |
| qkvg_projection | 336M | +0.156 GB | 4.99 GB |
| out_projection | 134M | +0.062 GB | 4.90 GB |

### 4B（base 2.27GB）

| 層種別 | パラメータ | 8bit 追加 | 合計 |
|---|---:|---:|---:|
| up_projection | 1,510M | +0.703 GB | 2.98 GB |
| in_proj | 759M | +0.353 GB | 2.63 GB |
| down_projection | 755M | +0.352 GB | 2.63 GB |
| out_proj | 252M | +0.117 GB | 2.39 GB |
| qkvg_projection | 210M | +0.098 GB | 2.37 GB |
| out_projection | 84M | +0.039 GB | 2.31 GB |

## 推奨配分

M3 16GB では 6GB 程度まで余裕がある（INT4 5.19GB は動作実績あり）。

### 9B

| 案 | 構成 | サイズ | 判断 |
|---|---|---:|---|
| 最小 | out_proj のみ 8bit | **5.02 GB** | 5.0GB 予算にほぼ一致 |
| **推奨** | out_proj + qkvg + out_projection | **5.24 GB** | 出力側を一通り保護 |
| 積極 | 上記 + down_projection | 5.99 GB | 16GB に余裕。要検証 |

### 4B

| 案 | 構成 | サイズ |
|---|---|---:|
| **推奨** | out_proj + qkvg + out_projection を 8bit | **2.52 GB** |
| 積極 | 上記 + down_projection | 2.87 GB |

## 根拠の限界

- 品質効果は **0.8B の down/out 8bit のみ実測**（d1e-3 で EN −1.06pp / JA −1.25pp）
- `qkvg_projection` / `out_projection` の 8bit は未検証
- 4B/9B では次元が違うため、同じ効果量が出る保証はない
- まず 0.8B で「out_proj + qkvg + out_projection」を試し、次に 4B で確認する順序が安全

## 推奨手順

1. 0.8B: `bits-down-out-8.json` を拡張し、out_proj / qkvg / out_projection も 8bit に
2. 0.8B: 100 chunks で評価（damping は最終設定に合わせる）
3. 効果が確認できたら 4B（2.52GB）→ 9B（5.24GB）へ
4. `up_projection` / `in_proj` は最後（サイズ増が大きい）
