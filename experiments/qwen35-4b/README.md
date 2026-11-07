# Qwen3.5-4B 量子化実験

対象: `huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated`
（BF16: 32層、hidden 2560、intermediate 9216、vocab 248,320）

## 取得元

- ローカル: `/tmp/huihui_4b_bf16/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated`
- Modal Volume `uzu-models` に BF16 のコピーあり
- 9.3GB（2 shards）

## 実行計画

| 段階 | 場所 | 状態 |
|---|---|---|
| RTN 変換 | ローカル | **完了**（2.44GB、129 Linear、embedding INT4） |
| RTN 評価 | ローカル | **不可**（OOM）→ Kaggle GPU |
| YAQA calibration | Kaggle CPU 16GB | **不可**（OOM）→ Kaggle GPU |
| YAQA calibration (GPU) | Kaggle T4 x2 | 明日 09:00 JST 以降 |
| YAQA 変換 | Kaggle GPU / Modal | 未実施 |

## YAQA calibration（Kaggle CPU）

- Kernel: `yamahei21python/uzu-quant-yaqa-4b-cpu`（kaggleAI リポジトリ）
- Dataset: `yamahei21python/uzu-quant-kit`
- GPU 無効・GPU クォータ非消費・32GB RAM
- HF から直接ロード（BF16 保存なし）
- `--calib-dtype bfloat16`
- 16 samples × 512 token
- H_O は out_dim <= 3072（out_proj/down）
- 出力: `/kaggle/working/calibration.npz`

```bash
# 明日以降 GPU を使う場合
kaggle kernels push -p kaggleAI/kaggle/quant-yaqa-cpu --accelerator NvidiaTeslaT4
```

## メモリ見積り

- BF16 model 8.4GB + grads 8.4GB + H ~2GB = 約19GB
- ローカル 16GB では不可。Kaggle 32GB で実行
- 変換後の INT4 は約 2.4GB（embedding BF16）または 1.5GB（embedding INT4）


## RTN 実測（ローカル）

- 出力: `/tmp/huihui_4b_rtn_lalamo`
- サイズ: 2,441.8 MB（embedding INT4 込み）
- HybridSpec: 129（Linear 128 + tied embedding 1）
- metadata 書き換え: 129 件
- BF16 保存なし（HF から直接 import）

## メモリ制約

- ローカル 16GB: BF16 8.4GB + INT4 2.4GB + JAX で OOM（評価不可）
- Kaggle CPU 16GB: 同様に OOM
- Kaggle GPU T4 x2: RAM 29〜32GB、VRAM 15.6GB × 2。TP で評価可能
