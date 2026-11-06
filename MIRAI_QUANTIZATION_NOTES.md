# Mirai Quantization 公式情報と再現メモ

最終更新: 2026-11。対象: Mirai-M (4-bit) / Mirai-L (8-bit)。

## 1. 公式ソース

| 情報 | URL |
|---|---|
| 技術ブログ（2026-06-03） | https://trymirai.com/blog/quantization |
| 0.8B-M モデルカード | https://huggingface.co/trymirai/Qwen3.5-0.8B-M |
| 2B-M モデルカード | https://huggingface.co/trymirai/Qwen3.5-2B-M |
| YAQA 論文 (ICML 2026) | https://arxiv.org/abs/2505.22988 |
| lalamo | https://github.com/trymirai/lalamo |
| uzu | https://github.com/trymirai/uzu |

論文タイトル: **Model-Preserving Adaptive Rounding**（YAQA = Yet Another Quantization Algorithm）。
Kronecker-factored Hessian 近似 `H ≈ H_out ⊗ H_in` を使い、GPTQ/LDLQ より誤差を約30%削減。

## 2. 公式フォーマット（Mirai-M）

モデルカードと配布チェックポイントの metadata から確認。

| 項目 | 値 |
|---|---|
| Linear 量子化 | 4-bit asymmetric integer |
| zero point | 4-bit |
| scales | BF16 |
| group size | 32 |
| Linear layout | `output_input` |
| RHT | block-diagonal, block size 32 |
| Linear RHT mode | `input_output` |
| Embedding 量子化 | あり |
| Embedding layout | `input_output` |
| Embedding RHT mode | `output` のみ |
| 推論 | uzu（Apple silicon） |
| 学習 | PTQ + QAD |

### 配布チェックポイントの実測（0.8B-M）

`safetensors` header のみ取得して確認。

- tensor 数: 617
- HybridSpec: 96（Linear と一致）
- Linear spec 例:
  ```json
  {"type":"HybridSpec",
   "quantization_spec":{"type":"IntSpec","bits":4,"group_size":32,
                        "is_symmetric":false,"layout":"output_input"},
   "adapter_spec":null,
   "incoherence_block_size":32,
   "incoherence_processing_mode":"input_output"}
  ```
- Embedding spec:
  ```json
  {"type":"HybridSpec",
   "quantization_spec":{"type":"IntSpec","bits":4,"group_size":32,
                        "is_symmetric":false,"layout":"input_output"},
   "adapter_spec":null,
   "incoherence_block_size":32,
   "incoherence_processing_mode":"output"}
  ```
- Embedding tensors:
  - `quantized.weights` U8 `[248320, 512]`
  - `quantized.scales` BF16 `[248320, 32]`
  - `quantized.zero_points` U8 `[248320, 16]`
  - `incoherence_signs.output_signs` I32 `[1024]`
- dtype 内訳: U8 194, I32 193, F32 133, BF16 97

## 3. 我々の 0.8B 変換との差分

| 項目 | 公式 0.8B-M | 現在の変換 |
|---|---|---|
| Linear spec | 一致 | 一致 |
| Linear scales | BF16 | BF16 |
| Embedding | INT4 + output RHT | **BF16 のまま** |
| tensor 数 | 617 | 614 |
| サイズ | 約 0.65 GB 常駐 | 764 MB |

差分は embedding の量子化のみ。BF16 embedding は 248320×1024×2 ≒ 508 MB。
`--quantize-embedding` で公式と同じ構成になる。

## 4. アルゴリズム（ブログ記載）

### PTQ

- 独自 JAX 実装の YAQA
- 数値安定性のための minor modifications（詳細非公開）
- 内部 calibration dataset から Fisher Information Matrix を推定

### QAD（Quantization-Aware Distillation）

- BF16 教師モデルの rollout を使用
- 量子化重み: Muon + deterministic straight-through estimator
- group scales / zero points: AdamW
- 品質はデータの domain coverage に敏感

### 評価

- MMLU-Pro（thinking 有効、81,920 token budget）
- mean KL: LMSYS-Chat-1M から 100 問、教師が生成した回答上の logits KL
- KL の noise floor は約 `8e-4`
- MMLU-Pro の有意差は約 1 ポイント未満では判定不能

## 5. 非公開・未確認

| 項目 | 状態 |
|---|---|
| Fisher 推定の内部データ | 非公開 |
| Fisher 推定の実装詳細 | 非公開 |
| 数値安定化の正確な変更 | 不明 |
| QAD の rollout 生成条件 | 不明 |
| Muon / STE / AdamW の学習率・スケジュール | 不明 |
| 公式チェックポイント生成コマンド | 不明 |

YAQA の丸めアルゴリズム、RHT、INT4 保存形式は公開されている。
Fisher と QAD のレシピが残りの壁。

## 6. 再現ロードマップ

| 段階 | 内容 | 難易度 |
|---|---|---|
| A | RHT + INT4 Group32 + YAQA | 高 |
| B | A + 公式に近い Fisher 推定・安定化 | 非常に高 |
| C | B + QAD（Muon/STE/AdamW） | 最高 |

まず A を 0.8B で完了させ、KL / PPL / 指示追従で RTN と比較する。
embedding 量子化は A に含める（公式構成に合わせる）。

## 7. 我々の実装との対応

| 公式 | 本リポジトリ |
|---|---|
| lalamo YAQA | `scripts/yaqa_calibration.py` + `HybridSpec.compress` |
| Sketch B | `H_I = mean G^T G`, `H_O = mean G G^T` |
| RHT Linear | `HadamardSpec(32, input_output)` |
| RHT Embedding | `HadamardSpec(32, output)` |
| INT4 Group32 | `IntSpec(4, 32, asymmetric, layout)` |
| embedding 量子化 | `rht_int4_convert.py --quantize-embedding` |
| KL 評価 | `scripts/eval_quantized.py` |

## 8. 論文メモ

- YAQA は layer 単体ではなくモデル出力の KL を最小化する
- Hessian の Kronecker 近似 `H_out ⊗ H_in` を使う
- block LDL で誤差を伝播
- GPTQ/LDLQ 比で誤差約30%減、QAT より低誤差の報告
- 推論時オーバーヘッドなし（RHT はカーネルに融合）

本実装の `H_I` / `H_O` はこの Kronecker 因子に対応する。

## 9. 配布チェックポイントの逆解析（0.8B-M vs 我々）

`scripts/compare_quantized_checkpoints.py` で 96 Linear を比較。

| 比較 | code 一致率 | scale 比 | code 分布 |
|---|---|---|---|
| 公式 vs RTN | 0.706 | 1.0021 | ずれあり |
| 公式 vs YAQA(H_in) | 0.620 | 0.99996 | **ほぼ一致** |

- 公式と YAQA の scale 平均比は 0.99996
- 4-bit code ヒストグラムは小数4桁まで一致
- 個別 code は 38% 異なる
  → 公式は YAQA 系の分布を持ち、QAD でコードを微調整した可能性
- zero point 分布も両者でほぼ同一（平均 7.50、std 1.27）

成果物:
- `research/checkpoint_compare_rtn.json`
- `research/checkpoint_compare_yaqa.json`
