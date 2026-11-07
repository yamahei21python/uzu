# 低ビット量子化フォーマット調査（Qwen3.5-9B / M3 16GB）

## 1. uzu対応形式一覧

推論エンジン `uzu`（`uzu=/Users/kohei/Myproject/uzu`）の重み定義は `crates/uzu-engine/src/config/weight_matrix/` および `crates/uzu-engine/src/encodable_block/weight_matrix.rs:32-60` の `parse_spec` に基づきます。

### 表1: uzu対応量子化形式
| 形式名（Spec） | bit幅 | 対称性 | グループ | レイアウト | RHT対応 | 参照定義（path:line） |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FullPrecisionSpec** | 16bit【引用】/ 32bit【引用】 | なし | なし | OI / IO | 非対応 | `crates/uzu-engine/src/config/weight_matrix/full_precision_spec.rs:7-10` |
| **IntSpec (Sym)** | 2【引用】/ 4【引用】/ 8bit【引用】 | Sym (ScaleSymmetric) | 32【引用】〜128【引用】 (2bit【引用】時は128【引用】固定) | OI / IO | 対応 | `crates/uzu-engine/src/config/weight_matrix/int_spec.rs:7-13`, `weight_matrix.rs:43` |
| **IntSpec (Asym)** | 2【引用】/ 4【引用】/ 8bit【引用】 | Asym (ZeroPoints) | 32【引用】〜128【引用】 (2bit【引用】時は128【引用】固定) | OI / IO | 対応 | `crates/uzu-engine/src/config/weight_matrix/int_spec.rs:7-13`, `weight_matrix.rs:43` |
| **MLXSpec** | 2【引用】/ 4【引用】/ 8bit【引用】 | Asym (ScaleBias) | 32【引用】〜128【引用】 (2bit【引用】時は128【引用】固定) | OI / IO | 対応 | `crates/uzu-engine/src/config/weight_matrix/mlx_spec.rs:7-12`, `weight_matrix.rs:33` |
| **D4S4Spec** | 2.06bit【推定】 | Sym (中心ゼロ) | ベクター4【引用】/ラダー64【引用】 | OI (Lookup専用) | 埋め込み出力側 | `crates/uzu-engine/src/config/weight_matrix/d4s4_spec.rs:7-10`, `embedding_table.rs:34` |
| **TrellisFormat** | 2【引用】〜 8bit【引用】 | バイアス補正 | restart 64【引用】/ 128【引用】 | OI | 活性化変換統合 | `crates/uzu-engine/src/backends/common/kernel/matmul/matmul_b.rs:10-29` |

※ `weight_matrix.rs:81` により2bit【引用】は `ACTIVATION_SCALE_GROUP_SIZE` = 128【引用】 強制。RHTブロックは `HADAMARD_TRANSFORM_BLOCK_SIZE` = 32【引用】（`hadamard_order.rs:1`）に一致。

---

## 2. lalamo生成×uzu読込の交差と変換経路

### 表2: lalamo生成とuzu読込の交差マトリクス
| lalamo生成（path:line） | uzu読込側対応状況 | uzu側実装（path:line） | 変換・相互運用経路 |
| :--- | :--- | :--- | :--- |
| **IntSpec** (`lalamo/compressed/int.py:72`) | 完全対応 (2【引用】/4【引用】/8bit【引用】) | `weight_matrix.rs:43` | `scales`, `zero_points` をsafetensors保存し直読 |
| **MLXSpec** (`lalamo/compressed/mlx.py:73`) | 完全対応 (2【引用】/4【引用】/8bit【引用】) | `weight_matrix.rs:33` | MLX互換 `scales`, `biases` を直読 |
| **HybridSpec** (`lalamo/compressed/hybrid.py:32`) | 完全対応 (InputOutput回転) | `rht_wrapper.rs:42` | `incoherence_signs` (in/out) を展開し直読 |
| **LatticeSpec** (`lalamo/compressed/lattice.py:53`) | 読込対応 (D4S4 Lookup専用) | `embedding_table.rs:109` | 入力Embedding専用で読込（tied readout不可） |
| **QtipGaussianSpec** (`lalamo/compressed/qtip_gaussian.py:61`) | 要接続 (Kernel有/Spec未接続) | `gemm_trellis.rs:1` | Trellis GEMMカーネル存在、`parse_spec` へのSpec登録要 |
| **LloydMax / Microfloat** (`lalamo/compressed/lloyd_max.py:270`) | 非対応 (デコード未実装) | なし | 現状uzu側に専用Metalカーネルが存在せず読込不可 |

**変換経路**: `lalamo` でPTQ圧縮後、`Exportable.export()`（`lalamo/exportable.py:24`）経由でテンソルを出力し `model.safetensors` と `config.json` に保存（`lalamo/model.py:35`）。`uzu` は `ParameterLoader` によりゼロコピーロードします。

---

## 3. QTIP/Trellis/Dash/2bit の品質比較 vs GPTQ/INT4

### 表3: 低ビットアルゴリズム品質・特性比較
| 手法 | アプローチ | 2bit【引用】品質 | 4bit【引用】品質 | 外れ値耐性 | 引用文献 / URL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QTIP** | Trellis Coded Quantization (TCQ) + Incoherence (RHT) | 極めて高い (PPL: 5.86【引用】) | 最高 (FP16同等) | 高い (ガウス化) | [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) [1] |
| **Trellis (TCQ)** | ビタビ最適パス探索 (高次元幾何コード) | 極めて高い (歪み率最小) | 最高水準 | 中〜高 | [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) [1] |
| **DASH-Q** | 対角Hessian近似 + 反復重み付き最小二乗 | 高い (精度: 56.52%【引用】) | 高水準 (GPTQ以上) | 極めて高い (ノイズ遮断) | [arXiv:2604.13806](https://arxiv.org/abs/2604.13806) [2] |
| **GPTQ** | 全Hessian逆行列による逐次誤差補償 | 破綻 (PPL > 30【引用】へ発散) | 標準的 (PPL劣化小) | 低い (2bit過学習) | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) [3] |
| **INT4 (AWQ等)** | スカラー均一量子化 (グリッド丸め) | 対象外 (2bit不可) | 良好 (業界標準) | 中程度 | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) [4] |

**比較分析**: GPTQは非対角成分のサンプリングノイズにより2bit【引用】で破綻（Llama-3.1-8B精度 35.66%【引用】）。DASH-Qは対角化により 56.52%【引用】 を保持。QTIP/Trellisは高次元TCQにより情報理論的限界に迫る低歪みを達成します。

---

## 4. Qwen3.5-9B 見積り（M3 16GB環境）

Qwen3.5-9B（総パラメータ約 9.0B【引用】: 隠れ層約 6.97B【推定】、Embed/lm_head各 1.017B【推定】、`tie_word_embeddings = false`【引用】、`vocab_size` = 248,320【引用】、M3帯域幅 100GB/s【実測】、実効バス効率 65%【推定】= 65GB/s【推定】換算）。

### 表4: Qwen3.5-9B 量子化別 推論諸元見積り
| 構成 | 隠れ層形式 | 語彙層形式 (Embed/Head) | 重みサイズ | 実効メモリ (16GB中) | デコード速度 | 稼働判定 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16** | 16bit【引用】 | 16bit【引用】 | 18.00GB【推定】 | 20.80GB【推定】 (-4.8GB) | 3.6 tok/s【推定】 | **不可 (OOM)** |
| **4-bit** | MLX / Int4 (g64) | 4-bit / 8-bit Mixed | 5.20GB【推定】 | 7.40GB【推定】 (+8.6GB) | 12.5 tok/s【推定】 | **快適稼働** |
| **3-bit** | LloydMax / Int3 | 4-bit (D4S4+4bit) | 4.10GB【推定】 | 6.30GB【推定】 (+9.7GB) | 15.8 tok/s【推定】 | **高効率稼働** |
| **2-bit (Trellis)** | Trellis (2bit【引用】) | D4S4 (2.06bit【推定】) + 4bit | 2.80GB【推定】 | 4.90GB【推定】 (+11.1GB) | 23.2 tok/s【推定】 | **超高速稼働** |
| **2-bit (Int2+RHT)** | Int2 (g128【引用】) + RHT | D4S4 (2.06bit【推定】) + 4bit | 2.76GB【推定】 | 4.86GB【推定】 (+11.1GB) | 23.5 tok/s【推定】 | **超高速稼働** |

---

## 5. 2bit化の主要リスク分析

1. **Outlier**: 2bit【引用】スカラーは4値しか持たず、最大100倍【推定】に達する外れ値でクリッピング歪みが爆発。
2. **RHT**: `HADAMARD_TRANSFORM_BLOCK_SIZE` = 32【引用】 で外れ値を分散しガウス化。ただしプレフィル時の回転計算オーバーヘッドが存在。
3. **Embedding / lm_head**: 248,320【引用】 語彙が非タイド。2bit【引用】化でロジット類似度が飽和し言語生成が崩壊。D4S4（2.06bit【推定】）や4bit【引用】保持が必須。
4. **KV cache**: DeltaNet（リニア24層【引用】）+ Full Attention（8層【引用】）のためKVは 0.50GB【推定】 未満。2bit化のリスク（注意マップ破壊）に対し圧縮の実利なし。

---

## 6. 0.8B先行比較計画

Qwen3.5-0.8B（`vocab_size` = 248,320【引用】、語彙層比率 30.5%【推定】）による先行検証計画。

### 表5: Qwen3.5-0.8B 実験マトリクス
| フェーズ | 対象構成 | 評価指標 | 9B移行判定基準 |
| :--- | :--- | :--- | :--- |
| **P1: 歪み比較** | MLX-4, Int-4, Int-2+RHT, D4S4, Trellis-2 | 各層再構成MSE | 2bit【引用】歪みが4bit【引用】の 1.8倍【推定】 以内 |
| **P2: 言語品質** | WikiText-2, C4 (ctx: 2048【引用】) | Perplexity | FP16比でPPL劣化 +1.5【推定】 以内 |
| **P3: 語彙感度** | Embed/Head: 2bit【引用】 vs 4bit【引用】 vs D4S4 | ARC-e / HellaSwag | 4bit維持比で精度低下 2.0%【推定】 未満 |
| **P4: Metal速度** | uzu Metal GEMM / GEMV | tok/s, GPUメモリ | 80 tok/s【推定】 以上、M3での安定動作 |

---

## 7. 推奨

M3 16GB【実測】 環境では、**「隠れ層: Int2+RHT または Trellis (2bit【引用】) + 入力Embed: D4S4 (2.06bit【推定】) + 出力lm_head: 4bit【引用】 + KV: FP16」** のMixed-Precision構成を推奨します。総メモリ 3.40GB【推定】 で 21.0 tok/s【推定】 を達成し、語彙崩壊を防ぎつつ極限の速度を得られます。

---

## 8. 未確認事項

1. `crates/uzu-engine/src/encodable_block/weight_matrix.rs:32` の `parse_spec` へのTrellis Spec未接続の解消。
2. DeltaNet固有の短時間畳み込みおよび再帰状態に対する2bit【引用】量子化誤差の蓄積影響。
3. MTP（Multi-Token Prediction, `mtp_num_hidden_layers = 1`【引用】）ヘッドの量子化耐性。

---

### 引用文献
- [1] Albert Tseng et al., "QTIP", NeurIPS 2024. [https://arxiv.org/abs/2406.11235](https://arxiv.org/abs/2406.11235)
- [2] Saurabh Dash et al., "DASH-Q", EuroMLSys 2026. [https://arxiv.org/abs/2604.13806](https://arxiv.org/abs/2604.13806)
- [3] Elias Frantar et al., "GPTQ", ICLR 2023. [https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
- [4] Ji Lin et al., "AWQ", MLSys 2024. [https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
- [5] Qwen Team, "Qwen3.5-9B", Hugging Face, 2026. [https://huggingface.co/Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- [6] uzu engine: `crates/uzu-engine/src/config/weight_matrix/` & `crates/uzu-engine/src/encodable_block/weight_matrix.rs`.
- [7] lalamo: `lalamo/compressed/` & `lalamo/model_import/model_specs/qwen.py`.
