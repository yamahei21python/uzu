# QAD 安定化レシピ調査レポート

本調査は lalamo による量子化認識蒸留（QAD）を 0.8B（20〜50 steps）から 4B/9B へ安定スケールするための最適化レシピ策定書である。結論として、2D 重みへの Muon（lr=1e-3, beta=0.95）と scale/zero-point への AdamW（lr=1e-4, betas=(0.9, 0.95)）の分離適用、Forward KL（T=1.0）、確定的 STE の維持が 50 step の初期値として妥当。4B/9B 拡張時は Newton-Schulz 直交化通信と bf16 状態管理が要点となる。

## 1. optax.contrib.muon の仕様と挙動分析

`optax.contrib.muon` は累積モメンタムを 5 次 Newton-Schulz 反復で直交化し、Schatten-p ノルムで最急降下を行う【引用】[1]。

| 主要引数 | デフォルト値 | 型 / 仕様 | QAD 制約・推奨値 |
| :--- | :--- | :--- | :--- |
| `learning_rate` | (必須) | `ScalarOrSchedule` | 推奨: `1e-3`〜`2e-3`【推定】 |
| `ns_coeffs` | `(3.4445, -4.7750, 2.0315)` | `tuple`/`str` | 5次係数。プリセットは `'muon'`, `'dion'`【実測】[1] |
| `ns_steps` | `5` | `int` | 直交化反復回数。5回で特異値整定【引用】[1] |
| `beta` | `0.95` | `float` | 減衰率。短ステップは `0.95` 維持【推定】 |
| `weight_decay` | `0.0` | `float` | 短期 QAD は `0.0` 推奨（事前学習保護）【推定】 |
| `mu_dtype` | `None` | `DTypeLike` | 4B/9B は省メモリのため `jnp.bfloat16` 推奨【推定】 |
| `nesterov` | `True` | `bool` | 既定 True。STE マスクとの相互作用は §8.2【引用】[1] |
| `consistent_rms` | `None` | `float` | 設定時のみ RMS 補正。QAD では未検証【引用】[1] |

- **非 2D 処理**: `muon_weight_dimension_numbers=None`（既定）では `ndim==2` のみ Muon、非 2D は内部 AdamW へ自動振分【実測】[1]。
- **非行列制約**: 明示的に Muon 次元指定した `ndim < 2` は `ValueError`。本実装は `multi_transform` で 1D を AdamW に分離済みのため該当しない【実測】。

## 2. QAD/QAT における Scale / Zero-Point 学習の安定化手法

scale $s$ と zero-point $z$ の学習は勾配乖離による Scale Collapse（微小化・発散）を招きやすい。

| 手法名 | Scale Collapse 対策 | クリッピング境界 | 学習率 (W / Scale) | AdamW Betas | 初期化方式 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LLM-QAT** | 勾配クリップ + 下限ガード | 符号付き整数射影 | W/S: `2e-5`（AdamW, wd=0）【引用】[4] | 既定 `(0.9, 0.999)` | Min-Max/Percentile |
| **BitDistiller** | 非対称クリップ学習 | 歪み最小化クリップ | W/S: `8e-6`（AdamW, wd=0）【引用】[5] | `(0.9, 0.999)`（既定） | 重み分布 MSE 探索 |
| **EfficientQAT** | Block-AP $\to$ E2E-QP 2段階 | 整数固定・scale 線形更新 | AP: W `2e-5` / S `1e-4`、E2E: `2e-5`(2bit)/`1e-5`(3bit)【引用】[6] | 既定 | RTN + Group Grid |
| **LR-QAT** | 低ランク射影正規化 | アダプタ直交制約 | A/B: `1e-5`〜`1e-2` 探索 / S: 探索【引用】[7] | `(0.9, 0.95)`, wd=0 | SVD 初期化 |
| **lalamo (現行)** | `jnp.clip(values, 0, qmax)` | STE 方向制限マスク | W: `1e-3` / S,Z: `1e-4`【実測】 | `(0.9, 0.999)`【推定】 | Pretrained 統計値 |

**処方箋**: 短ステップは $\beta_2=0.95$ へ短縮し、$s \ge 10^{-6}$ を下限ガード【推定】。

## 3. Deterministic STE vs. Stochastic Rounding の比較検証

| 評価軸 | Deterministic STE (lalamo 現行) | Stochastic Rounding (SR) | 20〜50 Step QAD 判定 |
| :--- | :--- | :--- | :--- |
| **順伝播計算** | $q = \text{round}(\text{clip}(x/s, 0, q_{max}))$【実測】 | 隣接 LUT レベルを確率比例サンプリング【引用】[12] | 決定論的で完全再現可能【実測】 |
| **勾配推定量** | バイアスあり（階段関数近似） | 不偏推定量 ($\mathbb{E}[q] = x/s$)【引用】[12] | 短期では低分散が最優先【推定】 |
| **ロジット分散** | ゼロ（安定的） | 高い（乱数起因ノイズ混入） | SR は 50 step で損失振動し不適【推定】 |

**判定**: 短期は Deterministic STE 推奨【推定】。`_mask_straight_through_gradients` は範囲外で戻る方向のみ許可するため、scale が格子外へ出ると勾配 0 化しやすい。

## 4. 蒸留損失関数（KL / CE / Reverse KL）と Teacher Rollout 条件

| 損失形式 | 挙動特性 | メリット / デメリット | Rollout 推奨 (50 steps) | 適合度 |
| :--- | :--- | :--- | :--- | :--- |
| **Forward KL** | Mode-covering【引用】[14] | 語彙脱落防止 / 低確率トークン過大評価 | 温度 $T=1.0$, Top-$p=0.95$【推定】 | **最推奨** |
| **Reverse KL** | Mode-seeking【引用】[14] | 幻覚抑制 / Mode Collapse リスク大 | 学生自己生成 (On-policy)【引用】 | 不適 |
| **Top-$k$ KL** | 上位包括 ($k=128 \sim 512$) | 語彙軸省メモリ、長尾遮断 / $k$ 調整要 | Off-policy (Teacher-forced)【推定】 | 推奨(9B) |
| **Hard CE** | 正解ラベル最尤 | 計算最軽量 / Teacher 暗黙知欠落 | 0.1M tok 程度（seq256×400）【推定】 | 補助利用 |

## 5. 0.8B モデル向け 20〜50 Step 推奨安定化レシピと診断表

| 項目 | 推奨値 | 調整範囲 | 項目 | 推奨値 | 調整範囲 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `steps` | `50` | `20`〜`100`【推定】 | `seq_len` | `256` (初期 `128`) | `128`〜`512`【推定】 |
| `global_bs` | `16` (Micro 4 $\times$ Accum 4) | `8`〜`32`【推定】 | `max_grad_norm` | `1.0` | `0.5`〜`2.0`【引用】[6] |
| `muon_lr` | `1.0e-3` | `5.0e-4`〜`1.5e-3`【推定】| `scale_lr` | `1.0e-4` | `3.0e-5`〜`2.0e-4`【推定】|
| `muon_beta` | `0.95` | `0.90`〜`0.98`【引用】[1]| `adam_betas` | `(0.9, 0.95)` | $\beta_2=0.95$【推定】 |

```yaml
# 0.8B E3 QAD 推奨設定 (qad_recipe_0.8b.yaml)
training: {steps: 50, seq_len: 256, micro_bs: 4, grad_accum: 4, max_grad_norm: 1.0}
optimizer:
  master_weights: {type: "muon", lr: 1.0e-3, beta: 0.95, ns_steps: 5}
  quant_params: {type: "adamw", lr: 1.0e-4, betas: [0.9, 0.95]}
distillation: {loss: "forward_kl", temperature: 1.0, scale_min: 1.0e-6}
```

| 観測症状 | 主な原因 | 監視指標 | 是正アクション |
| :--- | :--- | :--- | :--- |
| **1〜5 step で KL 急増** | Muon/Scale LR 過大 | `grad_norm` | Muon LR `5e-4`、Scale LR `3e-5` へ半減【推定】 |
| **KL が横ばい推移** | STE 勾配消失 / マスク率 100% | STE 逆伝播ノルム | Scale LR を `3e-4` に引上げ、初期値確認【推定】 |
| **Loss が NaN 化** | Log-Softmax 下限突破 / $s \to 0$ | `min(scales)`, Logit | FP32 実行、$s \ge 10^{-6}$ で固定【推定】 |
| **下流 PPL 悪化** | 短系列 (128) 過学習 | 検証長文 PPL | `seq_len` を 256/512 へ拡大【推定】 |
| **量子化値が不変** | 重み更新量が格子幅未満 | $\Delta W$ と $s$ の比 | Muon LR 引上げ、Warmup 3 step 短縮【推定】 |

```python
# optax 分割オプティマイザ構築
import jax, optax

def create_qad_tx(muon_lr=1e-3, scale_lr=1e-4):
    def label_fn(params):
        def label(path, value):
            name = ".".join(getattr(k, "name", str(k)) for k in path)
            if "scales" in name or "master_zero_points" in name:
                return "quant"
            return "muon" if value.ndim >= 2 else "quant"
        return jax.tree_util.tree_map_with_path(label, params)
    return optax.multi_transform(
        {"muon": optax.contrib.muon(learning_rate=muon_lr, beta=0.95, ns_steps=5),
         "quant": optax.adamw(learning_rate=scale_lr, b1=0.9, b2=0.95, eps=1e-8)},
        param_labels=label_fn
    )
```

## 6. 4B / 9B モデルへのスケールアップ拡張設計

| 項目 | 0.8B (E3 Student) | 4B (TP=2/4) | 9B (TP=4/8) | 留意点 |
| :--- | :--- | :--- | :--- | :--- |
| **重みメモリ (BF16, 生徒+教師)** | 約 3.2 GB | 約 16.8 GB | 約 36.4 GB | 生徒+教師推論用【推定】 |
| **Muon 状態メモリ (bf16)** | 約 1.6 GB (`mu` のみ) | 約 8.4 GB | 約 18.2 GB | AdamW の m+v より少ない【推定】 |
| **推奨 TP 並列度** | TP=1 (単一 GPU) | TP=2 または TP=4 | TP=4 または TP=8 | Modal A100/H100【推定】 |
| **Micro Batch / Accum** | Batch 4, Accum 4 | Batch 2, Accum 8 | Batch 1, Accum 16 | Global BS 16 維持【推定】 |
| **NS FLOPs / step** | 約 $1.2 \times 10^{11}$ | 約 $2.4 \times 10^{12}$ | 約 $8.1 \times 10^{12}$ | $T=5$、行列形状依存の概算【推定】 |
| **NS 通信オーバーヘッド** | ゼロ | All-Reduce 発生 | All-Reduce 発生 | Gram 形式で削減【引用】[3] |

**TP 拡張の要点**: 列・行並列下では $X X^T$ で $M \times M$ の All-Reduce が毎反復発生。Gram Newton-Schulz と `jax.checkpoint` を併用【推定】。

## 7. 公開 QAD/QAT 実装の網羅的ハイパーパラメータ比較

| フレームワーク | 対象ビット | 最適化手法 | 学習率 (Weight / Scale) | 損失関数形式 | バッチ・ステップ | 粒度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA ModelOpt** (NVFP4 QAD)【引用】[8][9] | NVFP4, FP8 | AdamW | W: `1e-5`〜`1e-6` / S: 固定【引用】[9] | Forward KL (logit) | BS 32〜64 / 1k〜5k st【推定】 | Group 16/32 |
| **Gemma 3 QAT** (Google)【引用】[15] | INT4 | 非公開 | 非公開 | QAT + distillation | 非公開 | Group-wise |
| **LR-QAT**【引用】[7] | INT2, INT3, INT4 | AdamW (wd=0) | A/B: `1e-5`〜`1e-2` 探索 / S: 探索 | Cross-Entropy | BS 32 / seq 1024 / 1e4 st | Group 128 |
| **torchtune QAT**【引用】[11][13] | INT4, INT8 | AdamW | レシピ既定（lr は config 依存） | Cross-Entropy | 1 epoch / max 2000 st per epoch | Group 32 |
| **Unsloth Gemma 4 QAT**【引用】[10] | INT4 | 配布のみ | 非公開 | - | - | - |

## 8. 未確認事項および今後の技術的検証課題

| 課題項目 | 調査・検証課題 | 潜在的リスク | 検証アプローチ | 優先度 |
| :--- | :--- | :--- | :--- | :--- |
| **8.1** | `consistent_rms` の影響 | STE 格子境界を乱す可能性 | `0.2` と通常を比較【推定】 | 中 |
| **8.2** | Nesterov と STE マスク | 外挿値が飽和帯へ突入 | `nesterov=False` 比較【推定】 | 高 |
| **8.3** | TP 下の Newton-Schulz 通信 | 4B/9B で律速化 | Modal TP=4/8 で計測【推定】 | 高 |
| **8.4** | Top-$k$ KL の語彙軸節約 | >128k 語彙で Softmax メモリ圧迫 | $k=256$ でメモリ・精度計測【推定】 | 中 |
| **8.5** | Zero-Point の格子境界逸脱 | デコード時オーバーフロー | 更新後 `jnp.clip(z, 0, qmax)`【推定】 | 高 |
| **8.6** | 引用 lr・バッチの一次確認 | [9] PDF と [10] は学習設定を明示しない | 原論文 Appendix / config 確認 | 中 |

**注**: 推奨値は初期値。実測ログ（KL, grad_norm, min(scales)）で調整する。`qad_finetune.py` の AdamW は既定 `b2=0.999` のため、`b2=0.95` には引数追加が必要（本番コード未変更）。

---

## References

[1] https://github.com/google-deepmind/optax/blob/main/optax/contrib/_muon.py  
[2] https://arxiv.org/abs/2502.16982  
[3] https://tridao.me/blog/2026/gram-newton-schulz/  
[4] https://arxiv.org/abs/2305.17888  
[5] https://arxiv.org/abs/2402.10631  
[6] https://arxiv.org/abs/2407.11062  
[7] https://arxiv.org/abs/2406.06385  
[8] https://nvidia.github.io/Model-Optimizer/guides/quantization_aware_training_and_distillation.html  
[9] https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf  
[10] https://unsloth.ai/docs/models/gemma-4/qat  
[11] https://arxiv.org/abs/2605.21442  
[12] https://arxiv.org/abs/2511.00874  
[13] https://docs.pytorch.org/torchtune/main/tutorials/qat_finetune.html  
[14] https://haryoa.github.io/blogs/distillation-part-1/  
[15] https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/
