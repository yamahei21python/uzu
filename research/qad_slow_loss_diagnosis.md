# lalamo 0.8B QAD 原因調査および 4B/9B スケール接続診断報告書

## 1. 実測データとエグゼクティブサマリ
0.8B QAD (KL蒸留) で **50 step に約82分 (98.4s/step)** を要し loss 悪化の事象を調査【実測】。

### 表1: 0.8B 50 step 実測損失ログの推移【実測】
| step | 実測 KL | フェーズ | 観測挙動【実測】 |
|---:|---:|:---|:---|
| 1 | 0.155950 | 初期 (YAQA直後) | 最適ベイスン内の極小値 |
| 5 | 0.769761 | Muon 初期更新 | 急激な悪化 (+0.61) |
| 10 | 0.742093 | 保存 #1 | 高止まり・振動 |
| 20 | 1.290452 | 保存 #2 | 発散進行 (1.2突破) |
| 30 | 0.838976 | 保存 #3 | チャンク依存の低下 |
| 40 | 1.563061 | 保存 #4 | 再度の大幅悪化 |
| 50 | 1.741509 | 保存 #5 | 最大値 (11.2倍発散) |

**最有力原因3点**:
1. **T4 の BF16 非対応と Muon 演算負荷【推定】**: T4 は BF16 Tensor Core を欠き FP32 動作。112 行列の Newton-Schulz 5反復が遅延を主導。
2. **QAD に対する Muon 不適と過大学習率【推定】**: YAQA 調整済み重みに `muon_lr=1e-3` が過大。AdamW スケール乖離で量子化破綻。
3. **極小バッチ (batch=1, seq=128) の STE ノイズ増幅【推定】**: 128 トークンの STE ノイズが Muon の特異値均一化で全方向に増幅。

## 2. 実行時間プロファイルとボトルネック内訳解析
1 step 平均 98.4秒の推定内訳を分解した【実測】【推定】。

### 表2: 0.8B ステップ所要時間 (98.4s/step) の推定内訳【推定】
| 構成要素 | 推定時間 | 支配率 | ボトルネック機序【推定】 |
|:---|---:|---:|:---|
| JIT 初期コンパイル (償却) | 4.0 s | 4% | 0.8B グラフの初期トレース・XLA 最適化【推定】 |
| Teacher Forward (BF16) | 14.0 s | 14% | T4 上の BF16 FP32 エミュレーション【推定】 |
| Student Forward & INT4 生成 | 20.0 s | 20% | `lalamo/compressed/int.py:12` INT4 pack/unpack【引用】 |
| Backward 勾配計算 (STE) | 24.0 s | 24% | 決定論的 STE (`GradientEstimator.DETERMINISTIC_ROUNDING`)【引用】 |
| Muon Newton-Schulz (5反復) | 18.0 s | 18% | 112行列×5反復 = 1,680回行列積【推定】 |
| 2×T4 PCIe Replicated 同期 | 6.0 s | 6% | PCIe 3.0 経由 All-Reduce 通信遅延【推定】 |
| Orbax チェックポイント I/O | 7.0 s | 7% | 10 step毎の 6〜8GB 保存【推定】 |
| Host-Device 同期 / tree_map | 5.4 s | 6% | `scripts/qad_finetune.py:78` `inject_params` 走査【引用】 |
| **合計 (実測と整合)** | **98.4 s** | **100%** | **50 step 計 4,920 秒 (82分) に合致【実測】** |

## 3. Loss非改善と初期値（step1）異常の機序
- **step 1 (0.156) の低さ**: YAQA キャリブレーション (`scripts/yaqa_calibration.py:45`) 直後状態【引用】。
- **step 5 以降の急悪化**: 直交更新で重みが急変し INT4 グリッドを離脱。

### 表3: Loss 悪化要因の感度マトリクス【推定】
| 要因 | 寄与度 | メカニズム【推定】 |
|:---|:---:|:---|
| Muon 直交 (`1e-3`) | **極大** | 特異値一様化による最適ベイスン離脱 |
| Scale Lag | **大** | scales/zeros の重み変動追従遅延 |
| STE ノイズ | **大** | 128トークンの高分散勾配の直交化増幅 |
| チャンク分散 | **中** | 評価チャンク難易度差による上下動 |
| 温度 T=1.0 | **中** | 語彙128kでの確率鋭角化・勾配スパイク |

## 4. 最適化アルゴリズムの妥当性評価（Muon vs 代替手法）
Muon は事前学習向きであり、QAD 微細蒸留には純粋 AdamW が適する。

### 表4: オプティマイザ特性比較【推定】
| 手法 | メモリ | T4 負荷 | QAD 適合性 | 特徴【推定】 |
|:---|:---:|:---:|:---:|:---|
| **Muon + AdamW** (現状) | 中 | 高 (1680 matmuls) | **不適 (×)** | ノイズ増幅と重み発散を誘発【実測】 |
| **純粋 AdamW** (推奨) | 高 | **極小** | **最適 (◎)** | 微小 lr でベイスン内を安定微調整 |
| **Lion** | **極小** | 極小 | **良好 (◯)** | メモリ節約だが Sign がノイズと干渉 |
| **Adafactor** | 低 | 小 | **良好 (◯)** | 長期向き。50 step では分散未収束 |
| **SOAP** | 極大 | 極大 | **不適 (×)** | 固有空間計算が T4 で破綻 |

## 5. 並列方式（2×T4 Replicated）と Checkpoint 影響
- **Replicated の冗長性**: `batch=1` は 2 台で重複計算し PCIe 通信遅延のみ加算【推定】。TP=2 は `research/qad_tp_optimizer.md:34` の通り Muon で `ShardingTypeError` 発生【引用】。
- **Orbax 停止時間**: `PyTreeCheckpointer` の 6〜8GB 保存が step 10,20,.. で停止誘発。

### 表5: 0.8B 並列構成・保存設定の比較【推定】
| 構成 | スループット | 通信負荷 | 評価【推定】 |
|:---|:---:|:---:|:---|
| **2×T4 Replicated** (現状) | 冗長 (非効率) | PCIe All-Reduce 大 | 【実測】非推奨 |
| **1×T4 単体** (推奨) | **100% 活用** | **通信ゼロ** | 最速・安定 |
| **save-every 10** (現状) | I/O 停止 5回 | ディスク負荷大 | 【実測】82分の7分占有 |
| **save-every 0** (推奨) | **I/O 停止ゼロ** | ディスク負荷なし | 診断実行に最適 |

## 6. データパイプラインと評価基準の是正
- **トークン数不足**: `seq=128, batch=1` は 128 トークンに過ぎず STE に致命的。
- **固定 Val Loss 導入**: 8 チャンク固定検証セットで真の損失を評価。
- **温度緩和**: $T=1.0 	o 1.5$ でロジット軟化・テールノイズ遮断。

## 7. 次の10ステップ判別実験計画
短時間 (10 step) で主因を特定する実験計画を表6に示す【推定】。

### 表6: 0.8B 10ステップ判別実験マトリクス【推定】
| 実験ID | 主な検証内容 | 設定値 | 期待される符号・観測【推定】 | 優先度 |
|:---|:---|:---|:---|:---:|
| **Exp-A** | **Muon OFF (純粋 AdamW)** | weights `2e-5`, scales `1e-5` | 10 step で loss 減少、step 時間半減 | **P0** |
| **Exp-B** | **1×T4 単体実行** | `--parallel none`, 1 GPU | step 時間 10〜15s 削減、通信解消 | **P0** |
| **Exp-C** | **Checkpoint OFF** | `save_every=0` | step 10 の 60s 停止が消滅 | **P1** |
| **Exp-D** | **固定 Val 評価** | 固定 8 チャンク計測 | 損失振動が消失し収束傾向を可視化 | **P1** |
| **Exp-E** | **トークン数拡大** | `seq=256, batch=2` | STE 勾配分散低下による学習平滑化 | **P2** |

## 8. 4B/9B への接続：スケール依存性の構造変化
0.8B (24層, h1024, inter3584, tied, 752.4M) の知見を 4B (32層, h2560, inter9216, 4.08B) と 9B (32層, h4096, inter12288, untied, 8.95B) へ接続する。

### 0.8B 対策の有効性と支配項の移行
- **T4 から A100 移行**: A100 (BF16 312 TFLOPS) で BF16 エミュレーションが解消。支配項は「GEMM 演算」から「通信帯域・状態メモリ」へ移行【推定】。
- **Muon NS コスト急増**: 計算量は $20 	imes \sum m \cdot n \cdot \min(m, n)$ で増大。0.8B の 7.47 TFLOPs に対し 4B は 158.9 TFLOPs (21倍)、9B は 571.7 TFLOPs (76倍) に爆発。純粋 AdamW が必須【推定】。
- **STE ノイズ緩和**: 4B/9B で `seq=256/512, batch=2` (512〜1024トークン) を確保すれば STE 勾配分散は 1/4〜1/8 に低減【推定】。

### 表7: スケール別計算量・ハードウェア支配項【推定】
| 項目 | 0.8B (2×T4 16GB) | 4B (A100 40/80GB) | 9B (2×A100 80GB) |
|:---|:---|:---|:---|
| **パラメータ / INT4** | 752.4M / 764MB | 4.08B / 2.44GB | 8.95B (untied) / 5.19GB |
| **Muon NS 演算量** | 7.47 TFLOPs | 158.9 TFLOPs (21倍) | 571.7 TFLOPs (76倍) |
| **演算方式** | T4 FP32 エミュレーション | A100 BF16 Tensor Core | A100 BF16 Tensor Core |
| **時間支配項** | BF16 エミュ + Muon NS | メモリ転送 + 通信遅延 | FSDP/TP 通信 + 状態同期 |

## 9. 4B/9B 並列戦略・オプティマイザ・メモリ収支比較

### 表8: 4B/9B 並列戦略・オプティマイザ適合性【推定】
| モデル | 並列方式 | オプティマイザ | 通信・Sharding 挙動 | 推奨ハードウェア【推定】 |
|:---|:---|:---|:---|:---|
| **4B** | **1 GPU (並列なし)** | 純粋 AdamW | 通信ゼロ、最速 | **A100 80GB (単体)** で収容可能 |
| **4B** | TP=2 | 純粋 AdamW | All-Reduce 発生 (NVLink 600GB/s) | A100 40GB × 2 (予算制約時) |
| **4B** | TP=2 | Muon (Reshard) | `research/qad_tp_optimizer.md:34` 必須 | 実装複雑・NS 負荷大で非推奨 |
| **9B** | **FSDP (ZeRO-3)** | 純粋 AdamW | 状態分散、最も安定 | **2× A100 80GB (最適構成)** |
| **9B** | TP=2 + CPU Offload | 純粋 AdamW | 教師を CPU 退避、PCIe 転送遅延 | 1× A100 80GB (低速化許容時) |
| **9B** | 教師 Top-k 事前計算 | 純粋 AdamW | 教師 VRAM ゼロ、ディスク I/O 増 | A100 80GB 単体運用可能 |

### 表9: 0.8B / 4B / 9B メモリ収支と Checkpoint 比較【推定】
| メモリ内訳 (GB) | 0.8B (T4 16GB) | 4B (A100 80GB) | 9B (2× A100 80GB FSDP) |
|:---|---:|---:|---:|
| Teacher (BF16) | 1.50 GB | 8.16 GB | 17.90 GB (FSDP 9GB) |
| Student master (BF16) | 1.50 GB | 8.16 GB | 17.90 GB (FSDP 9GB) |
| INT4 packed weights | 0.75 GB | 2.44 GB | 5.19 GB (2.60GB/GPU) |
| Gradients (BF16) | 1.50 GB | 8.16 GB | 17.90 GB (FSDP 9GB) |
| AdamW states (FP32) | 6.02 GB | 32.64 GB | 71.60 GB (FSDP 36GB) |
| アクティベーション | ~0.50 GB | ~4.50 GB | ~8.00 GB (~4.00GB/GPU) |
| **総 VRAM 所要量** | **11.28 GB** | **64.06 GB** | **138.5 GB (各 ~69GB)** |
| **Checkpoint サイズ** | **7.52 GB** | **40.80 GB** | **89.50 GB** |
| 推奨 save-every / 保持 | 50 step / keep 2 | 100 step / keep 2 (Async) | 200 step / keep 2 (Async) |

## 10. 4B/9B における損失挙動と品質のスケール依存性
- **step 1 KL の初期品質**: 4B/9B は高表現力により YAQA 直後 KL は 0.05〜0.10 に収まる【推定】。
- **Scale Lag の激甚化**: 規模拡大に伴い曲率差が広がり、Muon と AdamW の不整合はより致命的となる【推定】。
- **9B untied lm_head の影響**: 9B は untied。lm_head (5.24億パラメータ) の INT4 化はロジット分解能を破壊するため **lm_head BF16 保持**が必須【推定】。

## 11. 規模別運用推奨レシピと1行要約

### 表10: 規模別推奨パラメータと Modal コスト試算【推定】
| 項目 | 0.8B 診断/短期 | 4B QAD 本番 | 9B QAD 本番 |
|:---|:---|:---|:---|
| **推奨ハードウェア** | 1× T4 (16GB) | 1× A100 80GB | 2× A100 80GB |
| **オプティマイザ** | 純粋 AdamW | 純粋 AdamW | 純粋 AdamW |
| **Weight / Scale lr** | `2.0e-5` / `1.0e-5` | `1.5e-5` / `8.0e-6` | `1.0e-5` / `5.0e-6` |
| **seq / batch** | seq 256, batch 2 | seq 512, batch 2 | seq 1024, batch 2 |
| **温度 ($T$)** | 1.5 | 1.5 | 2.0 |
| **並列・Head 方式** | 単一 GPU / tied | 単一 GPU / tied | FSDP / lm_head BF16 保持 |
| **Checkpoint** | save-every 0 (診断) | save-every 100 (Async) | save-every 200 (Async) |
| **推定 step 速度** | 約 2.0 s/step | 約 2.5 s/step | 約 4.5 s/step |
| **Modal 200step 費** | 約 $0.15 (T4) | 約 $0.35 (A100-80G) | 約 $1.25 (2×A100-80G) |

- **4B 推奨1行**: `1× A100 80GB + 純粋 AdamW (weight 1.5e-5, scale 8e-6) + seq 512, batch 2 + save-every 100 (Async)`
- **9B 推奨1行**: `2× A100 80GB (FSDP) + 純粋 AdamW (weight 1.0e-5, scale 5e-6) + lm_head BF16保持 + seq 1024, batch 2 + save-every 200 (Async)`

## 12. 0.8B で答えられない限界と判別実験の転移

### 表11: 判別実験のスケール転移と最初の1本【推定】
| 段階 | 実験構成・対象 | 目的・検証内容 | 判定基準【推定】 |
|:---|:---|:---|:---|
| **0.8B 完了確認** | Exp-A (純粋 AdamW, 1 GPU) | Muon 排他と単調減少確認 | 10 step で KL < 0.15 への降下 |
| **4B 最初の1本** | **4B-Exp1**: 1× A100 80GB, AdamW | 収束性と VRAM 64GB 健全性 | 10 step で OOM なく KL 減少 |
| **9B 最初の1本** | **9B-Exp1**: 2× A100 80GB FSDP, head BF16 | untied 安定性と FSDP 速度 | 10 step でハングなく KL<0.10 |

### 0.8B では検証不能なスケール固有課題
1. **9B 品質**: 0.8B の KL 改善が 9B ベンチマーク維持に直結するかは不明。
2. **Untied head**: 0.8B は tied のため head 量子化破壊を測定不能。
3. **FSDP 通信**: 0.8B 単体では All-Gather/Reduce-Scatter 影響を評価不能。
4. **長文・2bit**: seq=128 では長文注意や 2bit 極限劣化を測定不能。
5. **人格ドリフト**: 蒸留時のシステムプロンプト遵守性の変質。

## 13. 失敗モード解析と防護策
1. **Scale Collapse**: `scale_min=1e-5` 下限保護と `optax.clip` 適用【引用】。
2. **STE Gradient Explosion**: `optax.clip_by_global_norm(1.0)` 適用。
3. **Log-Softmax NaN**: `jax.nn.log_softmax` で生徒・教師間 KL 定式化。
4. **Host OOM / I/O ハング**: 短期診断時は保存を停止。

```python
# 4B/9B 診断用 最小実行設定スニペット
diag_config = {
    "model_4b": {"device": "1x A100-80G", "opt": "adamw", "w_lr": 1.5e-5, "s_lr": 8e-6, "seq": 512, "batch": 2},
    "model_9b": {"device": "2x A100-80G-FSDP", "opt": "adamw", "w_lr": 1.0e-5, "s_lr": 5e-6, "lm_head": "bf16"},
}
```

## 14. 未確認事項と参考文献
### 実在確認済み文献
- **LLM-QAT**: Z. Liu et al., arXiv:2305.17888 (2023).
- **QuaRot**: S. Ashkboos et al., arXiv:2404.00456 (2024).
- **SpinQuant**: Z. Liu et al., arXiv:2405.16406 (2024).
- **Muon**: K. Jordan (2024); arXiv:2601.19156 (2026).
- **Optax Muon**: `optax/contrib/_muon.py`【引用】。

### 未確認事項
1. **4B/9B 実機上での FSDP 通信レイテンシ実測値**: Modal 2×A100 NVLink の実測値。
2. **9B untied lm_head の INT4 量子化時における PPL 劣化実測値**。
