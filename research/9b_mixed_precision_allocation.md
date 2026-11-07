# Qwen3.5-9B Mixed-Precision Layer Allocation 設計書

## 結論サマリ
- 推奨案: ハイブリッド Mixed-Precision 配分（GA 8層・lm_head・浅深層 MLP は INT8、GDN 24層・中間 MLP・embed は INT4、MTP は INT4）。
- 重み合計: 6.81 GB【推定】（積算 6,809,058,509 B）。設計目標 6.8〜7.3GB 内。
- 8k 文脈総メモリ: 11.57 GB【推定】。設計目標 11.0〜11.6GB 内。
- メモリ余裕: 16.0 GB【引用】に対し 4.43 GB【推定】空き。実効上限 12.5 GB【推定】（OS 3.5 GB【推定】除外）に対しても 4.43 GB【推定】確保。
- 推論速度: 9.55 tok/s【推定】（100 GB/s【実測】× 0.65【推定】÷ 6.81 GB【推定】）。
- 安定性: Trellis 排除、uzu `IntSpec` 準拠。GPU wired 8.07 GB【推定】< 10.7 GB【引用】維持。

## 9B コンポーネント別パラメータと精度別サイズ
safetensors 集計実測値に基づく精度別サイズ試算（GB: 10^9 B 換算）。

| コンポーネント | パラメータ【実測】 | BF16 (2.0B)【引用】 | INT8 (1.094B)【推定】 | INT4 (0.594B)【推定】 | INT2 (0.273B)【推定】 | D4S4 (0.258B)【推定】 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| embed_tokens | 1,017,118,720 | 2.03 GB | 1.11 GB | 0.60 GB | 0.28 GB | 0.26 GB |
| lm_head | 1,017,118,720 | 2.03 GB | 1.11 GB | 0.60 GB | 0.28 GB | 0.26 GB |
| MLP gate+up (32層) | 3,221,225,472 | 6.44 GB | 3.52 GB | 1.91 GB | 0.88 GB | 0.83 GB |
| MLP down (32層) | 1,610,612,736 | 3.22 GB | 1.76 GB | 0.96 GB | 0.44 GB | 0.41 GB |
| GDN in+out (24層) | 1,618,290,048 | 3.24 GB | 1.77 GB | 0.96 GB | 0.44 GB | 0.42 GB |
| GA qkvg+out (8層) | 469,824,000 | 0.94 GB | 0.51 GB | 0.28 GB | 0.13 GB | 0.12 GB |
| MTP head (1層) | 243,300,000 | 0.49 GB | 0.27 GB | 0.14 GB | 0.07 GB | 0.06 GB |
| Vision encoder | 456,000,000 | 0.91 GB | 0.50 GB | 0.27 GB | 0.12 GB | 0.12 GB |
| RMSNorm 等 | 300,000 | 0.6 MB | 0.3 MB | 0.2 MB | 0.1 MB | 0.1 MB |
| text-only 合計 | 8,953,800,000 | 17.91 GB | 9.79 GB | 5.32 GB | 2.45 GB | 2.31 GB |

※ 単価式: INT8=1.09375B (1+2/32+1/32), INT4=0.59375B (0.5+2/32+1/32), INT2=0.2734375B (0.25+2/128+1/128), D4S4=0.2575B (2.06bit)。

## 層別・部位別の感度根拠
線形変換特性・長文脈保持・層深度 U 字特性に基づく感度分析。

| 部位 | 感度 | 根拠論文 | 理由 |
| :--- | :--- | :--- | :--- |
| embedding | 中 | AWQ / QuaRot | Lookup 専用。INT4 + RHT で語彙表現維持。 |
| lm_head | 極高 | SmoothQuant / GPTQ | 出力ロジット直結。vocab 248,320【実測】の崩壊防止に INT8 必須。 |
| early MLP | 高 | arXiv:2603.25284 / SpinQuant | 入力直後特徴抽出（L0-3）。誤差伝播防止に INT8 維持。 |
| late MLP | 高 | arXiv:2608.28003 / DASH-Q | 出力直前統合層（L28-31）。U字終端高感度部位に INT8 維持。 |
| DeltaNet mixer | 低〜中 | arXiv:2609.04098 / arXiv:2412.06464 | GDN 射影は 4bit に強く gate α/β 低感度。線形再帰有界で INT4 適合。 |
| attention | 極高 | QuaRot / SpinQuant | 8層の大域注意機構。長距離想起（Needle）維持に INT8 必須。 |
| MTP | 低 | DASH-Q / QTIP | 投機実行ヘッド（243.3M【実測】）。本体品質影響なく INT4 軽量化。 |

## 推奨 allocation 表
16GB 制約下で品質最大化を図るハイブリッド配分。Trellis は未統合のため不採用。

| コンポーネント | ビット数 | 方式 (RHT/D4S4/Trellis) | 理由 | 期待MB (GB)【推定】 |
| :--- | :--- | :--- | :--- | :--- |
| embed_tokens | 4 bit | RHT (block 32) | Lookup 専用。INT4 g32 asym + RHT で表現保持 | 603.9 MB (0.60 GB) |
| lm_head | 8 bit | RHT (block 32, mode=Input) | ロジット精度保持。untied readout (HybridSpec mode=Input) | 1,112.5 MB (1.11 GB) |
| Early MLP (L0-3: 4層) | 8 bit | RHT (block 32) | 浅層 U 字領域の表現保持 | 660.6 MB (0.66 GB) |
| Late MLP (L28-31: 4層) | 8 bit | RHT (block 32) | 深層 U 字領域の表現保持 | 660.6 MB (0.66 GB) |
| Middle MLP (L4-27: 24層) | 4 bit | RHT (block 32) | 中間冗長層の INT4 化 | 2,151.7 MB (2.15 GB) |
| DeltaNet (GDN: 全24層) | 4 bit | RHT (block 32) | GDN 射影 1.618B。頑健性に基づき INT4 | 960.9 MB (0.96 GB) |
| Attention (GA: 全8層) | 8 bit | RHT (block 32) | 長文脈想起 8層。品質重視 INT8 | 513.9 MB (0.51 GB) |
| MTP head (1層) | 4 bit | RHT (block 32) | 投機実行層。INT4 で軽量化 | 144.5 MB (0.14 GB) |
| Norm / Biases | 16 bit | なし (BF16) | 300,000 パラメータ。RMSNorm は BF16 保持 | 0.6 MB (0.001 GB) |
| **合計 (推奨案)** | - | - | **単価積算 6,809,058,509 B【推定】** | **6,809.6 MB (6.81 GB)** |

## 16GB 制約の試算
Apple M3 16GB の実効上限 12.5 GB【推定】（OS 3.5 GB【推定】除外）の収支試算。

| 項目 | 4k context【推定】 | 8k context【推定】 | 16k context【推定】 | 備考・根拠 |
| :--- | :--- | :--- | :--- | :--- |
| モデル重み (weights) | 6.81 GB | 6.81 GB | 6.81 GB | 推奨ハイブリッド積算値【推定】 |
| DeltaNet state (24層) | 0.05 GB (50.3MB) | 0.05 GB (50.3MB) | 0.05 GB (50.3MB) | 24×32×128×128×4B【推定】文脈長不変 |
| KV cache (GA 8層 FP16) | 0.13 GB (131MB) | 0.26 GB (262MB) | 0.52 GB (524MB) | 32 KB/token × token 数【推定】 |
| 活性化バッファ (activation) | 0.60 GB | 0.60 GB | 0.60 GB | ピークテンソル領域【推定】 |
| Metal ランタイムバッファ | 0.35 GB | 0.35 GB | 0.35 GB | コマンドバッファ・ヒープ領域【推定】 |
| **GPU wired 計** | **7.94 GB** | **8.07 GB** | **8.33 GB** | **8.07 GB < 10.7 GB**【引用】達成 |
| OS 常駐領域 | 3.50 GB | 3.50 GB | 3.50 GB | macOS 常駐 2.8〜3.5GB【推定】 |
| システム総使用量 | 11.44 GB | 11.57 GB | 11.83 GB | 8k 時 11.57 GB は設計目標 11.0〜11.6GB 内 |
| 16GB 対比空き容量 | 4.56 GB | 4.43 GB | 4.17 GB | 既存 swap 6.5GB【実測】の再発回避を狙う【推定】 |
| 12.5GB 実効上限対比余裕 | 1.06 GB | 0.93 GB | 0.67 GB | recommendedMaxWorkingSetSize 安全域 |

## 品質リスクと 2bit の許容範囲
- Trellis mainline 未接続: uzu `parse_spec` は FullPrecisionSpec / MLXSpec / IntSpec のみ受付可能（`crates/uzu-engine/src/encodable_block/weight_matrix.rs:38-58`）。TrellisSpec 未登録で UnsupportedConfiguration となるため本番不可。
- 2bit 許容部位: 中間層 MLP（Layer 4-27）の `gate_proj` および `up_proj` のみ条件付き許容（INT2 g128 asym）。活性化非線形性により雑音吸収可能。
- 絶対禁止部位一覧:

| 部位 | 2bit 判定 | 禁止理由・リスク |
| :--- | :--- | :--- |
| embed_tokens | 否 (D4S4/INT4) | 語彙表現空間の直交性喪失 |
| lm_head | **絶対禁止** | トークン選択ロジット破壊・構文崩壊 |
| GA (Attention 8層) | **絶対禁止** | 長距離想起（Needle）の致命的喪失 |
| GDN (DeltaNet 24層) | **絶対禁止** | 線形状態更新の誤差発散・NaN リスク |
| Early / Late MLP | **絶対禁止** | U 字高感度領域であり PPL 急激悪化 |
| Middle MLP down | 否 (INT4 推奨) | 残差集約パスであり情報損失大 |
| Middle MLP gate+up | **条件付許容** | 高冗長性により INT2 適用可 |

## 代替案 A/B/C
運用要求（品質・速度・省メモリ）に応じた 3 代替案の比較。

| 案名 | 構成方針 | 重み GB【推定】 | tok/s【推定】 | 8k 総メモリ【推定】 | 主なリスク・トレードオフ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **推奨案** | ハイブリッド（GA/lm_head/浅深層 INT8、他 INT4） | 6.81 GB | 9.55 | 11.57 GB | 品質・速度・安全マージン最良バランス |
| **代替案 A** | 品質優先（推奨案 + 全 down INT8、MTP 削除） | 7.47 GB | 8.70 | 12.23 GB | 7.5GB 以下。16k 文脈時に実効上限へ接近 |
| **代替案 B** | 速度優先（全 Linear 層 INT4 + lm_head INT8） | 5.97 GB | 10.89 | 10.73 GB | 5.5〜6.2GB 範囲。GA 4bit 化で長文脈精度微減 |
| **代替案 C** | 省メモリ優先（中間24層 MLP を INT2、embed D4S4、他 INT4、MTP 削除） | 3.81 GB | 17.04 | 8.58 GB | 3.8〜4.3GB 範囲。2bit 化で推論品質低下 |

※ tok/s: 100 GB/s【実測】× 0.65【推定】÷ 重み GB【推定】。推奨案を採用。

## 0.8B 先行 ablation
Qwen3.5-0.8B（text 752.4M【実測】）を用いた 8 段階実験。0.8B は tied（lm_head なし）のため lm_head 分離実験は 4B/9B で実施。

| 順序 | 実験項目 | 構成詳細 | 判定基準・合否ライン |
| :--- | :--- | :--- | :--- |
| 1 | BF16 参照 | 全 BF16（752.4M×2B=1.51GB【推定】） | WikiText-2 PPL / KL の基準値確立。uzu 実測 840MB【実測】は別途照合 |
| 2 | 全層 INT4 均一化 | 全 Linear + embed INT4 g32 asym + RHT | PPL 悪化が BF16 比 +0.4 以内（423.2MB【推定】） |
| 3 | GDN 4bit 耐性 | GDN 射影 INT4、他 INT8 | GDN 起因の PPL 悪化 +0.08 以内、状態ノルム発散なし |
| 4 | 中間 MLP 4/2bit | gate+up INT2、down INT4 | gate+up 2bit 化の PPL 悪化 +0.25 以内 |
| 5 | RHT block 32 効果 | RHT 有効 vs 無効 (INT4) | 外れ値抑制効果と PPL 改善幅（-0.15 以上） |
| 6 | D4S4 embedding | embed_tokens を D4S4 化 | Lookup 速度向上、Cosine 類似度 > 0.98 |
| 7 | KV / State 伸長 | 4k / 8k / 16k キャッシュ負荷 | メモリリーク・Metal ヒープ断片化なし |
| 8 | untied lm_head 検証 | lm_head INT4 vs INT8 (4B/9B) | 0.8B は tied のため 4B/9B で実施。INT8 でロジット歪み防止 |

## 実装・変換手順と未確認事項
### 実装・変換手順
- lalamo 変換: `lalamo/compressed/int.py:72` でテンソルを asym INT4/INT8 量子化。`lalamo/compressed/hybrid.py:32` の `HybridQuantizer` で仕様定義。`lalamo/exportable.py:24` で safetensors 出力。
- uzu spec 構築: 未知 config キー禁止（`crates/uzu-engine-macros/src/uzu_config.rs:183-187`）。`parse_spec`（`crates/uzu-engine/src/encodable_block/weight_matrix.rs:38-58`）の `IntSpec` のみ利用。
- RHT / untied 制御: RHT block 32 固定（`crates/uzu-engine/src/backends/common/gpu_types/hadamard_order.rs:1`）、`ACTIVATION_SCALE_GROUP_SIZE=128`（`crates/uzu-engine/src/backends/common/kernel/activation_transform.rs:20`）。lm_head は `crates/uzu-engine/src/encodable_block/embedding.rs:90-134` および `crates/uzu-engine/src/encodable_block/linear/untied_readout.rs` の `HybridSpec mode=Input`。D4S4 は `crates/uzu-engine/src/encodable_block/embedding_table.rs:104-133` の lookup 限定。

### 未確認事項
- MTP head（243.3M【実測】）推論時投機実行の実効オーバーヘッド【実測不能】。
- Claude 4.6 Opus distill モデル特有の損失に対する各層量子化感度差異【実測不能】。
- Trellis カーネル（`crates/uzu-engine/src/backends/common/kernel/matmul/matmul_b.rs:10-42`）コード部（0.72b/weight）含む全体実効単価と精度【実測不能・未検証】。
- 長時間推論時における macOS メモリコンパクションとスワップ（実測 6.5GB【実測】）の再発性【実測不能】。

## 参考文献
- https://arxiv.org/abs/2609.04098
- https://arxiv.org/abs/2603.25284
- https://arxiv.org/abs/2608.28003
- https://arxiv.org/abs/2211.10438
- https://arxiv.org/abs/2306.00978
- https://arxiv.org/abs/2404.00456
- https://arxiv.org/abs/2405.16406
- https://arxiv.org/abs/2406.11235
- https://arxiv.org/abs/2210.17323
- https://arxiv.org/abs/2604.13806
- https://arxiv.org/abs/2412.06464
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json
- https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize
- https://github.com/ml-explore/mlx/issues/3346
