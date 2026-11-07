# uzu 推論エンジン内部実装 調査レポート

| 調査項目 | 実装の要点 | 主要ファイル・シンボル |
| :--- | :--- | :--- |
| **実行階層** | Engine → Decoder → Transformer → Layer → Mixer/MLP → Linear | `crates/uzu-engine/src/engine/language_model/mod.rs:27` |
| **DeltaNet/Attn** | 短期Conv+SSM更新。全ノード並列Gram逆行列木検証。INT8/FP8量子化KV | `crates/uzu-engine/src/encodable_block/mixer/delta_net.rs:51` |
| **投機・Trie統合** | DFlash+Weaver候補木生成、線形化Trieと文法ビットマスク検証 | `crates/uzu-engine/src/speculators/dflash_tfm.rs:87` |
| **バックエンド** | Metal 4 (`MTL4CommandBuffer`) / Slang (WebGPU) / CPU。2冪PoolAllocator | `crates/uzu-engine/src/backends/metal/command_buffer.rs:49` |
| **GEMV/GEMM境界** | $M \le 4$ は量子化GEMV、$M > 4$ (木検証) はMxuTiled GEMMでMXU稼働 | `crates/uzu-engine/src/backends/metal/kernel/matmul/gemv/policy/quantized.rs:43` |
| **RHT融合** | ブロック長 【実測】32 固定。Norm または GEMM Epilogue (D-Transform) 直結融合 | `crates/uzu-engine/src/backends/common/gpu_types/hadamard_order.rs:1` |
| **既知の制約** | BF16固定、RHTはInputOutput/32のみ、MoE未対応、safetensors全読込強制 | `crates/uzu-engine/src/engine/language_model/mod.rs:73` |

---

## 1. モジュール構成と実行フロー

### 主要モジュール・構造体・関数一覧

| 階層 | 主要 Struct / Trait | 主要関数・メソッド | 実装ファイル・行番号 |
| :--- | :--- | :--- | :--- |
| **Engine** | `Engine<B>` | `Engine::new`, `peak_memory_usage` | `crates/uzu-engine/src/engine/mod.rs:19`, `:33` |
| **LanguageModel** | `LanguageModel<B>` | `Engine::load_language_model` | `crates/uzu-engine/src/engine/language_model/mod.rs:27`, `:58` |
| **Stream / Runner** | `LanguageModelStream<B>` | `LanguageModelStream::generate` | `crates/uzu-engine/src/engine/language_model/stream/stream.rs:151`, `:380` |
| **Decoder** | `Decoder<B>` | `Decoder::encode` | `crates/uzu-engine/src/encodable_block/decoder.rs:36`, `:147` |
| **Transformer** | `Transformer<B>`, `TransformerState<B>` | `Transformer::encode`, `create_empty_state` | `crates/uzu-engine/src/encodable_block/transformer.rs:101`, `:242` |
| **TransformerLayer** | `TransformerLayer<B>` | `TransformerLayer::encode` | `crates/uzu-engine/src/encodable_block/transformer_layer.rs:69`, `:375` |
| **TokenMixer** | `Mixer<B>` (Trait), `MixerState<B>` | `Mixer::encode`, `MixerState::encode_accept` | `crates/uzu-engine/src/encodable_block/mixer/mod.rs:27`, `:60` |
| **MLP** | `Mlp<B>` (Trait), `DenseMlp<B>` | `DenseMlp::encode`, `new` | `crates/uzu-engine/src/encodable_block/mlp/mod.rs:26`, `dense.rs:17` |
| **Linear** | `Linear<B>` (Trait), `LinearMatmul<B>` | `Linear::encode_input`, `LinearMatmul::load` | `crates/uzu-engine/src/encodable_block/linear/mod.rs:35`, `matmul.rs:33` |
| **RHT Wrapper** | `RHTLinearWrapper<B>` | `RHTLinearWrapper::encode_input` | `crates/uzu-engine/src/encodable_block/linear/rht_wrapper.rs:28`, `:139` |
| **WeightMatrix** | `WeightMatrix<B>`, `ParsedWeightSpec` | `WeightMatrix::new` | `crates/uzu-engine/src/encodable_block/weight_matrix.rs:33`, `:96` |

### 実行フロー
1. **初期化**: `Engine::load_language_model` (`crates/uzu-engine/src/engine/language_model/mod.rs:58`) が重み・構造定義をロードし、各層の `TransformerLayer` を初期化。
2. **推論ループ (`stream.rs:380`)**:
   - `DFlashTfmSpeculator::propose_tree` (`crates/uzu-engine/src/speculators/dflash_tfm.rs:176`) で投機木を生成・平坦化。
   - `BatchTopology::new` (`crates/uzu-engine/src/encodable_block/batch_topology.rs:11`) で木トポロジを構築。
   - `Decoder::encode` (`crates/uzu-engine/src/encodable_block/decoder.rs:147`) → `Transformer::encode` で全層を一括実行。
   - `Embedding::encode_readout` (`decoder.rs:185`) で Logits 計算、`Sampling::encode` (`sampling/mod.rs:100`) で一括サンプリング。
   - `FlatTrie::accept` (`crates/uzu-engine/src/trie.rs:268`) で合致列を受託し、`MixerState::encode_accept` で状態確定。

---

## 2. コアレイヤ・演算の実装方式

### 各コンポーネントの構造と動作比較

| コンポーネント | 内部保持状態 | Prefill処理 | 通常Decode ($M=1$) | 投機木検証 ($M>1$) |
| :--- | :--- | :--- | :--- | :--- |
| **DeltaNet (GDN)** | `conv_state`, `ssm_state` (`delta_net.rs:51`) | `conv_pack` + `conv_scan`, `chunked.encode` (`delta_net.rs:510`) | `conv_update` + `delta_net_update` 融合 (`delta_net.rs:483`) | `tree_prep` + `TreeVerify` カーネル (`tree_verify.metal:30`) |
| **Gated Attention** | `KVCache` (`codes`, `scales`) (`attention/state.rs:161`) | シーケンス一括投影、KV追記 (`attention/mod.rs:334`) | 単一 `prepare` + `vector.metal` (GEMV) | `ancestor_attention.metal` 祖先参照 (`ancestor_attention.metal:18`) |
| **RMSNorm** | ステートレス（重み、Hadamard係数） | シーケンス長並列。残差加算融合 | バッチ並列実行 | バッチ次元（ノード数）並列実行 |
| **RoPE** | CPU事前計算 Cos/Sin バッファ (`rope.rs:6`) | 全入力位置一括割り当て | 単一位置周波数参照 | ノード深さ (`height`) に応じた周波数適用 |
| **MLP** | ステートレス（Up/Gate/Down 重み） | バッチ GEMM + SwiGLU / GeGLU 乗算 | GEMV または Small GEMM | 木ノード数 $M$ のバッチ GEMM 実行 |

- **DeltaNet 状態更新**: Decode ($M=1$) は短畳み込みと SSM 更新 $S_t = \alpha_t S_{t-1} + \beta_t v_t k_t^T$ を実行 (`delta_net.rs:483`)。Prefill は 【実測】`CONV_TILES = 16`, `CONV_MIN_TILE_TOKENS = 16` (`delta_net.rs:34-35`) で並列スキャン。木受託時は `StateAdvanceKernel` (`delta_net.rs:102`) で採用パスのみ前進更新。
- **Gated Attention & KV**: `KVCache` はコード値 (INT8/FP8) とスケール値 (BF16) を分離管理 (`attention/state.rs:163`)。受託時は `KVCacheState::accept` (`state.rs:105`) が `Copy` を発行し、`KVCacheUpdateKernel` で詰め替え。

---

## 3. バッチトポロジ・Trie・推論パイプライン統合

### 構成要素と責務

| モジュール | 主要型 / 特性 | 責務とアルゴリズム | 実装コード位置 |
| :--- | :--- | :--- | :--- |
| **BatchTopology** | `BatchTopology<'a>` | 木の親子関係 (`parents: Box<[i32]>`) および線形フラグを管理 | `crates/uzu-engine/src/encodable_block/batch_topology.rs:3` |
| **TrieNode / FlatTrie** | `TrieNode`, `FlatTrie<'a>` | 投機木トークンを格納し、深さ優先順で平坦化 | `crates/uzu-engine/src/trie.rs:24`, `:39` |
| **DFlashTfmSpeculator** | `DFlashTfmSpeculator<B>` | DFlash ブロック並列ドラフト + Weaver 木探索 | `crates/uzu-engine/src/speculators/dflash_tfm.rs:87` |
| **Grammar** | `Grammar` (`xgrammar`) | JSON Schema / 正規表現に沿った語彙ビットマスク生成 | `crates/uzu-engine/src/engine/language_model/grammar/grammar.rs:6` |
| **Sampling** | `Sampling<B>` | Temperature / Top-K / Top-P / Penalty / Mask 一括適用 | `crates/uzu-engine/src/encodable_block/sampling/mod.rs:26`, `:46` |

- **統合サイクル**: DFlash がブロック候補生成 (`dflash_tfm.rs:256`) → Weaver 木構築・平坦化 (`dflash_tfm.rs:301`) → `FlatTrie::fill_bitmasks` (`trie.rs:224`) 文法マスク付与 → `BatchTopology` で 1 回検証 (`stream.rs:649`) → `FlatTrie::accept` (`trie.rs:268`) で受託し状態確定。

---

## 4. バックエンド抽象化とハードウェア最適化

### バックエンド比較

| バックエンド | コマンドバッファ実装 | メモリ・バッファ管理 | カーネル記述 | 対象プラットフォーム |
| :--- | :--- | :--- | :--- | :--- |
| **Metal** | `MTL4CommandBuffer` (`command_buffer.rs:49`) | `MetalHeaps` (`MTLSparsePageSize`) + `PoolAllocator` | MSL (`.metal`) | Apple Silicon (macOS / iOS) |
| **Slang / WebGPU** | WebGPU コマンドエンコーダ | WebGPU Buffer 管理 | Slang (`.slang`) → WGSL/SPIR-V | Web ブラウザ / 各種 GPU |
| **CPU** | 同期エミュレーションバッファ | ホストヒープ (`Vec<u8>`) | Pure Rust | テスト・フォールバック |

- **PoolAllocator**: 【実測】$2^0 \sim 2^{31}$ バイトの 32 本の Free List (`pages: [Mutex<VecDeque<T>>; 32]`, `allocator/pool.rs:8`) で一時確保コストを排除。
- **重みレイアウト変換**: ロード時に符号反転 (`*word ^= 0x8080_8080_8080_8080`, `matmul_metal_kernel.rs:284`) を行い、MXU 向け `GemmBLayout::MxuTiled` (`backends/metal/gpu_types/gemm.rs:17`) へ再配置。

---

## 5. 性能特性とボトルネック分析

### GEMV / GEMM 選択境界とハードウェア特性

| 実行モード | バッチサイズ $M$ | 選択カーネル | ボトルネック要因 | MXU 行列ユニット利用 |
| :--- | :--- | :--- | :--- | :--- |
| **通常 Decode** | $M = 1$ | `gemv_mxu_tiled.metal` / `gemv.metal` | メモリ帯域幅飽和 (Memory-BW Bound) | 条件付可 (`mxu_tiled_kernel`) |
| **小バッチ / 狭木** | $2 \le M \le 4$ | `gemv.metal` (Quantized GEMV) | メモリ帯域幅 + 起動オーバーヘッド | 不可 (SIMDGroup リダクション) |
| **投機木検証** | $M \ge 8$ (典型的 【実測】16〜64) | `gemm.metal` (`GemmBLayout::MxuTiled`) | 演算スループット (Compute Bound移行) | **フル活用可能** |
| **Prompt Prefill** | $M \gg 64$ | `gemm.metal` (大タイル GEMM) | 演算器稼働率・Tiling効率 | **フル活用可能** |

- **GEMV の制限**: `crates/uzu-engine/src/backends/metal/kernel/matmul/gemv/policy/quantized.rs:43` にて `if !bf16_io || bits == 2 || (m > 4 && n > 64) { return None; }` とされ、量子化重みでは $M > 4$ で GEMM へフォールバックする。
- **RHT 融合**: 外れ値を直交ランダム回転で分散させ、INT4/INT8 量子化精度を担保。`GemmDTransform::RHT` (`gemm.rs:19`) または `Normalization::encode` (`normalization.rs:47`) 内で直結計算し、VRAM 帯域浪費を排除。

### Qwen3.5-0.8B と Qwen3.5-9B における実測対応と分析

| 評価項目 | Qwen3.5-0.8B (小型モデル) | Qwen3.5-9B (中型モデル) |
| :--- | :--- | :--- |
| **隠れ層次元 / 重み容量 (INT4)** | 【推定】$D$ 約 1024〜2048 / 約 0.5 GB | 【推定】$D$ 約 4096 / 約 5.0 GB |
| **ベースライン通常速度** | 【推定】80〜120 tok/s | 【引用】約 29 tok/s (M5 帯域 【推定】~150GB/s 上限) |
| **投機推論時速度** | 【推定】120〜150 tok/s | 【引用】**92.1 tok/s** (M5 MacBook Pro) |
| **速度向上倍率** | 【推定】約 1.2〜1.4x | 【引用】約 **3.17x** (約 4x レベルの大幅加速) |
| **主要ボトルネック** | カーネルディスパッチレイテンシ、投機の相対負荷 | メモリ帯域幅飽和。投機による GEMM 化が劇的な効果を発揮 |
| **平均受託トークン数** | 【推定】4.0〜5.5 トークン/パス | 【引用】平均 **7.5 トークン/パス** |

---

## 6. エンジン拡張ポイントと実装ガイド

### 拡張項目と具体的挿入位置

| 拡張要件 | 変更ファイル | 対象 struct / 関数 | 具体的な実装アプローチ |
| :--- | :--- | :--- | :--- |
| **Raw Logits API** | `crates/uzu-engine/src/engine/language_model/stream/stream.rs` | `generate`, `DecodingState` | `decoder_output.logits` (`stream.rs:657`) を `context.copyout::<f32>` で取得する `generate_with_logits` を追加 |
| **Prefill 全文評価** | `crates/uzu-engine/src/encodable_block/decoder.rs` | `Decoder::encode` | `output_range` 引数 (`decoder.rs:151`) に `Some(0..batch_dim.size())` を渡し全トークンの Logits を計算 |
| **新量子化形式追加** | `crates/uzu-engine/src/config/weight_matrix/`, `backends/metal/kernel/matmul/` | `AnyWeightMatrixSpec`, `quant_unpack.h` | 独自 Spec 定義、`LinearMatmul::load` への分岐追加、Metal 側 dequant アンパック関数の追加 |
| **MTP 追加** | `crates/uzu-engine/src/speculators/`, `prediction_head.rs` | 新設 `mtp.rs`, `PredictionHead` | `Transformer::encode` の `hidden_feature_layer_indices` (`transformer.rs:302`) を受ける MTP Speculator を実装 |
| **評価用 CLI コマンド** | `crates/legacy/cli/src/main.rs`, `bench/runner.rs` | `Commands`, `BenchRunner::run` | `Commands::EvalLogits` サブコマンドを追加し、`bench/runner.rs:42` で Perplexity を算出 |

---

## 7. 既知の制約事項・落とし穴

### 制約事項一覧とコード根拠

| 制約項目 | 具体的な内容 | 違反時の挙動 | コード根拠（ファイル・行番号） |
| :--- | :--- | :--- | :--- |
| **RHT 設定制約** | 処理モードは `InputOutput`、ブロック長は 【実測】32 のみ対応 | `UnsupportedConfiguration` エラー | `crates/uzu-engine/src/encodable_block/linear/mod.rs:117-134` |
| **データ型固定** | モデル全体の実行データ型が `BF16` にハードコード固定 | コンパイル・初期化時に他型不可 | `crates/uzu-engine/src/engine/language_model/mod.rs:73` |
| **GEMV I/O 制約** | GEMV カーネルの入出力型が BF16 かつ転置 B のみ許可 | GEMV 不成立、GEMM へフォールバック | `crates/uzu-engine/src/backends/metal/kernel/matmul/gemv/kernel.rs:94` |
| **MoE 未対応** | `MixtureOfExpertsConfig` が読み込まれると即座にエラー | 初期化失敗 | `crates/uzu-engine/src/encodable_block/mlp/mod.rs:40`, `:60` |
| **厳格な Serde** | 想定外フィールドが存在すると `Unsupported` 型によりパース失敗 | デシリアライズエラー | `crates/uzu-engine/src/utils/strict_serde.rs:5`, `config/model/generation.rs:3` |
| **全テンソル読込強制** | safetensors 内に 1 つでも未読込のテンソルが残っていると停止 | `ParameterLoaderError::UnreadTensors` | `crates/uzu-engine/src/safetensors.rs:110-124` |
| **接尾辞容量固定** | Attention State の接尾辞バッファ容量が 【実測】1024 に固定 | 容量超過時の不正動作 | `crates/uzu-engine/src/encodable_block/mixer/attention/state.rs:12` |

---

## 8. 未確認事項

1. **Slang / WebGPU バックエンド完成度**: `build/slang/compiler.rs` と `.slang` カーネルはあるが、ブラウザ実機での INT4/RHT 実行性能・安定性は未確認。
2. **極小ビット (2-bit / Trellis) 量子化**: `gemm_trellis.metal` 等はあるが、Qwen3.5-9B での Perplexity 劣化と Metal 4 実測特性は未確認。
3. **Multi-GPU / 分散推論**: 単一 Unified Memory 前提であり、Tensor/Pipeline Parallelism への対応は未確認。
4. **Weaver 枝刈りパラメータ (`prune_sigma`) 感度**: デフォルト値 【実測】1.5 (`crates/uzu-engine/src/speculators/dflash_tfm.rs:76`) の 0.8B/9B 受託率への最適値カーブは未確認。
