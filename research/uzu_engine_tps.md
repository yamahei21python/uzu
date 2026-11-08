# uzu エンジンの TPS 構造（代替実装で同等を出すための要件）

2026-11-08 / 対象: `crates/uzu-engine`（Rust + Metal）

目的: uzu の TPS がどこから来ているかを分解し、
**代替エンジンで同じ TPS を出すために必須な要素**を特定する。
既存の `research/uzu_engine_internals.md`（実装の地図）を、速度の観点で掘り下げたもの。

---

## 0. 要約

uzu の decode 速度は、ほぼ **「重みのバイト数 ÷ 実効帯域」** で決まる。
理論上限に対する実測効率は 33〜54%。差を埋めているのは次の 4 つ:

1. **量子化 GEMV が dequant を融合**している（別パスで f16 に展開しない）
2. **RHT を GEMV のプロローグ/エピローグに融合**している
3. **1 トークン = 1 コマンドバッファ**で、dispatch と同期のオーバーヘッドが小さい
4. **投機デコードで「重み 1 回読み出し → 複数トークン」**にできる

代替実装が同等を出すには、1〜3 が必須。4 は倍率を狙う場合のみ。

---

## 1. 1 トークンのコストモデル

Decode（batch 1）は次の和で決まる。

```
t_token = Σ_layers ( bytes_weights_layer / BW_effective )
        + bytes_kv / BW
        + N_dispatch * t_dispatch
        + t_sync
```

- `bytes_weights_layer` = 量子化重み + scale/zero。INT4 group32 なら約 0.578 B/param
- `t_dispatch` = Metal のエンコード〜実行開始の固定費（µs 台）
- `t_sync` = 1 トークンあたりの待ち（uzu は次トークンの encode を先行させる）

**モデルサイズだけで決まる部分が支配的**。だから量子化のビット数・group が直接効く。

### 実測と理論（M3 16GB、帯域 ~100GB/s）

| モデル | 重み | 理論上限 | 実測 | 効率 |
|---|---:|---:|---:|---:|
| 4B INT4 | 2.40GB | 41.7 t/s | 14.0（EN）/ 18.6（JA） | 34〜45% |
| 9B INT4 | 5.19GB | 19.3 t/s | 6.3（EN）/ 11.5（JA） | 33〜60% |

JA が速いのは 1 トークンあたりの文字数（fertility）が高く、
**「トークン/秒」が同じでも体感が違う**ため。TPS 比較は同一言語で行う。

---

## 2. uzu が速い理由（コード根拠）

### 2.1 量子化 GEMV に dequant を融合

- `backends/metal/kernel/matmul/gemv/gemv.metal`
  - `GemmBPrologueKind::{ScaleBiasDequant, ScaleZeroPointDequant, ScaleSymmetricDequant}`
  - `BITS` は 0/2/4/8、`GROUP_SIZE` は 0/32/64/128 を variant として特殊化
- 重みを f16 に展開する中間バッファを作らない → 帯域が「量子化バイト数」で済む

**代替実装の必須条件**: dequant を matmul 内で行い、展開済み重みを保持しない。

### 2.2 RHT の融合

- `backends/common/gpu_types/hadamard_order.rs`（ブロック長 32）
- `encodable_block/linear/rht_wrapper.rs`
- RHT を「Norm」または「GEMM epilogue（D-Transform）」に直結

**必須条件**: 量子化形式が RHT 前提なら、変換を別カーネルにしない（1 パス増える）。

### 2.3 dispatch と同期の削減

- `engine/language_model/stream/stream.rs:380` 以降
  - `DecodingState::ForwardPassPending` / `InFlight` で**次トークンの forward を先行 encode**
  - `CommandBuffer::create` をトークン単位で 1 回
- `backends/metal/command_buffer.rs` に copy/fill の blit 融合、alignment variant

**必須条件**: 層ごとに command buffer を分けない。CPU 側の encode と GPU 実行を重ねる。

### 2.4 GEMV/GEMM の切り替え

- `gemv/policy/quantized.rs`: `m ≤ 4` は GEMV、`m > 4`（投機の木検証）は tiled GEMM
- `gemv/kernel.rs`: `MXU_TILED_KERNEL_MIN_K = 2048`、`k >= 2048` かつ BF16 重みで MXU 経路

**必須条件**: batch 1 用の GEMV と、投機検証用の GEMM を両方持つ。

### 2.5 投機デコード

- `speculators/dflash_tfm.rs`, `encodable_block/weaver.rs`
- 木検証は `m > 4` の GEMM に落ちるので、**重み読み出し 1 回で複数候補を評価**できる
- 受理率が高いほど TPS が上がる（理論上限を超えられる唯一の手段）

**必須条件（倍率狙いの場合）**: draft モデル + 木検証 + 受理ロジック。

### 2.6 unified memory

- Apple Silicon では重みが CPU/GPU 共有。コピー無しで読める
- 代替実装が CUDA 前提だと、ここで不利（PCIe 転送）

---

## 3. 代替実装の同等 TPS チェックリスト

| # | 要件 | 必須度 | 備考 |
|---|---|---|---|
| 1 | 量子化 GEMV（4bit group32、dequant 融合） | **A** | これが無いと 2〜3 倍遅い |
| 2 | RHT の融合（別パス禁止） | **A** | 形式が RHT 前提なら必須 |
| 3 | 1 トークン 1 command buffer + 先行 encode | **A** | dispatch 律速を避ける |
| 4 | GEMV / GEMM の batch 別切り替え | **A** | batch 1 と投機検証の両立 |
| 5 | KV cache の量子化（INT8/FP8） | B | 長文で効く。短文では誤差 |
| 6 | 投機デコード（draft + 木検証） | B | 最大 2 倍。draft の学習コスト |
| 7 | Metal（or 同等の低オーバーヘッド API） | A | WebGPU は dispatch が重い |
| 8 | safetensors の mmap（全読みしない） | B | 起動時間。TPS には無関係 |

**A が揃わないと uzu の TPS に届かない。** B は「超える」ための要素。

---

## 4. uzu 側の既知の制約（代替が差別化できる点）

| 制約 | 根拠 | 代替での扱い |
|---|---|---|
| BF16 固定（活性） | `engine/language_model/mod.rs` | f16 で帯域を半分にできる可能性（要検証） |
| RHT はブロック 32 固定 | `hadamard_order.rs` | 64/128 で演算量削減（品質影響要確認） |
| MoE 未対応 | internals の記載 | — |
| safetensors 全読み込み | internals の記載 | mmap で起動改善 |
| 投機の draft がモデル依存 | `speculation_supported()` | 自前 draft が必要 |

---

## 5. 計測プロトコル（TPS を語るときの前提）

- decode t/s = `(N_out − 1) / (T_last − T_first)`（**第 1 トークンを除外**）
- prefill と decode を分ける
- 同一プロンプト・同一 max_tokens・同一言語
- 5 回の中央値。MLX 系は `mx.eval()` の同期漏れで数倍過大になるので注意
- 参考: `research/apple_silicon_benchmarking.md`

---

## 8. 実測プロファイル（2026-11-08 / MacBook Air M3 16GB）

### 方法

- `target/release/cli bench <model> <task> <out>` で TPS を取得（UI 無し）
- `xctrace record --template 'Metal System Trace'` で command buffer を取得
- `xctrace export` で `metal-application-command-buffer-submissions` / `metal-gpu-intervals` を XML 化して集計

### 結果

| モデル | 使用メモリ | generate t/s | 実効帯域 | prompt t/s |
|---|---:|---:|---:|---:|
| 0.8B INT4 | 0.92 GB | **87.9** | **81 GB/s** | 681 |
| 9B INT4 | 5.37 GB | **15.3** | **82 GB/s** | 96.8 |

**両モデルで実効帯域が 81〜82 GB/s に揃う。** M3 の公称 ~100GB/s に対して **82%**。
decode は完全に帯域律速で、per-token 時間 = 重みバイト数 ÷ 82GB/s でほぼ説明できる。

### 計測の落とし穴（重要）

command buffer の `duration` は **GPU 実行時間ではない**。

| 指標 | 0.8B | 9B |
|---|---:|---:|
| CB duration（中央値） | 2.59 ms | 3.77 ms |
| 実 wall / token | 11.4 ms | 65 ms |

CB duration は encode〜submit の側で、GPU 実行は wall の大半を占める。
これを GPU 時間と誤認すると「GPU 稼働率 20%、CPU がボトルネック」と誤結論する。
実際は CPU 側は 1 トークンあたり数 ms で、帯域が支配的。

### 実効帯域から見た上限

| 構成 | 0.8B | 9B |
|---|---:|---:|
| BF16 | 1.6GB → 約 51 t/s | 18GB → 約 4.6 t/s |
| INT4 uniform | 0.58GB → 約 141 t/s（理論）/ 実測 88 | 5.0GB → 約 16.4 / 実測 15.3 |
| INT4 wide8 | 0.66GB → 約 124 t/s（理論） | 5.24GB → 約 15.6 |
| INT4 + 投機（受理 1.5 tok/pass） | ×1.5 | ×1.5 |

- 4bit の実測が理論の 62% なのは、実効帯域 82% と KV/活性のぶん
- **8bit 化は理論値で −8〜14%**（サイズ増のぶんだけ素直に遅くなる）
- 帯域 82% を 90% に上げられれば +10% だが、カーネル改善の余地は小さい

### 結論の更新

1. uzu の decode は **帯域律速（実効 82GB/s）**。CPU はボトルネックではない
2. TPS を上げる手段は **「1 トークンあたりの読み出しバイトを減らす」か「1 回の読み出しで複数トークン出す」** の2つだけ
3. 前者は 4bit 維持・8bit 最小化・group 64（メタデータ減）
4. 後者は投機デコード。**帯域の壁を越える唯一の手段**

---

## 6. 未調査（次の深掘り候補）

1. `gemv.metal` の内側ループ（lane staging 8 bytes の実装）と実効帯域の実測
2. KV cache の量子化形式（INT8/FP8）と長文での効き方
3. `Weaver` の受理率（draft 品質）と実 TPS の関係
4. command buffer の encode 時間（CPU 側が律速になる境界）
5. f16 活性の可否（BF16 固定を外したときの数値影響）

---

## 7. 結論

- uzu の TPS は **「量子化バイト数 × 融合カーネル × dispatch 削減」** で説明できる
- 代替実装で同等を狙うなら、**A の 4 項目（dequant 融合 GEMV / RHT 融合 / 単一 CB / GEMV-GEMM 切替）が必須**
- それだけで理論上限の 35〜50% 程度までは到達する見込み
- 上回るには **投機デコード**か **f16 活性**が候補
