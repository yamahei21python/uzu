# Apple Silicon (M3 16GB) LLM 量子化ベンチマーク計測仕様

Apple Silicon (M3 16GB) で JAX ([lalamo](https://github.com/trymirai/lalamo)) と Rust/Metal ([uzu](https://github.com/trymirai/uzu)) のメモリ・速度を比較計測する仕様。

> **数値タグ**: **【実】** 本機実測 (M3 / 16GiB / macOS 26.5.2, 2026-11-07) ・ **【推】** 推定 (計算式併記) ・ **【引】** 外部引用 (URL 付)。

---

## 1. macOS メモリ計測と「実使用メモリ」の定義

UMA で CPU/GPU が物理 RAM を共有 → 指標の使い分け必須。

### 1.1 メモリ計測指標の比較

| 指標 (`コマンド`) | 取得対象 | デメリット | 判定 |
| :--- | :--- | :--- | :--- |
| **Footprint** (`footprint -p PID`) | ダーティ匿名 + 圧縮 + Metal ダーティ | Xcode CLT 要。瞬間値 | **プロセス実使用に最善** |
| **Metal** (`currentAllocatedSize`) | バッファ/テクスチャ/ヒープ | CPU ヒープ除外 | **GPU 観点で最善** |
| **RSS** (`ps -o rss`) | 常駐ページ総量 | クリーン mmap 含む・圧縮除外 | **不適（過大）** |
| **`vm_stat`** | Mach VM 統計 | 全体値・16KB ページ | 健全性確認 |
| **Swap** (`sysctl vm.swapusage`) | SSD swap | プロセス別不明 | **必須** |

### 1.2 内訳とキャッシュ
- **Wired** = カーネル/Metal 固定領域（swap 不可）。**Active/Inactive** = 使用中/解放可。**Compressed** = RAM 内圧縮。
- **Page Cache**: `safetensors`/`gguf` の `mmap` クリーンページは破棄可 = Footprint 外。Metal バッファ分のみ実使用。
- **swap 検出**: `sysctl vm.swapusage` の `used > 0` で速度劣化確定。**【実】** 検証機は測定時 `total=7168M, used=6544M` = 常時 swap 状態。ベースラインは再起動推奨。
- ページサイズ **【実】** `vm.pagesize=16384` → `vm_stat` の値は ×16384 でバイト換算。

---

## 2. JAX (lalamo) / Metal のピークメモリ計測

### 2.1 JAX/XLA 環境変数と API

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # 全メモリ事前確保(75-90%)を無効化
export XLA_PYTHON_CLIENT_ALLOCATOR=platform # プラットフォーム標準アロケータを使用
```

| 手法 / API | 対象 | 注意点 |
| :--- | :--- | :--- |
| `jax.live_arrays()` | 生存 `jax.Array` の `sum(nbytes)` | XLA 一時バッファは除外 |
| `jax.profiler` | `start_trace()`/`stop_trace()` | Perfetto でタイムライン可視化 |
| `jax.devices()[0].memory_stats()` | JAX デバイス統計 | `jax-metal` では未実装の可能性（§8-1） |
| `MTLDevice.currentAllocatedSize` | Metal 現在確保バイト数 | ポーリングでピーク捕捉 |
| `recommendedMaxWorkingSetSize` | 推奨上限 **【推】** 16GiB×0.7〜0.75 ≒ 11.2〜12GB | 超過で性能低下・swap |

### 2.2 0.8B / 9B モデル試算 (M3 16GB)
推奨計測手順（JAX）:
```python
import jax  # 事前に PREALLOCATE=false / ALLOCATOR=platform を export
out = model.apply(...); jax.block_until_ready(out)   # 非同期実行の同期 必須
print(sum(a.nbytes for a in jax.live_arrays()) / 2**30, "GiB")
```
```bash
# ピーク捕捉: 別プロセスで 200ms ポーリング
while :; do footprint -p $(pgrep -f lalamo) | grep -m1 phys_footprint; sleep 0.2; done
```
0.8B は上記で十分。9B は BF16 不可、INT4/YAQA のみ測定可。
**【実】** `hw.memsize=17179869184`（=16GiB）。OS/WindowServer 常時 3〜4GiB → 実質空き **約 11〜12GiB**。

重み概算式 `params × bytes/param`（BF16=2B、INT4/YAQA≈0.56B 実効）。KV は `2 × layers × kv_heads × head_dim × ctx × 2B`。

| 規模 | 量子化 | 重み概算 | KV (ctx=2048) | 合計推定 | 判定 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.8B** | BF16 | **【推】** 0.8e9×2 = 1.6GB | ~0.3GB | ~1.9GB | 余裕 |
| **0.8B** | INT4 / YAQA | **【推】** 0.8e9×0.56 = 0.5GB | ~0.3GB | ~0.8GB | 余裕 |
| **9B** | **BF16** | **【推】** 9e9×2 = **18.0GB** | ~1.5GB | ~19.5GB | **不可**（swap/SIGKILL） |
| **9B** | INT4 / YAQA | **【推】** 9e9×0.56 = 5.0GB | ~1.5GB | ~6.5GB | 安定 |

> **推奨**: 9B BF16 は 16GiB 超過。主軸 INT4 / YAQA、BF16 は swap 監視付き限界試験のみ。

---

## 3. uzu のメトリクス実装の特定

[trymirai/uzu](https://github.com/trymirai/uzu) のソースコード調査結果。

### 3.1 "memory used" の実態
**【引】** `memory used` は OS RSS ではなく **Metal の `currentAllocatedSize`**。

- **実装箇所**（行番号はローカル clone で一致確認）:
  - `crates/uzu-engine/src/backends/metal/buffer/dense.rs:37` / `heaps.rs:134`
```rust
context.peak_memory_usage.fetch_max(context.device.current_allocated_size(), Ordering::Relaxed);
```
  - `metal/context.rs:229` `peak_memory_usage()` が Atomic を返却
  - `legacy/nagare/src/chat/token.rs:184` `state.memory_usage = instance.peak_memory_usage();`
  - `chat/token.rs:522` → `ChatReplyStats` の `memory_usage`
  - `legacy/cli/.../history_cell.rs:553` `format_memory_used`
- **内訳**: Metal の重み + KV + 中間バッファのみ。**ホスト CPU ヒープ・OS キャッシュ・mmap クリーンページは含まない**。
- **単位の罠**: `format_memory_used` は `bytes/1024³`（=GiB）を **"GB"** と表示。数値は GiB として読む。

### 3.2 速度メトリクス（`chat/token.rs:482-552`、要約）
```rust
let ttft_duration = time_first_token.map(|t| t.duration_since(self.time_prefill_start));
let prefill_tps = self.total_tokens_input as f64 / ttft_duration.as_secs_f64();
let generate_tps = calculate_rate(self.total_tokens_output, generate_duration);
// calculate_rate: tokens < 2 は None、それ以外は (tokens - 1) / secs
```
- **prefill_tps = prompt トークン数 / TTFT**（初回 forward 込み）
- **generate_tps = (出力トークン数 − 1) / 生成時間**（第1トークン＝prefill 分を除外）

### 3.3 `TokenStreamMetrics` の構造
```rust
// 要約: crates/legacy/shoji/src/traits/backend/chat_token.rs:18-26
pub struct TokenStreamMetrics {   // フィールドは全て usize
    num_prefill_forward_passes, num_decode_forward_passes,
    num_tokens_prefilled, num_tokens_proposed,
    num_tokens_accepted, num_tokens_returned,
}
```
投機的デコード時 `tokens_per_forward_pass = num_tokens_accepted / num_forward_passes`。

---

## 4. 速度計測の定義と公平比較

### 4.1 指標の厳密な定義

| 指標 | 定義式 | 留意点 |
| :--- | :--- | :--- |
| **TTFT** | `T_first − T_start` | Prompt 評価 + 初回 token 生成。tokenize と Metal PSO 初回コンパイル込み |
| **Prefill t/s** | `N_prompt / T_prefill` | プロンプト長で並列度が変動 |
| **Decode t/s** | `(N_out − 1) / (T_last − T_first)` | **第1トークン除外**。含めると過小評価 |

### 4.2 測定プロトコルと他エンジン比較
- **Warmup**: 3 回（Metal PSO コンパイル + 重み展開）。1 回目破棄。
- **サンプリング**: Greedy（`temperature=0`, `top_p=1`, `top_k=1`）→ **seed 不要**。非 greedy 時のみ sampler seed 固定 + `-r 3`。
- **固定条件**: プロンプト長・生成長・chat template・system prompt・stop 条件を記録。
- **llama.cpp**: Prefill `-p 512 -n 0`、Decode `-p 512 -n 128`、`-r 5` 中央値、`-ngl 99` 固定。
- **MLX**: 計測前後に `mx.eval()` 同期必須。未同期は decode t/s が数倍過大。
- **uzu**: 投機的デコード既定有効。帯域比較時は無効化（§8-4）。
- **公平性**: 同一 bit 幅・同一 ctx・同一出力長・同一 tokenizer。主指標は decode t/s。

---

## 5. 再現性の確保（熱・電力・OS キャッシュ制御）

| 要因 | 影響 | 対策・コマンド |
| :--- | :--- | :--- |
| **熱** | クロック低下で t/s 2 桁変動 | 試行間冷却。`pmset -g therm` で `CPU_Speed_Limit=100` 確認 |
| **電力** | バッテリー駆動で GPU 制限 | AC 接続 + `sudo pmset -a lowpowermode 0` |
| **スリープ** | タイマ分解能悪化 | `caffeinate -dimsu <cmd>` |
| **ページキャッシュ** | 初回 mmap で I/O 遅延 | コールド時 `sudo purge`。ホット時は破棄しない |
| **他プロセス** | ブラウザ等が数 GiB 占有 | 全アプリ終了。`ps -Ao rss,comm -r` で上位確認 |
| **電力推移** | 熱設計消費の把握 | `sudo powermetrics --samplers cpu_power,gpu_power,thermal -i 1000 -n 5` |

> 5 回以上反復し **中央値 + 最小/最大**を記録。初回は warmup として破棄。

---

## 6. ベンチマーク表テンプレート

| Size | Quant | Lang | Prompt | Gen | TTFT (ms) | Prefill (t/s) | Decode (t/s) | Metal (MB) | Footprint (MB) | Swap (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.8B | BF16 | EN | 512 | 128 | | | | | | 0 |
| 0.8B | BF16 | JA | 512 | 128 | | | | | | 0 |
| 0.8B | INT4 | EN | 512 | 128 | | | | | | 0 |
| 0.8B | INT4 | JA | 512 | 128 | | | | | | 0 |
| 0.8B | YAQA | EN | 512 | 128 | | | | | | 0 |
| 0.8B | YAQA | JA | 512 | 128 | | | | | | 0 |
| 9B | BF16 | EN | 512 | 128 | *(不可)* | *(不可)* | *(不可)* | *(18GB超)* | — | *(記録)* |
| 9B | BF16 | JA | 512 | 128 | *(不可)* | *(不可)* | *(不可)* | *(18GB超)* | — | *(記録)* |
| 9B | INT4 | EN | 512 | 128 | | | | | | 0 |
| 9B | INT4 | JA | 512 | 128 | | | | | | 0 |
| 9B | YAQA | EN | 512 | 128 | | | | | | 0 |
| 9B | YAQA | JA | 512 | 128 | | | | | | 0 |

> 併記: 実効 bit/param、`n_ctx`、`recommendedMaxWorkingSetSize`、commit hash、`sw_vers`、`pmset -g therm`。

---

## 7. 具体的なコマンド例 (macOS)

```bash
# 1. プロセス実使用 (footprint = phys_footprint, 瞬間値)
footprint -p <PID> | grep -E "Footprint|Dirty"

# 2. VM 統計を MB 換算 (page size 16KB を自動取得)
vm_stat | awk 'BEGIN{FS=":"} /page size of/ {gsub(/[^0-9]/,"",$2); ps=$2} /^Pages (wired|active|inactive)/ {gsub(/[^0-9]/,"",$2); printf "%-20s %8.1f MB\n", $1, $2*ps/1048576}'

# 3. swap / メモリ圧力
sysctl vm.swapusage; memory_pressure | tail -3

# 4. 熱スロットリング (CPU_Speed_Limit=100 が正常)
pmset -g therm

# 5. 低電力解除 + キャッシュ破棄
sudo pmset -a lowpowermode 0; sudo purge

# 6. 電力・周波数サンプリング
sudo powermetrics --samplers thermal,cpu_power,gpu_power -i 1000 -n 5

# 7. スリープ抑止で実行
caffeinate -dimsu ./run_benchmark.sh

# 8. llama.cpp 比較 (-p prefill / -n decode)
llama-bench -m ./models/qwen-0.8b-q4.gguf -p 512 -n 128 -r 5 -ngl 99
```

---

## 8. 未確認事項

以下は確定できなかった点。**§8 以外はコード引用または実測ベース。**

| # | 未確認項目 | 現状の推定・代替手段 |
| :--- | :--- | :--- |
| 1 | `jax-metal` の `memory_stats()` | PJRT 依存で未実装例外の可能性。代替: `jax.live_arrays()` 合計 + `footprint` |
| 2 | YAQA カーネルの Metal 実行時分岐 | コード存在、量子化構成ごとのディスパッチ未追跡。実測は `currentAllocatedSize` |
| 3 | `recommendedMaxWorkingSetSize` の本機実値 | Apple 非公開。**【推】** 0.7〜0.75 × RAM。Swift 数行で取得可 |
| 4 | uzu の投機的デコード既定値・無効化フラグ | `speculator_stats` は提案時のみ算出。既定値は未確認 |
| 5 | 9B BF16 の実測挙動 | **【推】** 18GiB > 16GiB で不可。実測未実施 |
| 6 | `footprint` と `currentAllocatedSize` の差分 | 前者は圧縮含む全体、後者は GPU リソースのみ。両者併記 |

---

## 9. 参考文献・引用リンク

- [trymirai/uzu](https://github.com/trymirai/uzu) ・ [trymirai/lalamo](https://github.com/trymirai/lalamo)
- [MTLDevice.currentAllocatedSize](https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize) ・ [recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
- [JAX GPU Memory Management](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) ・ [llama.cpp llama-bench](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench)
