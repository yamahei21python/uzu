# Device Profile — uzu ベンチ用

計測日: 2026-10-28

## マシン

| 項目 | 値 |
|---|---|
| 機種 | **MacBook Air 13-inch M3** (Mac15,12) |
| 冷却 | **ファンレス** → 持続負荷でサーマルデグレーディング注意 |
| CPU | Apple M3 8コア (4P + 4E) |
| GPU | Apple M3 **10コア** |
| メモリ | 16GB 統合メモリ |
| 帯域 | **100 GB/s** (M3公称、APIからは取得不可) |
| 統合メモリ | `hasUnifiedMemory = true` |
| GPU低電力 | `isLowPower = false` |

## OS / Toolchain

| 項目 | 値 |
|---|---|
| macOS | 26.5.2 (25F84) |
| Metal | **Metal 4** / metalfe-32023.883 |
| Xcode | /Applications/Xcode.app/Contents/Developer |
| Metal Toolchain | 17F109 (`xcodebuild -downloadComponent MetalToolchain` で導入) |
| rustc | 1.93.1（uzuは`rust-toolchain.toml`でnightly強制） |
| uzu | v0.6.13 / `target/release/cli` (50MB) |

## Metal 実機プローブ (MTLCreateSystemDefaultDevice)

```
name = "Apple M3"          ← uzuの device.name() が読む実値
registryID = 4294968444
recommendedMaxWorkingSetSize = 12.71 GB   (16GB中の上限)
maxThreadsPerThreadgroup = 1024 x 1024 x 1024
argumentBuffersSupport = tier 1
readWriteTextureSupport = tier 2
GPU family: raw 1001..1009 = true, 1010..1012 = false  → Apple1〜Apple9対応 (Apple10=非対応)
common1/2/3, metal3 = 全て true
```

## uzu への影響（重要）

### 1. `shapes.json` に `Apple M3` が無い → 投機的デコード無効

`crates/uzu-engine/src/speculators/dflash_tfm.rs:104-110`
```rust
let Some(shape) = context.device_name()          // "Apple M3"
    .and_then(|s| shapes.remove(s))               // None（M5/M6のみ収録）
    .or_else(|| shapes.remove("default"))          // None（defaultも無い）
else { return Ok(None); };                        // → speculator = None
```
`crates/uzu-engine/src/engine/language_model/mod.rs:89-93` で `.flatten()` → **黙って無効化**。
エラーも警告も出ない。**9Bを積んでも `t/f = 1.00`**。

**修正**: `speculator/shapes.json` に追加
```json
"Apple M3": {
  "tree_budget": 16,
  "max_tree_depth": 16,
  "dflash_depth_override": null,
  "construction_method": { "type": "Weaver", "rounds": 16, "expand_per_round": 4, "expand_width": 4 }
}
```
（M5の値を流用。M3でも `tree_budget 16` は現状最大の受け入れ幅）

### 2. `qmv/routes.rs` にも `Apple M3` 行が無い → ただし実害なし

`measured_device_name!` マクロは M1/M2/M2Pro/M3Max/M4/M4Pro/M5Max のみ。
→ `routes.rs:220-225` の **GPU family フォールバック**（`apple_gpu_family`一致）で解決。
M3 = Apple9 → `Apple M3 Max` 行に一致するためカーネルは正常動作。
**4Bで 31.9 t/s 出ているのが証拠。shapes.jsonだけフォールバックが無い。**

### 3. メモリ帯域から理論値を逆算

| モデル | 重み | 帯域律速上限 (100GB/s) | 投機あり (t/f 5〜7.5) |
|---|---|---|---|
| Qwen3.5 4B-M | 2.44 GB | ~41 t/s | （spec無し） |
| Qwen3.5 9B-M | 5.19 GB | **~19 t/s** | **~96〜145 t/s** |

→ 投稿の **92 t/s = 4.8x** はこの帯域計算と一致。llama.cpp 22 t/s も同オーダー。
**つまり投稿の数字はM3帯域でも成立する。ただし `shapes.json` 修正が前提。**

## 電源・熱

| 項目 | 値 |
|---|---|
| lowpowermode | 0 (OFF) |
| 給電 | AC |
| サーマル警告 | なし |
| 性能警告 | なし |

## メモリ状態（計測時の問題点）

| 時点 | swap 使用 | 空き |
|---|---|---|
| Ollama 9B常駐時 | **7.6G / 8G** | 19% |
| Ollama停止後 | 6.5G / 8G | 68% |

`Pages reactivated = 2.3億` → **スワップthrashingが速度を壊していた**。
uzu計測時は Ollama を停止すること（`ollama stop <model>`）。

## 実測ベンチ（qwen3.5:4b, `-m` 1発）

| 条件 | generation | prefill | TTFT |
|---|---|---|---|
| Ollama常駐時 | 3.16 t/s | 14 t/s | 1.63 s |
| Ollama停止後 | **31.89 t/s** | **164 t/s** | **0.14 s** |

`t/f = 1.00`（4Bはspeculatorファイル自体が存在しない）
