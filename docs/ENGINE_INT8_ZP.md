# int8 zero-point GEMM を uzu に入れる（進行中 / 2026-11-08）

## なぜ

asymmetric int8 は symmetric より重み MSE が 1.39x 良い（実測 246 行列）。
wide8 の品質は asym 1.13/1.24 に対し sym 1.73/2.10。**uzu が int8 ZP を持てば 0.6pp 戻る。**

## 入れた変更

| ファイル | 変更 |
|---|---|
| `gemm.metal` | `ScaleZeroPointDequant` を BITS 8 でも許可。ただし **A が int8 のときは除外**（`IntegerSchedule` に該当 launch が無い） |
| `gemv.metal` | ZP の BITS 制約を 4 → 4/8 |
| `quant_unpack.h` | `decode_zero_point<8>` を追加（8bit は 1 byte = 1 個） |
| `operands.h` | `ZERO_POINTS_PER_BYTE` を定数 2 から `8 / BITS` に修正 |
| `selection.rs` | (ZP, 8) を symmetric と同じ `Tile8x32x128_Simdgroups1x2x4` に |
| `quant_dispatch_test.rs` | w8 ZP の parity ケースを追加 |

## 原因（2026-11-08 特定済み）

**split-k の zero-point bias 二重計上**だった。

- bias はステージングされた全要素（group 境界のパディング含む）に加算される
- k-split すると、そのパディング分の bias が **split の数だけ重複**する
- `UZU_FORCE_NO_SPLIT_K=1` で全ケース一致 → 確定

修正: int8 zero-point では `split_k = 1` に固定（`selection.rs`）。
実形状では n が大きく base_tiles が十分あるため、実害はほぼ無い見込み。

## 現状（parity の実測）

| 条件 | 結果 |
|---|---|
| m=1（decode, k=1024） | ✅ 一致 |
| m=2（k=256） | ✅ 一致 |
| m=16（k=256） | ✅ 一致 |
| m=16 / m=33（k=1024, k=768, k=512） | ✅ 一致（修正後） |
| symmetric int8（同じ k=1024, m=16/33） | ✅ 一致 |

**k が大きいときだけ壊れる。symmetric は壊れない** ので、zero-point 固有の経路に原因がある。

## 未確認

- **実モデルでの速度と品質**。parity テストは合成データなので、
  実変換モデル（wide8 asymmetric）で bench が通るかは別途確認中
  （`uzu-asym8-wide8` を Kaggle CPU でビルド）
- split_k=1 固定による prefill 速度の低下量（実形状では小さいと予想）
