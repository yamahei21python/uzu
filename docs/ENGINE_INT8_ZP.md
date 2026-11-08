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

## 現状（parity の実測）

| 条件 | 結果 |
|---|---|
| m=1（decode, k=1024） | ✅ 一致 |
| m=2（k=256） | ✅ 一致 |
| m=16（k=256） | ✅ 一致 |
| **m=16 / m=33（k=1024）** | ❌ 相対 10% の差（0.59〜0.69） |
| symmetric int8（同じ k=1024, m=16/33） | ✅ 一致 |

**k が大きいときだけ壊れる。symmetric は壊れない** ので、zero-point 固有の経路に原因がある。

## 疑っているところ

- `quant_scale_zero_point.h` の `next()` は group step ごとに
  `zero_point_index += zero_point_group_stride` する。k=1024 では 32 回進む
- split-k が選ばれたときの bias の扱い（`out_bias = -scale * zp` を分割ごとに加算していないか）
- symmetric は同じ構造なのに通るため、split-k 説は弱い

## 次の一手

1. k=512 で試す（境界を絞る）
2. `split_k` を強制的に 1 にして比較
3. `quant_scale_zero_point.h` の `next()` と Rust 側の stride 計算を突き合わせる

**注意**: このブランチのテストは 2 件落ちている（m=16/m=33 の k=1024）。
それ以外の 9 件は通る。decode は正しいので、prefill を直せば実用になる。
