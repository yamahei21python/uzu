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

## 実モデルでの確認（2026-11-08 / ローカル M3）

Kaggle のカーネルが詰まっていたため、**ローカルで RTN 変換して確認**した
（calibration 無しなので品質は低いが、エンジンの経路は同じ）。

```
lalamo convert Qwen/Qwen3.5-0.8B --output-dir /tmp/bf16_lalamo
rht_int4_convert --load-from /tmp/bf16_lalamo --bits 4 --group-size 32 \
  --quantize-embedding --embedding-bits 4 \
  --layer-bits-json lanes/bits-wide-8.json --symmetric-bits ""
```

| 項目 | 結果 |
|---|---|
| ロード | ✅ 0.602GB（asymmetric int8 の wide8 + emb4） |
| prompt | 883 t/s（load 6.6 のため参考値） |
| generate | 128.9 t/s（同条件の symmetric wide8 は 124.4。バイト数が同じなので妥当） |
| 生成 | ✅ 指示に従う（"Repeat exactly this word: banana" → `banana`） |
| カーネルエラー | 無し |

**int8 zero-point は uzu で動く。** 残る `split_k = 1` 固定は性能上の制約のみ。

注意: この確認モデルは YAQA 校正なしの RTN。品質の数字（1.13 / 1.24）は
校正ありのモデルで測ったもので、このローカルモデルの品質ではない。

## split-k の解決（2026-11-09 / 委任先の実測にもとづく）

### 原因は丸め精度だった

幾何・メタデータ・インデックスは最初から正しかった。誤差の正体は
**split-k の部分和を bf16 で保存していたこと**。

- asymmetric ZP の重みは群平均が `scale*(127.5 - zp)` だけゼロからずれる
- 部分和は ±100 規模になり、reduce で相殺されて 6 程度になる
- bf16 は |S| ∈ [64,128) で ulp = 0.5 → 8 分割で最大 ~0.9 の誤差
- 実測: 2048 要素中 1〜2 要素だけ ±0.6 ずれる（相対 10% に見えていた）
- symmetric は平均 ≈ 0 で部分和が小さく、誤差が許容内

これで「案A（group 境界揃え）が無効だった」ことも説明できる。
`align = 128` が既に group 境界を包含しており、案A は数学的に no-op だった。

### 修正

1. `kernel.rs`: int8 ZP のとき **partial スクラッチを F32** に
2. `selection.rs`: `split_k = 1` の暫定回避を削除

### ついでに見つかった別バグ（これも修正）

`small_m_weights` が (ZP, 8) を `Tile8x32x128`（small-batch DirectSchedule）に割り当てていたが、
**そのスケジュールは symmetric 専用**（`magic_code_float<128>` を使い zero point を読まない）。
m ∈ [4,8] で **41% の誤差**（実測: 期待 −150 → 実際 −87.5）。

修正: この tile は symmetric のみに限定。ZP int8 は staged GEMM / GEMV に落ちる。

### 検証

- parity: 16 ケース（m=1/2/5/16/33, k=256〜1024, g32/64, symmetric 対照）全一致
- matmul モジュール: **209 passed / 0 failed**
- 実形状（m=512/k=4096/n=4096）は選択ロジックを unit test で担保。
  MXU 実行は M3 に MXU が無いため未検証（M5 で要確認）

### コスト

F32 partial は split_k>1 の int8 ZP 層のみ。scratch が 2 倍（9B で 32MB → 64MB）。
重み 5.37GB に対して 0.6% なので、速度への影響はほぼ無い。
