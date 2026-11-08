# int8 zero-point GEMM の split-k（Antigravity 委任 / 2026-11-08）

> **検証結果（委任元）**
>
> - 原因の説明は説得力がある。特に **symmetric が壊れない理由**は明白:
>   パディング要素は符号なしコード 128 になり、`128*scale - 128*scale = 0` で完全に相殺される。
>   asymmetric は `-scale*zp` が残るため、パディングが split ごとに加算される
> - しかし **推奨案A（split 境界を group 境界に揃える）を実装して測ったところ、
>   parity の誤差は 1 ビットも改善しなかった**（同じ 3 ケースが同じ値で落ちる）
> - したがって「分割点が group を跨ぐこと」だけが原因ではない。
>   tile ステージング側にも境界を跨ぐパディングがあり、`split_k=1` 以外では bias が重複する
> - 現状は `selection.rs` で **int8 zero-point の split_k を 1 に固定**して回避している。
>   実形状では n が大きく base_tiles が足りているため、速度への影響は小さいと見ている
>
> 以下は委任先の報告原本（未修正）。

# int8 Zero-Point GEMM における Split-K 復活の設計とパッチ

## §1 原因の厳密な特定

### 1. Split-K の実装パイプライン
1. **計画選定 (`selection.rs:159-192`)**: `select_split_k` が `target_tiles / base_tiles` と `k / align` から `split_k` を選定。
2. **ディスパッチ制御 (`kernel.rs:346-369, 441-575`)**: `plan.split_k > 1` のとき `encode_split_k` を呼び出し、スクラッチバッファ `temp` (`split_k * M * N * sizeof(T)`) を確保。`params.aligned_inner_iterations = (k / split_k) / k_step / simdgroups_k` を設定し、Grid Z = `split_k` で GEMM カーネルを実行。
3. **GEMM 実行 (`gemm.metal:162`, `simdgroup_mma_core.h:205-223`, `mxu_mma_core.h:86-109`)**: 各スレッドグループが `partition = group_z` を取得し、`k_offset = partition * aligned_inner_iterations * THREADGROUP_BLOCK_K` から K 軸の部分和を計算して `d[partition * M * N + ...]` に出力。
4. **リダクション (`gemm_split_k_reduce.metal:46-49`)**: `GemmSplitKReduce` が各パーティションの要素を加算:
   ```metal
   for (uint partition = 0u; partition < partition_count; ++partition) {
     accumulator += float4(partial_sum_vectors[partition * vector_count + vector_index]);
   }
   ```

### 2. パディングの発生と Bias 畳み込み
- **Bias の畳み込み (`quant_scale_zero_point.h:118-130`, `quant_unpack.h:112, 137`)**:
  `current_scale_bias` で `out_bias = -scale * float(zero_point)` を算出。`dequantize` 内の `fma(code, scale, bias)` により、重みステージングメモリ (`dst`) 上の全要素に bias が加算される。
- **グループ境界不整合によるパディング重複**:
  `staged.h:32` では `first_group = k_offset / GROUP_SIZE`、`quant_scale_zero_point.h:77` では `group_step_counter = 0` と初期化される。これは各パーティションの `k_offset` がグループ境界に完全に一致していることを前提とする。
  もし $K / \text{split\_k}$ が `group_size` の整数倍でない場合、パーティション境界がグループの途中に跨がる。タイルスライスを埋めるためのパディング要素にも `out_bias` が付与され、リダクション時に本来 1 回だけ計上されるべき bias がパーティション数分だけ重複加算され、実測で約 10% の相対誤差（0.59〜0.69）となって現れる。

### 3. なぜ Symmetric int8 は壊れないのか
- `quant_unpack.h:18` より、Symmetric では `symmetric_zero_point<8>() = 128`（符号なしバイトの中央値）。
- 重みコードは `code ^ 0x80` で符号付き $[-128, 127]$ を $[0, 255]$ に写像する。
- パディング要素（数値 0）は符号なしコード 128 に復元される。
- 逆量子化計算:
  $$\text{dequant}(128) = 128 \cdot \text{scale} + (-128 \cdot \text{scale}) = 0.0\text{f}$$
- つまり、Symmetric ではパディング要素が**数学的・ビット表現的に完全な 0.0f に相殺**されるため、$A \cdot 0.0\text{f} = 0$ となり不要なバイアスが累積しない。
- 一方 Asymmetric Zero-Point では、Zero-Point $Z \in [0, 255]$ が任意の値であるため、パディング要素（コード 0）であっても $0 \cdot \text{scale} - \text{scale} \cdot Z = -\text{scale} \cdot Z \ne 0$ が残り、パーティションを跨いでバイアスが重複加算される。

---

## §2 修正案の比較

| 項目 | 案A: split 境界を group 境界に揃える | 案B: reduce 時にパディング bias を補正 | 案C: activation sum による bias 外出し |
|---|---|---|---|
| **概要** | `k % (split_k * group_size) == 0` を要求し、パディングを根絶 | `gemm_split_k_reduce` で余剰パディング要素分の bias を減算 | $A$ の row_sum を計算し、bias 畳み込みを GEMM 外で補正 |
| **変更ファイル** | `crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/selection.rs` | `gemm_split_k_reduce.metal`, `kernel.rs` | `gemm.metal`, 各 core ヘッダ, `kernel.rs` |
| **実装難易度** | **低**（Rust 側の選択ロジック 1 箇所） | **高**（メタデータ・スレッド位置再計算で複雑怪奇） | **中〜高**（BF16 A 向け row_sum パイプライン新設） |
| **性能への影響** | **ほぼ皆無**（実モデルの K は group_size の大きな倍数） | **悪化**（reduce の帯域・レジスタ圧迫、レイテンシ増） | **悪化**（activation sum カーネルの追加実行） |
| **正しさの担保** | **容易**（境界跨ぎが構造上消滅、数学的等価性自明） | **極めて困難**（端数計算・丸め誤差の検証が難航） | **中**（数式は確立しているがコード規模大） |
| **推奨度** | **◎（強く推奨）** | ×（非現実的） | △（過剰設計） |

### 各案の評価
- **案A（推奨）**: パディング要素が発生する根本原因（グループ途中での分割）を遮断する。LLM の $K$（2048, 4096, 8192 等）は $g \in \{32, 64\}$ の倍数であるため、`split_k`（2〜8）で割っても常に割り切れ、性能ペナルティなしに安全性を確保できる。
- **案B**: reduce 側に scales / zero_points バッファを渡し、部分グループの端数要素数を逆算して差し引く必要がある。実装が極度に肥大化しバグの温床となる。
- **案C**: A8W4 の `NEEDS_ASYMMETRIC_WEIGHT_CORRECTION` と類似の仕組みだが、全精度 BF16 activation に対して group_sums を追加計算・保持する必要があり、メモリ帯域と実行時間を無駄に消費する。

---

## §3 推奨案（案A）のパッチ

現状の `selection.rs` にある int8 ZP 時の `split_k = 1` 固定（暫定回避）を撤廃し、量子化重み（特に zero-point）において各パーティションが完全なグループの整数倍（`k % (split_k * group_size) == 0`）となることを保証する。

```diff
--- a/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/selection.rs
+++ b/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/selection.rs
@@ -161,12 +161,6 @@ impl GemmProblem {
         engine: GemmEngine,
         tiling: GemmTiling,
     ) -> u32 {
-        // A zero-point bias is folded into every staged element, including the
-        // padding at a group boundary, so a k-split would count that padding
-        // once per split. int8 zero-point weights therefore stay unsplit.
-        if self.shape.b_bits == Some(8) && self.shape.b_prologue == GemmBPrologueKind::ScaleZeroPointDequant {
-            return 1;
-        }
         let shape = self.shape;
         let splittable = shape.is_quant() || (shape.b_transpose && shape.b_leading_dimension.is_none());
         if !splittable || !self.split_k_output_supported() {
@@ -182,10 +176,18 @@ impl GemmProblem {
         let align = step.max(ACTIVATION_SCALE_GROUP_SIZE) * tiling.simdgroups_k();
         let target_tiles = policy::split_k_target_tiles(shape.a_is_quantized(), tiling, shape.b_bits);
         let mut split_k = (target_tiles / base_tiles).max(1).min((shape.k / align).max(1));
+        let group_size = if shape.is_quant() {
+            shape.b_group_size.unwrap_or(1)
+        } else {
+            1
+        };
         if shape.a_is_quantized() && engine == GemmEngine::Mxu && tiling.block_k() != 0 {
             split_k = split_k.min((shape.k / tiling.block_k()).max(1));
         }
-        while split_k > 1 && !shape.k.is_multiple_of(split_k * align) {
+        while split_k > 1
+            && (!shape.k.is_multiple_of(split_k * align)
+                || !shape.k.is_multiple_of(split_k * group_size))
+        {
             split_k -= 1;
         }
         split_k
```

---

## §4 テスト計画

### 1. 既存テストケースの split-k 到達状況
`crates/uzu-engine/unit/backends/metal/kernel/matmul/quant_dispatch_test.rs` 内の `signed_weights_full_precision_activations_parity_bf16`:
- `w8_zp_m1` ($m=1, k=1024$): GEMV パスにディスパッチ（GEMM split-k は非該当）
- `w8_zp` ($m=2, k=256$): batched GEMV パスにディスパッチ
- `w8_zp_m16_k256` ($m=16, k=256, g=32$): **`split_k = 2`** を通過 ($kp = 128 = 4 \times 32$)
- `w8_zp_m16_k512` ($m=16, k=512, g=32$): **`split_k = 4`** を通過 ($kp = 128 = 4 \times 32$)
- `w8_zp_m16_k768` ($m=16, k=768, g=32$): **`split_k = 6`** を通過 ($kp = 128 = 4 \times 32$)
- `w8_zp_m16` ($m=16, k=1024, g=32$): **`split_k = 8`** を通過 ($kp = 128 = 4 \times 32$)
- `w8_zp_m16_k1024_g64` ($m=16, k=1024, g=64$): **`split_k = 8`** を通過 ($kp = 128 = 2 \times 64$)
- `w8_zp_m33` ($m=33, k=1024, g=32$): **`split_k = 8`** を通過 ($kp = 128 = 4 \times 32$)

### 2. 追加すべきテストケース提案
境界条件および非整除フォールバックの動作を担保するため、以下のケースの追加を推奨する:
```rust
// 1. group_size=64 で split_k > 1 を踏む中規模 K
#[case::w8_zp_m16_k512_g64(8, QuantizationMethod::ScaleZeroPoint, 512, 16, 64)]
// 2. K が (split_k * group_size) の倍数にならず、split_k がデクリメント調整されるケース (推測値)
//    k=384, g=64: split_k=3 (kp=128) または split_k=2 (kp=192) へ正しく収束するか
#[case::w8_zp_m16_k384_g64(8, QuantizationMethod::ScaleZeroPoint, 384, 16, 64)]
// 3. m=64 以上の prefill 形状で split_k=1 に自然フォールバックするケース
#[case::w8_zp_m64_k1024_g32(8, QuantizationMethod::ScaleZeroPoint, 1024, 64, 32)]
```

---

## §5 残るリスクと注意点

1. **実モデル形状における split_k の発動頻度**:
   LLM の実推論（Prefill 時）では $N$（中間層・語彙サイズ）が 4096〜12288 と大きいため、`base_tiles` が大きくなり `split_k` は元々 1 に収束することが多い（性能寄与は主に小さな $N$ やベンチマーク合成形状に限定される可能性、推測）。
2. **`ACTIVATION_SCALE_GROUP_SIZE` (128) との結合関係**:
   `selection.rs:182` では `align = step.max(ACTIVATION_SCALE_GROUP_SIZE) * tiling.simdgroups_k()` となっており、全精度 activation 時にも 128 のアラインメントが残存している。現状のパッチは既存のアラインメントを包含しつつ安全側（`group_size` の整数倍）に倒すため後方互換性は完全だが、将来 `align` の定義が変更された際にも `k.is_multiple_of(split_k * group_size)` の制約が防壁となる。
3. **W4 Zero-Point への影響**:
   本パッチは `shape.is_quant()` 全体に `group_size` アラインメントを要求するため、W4 Zero-Point に対しても同様にグループ境界跨ぎを防止し、健全性を向上させる。
