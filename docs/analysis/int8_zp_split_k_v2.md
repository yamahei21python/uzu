# int8 zero-point GEMM の split-k 復元（第2次委任 / 2026-11-09）

> **検証環境と前提の補正**
>
> - 委任状は「Linux sandbox・Metal 無し」を想定していたが、実際の実行環境は macOS (Apple M3, Metal 32023.883,
>   cargo 1.101.0-nightly) で、**Metal 実機での再現・計測・修正検証を実施した**。
> - 本機は `context.supports_mxu == false` を返す（実測）。したがって MXU カーネルの**実行**検証は不可能
>   （既存の A8W\* MXU 系テストも本機ではすべて early-return している）。MXU については選択ロジックを
>   unit test（`GemmProblem::new(..., supports_mxu=true)` を注入）で検証した。
> - 数値はすべて実測値。捏造はない。不明点は「不明」と明記する。
> - 修正本体は委任元が commit 82166ff0 として取り込み済み（§5.4 の small-batch 対処も同時に入った）。
>   本ドキュメントと「残タスク」追記分は未コミット。

## §1 失敗の再現条件

`selection.rs` の int8 ZP に対する `split_k = 1` 強制を外すと、報告どおり 3 ケースが落ちる（実機で再現済み）。
選択される plan は次のとおり（`encode_split_k` に一時ロギングを入れて実測。いずれも n=128、Simdgroup エンジン）。

| ケース | tile (Simdgroup) | split_k | kp | aligned_iters | 結果 |
|---|---|---|---|---|---|
| w8_zp_m16_k256 | Tile8x32x32_Simdgroups1x1 | 2 | 128 | 4 | **通る** |
| w8_zp_m16_k512 | 同上 | 4 | 128 | 4 | **通る** |
| w8_zp_m16_k768 | 同上 | 6 | 128 | 4 | **通る** |
| w8_zp_m16 (k=1024) | 同上 | 8 | 128 | 4 | **落ちる** |
| w8_zp_m16_k1024_g64 | 同上 (g=64) | 8 | 128 | 4 | **落ちる** |
| w8_zp_m33 | Tile64x64x32_Simdgroups2x2 | 8 | 128 | 4 | **落ちる** |
| w8_zp_m1 (k=1024) | Tile8x32x32 | 8 | 128 | 4 | **通る** |
| w8_sym_m16 / m33 | （同形状） | 8 | 128 | 4 | **通る** |

要点:

- **失敗・成功で k 幾何は完全に同一**（kp=128、aligned=4、8 分割）。split_k の数（2/4/6 なら通る）と
  データ（symmetric か ZP か）だけが違う。
- 症状の正確な姿は「全体が 10% ずれる」ではなく、**全出力のうち 1〜2 要素だけが ±0.5〜0.7 ずれる**。
  実測（許容差 `max(0.5, 0.05·|expected|)`、`assert_parity` のミスマッチ全列挙による）:
  - `w8_zp_m16` k=1024 g32: 2048 要素中 **1 件**。`[1947] (row15, col27) 6.4375 → 7.125 (+0.6875)`
  - `w8_zp_m33` k=1024 g32: 4224 要素中 **2 件**。`[1545] 4.59375 → 5.1875 (+0.59375)`、
    `[2153] 9.875 → 9.3125 (−0.5625)` — **符号が混在**
  - `w8_zp_m16_k1024_g64`: 2048 要素中 **1 件**。`[1204] (row9, col52) 1.5 → 0.96875 (−0.53125)`
- 出力が小さい要素（〜6 程度）に ±0.6 が乗るため相対誤差 10% に見えていた。

## §2 原因の特定

### 2.1 まず幾何・メタデータは正確だった（初回報告の仮説は反証）

- `k_offset = partition * aligned_inner_iterations * THREADGROUP_BLOCK_K`
  (`simdgroup_mma_core.h:214`)、`aligned = kp / k_step / sgk = 128/32/1 = 4`。各 partition は
  k ∈ [128p, 128p+128) を**過不足なく**カバーし、8 分割で k=1024 をちょうど埋める。
  K は BLOCK_K=32 の倍数なのでステージングに k 方向のパディングは発生しない。
  `encode_split_k` は `GemmAlignment` の K を常時 true にするため、k_loop の leftover 端数処理
  (`simdgroup_mma_core.h:128-140`) も走らない（ここで端数が落ちる・ずれる仮説 1/2 は不成立）。
- スケール/ZP の選択も正確: `first_group = k_offset / GROUP_SIZE` (`staged.h:32`) は 128p で常に整数、
  loader の `group_step_counter` は初期 0 で GROUP_STEPS_PER_BLOCK 単位で進む
  (`quant_scale_zero_point.h:188-201`) ので、k_offset が group 境界にある前提と矛盾しない（仮説 3 不成立）。
  `gemm_split_k_reduce.metal` の部分和インデックス（`partition * vector_count + vector_index`）も
  スクラッチ `split_k·M·N` と一致する（仮説 4 不成立）。
- **決定的な反証**: スクラッチの dtype だけを bf16→f32 にすると（幾何は無変更のまま）14 ケース全部が一致した。
  幾何・メタデータ・インデックスの誤りなら f32 化では直らない。

### 2.2 真因: split-k スクラッチ bf16 保存の丸め誤差の積算

`encode_split_k` (`kernel.rs:515-519`) は partial 用スクラッチを `a_int8 || is_small_batch_variant()`
のときだけ f32、それ以外は**出力 dtype（ここでは bf16）**で確保する。問題の構造:

1. asymmetric ZP の重みは `w = scale·(code − zp)`。code は一様乱数なので**グループ平均が
   `scale·(127.5 − zp)` となりゼロから大きくずれる**。
2. その結果、パーティション部分和 `S_p = Σ_{k∈[128p,128p+128)} x·w` は巨大になる。失敗要素
   (`w8_zp_m16` row15/col27、seed=0 のテストデータを CPU で再構成、`total=6.44 ≒ expected 6.4375` で解釈検証済み) の実測:
   `S_p = [−60.9, +80.8, +13.6, −114.1, +2.1, +5.9, +6.2, +72.8]` — **部分和が ±100 規模なのに合計は 6.4**
   という強い相殺が起きている。
3. bf16（仮数 8bit）では `|S| ∈ [64,128)` の ulp = 0.5。1 partial あたり最大 ±0.25 丸め、sk=8 で累積し
   最悪 ~0.9。観測された ±0.53〜0.69 と整合する。
4. **symmetric が無事な理由**: `E[code − 128] ≈ 0` で重み平均がほぼゼロ → partial が小さく（実測でも数〜20 規模）
   ulp も小さく、誤差が許容内に収まる。**k≤768 が通る理由**: kp は同一 (128) なので partial の規模は同じだが、
   丸め回数が 2/4/6 回と少ないことに加え、どの要素が abs 0.5 を超えるかはデータ運（2048 要素中 1〜2 だけ超過）。
   `m=1` が通ったのも同様のデータ運である。

### 2.3 案A（group 境界揃え）が効かなかった理由

- 選択制約の `align = max(BLOCK_K, ACTIVATION_SCALE_GROUP_SIZE)·sgk = 128` (`selection.rs:176`) であり、
  既存の `k % (split_k·128) == 0` 要求は group_size ∈ {32, 64} に対して
  `k % (split_k·group_size) == 0` を**既に含意する**。つまり案Aの追加条件は最初から常に満たされており、
  数学的に no-op だった。実測「1 ビットも改善しなかった」はこの通り。
- そもそも「分割境界を跨ぐ bias 重複」は起きていなかった（§2.1）。パディングが仮に発生しても
  A 側パディングは 0 (`load_safe`) で積はゼロになる。誤差は丸め精度の問題なので、境界を揃える系統の
  修正では決して直らない。

## §3 修正案（unified diff）

方針: **int8 zero-point 重みの split-k partial は f32 スクラッチで積算する**。partial 幾何は既に正確なので、
精度だけを integer-activation パスと同じ水準に引き上げる。`split_k = 1` 強制は撤去する。

```diff
--- a/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/kernel.rs
+++ b/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/kernel.rs
@@ -512,7 +512,15 @@ impl GemmKernel {
         )?;
 
         let elem = (m as usize) * (n as usize);
-        let partial_data_type = if a_int8.is_some() || tiling.is_small_batch_variant() {
+        // Zero-point int8 weights carry a per-group mean of scale * (2^(bits-1) - zp), so each
+        // partition's partial sum is dominated by that offset and cancels across partitions in
+        // the reduce. Storing those large partials in bf16 loses up to half an ulp per split
+        // (measured: ~10% drift at split_k=8, k=1024), so they accumulate in f32 — the scratch
+        // precision the int8-activation path already uses.
+        let partial_data_type = if a_int8.is_some()
+            || tiling.is_small_batch_variant()
+            || (shape.b_bits == Some(8) && shape.b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)
+        {
             DataType::F32
         } else {
             self.output_data_type
```

```diff
--- a/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/selection.rs
+++ b/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/selection.rs
@@ -176,13 +176,6 @@ impl GemmProblem {
         let align = step.max(ACTIVATION_SCALE_GROUP_SIZE) * tiling.simdgroups_k();
         let target_tiles = policy::split_k_target_tiles(shape.a_is_quantized(), tiling, shape.b_bits);
         let mut split_k = (target_tiles / base_tiles).max(1).min((shape.k / align).max(1));
-        // Zero-point bias is folded into every staged element, and aligning the
-        // split to group boundaries is not enough to keep it counted once
-        // (measured: int8 zero-point still drifts 10% with aligned splits), so
-        // these weights stay unsplit until the staging path is reworked.
-        if shape.b_bits == Some(8) && shape.b_prologue == GemmBPrologueKind::ScaleZeroPointDequant {
-            return 1;
-        }
         if shape.a_is_quantized() && engine == GemmEngine::Mxu && tiling.block_k() != 0 {
             split_k = split_k.min((shape.k / tiling.block_k()).max(1));
         }
```

テスト追加（2 ファイル、全文はワーキングツリーの `git diff` 参照）:

- `unit/backends/metal/kernel/matmul/gemm/selection_test.rs`: `zero_point_int8_weights_select_split_k` —
  W8 ZP が (k=256,512,768,1024) → sk=(2,4,6,8)、m=33 → 8、W4 ZP → 8、
  実形状 m=512/n=4096/k=4096 → **Mxu かつ sk=4**、m=512/n=512/k=4096 → Simdgroup sk=8 をアサート。
  暫定回避が再発した場合に即座に検出できる。
- `unit/backends/metal/kernel/matmul/quant_dispatch_test.rs`:
  `w4_zp_m16_k1024`（W4 ZP が bf16 partial のまま sk=8 を通ることの対照ケース）、
  `w8_zero_point_prefill_shape_split_k_parity_bf16`（実形状 m=512/k=4096/n=4096/g32 の end-to-end）、
  `w8_zero_point_prefill_scale_split_k8_parity_bf16`（m=512/k=4096/n=512/g32、Simdgroup で実際に sk=8 が走る規模）。

### 採用しなかった代替案

| 案 | 判断 |
|---|---|
| partial を**常時** f32 | 正しいが、symmetric/W4 までスクラッチ 2 倍化の無駄。対象は ZP int8 で十分（W4 ZP は §4 実測で bf16 のまま合格） |
| zp 由来の平均を activation group sums で外出し（A8W4 ScaleBias の `NEEDS_ASYMMETRIC_WEIGHT_CORRECTION` 相当） | 部分和自体を小さくできるが、A8 専用の補正経路の FP-A 版を新設する大改修。丸め問題に対しては過剰 |
| bf16 のままテスト許容差を緩和 | 真の出力に対して 10% の誤差を常態化させるのは本末転倒 |

## §4 テスト計画と実測結果

### split_k > 1 を踏む既存ケース（修正後）

| ケース | sk | partial dtype | 判定 |
|---|---|---|---|
| w8_zp (m2 k256) / w8_zp_m16_k256 | 2 | f32 | ✅ |
| w8_zp_m16_k512 | 4 | f32 | ✅ |
| w8_zp_m16_k768 | 6 | f32 | ✅ |
| w8_zp_m1 / w8_zp_m16 / m33 / m16_k1024_g64 | 8 | f32 | ✅ |
| w4_zp_m16_k1024（追加） | 8 | bf16 のまま | ✅ |
| w8_sym_m16 / w8_sym_m33 / w4_bias / w4_zp（対照） | 2〜8 | 変更なし | ✅ |

注: 本テスト構成では m=1 も GEMV ではなく GEMM split-k 経路を通る（実測ログ）。m=1 が修正前も
通っていたのは partial がデータ運で小さかっただけで、幾何の証拠ではない。

### 実行結果（最終状態、本機）

- `backends::metal::kernel::matmul` モジュール全体（追加テスト込み、最終状態）: **192 passed / 0 failed**。
  `quant_dispatch_test` ファイル単体でも 192 ケース中全パス（修正直後の追加前状態でも 169/169 を確認済み）。
  修正取り込み後（commit 82166ff0、small-batch 対処込み）は委任元の実行で **209 passed**、
  本追記時点の対象テスト（parity 16 + 選択 1）も全パスを再確認済み。
- 実形状 m=512/k=4096/n=4096/g32: 選択は **Mxu sk=4**（selection test でアサート、supports_mxu=true 注入）。
  本機は MXU 実行不可のため、end-to-end 実行は (a) 同形状での sk=1 実行（パス）と
  (b) Simdgroup で sk=8 が実際に走る同規模形状 m=512/k=4096/n=512/g32（パス、CPU リファレンス含め 248s）で
  検証した。MXU 搭載機での sk=4 実行確認は未実施（§5）。
- テスト 1 本あたりの所要: 実形状 2 本は CPU リファレンス起因で 76s / 248s。CI 時間が問題になるなら
  n を落とすか seed を絞ってよい。

## §5 残るリスク

1. **性能（追加計測ずみ）**: criterion（GPU 実行時間）で W8 ZP gs32 の 5 形状 —
   sk8_m8/n2048/k2048、sk4_m16/n2048/k4096、sk2_m512/n2048/k4096、sk8_m512/n512/k4096、
   コントロール sk1_m16/n14336/k4096 — について、f32 / bf16 partial を同一プロセス内で交互に、
   かつラウンド順序を反転して計 10 回起動して比較した。ただし**コントロール（sk=1 は両者同一コード）
   の比率が 0.81〜0.96 と変動し、本機のノイズフロアは ±10〜20%**。観測された f32/bf16 比率
   0.89〜1.44 はすべてこの中に収まり、**有意差は検出不能**だった。理論上限は追加トラフィック
   4·sk·M·N バイト/encode（書き+読み、0.5〜8MB）で全体の 0.004〜1.6% — ノイズ以下。
   結論: 実用上の影響は理論 bound でも 2% 未満。精密値が必要なら静かな環境での再計測を推奨。
2. **MXU 実行は M5+ 実機で要確認（本機では不可能と実証）**: `supports_mxu` は非公開 selector
   `supportsMXU` を読む拡張 (`metal_extensions/device_extensions.rs`) で、同所に
   「True on M5+ (Gen18+), false on M1-M4」と明記されている。本機 M3 で supports_mxu を強制 true に
   して MXU を実行させたところ、**パイプライン作成・実行はエラーなく通るが数値は崩壊した**
   （m=16/n=128/k=1024: W8 symmetric sk=8 で 2009/2048 不一致・max差358。W8 ZP は sk=1 / sk=2 / sk=8 ×
   f32/bf16 partial の**全構成**で同一の崩壊（2012/2048・max差538））。つまり M3 では
   `mpp::tensor_ops::matmul2d` が split-k と無関係に不正動作し、Metal は実行を拒否しない。
   upstream の M5+ ゲートは正当であることが実証された。**sk=4 の実行確認には M5+ 実機が必須**。
   修正は engine 共通の `encode_split_k` のみでカーネル本体は無変更のため、M5+ でのリスクは
   低いと見るが未確認。
3. **W4 ZP は bf16 partial のまま**: 仕組み上 partial は W8 の ~1/16 以下（code 範囲 16 vs 256）で、
   k=1024 (sk=8) に加え **k=8192 (sk=64、kp=128 で 64 回の bf16 丸め) の極端分割でも実測パス**
   （追加テスト `w4_zp_m16_k8192` / `w8_zp_m16_k8192`、未コミット）。問題が出たら
   `partial_data_type` の条件を `b_bits` 無関係の `ScaleZeroPointDequant` に広げればよい（1 行）。
4. **別件として実測で発見したバグ（対応済み: commit 82166ff0 で本修正と同時に取り込まれた）**: W8 ZP で **m ∈ [4, 8] かつ MXU 非搭載機**では、
   `small_m_weights` が (ZP, 8) を Tile8x32x128 に割り当てたせいで small-batch DirectSchedule に乗るが、
   そのデコードは symmetric 専用（`schedules/small_batch/direct.h` の `magic_code_float<128>` +
   `magic_dequant_bias`、zp を読まない）。本機で m=5/k=1024/n=128/g32 を実測:
   **expected −150 → got −87.5（41% 誤差）**。出所は我々のコミット e2cef535（upstream は symmetric のみ）。
   対処案（1 行）:
   ```diff
   --- a/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/selection.rs
   +++ b/crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/selection.rs
   @@ -53,7 +53,7 @@
   -        (GemmBPrologueKind::ScaleSymmetricDequant | GemmBPrologueKind::ScaleZeroPointDequant, Some(8)) => {
   +        (GemmBPrologueKind::ScaleSymmetricDequant, Some(8)) => {
                Some((ActivationFormat::Bf16, GemmTiling::Tile8x32x128_Simdgroups1x2x4, 256))
            },
   ```
   これで W8 ZP の m∈[4,8] は staged GEMM (Tile8x32x32) / GEMV に落ち、正しくなる。
   本バグは split-k 修正と独立。0.8B モデルを本機でバッチデコード (m=4..8) する場合に実害となる。
5. **gemv_test に W8 ZP の直接ケースが無い**: `gemv_test.rs` は W4 ZP と W8 symmetric のみ。
   W8 ZP の gemv は non-MXU 機で dispatch テストの m=2 ケースが間接的に踏む（実測パス）が、
   直接ケースの追加を推奨。
6. **bf16 partial の残余リスク**: symmetric W8 / W4 全般は bf16 partial のまま。データ運で
   許容差 (abs 0.5) を超える可能性は構造的にゼロではないが、既存テスト範囲では問題なし。

### 受入条件との対応

| 条件 | 状況 |
|---|---|
| 1. split_k>1 で w8_zp_* 全一致 | ✅ 実機で 14（+追加 1）ケース全一致 |
| 2. w8_sym_* と 4bit の退行なし | ✅ matmul モジュール全数 192/0 |
| 3. 実形状で split_k>1 が選ばれても正しい | ✅ 選択（Mxu sk=4）は unit test で担保 / 実行は Simdgroup sk=8 の同規模形状で検証 / MXU 実行は M3 では不正動作と実証済みのため M5+ 実機で要確認（§5.2） |
| 4. 難しければ撤退 | 不要 — 原因は丸め精度で、幾何は最初から正確だった |
