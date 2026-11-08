# 委託: uzu の int8 zero-point GEMM で split-k を正しく動かす（第2次）

## 依頼の背景

我々は uzu（Apple silicon 向けの Rust + Metal LLM 推論エンジン）に
**int8 の zero-point 量子化**を追加した。decode も prefill も数値一致まで確認済みで、
実モデル（wide8 asymmetric int8, 0.8B）もロード・生成できる。

ただし **`split_k` を int8 zero-point で有効にすると数値が壊れる**ため、
現在は `split_k = 1` に固定して回避している。これを正しく動かしたい。

## リポジトリ

```
git clone https://github.com/yamahei21python/uzu
```

（upstream は trymirai/uzu。公開フォーク）

## 現在の状態（我々が入れた変更）

1. `gemm.metal`: `ScaleZeroPointDequant` を BITS 8 でも許可（**A が int8 のときは除外**）
2. `gemv.metal`: ZP の制約を 4 → 4/8
3. `quant_unpack.h`: `decode_zero_point<8>` を追加（8bit は 1 byte = 1 個）
4. `gemm/common/operands.h`: `ZERO_POINTS_PER_BYTE` を定数 2 から `8 / BITS` に修正
5. `selection.rs`: (ZP, 8) を symmetric と同じ `Tile8x32x128_Simdgroups1x2x4` に割り当て、
   **int8 ZP では `split_k = 1` を強制**（暫定回避。コメントに理由あり）

テスト: `crates/uzu-engine/unit/backends/metal/kernel/matmul/quant_dispatch_test.rs`
の `signed_weights_full_precision_activations_parity_bf16` に次のケースがある。

```
w8_zp_m1  (k=1024, m=1,  g32)
w8_zp     (k=256,  m=2,  g32)
w8_zp_m16_k256 / k512 / k768 / k1024(g32) / k1024(g64)
w8_zp_m16 / m33 (k=1024, g32)
w8_sym_* (対照)
```

現状は `split_k=1` なので **14 件すべて通る**。

## 症状（実測）

- `split_k > 1` を許すと、**m=16 / m=33・k=1024 で相対 10% の誤差**
  （例: 期待 6.4375 → 実際 7.125）
- k=256/512/768 では誤差なし
- **symmetric int8 は同じ条件で誤差なし**

## 分かっていること

- bias は `out_bias = -scale * zero_point` として **ステージングされた全要素**に畳み込まれる
  （`gemm/common/quant_scale_zero_point.h` の `current_scale_bias`）
- symmetric が壊れない理由は説明できる:
  パディング要素は符号なしコード 128 に復元され、
  `128 * scale + (-128 * scale) = 0.0` で**完全に相殺**される。
  asymmetric は `-scale*zp` が残るため、パディングが split ごとに加算される

## 既に試して駄目だったこと（重要）

**案A: split 境界を group 境界に揃える**

```rust
while split_k > 1
    && (!shape.k.is_multiple_of(split_k * align)
        || !shape.k.is_multiple_of(split_k * group_size))
{
    split_k -= 1;
}
```

→ **parity の誤差は 1 ビットも改善しなかった**（同じ 3 ケースが同じ値で落ちる）。
「分割点が group を跨ぐこと」だけが原因ではない。

## 仮説（検証してほしい順）

1. `aligned_inner_iterations` の切り捨て
   （`kernel.rs` の `encode_split_k` で `(k / split_k) / k_step / simdgroups_k`）
   → 端数が落ちて k の一部が計算されない、または `k_offset` とずれる
2. `k_offset` の起点（`partition * aligned_inner_iterations * THREADGROUP_BLOCK_K`）が
   実際のステージング開始位置と一致しない
3. `quant_scale_zero_point.h` の `group_step_counter = 0` 初期化が
   `k_offset` が group 境界にある前提になっている（案A で直るはずだが直らなかった）
4. split-k のスクラッチバッファ（`temp`, `split_k * M * N`）と
   `gemm_split_k_reduce.metal` のインデックス計算が group を跨ぐケースで狂う

## 環境制約

- Linux sandbox。**Metal が無いのでビルド・実行はできない**。コード読解とパッチ作成のみ
- 数値の捏造禁止。不明は「不明」と書く
- 変更は提案（diff）まで。コミットしない

## 成果物

`docs/analysis/int8_zp_split_k_v2.md`（日本語、250 行以内）

- §1 失敗の再現条件（どの形で、どの値が、どれだけずれるか）
- §2 原因の特定（ファイル:行つき。案Aが効かなかった理由の説明を含む）
- §3 修正案（unified diff）。`split_k=1` の暫定回避を置き換えるもの
- §4 テスト計画（既存ケースのどれが split_k>1 を踏むか、追加すべきケース）
- §5 残るリスク（4bit 経路への影響、性能）

## 受け入れ条件

1. `split_k > 1` で `w8_zp_*` が全一致（許容差は既存テストと同じ）
2. `w8_sym_*` と 4bit の既存ケースが退行しない
3. 実形状（m=512, k=4096, n=4096, g32）で `split_k > 1` が選ばれても正しい
4. もし本質的に難しければ「なぜ難しいか」を明示して撤退してよい
   （現状の `split_k=1` は実害が小さいため、無理に直す必要はない）
