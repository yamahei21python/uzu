# uzu INT4 / RHT 内部調査

対象: uzu（`origin/main`、2026-11 時点）。Qwen3.5-0.8B の RHT+INT4 検証で確認した内容。

## 1. raw logits は公開 API に無い

`LanguageModelStream::generate()` はトークン ID のみ返す。
logits は内部でサンプリングに使われた後、外部へ出ない。

- `crates/uzu-engine/src/engine/language_model/stream/stream.rs`
  - `decoder_output.logits` を取得（prefill: 264 行付近、decode: 657 行付近）
  - そのまま `model.sampling.encode(...)` に渡す
- `LanguageModelStreamOptions` に logits 出力先が無い
- CLI / server にも logprobs 相当の API が無い

### 拡張する場合

1. `LanguageModelStreamOptions` に `logits_sink`（例: `std::sync::mpsc::Sender<Box<[f32]>>`）を追加
2. `logits.unwrap()` をサンプリング前にコピーし、command buffer 完了後に `copyout::<f32>` で取得
3. CLI に debug サブコマンドを追加

コスト: vocab 248k × f32 ≒ 1 MB/token。teacher-forced 評価なら 1 回の forward で
全位置分をまとめて取得できる（例: 64 token で 64 MB）。

## 2. RHT カーネルの符号順序

Metal 実装は lalamo と一致している。

- 入力側: `value * sign` → Hadamard
  - `backends/metal/kernel/activation_transform/activation_transform.metal`
- 出力側: Hadamard → `value * sign`
  - `backends/metal/kernel/hadamard_transform/hadamard_transform.h`
- block size は 32 固定
  - `backends/common/gpu_types/hadamard_order.rs`
- 出力側の符号は matmul epilogue で適用
  - `backends/metal/kernel/matmul/gemm/common/gemm_rht.h`

## 3. INT4 の復元式

- 重み: `(code - zero_point) * scale`
- nibble は low-first
  - `backends/metal/kernel/common/packed.h: read_packed<4>`
- `zero_point` は `-scale * zp` の bias として加算
  - `backends/metal/kernel/matmul/gemm/common/quant_scale_zero_point.h`
- scales / zero_points のレイアウトは `GroupOutput`（`[G, N]`、zero point は N 軸に pack）
  - `backends/common/kernel/matmul/mod.rs`

lalamo の export と一致する。

## 4. 対応する RHT 構成は限られる

`has_input_output_rht` は以下だけを RHT ラッパーとして扱う。

- `incoherence_block_size == Some(32)`
- `incoherence_processing_mode == InputOutput`

それ以外（block 64/128、Input のみ、Output のみ、`None`）は通常の `WeightMatrix` として読まれる。
`None`（RHT なし）は正しいが、Input/Output のみのモデルは**警告なしで誤った解釈**になる。

- `crates/uzu-engine/src/encodable_block/linear/rht_wrapper.rs`

現状の lalamo 出力（InputOutput, 32）は問題ない。
他構成を配布する場合は uzu 側の対応追加か、変換時の拒否が必要。

## 5. 性能の観測

0.8B の同一 M3 で:

| 条件 | BF16 | RHT+INT4 |
|---|---|---|
| 通常 sampling | 8.5 tok/s | 12.5 tok/s |
| top_k=1 + repetition penalty | 43〜57 tok/s | 6〜14 tok/s |

同じ条件で逆転するため、モデル固有の速度差とは言えない。
可能性:

- 生成長の違いによる平均の偏り
- `top_k=1` + repetition penalty でサンプリング経路が変わる
- INT4 GEMV が full-precision activation を要求し、余分な変換が入る
  - `gemv/kernel.rs`: prepared int8 activations require GEMM
- 出力 RHT が fused されず、別カーネルになるケース
  - `fuses_rht()` が false のとき `deferred_factors` 経由

現時点では未確定。速度を主張するなら同一生成長・同一 sampling での再測定が必要。

## 6. メモリ

uzu の表示:

- BF16: 1.49 GB
- RHT+INT4: 0.84 GB

モデルファイルは 1.4 GB / 764 MB。表示は重み＋実行時バッファを含む。

## 7. config の strict 仕様

`#[uzu_config]` マクロが全構造体に付与する。

- `#[serde(deny_unknown_fields)]`
- 全フィールド必須（`Option<T>` でもキー省略不可）

lalamo の `token_codec_config` はこの2点で不一致。
`scripts/normalize_lalamo_config.py` で吸収する。

## 8. 結論

- RHT / INT4 の数値解釈は lalamo と一致
- raw logits 比較を uzu で行うにはエンジン拡張が必要
- RHT 構成は InputOutput/32 のみ実質サポート
- 速度は条件依存。品質評価と切り分けて測る
