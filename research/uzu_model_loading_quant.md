# uzu モデル形式・ロード・量子化互換性 調査レポート

## TL;DR
- 【実測】lalamo の RHT+INT4 は `HybridSpec`（`IntSpec` 内包）として uzu に適合。Linear は `InputOutput`、Embedding は `Output`、Readout は `Input` の RHT モードに対応。
- 【実測】Embedding は tied/untied 両対応。tied 時は safetensors 上で `decoder.embedding.embedding.*` の二重プレフィックスになる。
- 【実測】設定 JSON / metadata は `#[serde(deny_unknown_fields)]` で完全一致が必須。未知キーは即失敗。
- 【実測】`scales` は BF16、`zero_points` は U8。`OutputInput` 重みの scales/zero_points は 4 列パディングが必須。
- 【実測】Speculator（DFlash+Weaver）は `speculator/` 下に配置し、実行デバイス名または `default` を含む `shapes.json` が必須。

### 互換性マトリクス表
|コンポーネント|対象|形式 / Spec|uzu 0.6.17|主な要件・制約|
|---|---|---|---|---|
|Linear|全層|HybridSpec(IntSpec 4bit)|【実測】対応|`incoherence_block_size: 32`, `mode: input_output`|
|Embedding(Tied)|`decoder.embedding`|HybridSpec(IntSpec 4bit)|【実測】対応|`mode: output`, `embedding_dim % 32 == 0`|
|Embedding(Untied)|`decoder.input_embedding`|HybridSpec(IntSpec 4bit)|【実測】対応|`mode: output`, `Layout::InputOutput`|
|Embedding(D4S4)|`decoder.input_embedding`|D4S4Spec|【実測】対応|Untied専用Lookup。Tied拒否|
|Readout(Tied)|`decoder.embedding`|Tied (Matrix共有)|【実測】対応|出力サインを最終段入力 RHT として適用|
|Readout(Untied)|`decoder.output_embedding`|HybridSpec(IntSpec 4bit)|【実測】対応|`mode: input`, `incoherence_block_size: 32`|
|Speculator(DFlash)|`speculator/`|DFlashSpeculatorConfig|【実測】対応|`shapes.json`（デバイス別設定）必須|
|Speculator(Weaver)|`speculator/`|WeaverConfig|【実測】対応|任意。候補プールから自己回帰探索|
|新量子化(Trellis)|全層|-|【実測】未対応|Mainline未登録（Metal先行）|

---

## 1 ディレクトリ構成

### 1.1 構成要素一覧と必須・任意
【実測】`load_language_model` における配置規則。

|パス|区分|形式|探索コード（パス:行）|
|---|---|---|---|
|`config.json`|**必須**|JSON (`LanguageModelConfig`)|`crates/uzu-engine/src/engine/language_model/mod.rs:62`|
|`model.safetensors`|**必須**|Safetensors（重み/メタデータ）|`crates/uzu-engine/src/engine/language_model/mod.rs:64`|
|`tokenizer.json`|**必須**|HF Tokenizer JSON|`crates/uzu-engine/src/engine/language_model/mod.rs:70`|
|`speculator/`|任意|ディレクトリ|`crates/uzu-engine/src/engine/language_model/mod.rs:67-68`|
|`encoding.json`|任意|JSON|`crates/legacy/uzu/src/registry/local/mod.rs:88`|

### 1.2 探索コードと振る舞い
- `config.json`: 【引用】`file::read(model_path.join("config.json"))` (`crates/uzu-engine/src/engine/language_model/mod.rs:62`) で読込。種別判定 (`crates/uzu-engine/src/engine/mod.rs:63-69`) でも使用。
- `model.safetensors`: 【引用】`ParameterLoader::new(...)` (`crates/uzu-engine/src/engine/language_model/mod.rs:64`) で開く。未読テンソル残留時は `assert_all_tensors_read()` (`同:98`) でエラー。
- `tokenizer.json`: 【引用】`Tokenizer::from_bytes(...)` (`crates/uzu-engine/src/engine/language_model/mod.rs:70`) で読込。
- `speculator/`: 【引用】`file::exists(&speculator_path)...` (`crates/uzu-engine/src/engine/language_model/mod.rs:68`) で存在時ロード。
- `encoding.json`: 【実測】コア非参照。レガシーCLI/GUI（`crates/legacy/uzu/src/registry/local/mod.rs:88`）のみ読込。

---

## 2 config schema

### 2.1 #[uzu_config] の strict 仕様
【実測】`crates/uzu-engine-macros/src/uzu_config.rs:183-187` の仕様:
1. `#[serde(deny_unknown_fields)]`: 未知キー混入時は即エラー。
2. 全フィールドに `strict_serde::required` が注入され省略禁止。
3. 継承時 `ty: monostate::MustBe!(<StructName>)` を `type` に注入し完全一致検査。
4. `Unsupported` 型 (`crates/uzu-engine/src/utils/strict_serde.rs:5`): 空列挙型。値指定時はデシリアライズ失敗し未対応機能を静的遮断。

### 2.2 主要コンフィグの Schema 一覧表
|構造体名|定義箇所|主要フィールド|説明|
|---|---|---|---|
|`AnyModelConfig`|`crates/uzu-engine/src/config/model/mod.rs:10`|`LanguageModelConfig`, `ClassifierModelConfig`, `DecisionModelConfig`|untagged enum|
|`LanguageModelConfig`|`crates/uzu-engine/src/config/model/language_model.rs:6`|`type`, `token_codec_config`, `decoder_config`, `generation_config`|言語モデル最上位|
|`DecoderConfig`|`crates/uzu-engine/src/config/decoder.rs:9`|`embedding_config`, `transformer_config`, `vocab_size`, `ple_model_config`, `embedding_norm_config`|デコーダ構成|
|`TransformerConfig`|`crates/uzu-engine/src/config/transformer.rs:6`|`layer_configs`, `output_norm_config`, `model_dim`, `hidden_dim`|Transformer構成|
|`AnyWeightMatrixSpec`|`crates/uzu-engine/src/config/weight_matrix/mod.rs:16`|`FullPrecisionSpec`, `MLXSpec`, `IntSpec`, `HybridSpec`, `D4S4Spec`|重み量子化仕様|

---

## 3 WeightMatrix ロード

### 3.1 AnyWeightMatrixSpec の仕様対応表
【実測】`WeightMatrix::load` (`crates/uzu-engine/src/encodable_block/weight_matrix.rs:102-177`) と `parse_spec` (`同:38-85`) の対応表。

|Spec名|Layout|bits|group_size|symmetric|zero_point|scales dtype|RHT modes|
|---|---|---|---|---|---|---|---|
|`FullPrecisionSpec`|`output_input` / `input_output`|-|-|-|なし|なし|単体不可(Hybrid経由)|
|`MLXSpec`|`output_input` / `input_output`|2,4,8|>0|false|なし(`biases`)|BF16/F32|単体不可(Hybrid経由)|
|`IntSpec`|`output_input` / `input_output`|2,4,8|>0|`is_symmetric`|U8 / なし|BF16/F32|単体不可(Hybrid経由)|
|`HybridSpec`|内部Spec準拠|内包|内包|内包|内包|内包|`input`,`output`,`input_output`|
|`D4S4Spec`|`input_output`のみ|4 (code)|4/64 col|固定|なし(ladder)|F16, BF16|`output`(Lookup専用)|
|`Trellis`|-|2〜4 bit|可変|【実測】未対応|-|-|Spec未登録(Metal先行)|

### 3.2 各 Spec の特徴と制約
- `IntSpec`: 【実測】bits は 2,4,8。bits=2 は group_size 32 固定。is_symmetric: true は zero_points 不要、false は U8 の zero_points 必須。
- `MLXSpec`: 【実測】Scale + Bias 方式。`scales` と `biases` テンソルを使用。
- `HybridSpec`: 【実測】内部Specを内包。adapter_spec は null のみ。block_size は 32 固定。

---

## 4 embedding / readout / tied・untied

### 4.1 EmbeddingTable::load が受ける spec
【実測】`crates/uzu-engine/src/encodable_block/embedding_table.rs:104-133` の受入条件。

|Spec|受入れ条件|ストレージ種別|output_hadamard_factors|tied 可否|
|---|---|---|---|---|
|`D4S4Spec`|`layout == InputOutput` かつ `embedding_dim % 64 == 0`|`Storage::D4S4`|あり（`output_hadamard_factors`）|**不可** (`as_matrix() == None`)|
|`HybridSpec`|`mode == Output`, `block_size == 32`, `embedding_dim % 32 == 0`|`Storage::Matrix`|あり（`incoherence_signs.output_signs`）|**可能** (`Storage::Matrix`)|
|`IntSpec`/他|`layout == InputOutput`|`Storage::Matrix`|なし (`None`)|**可能**|

### 4.2 tied / untied 判定と RHT output 有効条件
【実測】`crates/uzu-engine/src/encodable_block/embedding.rs:90-134` の判定と挙動。

|モード|Config type|Safetensors Subtree|動作・RHT 適用規則|
|---|---|---|---|
|**Tied**|`TiedEmbeddingConfig`|`decoder.embedding`|`EmbeddingTable::load` を実行（D4S4 拒否）。<br>spec が `HybridSpec` の場合、`readout_input_hadamard_factors` として `embedding_tree.subtree("incoherence_signs").output_signs` を抽出し Transformer 最終層出力に適用。|
|**Untied**|`UntiedEmbeddingConfig`|入力: `decoder.input_embedding`<br>出力: `decoder.output_embedding`|入力は `EmbeddingTable`、出力は `UntiedReadout` で独立ロード。<br>Readout は `mode == Input` の `HybridSpec` で `incoherence_signs.input_signs` を適用。|

---

## 5 speculator

### 5.1 speculator/ の構成とファイル仕様
【実測】`crates/uzu-engine/src/speculators/dflash_tfm.rs:97-156` に基づく仕様。

|ファイル|区分|データ構造 / 仕様|目的・内容|
|---|---|---|---|
|`config.json`|**必須**|`SpeculatorModelConfig`|`speculator_config: AnySpeculatorConfig::DFlashSpeculatorConfig`|
|`shapes.json`|**必須**|`HashMap<String, DFlashTfmTreeShape>`|デバイス名（`"Apple M5"` 等）または `"default"` をキーとする形状|
|`model.safetensors`|**必須**|Safetensors|`speculator.draft_model` と `speculator.weaver`（任意）の重み|

### 5.2 DFlash と Weaver、および target_layer_ids
- **DFlash (`draft_model`)**: 【実測】必須（`crates/uzu-engine/src/config/dflash.rs:9-22`）。メイン埋め込み RHT 係数を共有可能。
- **Weaver (`weaver`)**: 【実測】任意（`crates/uzu-engine/src/config/weaver.rs:6-18`）。候補プールから自己回帰ツリー探索。
- **target_layer_ids**: 【実測】指定中間層出力をドラフトの `context_projection` に入力。
- **shapes.json**: 【実測】tree_budget, max_tree_depth 等を定義。該当キー欠落時は自動スキップ。

---

## 6 変換時の落とし穴

【実測】lalamo から uzu へのエクスポート時の必須注意事項。

|項目|落とし穴の内容|発生エラー|回避策・要件|
|---|---|---|---|
|**未知フィールド**|余分なキーの混入|`unknown field`|`#[serde(deny_unknown_fields)]` のため完全一致必須|
|**metadata 階層**|`__metadata__` を nested 化|`HeaderDeserialization`|flat な `HashMap<String, String>`（値はJSON文字列）|
|**未読テンソル**|未使用テンソルの残留|`UnreadTensors`|`assert_all_tensors_read()` があるため不要テンソル厳禁|
|**Tied プレフィックス**|Tied Embedding の命名ミス|`KeyNotFound`|`decoder.embedding.embedding.*` と二重prefixになる|
|**scales dtype**|scales の型不一致|`InvalidTensor`|基本型の **BF16** で保存|
|**zero_points dtype**|INT4 zero_points の型不一致|`InvalidTensor`|storage_type の **U8**（2要素 pack）で保存|
|**incoherence signs**|RHT サインの型不一致|`InvalidTensor`|`DataType::I32` で保存|
|**4列パディング**|`OutputInput` 行列の列数不一致|`InvalidTensor`|4列パディング (`ALIGNMENT=4`)|
|**Embedding Layout**|EmbeddingTable の Layout ミス|`UnsupportedConfiguration`|`Layout::InputOutput` (`[emb_dim, vocab]`) 必須|

---

## 7 新量子化形式・モデル追加時の変更箇所

【実測】新量子化形式や新規モデル追加時の変更ファイル・関数一覧。

|レイヤー|役割|変更ファイルパス|対象関数 / 構造体|
|---|---|---|---|
|**Config**|新 Spec 定義|`crates/uzu-engine/src/config/weight_matrix/<spec>.rs`|`#[uzu_config(super::WeightMatrixSpec)] pub struct <New>Spec`|
|**Config**|Spec 登録|`crates/uzu-engine/src/config/weight_matrix/mod.rs`|`#[uzu_config_abstract(...)]` に新 Spec を追加|
|**Loader**|Spec パース|`crates/uzu-engine/src/encodable_block/weight_matrix.rs`|`parse_spec()` の match 分岐|
|**Loader**|重み読み込み|`crates/uzu-engine/src/encodable_block/weight_matrix.rs`|`WeightMatrix::load()`, `QuantizationInfo`|
|**Types**|量子化モード|`crates/uzu-engine/src/backends/common/gpu_types/quantization.rs`|`QuantizationMode`, `packing_divisor()`, `storage_type()`|
|**Linear**|実行ラッパー|`crates/uzu-engine/src/encodable_block/linear/mod.rs`|`Linear::new_mixed_precision()` の match 分岐|
|**Embedding**|埋め込み対応|`crates/uzu-engine/src/encodable_block/embedding_table.rs`|`EmbeddingTable::load()` の match 分岐|
|**Kernel IF**|Matmul 引数|`crates/uzu-engine/src/backends/common/kernel/matmul/matmul_b.rs`|`MatmulB`, `QuantizedB`|
|**Metal**|GPU カーネル|`crates/uzu-engine/src/backends/metal/kernel/matmul/gemm/`|Metal シェーダー (`.metal`), ディスパッチ Rust コード|
|**CPU**|リファレンス|`crates/uzu-engine/src/backends/cpu/kernel/matmul/reference.rs`|CPU デコード・行列積ロジック|
|**Converter**|lalamo 側|`lalamo/compressed/<spec>.py`, `weight_matrix.py`|`WeightMatrixSpec`, `WeightMatrix.export()`|

---

## 8 未確認事項

|項目|未確認の内容|推定・リスク|今後の確認方針|
|---|---|---|---|
|**Trellis Mainline 化**|Mainline の `AnyWeightMatrixSpec` に TrellisSpec が未登録の理由|【推定】カーネル実装先行でメタデータ仕様策定ブランチ合流待ち。|ブランチコミット履歴および PR の追跡|
|**WebGPU 最適化**|WebGPU バックエンドでの RHT+INT4 GEMM 実行性能|【推定】Metal向けSIMD最適化と異なりWGSL側フォールバックの可能性。|WebGPU 環境でのプロファイリング|
|**RoPE スケーリング**|Qwen3.5 長文コンテキスト時の動的 RoPE スケーリング完全互換性|【推定】`AnyRoPEConfig` はあるがQwen3.5固有パラメータ整合は要検証。|`crates/uzu-engine/src/config/rope/` の実装検証|

---

## 根拠ファイル一覧（パス:行）
- `crates/uzu-engine/src/engine/mod.rs:63-96`
- `crates/uzu-engine/src/engine/language_model/mod.rs:62-98`
- `crates/legacy/uzu/src/registry/local/mod.rs:88-100`
- `crates/uzu-engine-macros/src/uzu_config.rs:10-93, 95-190`
- `crates/uzu-engine/src/utils/strict_serde.rs:4-29`
- `crates/uzu-engine/src/config/model/mod.rs:10-17`
- `crates/uzu-engine/src/config/model/language_model.rs:5-9`
- `crates/uzu-engine/src/config/decoder.rs:8-16`
- `crates/uzu-engine/src/config/transformer.rs:5-12`
- `crates/uzu-engine/src/config/weight_matrix/mod.rs:9-24`
- `crates/uzu-engine/src/encodable_block/weight_matrix.rs:38-85, 102-177`
- `crates/uzu-engine/src/encodable_block/linear/mod.rs:98-135`
- `crates/uzu-engine/src/encodable_block/linear/rht_wrapper.rs:39-48, 114-197`
- `crates/uzu-engine/src/encodable_block/linear/matmul.rs:73-100`
- `crates/uzu-engine/src/encodable_block/linear/untied_readout.rs:24-75`
- `crates/uzu-engine/src/encodable_block/embedding.rs:42-51, 82-134`
- `crates/uzu-engine/src/encodable_block/embedding_table.rs:104-133, 199-236`
- `crates/uzu-engine/src/config/speculator/model.rs:5-8`
- `crates/uzu-engine/src/config/speculator/dflash.rs:5-9`
- `crates/uzu-engine/src/config/dflash.rs:8-22`
- `crates/uzu-engine/src/config/weaver.rs:5-18`
- `crates/uzu-engine/src/speculators/dflash_tfm.rs:62-85, 97-156`
- `crates/uzu-engine/src/safetensors.rs:60-65, 110-124, 143-154`
- `crates/uzu-engine/src/backends/common/kernel/matmul/mod.rs:92-129`
- `crates/uzu-engine/src/backends/common/gpu_types/quantization.rs:15-28`
- `lalamo/modules/embedding.py:162-171, 216-226`
- `lalamo/compressed/quant_params.py:12-30, 53-56`
