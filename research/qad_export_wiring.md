# 0.8B QAD Export Wiring

対象: `/Users/kohei/Myproject/uzu`。根拠 = 【実測】(本機実行) / 【コード】 / 【推定】。

## 0. 結論

1. `save()` 出力は **3ファイルのみ**。`encoding.json` は無い → そのまま CLI に渡すと即死【実測】。
2. config は lalamo 形式 → normalize 必須。safetensors は `incoherence_spec` 入れ子 → rewrite 必須【実測】。
3. dtype は **quantized.scales=BF16 / norm.scales=F32** が正解。一括 BF16 化は壊す【実測】。
4. 2step 学習は**未実行**（検証環境 `/tmp` 消失）。他は実走確認済み。

## 1. 現状フロー

| # | 段階 | 実体 | file:line |
|:-:|:--|:--|:--|
| 1 | QAD学習 | `scripts/qad_finetune.py` | :95-214 |
| 2 | export | lalamo `BaseModel.save` → 3ファイル | `lalamo/model.py:35-52`, `:107-109` |
| 3 | spec 平坦化 | `rewrite_lalamo_hybrid_spec.py`（safetensors のみ） | :51-101 |
| 4 | config 正規化 | `normalize_lalamo_config.py`（config のみ） | :26-45 |
| 5 | 生成 | `target/release/cli` | `cli/.../non_interactive.rs:23-32` |

3→4 / 4→3 どちらでも可（対象ファイル直交）。`qad_finetune.py:213-214` は rewrite→normalize。

## 2. save() 生成物

### 2.1 ファイル

| ファイル | 生成元 | 必須 |
|:--|:--|:--|
| `model.safetensors` | `lalamo/model.py:44-49`（arrays+metadata） | 必須 |
| `config.json` | `:51-52`（`to_json()`、indent=4） | 必須 |
| `tokenizer.json` | `:107-109`（`tokenizer.save`） | 必須 |
| `encoding.json` | **生成されない** | 実質必須（§5-3） |

【実測】出力3ファイルのみ。`model.safetensors` = 800,718,936 B、tensor 614、metadata 193。

### 2.2 tensor 名 / dtype（0.8B YAQA INT4 実測）

| キー例 | dtype | shape 例 | 備考 |
|:--|:--|:--|:--|
| `decoder.transformer.layers.0.mlp.up_projection.weights.quantized.weights` | U8 | パック済 | 4bit 詰め【実測: 96本】 |
| `…weights.quantized.scales` | **BF16** | [rows, groups] | 96本すべて BF16【実測】 |
| `…quantized.zero_points` | U8 | [rows, groups] | `is_symmetric:false` で存在【実測】 |
| `…incoherence_signs.input_signs` | I32 | [in_dim] | `rht_wrapper.rs:173-174` |
| `…incoherence_signs.output_signs` | I32 | [out_dim] | 同 `:176` |
| `…norm.scales` | **F32** | [dim] | 79本【実測】。`normalization.rs:73` |
| `decoder.embedding.embedding.weights` | BF16 | [vocab, dim] | spec は別キー |

scales dtype は重み dtype 由来（`compressed/int.py:68-85`）。`Model.load(dtype=bfloat16)` 後も norm 系は F32 のまま【実測: F32 79 / BF16 96】。export は dtype 変換しない（`int.py:410-436`）。よって guard は**一括キャストではなく quantized.scales=BF16 の検査**。

### 2.3 metadata

`…weights.spec` = `HybridSpec`、`…weights.quantized.spec` = `IntSpec`。signs は `incoherence_signs.*`（`hybrid.py:105`）。save 直後は 96 本すべて入れ子【実測】。

## 3. rewrite / normalize

| スクリプト | 変更対象 | Before → After | file:line |
|:--|:--|:--|:--|
| rewrite | `model.safetensors` の `__metadata__` のみ | `incoherence_spec:{block_size,processing_mode}` → `incoherence_block_size`, `incoherence_processing_mode` | :34-48, :60-70 |
| normalize | `config.json` のみ | `response_format`/`reasoning_effort` 削除、`output_parser_regex`/`end_of_thinking_tag` を null 追加 | :22-23, :34-44 |

- 順序依存なし（rewrite=safetensors :51-101 / normalize=config :27）。
- 冪等【実測】: rewrite 2回目 **md5 不変**（`bf95a7…`）。normalize 2回目は `already normalized`（:39-41）。
- 副作用: `model.safetensors.lalamo.bak` を残す（:88-92、約800MB）。`--no-backup` で抑止。
- **穴**: normalize は `decoder_config` 未知キーを消さない（`uzu_convert.py:249-260` は削除）。0.8B は 5 キー一致で無害【実測】。両者とも `encoding.json` は作らない。

## 4. uzu のロード

### 4.1 探索順

| 順 | 動作 | file:line |
|:-:|:--|:--|
| 1 | `Engine::model` がディレクトリ判定 → 一致なしなら `model_by_path` | `legacy/uzu/src/engine/mod.rs:484`, `:552-557` |
| 2 | `config.json` ありで単一、無ければ**サブディレクトリ走査** | `registry/local/mod.rs:57-77` |
| 3 | `config.json` を `AnyModelConfig` としてパース | `uzu-engine/src/engine/mod.rs:93-96` |
| 4 | `config.json`→`model.safetensors`→`speculator/`→`tokenizer.json` | `engine/language_model/mod.rs:63-71` |
| 5 | Decoder 構築 → `assert_all_tensors_read()` | 同 `:75-99` |
| 6 | `encoding.json` は任意読込（不正なら warn） | `local/mod.rs:120-133` |

必須: `config.json`, `model.safetensors`, `tokenizer.json`。`encoding.json` はロード時任意だが**生成時に必須**（§5-3）。

### 4.2 strict 挙動

| 機構 | 内容 | file:line |
|:--|:--|:--|
| `deny_unknown_fields` | 全 config 構造体に付与。未知キー1個で失敗 | `uzu_config.rs:184` |
| `required` | 全フィールド必須。`Option<T>` でもキー欠落は失敗 | `:172`, `strict_serde.rs:26-28` |
| `MustBe` | `"type"` が型名と完全一致必須 | `:152-153` |
| tensor 検査 | shape と dtype の**完全一致**要求 | `safetensors.rs:165-173` |
| 未読検査 | 読まなかった tensor があるとエラー（一覧付き） | `safetensors.rs:110-124` |

### 4.3 Spec 必須フィールド

| 型 | 必須 | 制約 | file:line |
|:--|:--|:--|:--|
| HybridSpec | `type`, `quantization_spec`, `adapter_spec`, `incoherence_block_size`, `incoherence_processing_mode` | block=32・mode=`input_output` 必須（RHT 経路） | `hybrid_spec.rs:14-19`, `linear/mod.rs:117-134` |
| IntSpec | `type`, `bits`, `group_size`, `is_symmetric`, `layout` | `layout` は `output_input`/`input_output` | `int_spec.rs:6-11` |
| Layout | `output_input` / `input_output` | Linear は OutputInput 固定 | `weight_matrix/mod.rs` |

embedding も `decoder.embedding.embedding.spec` 必須（`embedding_table.rs:107`）。

## 5. 失敗トップ10

| # | 事象 | 原因 | 実測メッセージ / 対策 |
|:-:|:--|:--|:--|
| 1 | config パース失敗 | `response_format` 残存 | `data did not match any variant of untagged enum AnyModelConfig` → normalize【実測】 |
| 2 | spec パース失敗 | `incoherence_spec` 入れ子 | `Unable to deserialize metadata "...weights.spec"` → rewrite【実測】 |
| 3 | encoding 無し | save が出さない | `Unable to load model: can not get encoding config` → コピー【実測】 |
| 4 | 未読 tensor | 余剰 tensor（master_weights 等） | `Unread tensors: [...]` → INFERENCE 後 save |
| 5 | dtype 不一致 | norm.scales まで BF16 化 | `InvalidTensor{expected:F32}` → F32 維持（normalization.rs:73） |
| 6 | block_size≠32 | Hadamard 64/128 | `UnsupportedConfiguration(HybridSpec…)` → 32 固定 |
| 7 | embedding spec 欠落 | metadata 欠損 | `KeyNotFound("…embedding.spec")` |
| 8 | パス1段ズレ | `config.json` が下位 | `Model not found` → 親指定（local/mod.rs:57-77） |
| 9 | tokenizer 破損 | 別モデル混在 | `Tokenizer error` → save 出力を使う |
| 10 | 正規化済みを lalamo に戻す | 未知キー | `ForbiddenExtraKeysError` / `KeyError: 'response_format'`【実測】→ **QAD 入力は raw 必須** |

## 6. 最小検証手順

### (a) 2step / seq32 学習

```bash
cd /Users/kohei/Myproject/uzu
JAX_PLATFORM_NAME=cpu /tmp/lalamo-direct/.venv/bin/python scripts/qad_finetune.py \
  --teacher-dir /tmp/qwen35_0.8b_bf16_lalamo_lalamo_origcfg \
  --student-dir /tmp/qwen35_0.8b_yaqa_hin_hout_lalamo_eval \
  --corpus experiments/qwen35-0.8b/data/calibration_en_wikitext2.txt \
  --output-dir /tmp/qad_run_2step \
  --steps 2 --seq 32 --lr 1e-4
```

期待: `chunks: N x 32 tokens` → `step 1/2 kl=…` → `saved QAD model -> …`。`--student-dir` は**未正規化**必須（§5-10）。

### (b) 生成物検査

```bash
python3 - <<'PY'
import json,struct,collections
p="/tmp/qad_run_2step/model.safetensors"
f=open(p,'rb'); n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n))
md=h.pop("__metadata__",{})
print("tensors",len(h),"meta",len(md))
print("q",collections.Counter(v["dtype"] for k,v in h.items() if k.endswith("quantized.scales")))
print("n",collections.Counter(v["dtype"] for k,v in h.items() if k.endswith("norm.scales")))
print("nested",sum('incoherence_spec' in v for v in md.values()),
      "flat",sum('incoherence_block_size' in v for v in md.values()))
PY
ls /tmp/qad_run_2step   # config.json model.safetensors tokenizer.json のみ
```

合格基準: nested=0 / flat=96 / quant scales={BF16:96} / norm に F32 を含む。

### (c) CLI

```bash
cp /tmp/qwen35_0.8b_yaqa_hin_hout_lalamo_eval/encoding.json /tmp/qad_run_2step/
cd /Users/kohei/Myproject/uzu
target/release/cli --model /tmp/qad_run_2step --message "Say hi in one word." --seed 0
```

【実測】同配線（save→normalize→rewrite→encoding）で TTFT 0.07 s / 52.7 t/s / 0.84 GB。

### (d) 切り分け

| 症状 | 一次確認 | 原因 | 処置 |
|:--|:--|:--|:--|
| `AnyModelConfig` | `python3 -m json.tool <dir>/config.json` | normalize 未実行 | §3 実行 |
| `AnyWeightMatrixSpec` | (b) の nested 数 | rewrite 未実行 | §3 実行 |
| `can not get encoding config` | `ls <dir>/encoding.json` | 未コピー | encoding.json 配置 |
| `InvalidTensor … data_type` | (b) の dtype | 一括 cast / 手編集 | 元 dtype に戻す |
| `Unread tensors` | エラー内キー | 余剰 tensor | 再 export |
| `Model not found` | パス階層 | 1段ズレ | 親ディレクトリ指定 |

## 7. フォールバック

1. **再 normalize**: `python3 scripts/normalize_lalamo_config.py <dir>`（冪等）。
2. **再 rewrite**: `python3 scripts/rewrite_lalamo_hybrid_spec.py <dir>/model.safetensors --no-backup`（冪等）。
3. **手動 1 キー修復**: `rewrite(path, dry_run=True)` で差分確認後、ヘッダのみ差し替え（offset 連動のため `rewrite_lalamo_hybrid_spec.py:83-99` の関数を再利用）。
4. **再 export**: raw から `Model.load`→`save`→§3→encoding。**QAD 出力からの再 export は不可**（config が uzu 化済み、§5-10）。
5. **最終手段**: `uzu_convert.py --source hf-mlx-4bit|hf-bf16`。QAD 重みは失われる。

## 8. 観測性

| 時点 | 検査 | 合格 | 異常時 |
|:--|:--|:--|:--|
| 学習前 | student/teacher が **raw** config か | `incoherence_spec` あり | 正規化済みは別コピー |
| 学習中 | `kl` が有限 | NaN/Inf なし | 即中断（clamp は下限のみ :162-168） |
| save 直前 | 全葉 `isfinite` | True | 中断（NaN も保存される） |
| save 直後 | 3ファイル / 614 tensor / nested 96 | 一致 | export 経路を疑う |
| rewrite 後 | `flat=96, nested=0` | 一致 | 再実行 |
| normalize 後 | `token_codec_config` キー集合 | §3 の差分のみ | 手動パッチ |
| ロード時 | `assert_all_tensors_read` 通過 | エラーなし | Unread 一覧 |
| 生成時 | 固定 seed で反復崩壊なし | 意味のある文 | 軸/符号ズレ（**無音破壊**） |

無音破壊の本命 = scales 軸入替 / signs 符号反転。エラーにならず perplexity のみ崩れる。同一プロンプト・同一 seed で JAX と uzu の top-1 一致率を測るのが本筋（未実装）。

## 9. 未確認事項

1. **2step/seq32 学習の実走**: `/tmp` 消滅で未実行。lowering（JIT 定数 2.02GB）まで確認、完走・KL は未確認。
2. **QAD 後品質**: 未測定。
3. **Metal 数値一致**: CLI は実測 1 回。JAX logits 比較は未実施。
4. **encoding.json の正解 name**: 既存 `{"type":"hanashi","name":"qwen3.5"}` 流用。0.8B 用は未確定。
5. **uzu_convert 経由の再構築**: HF 入力前提のため lalamo 出力からは不可（コード読解のみ）。
6. **他モデルの decoder_config 余剰キー**: 0.8B では非発生。normalize 非対応のまま。
