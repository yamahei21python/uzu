# E0〜E3 実行手順

対象: Qwen3.5-0.8B。条件は `README.md` のマトリクスに固定する。

## 0. ディレクトリ

```
DATA=experiments/qwen35-0.8b/data
EXP=experiments/qwen35-0.8b
```

評価コーパスと Calibration コーパスは必ず別ファイル。

## 1. E1: RHT + INT4 RTN（基準）

```bash
JAX_PLATFORM_NAME=cpu python scripts/rht_int4_convert.py \
  --spec-name Qwen3.5-0.8B \
  --load-from /tmp/qwen35_0.8b_bf16_lalamo_lalamo_origcfg \
  --lalamo-path /tmp/lalamo-direct \
  --output-dir /tmp/qwen35_0.8b_rht_int4_lalamo
```

`--load-from` は保存済み lalamo モデルから開始する（HF 再インポート不要）。
初回だけ `--bf16-dir` で BF16 を保存する。

`--bf16-dir` は lalamo 形式のまま保存される。uzu 用と lalamo 評価用で config が異なるため、
変換後に評価用コピーを作る。

```bash
python scripts/make_lalamo_eval_copy.py \
  --source /tmp/qwen35_0.8b_rht_int4_lalamo \
  --dest /tmp/qwen35_0.8b_rht_int4_lalamo_eval
```

## 2. E0: RHT なし INT4

```bash
JAX_PLATFORM_NAME=cpu python scripts/rht_int4_convert.py \
  --spec-name Qwen3.5-0.8B \
  --lalamo-path /tmp/lalamo-direct \
  --output-dir /tmp/qwen35_0.8b_plain_int4_lalamo \
  --no-rht
```

注意: この場合 `incoherence_signs` は保存されない。`verify_rht_int4_model.py` は RHT なしを自動判定する。

## 3. E2: RHT + INT4 + YAQA(H_in)

```bash
JAX_PLATFORM_NAME=cpu python scripts/yaqa_calibration.py \
  --model-dir /tmp/qwen35_0.8b_bf16_lalamo_lalamo_origcfg \
  --corpus $DATA/calibration_en_wikitext2.txt \
  --output $EXP/yaqa-hin-int4-g32/calibration.npz \
  --json-out $EXP/yaqa-hin-int4-g32/calibration.json \
  --samples 32 --max-tokens 128 \
  --output-block none \
  --damping 1e-3

JAX_PLATFORM_NAME=cpu python scripts/rht_int4_convert.py \
  --spec-name Qwen3.5-0.8B \
  --lalamo-path /tmp/lalamo-direct \
  --output-dir /tmp/qwen35_0.8b_yaqa_hin_lalamo \
  --preconditioners $EXP/yaqa-hin-int4-g32/calibration.npz
```

## 4. E3: RHT + INT4 + YAQA(H_in + H_out)

H_O は `out_dim^2`。`in_proj` (8224) を含めると 1 層で 270 MB なので、まず `small` で out_dim <= 2048 の層だけ両側にする。

```bash
JAX_PLATFORM_NAME=cpu python scripts/yaqa_calibration.py \
  --model-dir /tmp/qwen35_0.8b_bf16_lalamo_lalamo_origcfg \
  --corpus $DATA/calibration_en_wikitext2.txt \
  --output $EXP/yaqa-hin-hout-int4-g32/calibration.npz \
  --json-out $EXP/yaqa-hin-hout-int4-g32/calibration.json \
  --samples 32 --max-tokens 128 \
  --output-block small --output-block-limit 2048 \
  --damping 1e-3
```

## 5. 重み再構成

```bash
python scripts/verify_rht_int4_model.py \
  --quantized-dir /tmp/qwen35_0.8b_rht_int4_lalamo \
  --reference-dir /tmp/qwen35_0.8b_bf16_lalamo \
  --json-out $EXP/rtn-int4-g32/reconstruction.json
```

## 6. logits / PPL

```bash
JAX_PLATFORM_NAME=cpu python scripts/eval_quantized.py \
  --bf16-dir /tmp/qwen35_0.8b_bf16_lalamo_lalamo_origcfg \
  --candidate-dir /tmp/qwen35_0.8b_rht_int4_lalamo_lalamo_origcfg \
  --corpus $DATA/eval_en_wikitext2.txt \
  --max-sequences 100 --max-tokens 128 \
  --json-out $EXP/rtn-int4-g32/eval_en_wikitext2.json
```

日本語も同じコマンドで `--corpus $DATA/eval_ja_wikipedia.txt`。

## 7. 実験記録

```bash
python scripts/experiment_record.py append experiments/qwen35-0.8b/records.jsonl record.json
python scripts/experiment_record.py validate experiments/qwen35-0.8b/records.jsonl
```

## 8. uzu 側

uzu は lalamo 形式の config を読めない。変換後のモデルディレクトリを uzu 用に正規化し、
raw logits が必要な場合は現状 API がないため、まずは生成品質で比較する。

```bash
python scripts/normalize_lalamo_config.py /tmp/qwen35_0.8b_rht_int4_lalamo
./target/release/cli --model /tmp/qwen35_0.8b_rht_int4_lalamo -m "..." --reasoning-effort disabled
```
