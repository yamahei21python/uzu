# Qwen3.5-0.8B 量子化実験

比較条件を固定するための実験ディレクトリ。

## レイアウト

```
experiments/qwen35-0.8b/
  data/
    wikitext2_test.txt        # 評価コーパス（差分評価用。Calibration に使わない）
    calibration.txt           # Preconditioner 推定用（評価と分離）
  bf16/
    model_info.json
    eval_wikitext2.json
  rtn-int4-g32/
    quant_config.json
    reconstruction.json       # scripts/verify_rht_int4_model.py --json-out
    eval_wikitext2.json       # scripts/eval_quantized.py --json-out
  yaqa-hin-int4-g32/
    calibration.json
    reconstruction.json
    eval_wikitext2.json
  yaqa-hin-hout-int4-g32/
    calibration.json
    reconstruction.json
    eval_wikitext2.json
```

## 固定する条件

- 元モデルの revision
- lalamo commit（`e401e17761f2fcd337b0cf20218d67d656e6ecba`）
- uzu commit
- tokenizer / チャットテンプレート
- トークン列（teacher forcing）
- 量子化対象層、bits、group size、symmetric、layout
- Calibration / Validation / Test の分割

## 実行例

```bash
# 1. 評価コーパスを固定
HF_HUB_ENABLE_XET=0 python scripts/fetch_eval_corpus.py \
  --dataset wikitext --config wikitext-2-raw-v1 --split test \
  --output experiments/qwen35-0.8b/data/wikitext2_test.txt \
  --max-characters 2000000

# 2. 重み再構成（層別）
python scripts/verify_rht_int4_model.py \
  --quantized-dir /tmp/qwen35_0.8b_rht_int4_lalamo \
  --reference-dir /tmp/qwen35_0.8b_bf16_lalamo \
  --json-out experiments/qwen35-0.8b/rtn-int4-g32/reconstruction.json

# 3. teacher-forced PPL / logits
JAX_PLATFORM_NAME=cpu python scripts/eval_quantized.py \
  --bf16-dir /tmp/qwen35_0.8b_bf16_lalamo_lalamo_origcfg \
  --candidate-dir /tmp/qwen35_0.8b_rht_int4_lalamo_lalamo_origcfg \
  --corpus experiments/qwen35-0.8b/data/wikitext2_test.txt \
  --max-sequences 100 --max-tokens 128 \
  --json-out experiments/qwen35-0.8b/rtn-int4-g32/eval_wikitext2.json
```

## 実験マトリクス

| 実験 | RHT | YAQA | ディレクトリ |
|---|---|---|---|
| E0 | なし | なし | `plain-int4-g32/` |
| E1 | あり | なし | `rtn-int4-g32/` |
| E2 | あり | H_in | `yaqa-hin-int4-g32/` |
| E3 | あり | H_in + H_out | `yaqa-hin-hout-int4-g32/` |

E0 は `scripts/rht_int4_convert.py --no-rht` で生成する。

## 評価の順序

1. Weight reconstruction（全 Linear）
2. teacher-forced logits / PPL
3. 指示追従（M-IFEval / JFBench）
4. 日本語タスク（llm-jp-eval の一部）
5. 長文（RULER / LongBench v2）

結果は `scripts/experiment_record.py` の JSONL にも追記する。

## 実測結果（2026-11）

### logits（lalamo、59位置）

| モデル | Top-1 | Top-5 | KL | cosine | サイズ |
|---|---|---|---|---|---|
| RTN | 0.780 | 0.719 | 0.364 | 0.966 | 800MB |
| YAQA H_in（旧条件） | 0.797 | 0.776 | 0.157 | 0.974 | 800MB |
| E3（H_in + H_O） | 0.780 | 0.759 | 0.157 | 0.974 | 800MB |
| E3b（+ embedding INT4） | 0.746 | 0.759 | 0.193 | 0.970 | 439MB |
| 公式 Mirai 0.8B-M | 0.881 | 0.793 | 0.108 | 0.983 | 419MB |

### 再構成（Hessian 重み付き）

- YAQA H_in は全96層で RTN より改善
- RTN 比の中央値 0.244x

### 公式チェックポイント

- 419MB（我々は 764MB、embedding BF16 の差）
- scale 比 0.99996、code 分布ほぼ一致
- 詳細: `research/checkpoint_compare_yaqa.json`

### ディレクトリ追加

- `rtn-int4-g32/` — RTN の再構成・logits
- `yaqa-hin-old-int4-g32/` — 旧条件 YAQA（token sum, 64 token, H_in）
- `yaqa-hin-hout-int4-g32/` — 修正版 calibration（512 token, H_O small）
- `official-0.8b-m/` — 公式モデル比較
