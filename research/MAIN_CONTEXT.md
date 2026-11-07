# MAIN CONTEXT (subagent bootstrap)

このファイルはサブエージェントが最初に読む。ここに書かれた状態を前提に調査する。本番コードは指示がない限り変更しない。

## 目的
Huihui Qwen3.5（0.8B/4B/9B、abliterated + Claude 4.6 Opus distill）を uzu 用 RHT+INT4+YAQA/QAD に変換し、M3 16GB で動かす。0.8B は base `Qwen/Qwen3.5-0.8B` を使う検証台。4B/9B が本命。

## 現在の実測
### 0.8B base PPL
| 条件 | EN | JA |
|---|---:|---:|
| BF16 | 40.81 | 36.58 |
| RTN | 45.29 | 42.47 |
| E3 EN校正 | 43.81 | 43.05 |
| mixed50 | 43.17 | 41.70 |
| mix65 (JA65/EN35) | 41.13 | 47.58 |
| 公式 0.8B-M | 41.97 | 42.14 |
| E3b emb4 | 45.07 | 44.37 |

mix65 は「JA を増やしたのに JA が悪化」する異常。原因調査中。

### uzu/lalamo parity
同一 0.8B RTN 重みで cosine 0.999775 / symmetric KL 0.00227 / top5 0.96。export/読み込みは正常。

### 4B/9B
- 4B RTN: `models/huihui-4b-int4-rtn` 2.40GB、EN 14.0 t/s / JA 18.6 t/s。
- 9B RTN: `models/huihui-9b-int4-rtn` 5.19GB、EN 6.28 t/s / JA 11.47 t/s。
- 4B calibration は Kaggle で SIGKILL（H_I/モデルロードで OOM）。Modal へ格上げ。

## 重要パス
- repo: `/Users/kohei/Myproject/uzu`
- 0.8B base BF16: `/tmp/qwen35_0.8b_bf16_lalamo_lalamo_origcfg`
- 0.8B base RTN: `/tmp/qwen35_0.8b_rht_int4_lalamo`
- 0.8B Huihui abliterated: `models/huihui-0.8b-abliterated-bf16-lalamo-eval`（準備中）
- 4B RTN: `models/huihui-4b-int4-rtn`
- 9B RTN: `models/huihui-9b-int4-rtn`
- scripts: `scripts/{rht_int4_convert,yaqa_calibration,qad_finetune,eval_quantized,compare_engines}.py`
- logs: `logs/2026-11-07/`

## 制約
- M3 16GB。9B は INT4 5.19GB で動作。BF16 不可。
- Kaggle GPU は同時2本まで。4B calibration は不可。
- Modal は従量課金。0.8B の結果を見てから 4B/9B に投下。
- `rht_int4_convert.py` は `--embedding-bits` 対応済み。Kaggle dataset は更新が必要な場合あり。
- QAD は 0.8B で `replicated`、4B/9B は TP+AdamW or reshard Muon。

## 未解決
1. mix65 の JA 悪化原因（校正データ / preconditioner / モデルクセ）
2. QAD 50step の loss / PPL / uzu generate
3. emb8 の EN/JA PPL
4. 4B/9B の YAQA と layer 別 mixed precision
5. 無検閲・拒否の評価ハーネス

## サブエージェントの使い方
- 最初にこのファイルを読む。
- 数値は【実測】【推定】【引用】を分離。
- パスは `file:line`。
- 成果物は `research/` に 8〜12KB で出す。
- 本番コード変更は禁止。sandbox 検証のみ。
