# 無効な結果（参照しない）

2026-11-07 の監査で、プロトコル（100 chunks / dedup コーパス）を満たさない結果を検出した。
**数値を引用しない。**

## 50 chunks（旧 eval）

| ディレクトリ | 言語 | BF16 | 候補 | 備考 |
|---|---|---:|---:|---|
| `rtn-int4-g32` | en/ja | 37.18 / 42.17 | — | 旧コーパス（sha 35c30…, f1f6b…） |
| `yaqa-hin-hout-int4-g32` | en/ja | 同上 | — | 同上 |
| `yaqa-hin-hout-emb-int4-g32` | en/ja | 同上 | — | 同上 |
| `yaqa-hin-hout-mixed-int4-g32` | en/ja | 同上 | — | 同上 |
| `official-0.8b-m` | en/ja | 同上 | — | 同上 |

同じコーパス sha でも clean 実行（`342e8f6a…` / `707c9a6c…`）の BF16 は 35.38 / 43.42 で一致している。
旧結果の 37.18 / 42.17 は**別コーパス**（dedup 前）なので比較できない。

## 有効な基準（clean / dedup / 100 chunks）

| 条件 | EN BF16 | JA BF16 | 参照 |
|---|---:|---:|---|
| clean enonly | 35.38 | 43.42 | `clean-enonly-int4-g32/` |
| clean mixed50 | 35.38 | 43.42 | `clean-mixed50-int4-g32/` |

新しい結果は必ず `check_experiment.py` を通す。
