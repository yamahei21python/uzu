# Qwen3.5-0.8B Huihui abliterated 同一実験

目的: `Qwen/Qwen3.5-0.8B` ベースで起きた **mix65（JA65/EN35）で JA が悪化する異常**が、無検閲 abliterated モデルでも再現するか確認する。

## モデル
| モデル | 用途 | パス |
|---|---|---|
| Qwen3.5-0.8B base | 既存結果 | `/tmp/qwen35_0.8b_bf16_lalamo_lalamo_origcfg` |
| Huihui-Qwen3.5-0.8B-abliterated | 今回 | `models/huihui-0.8b-abliterated-bf16-lalamo-eval` |

Huihui 0.8B は **abliterated のみ**。Claude 4.6 Opus 蒸留は入っていない。4B/9B の「蒸留 + 無検閲」とは切り分けられる。

## 条件
| 条件 | 校正コーパス | 既存 base 結果 (EN/JA) |
|---|---|---|
| EN-only | `calibration_en_wikitext2_dedup.txt` | 43.81 / 43.05 |
| mixed50 | `calibration_mixed_en_ja_dedup.txt` | 43.17 / 41.70 |
| mix65 | `calibration_mixed_ja65_en35_dedup.txt` | 41.13 / 47.58 |
| mix35 | `calibration_mixed_ja35_en65_dedup.txt` | 実行中 |
| JA-only | `calibration_ja_wikipedia_dedup.txt` | 実行中 |

YAQA 設定は base と同一:
`samples=32`, `max_tokens=128`, `output_block=small`, `output_block_limit=2048`, `damping=1e-3`, RHT block 32, INT4 g32 asym.

## 判定
| 結果 | 解釈 |
|---|---|
| Huihui でも mix65 で JA 悪化 | 無検閲モデル固有ではなく、校正/preconditioner 側 |
| Huihui では再現しない | abliteration が Hessian / 活性化分布に影響 |
| Huihui で悪化が増幅 | 無検閲 + 言語非対称が原因候補 |

## 実行
```bash
cd /Users/kohei/Myproject/uzu
zsh experiments/qwen35-0.8b-huihui/run_ratio_suite.sh
```

## 出力
```
experiments/qwen35-0.8b-huihui/
  yaqa-hin-hout-{enonly,mixed50,mix65,mix35,jaonly}-int4-g32/
    calibration.json
    ppl_en.json
    ppl_ja.json
logs/2026-11-07/local/huihui08b_ratio.log
```

## 注意
- `calibration.npz` は変換後に削除する（1.9GB/条件）
- 評価コーパスは base と同一。モデルだけ差し替える
- まず mixed50 と mix65 の2条件だけでも判別可能
