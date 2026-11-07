# 0.8B 測定監査（2026-11-07）

対象: `experiments/qwen35-0.8b/**` の PPL / calibration / logits。目的は「設定ドリフトで比較が壊れていないか」の確認。

## 結論

**比較表は3つの軸で壊れている。** 旧結果と新結果を同じ表に載せてはいけない。

1. **評価 chunk 数**: 50 chunks（6,350 tokens）と 100 chunks（12,700 tokens）が混在
2. **校正設定**: E3 は `samples=16, max_tokens=512, damping=1e-2`、比率系は `32, 128, 1e-3`
3. **モデル dir**: 旧 candidate は `/tmp` 消滅で再評価不能。JSON だけ残っている

## 1. 評価 chunk の混在

| 条件 | chunks | tokens | BF16 EN | BF16 JA |
|---|---:|---:|---:|---:|
| official 0.8B-M | 50 | 6,350 | 40.812 | 36.580 |
| RTN | 50 | 6,350 | 40.812 | 36.580 |
| E3 | 50 | 6,350 | 40.812 | 36.580 |
| E3b emb4 | 50 | 6,350 | 40.812 | 36.580 |
| mixed50 | 50 | 6,350 | 40.812 | 36.580 |
| mix65 | 100 | 12,700 | 37.181 | 42.166 |
| mix35 | 100 | 12,700 | 37.181 | 42.166 |
| JA-only | 100 | 12,700 | 37.181 | 42.166 |

同じ BF16 モデル・同じ corpus なのに BF16 PPL が 40.81/36.58 と 37.18/42.17 で違う。原因は chunk 数。**50 と 100 の candidate PPL は直接比較不可**。

## 2. 校正設定の混在

| 条件 | corpus | samples | max_tokens | output_block | damping |
|---|---|---:|---:|---|---:|
| E3 | EN WikiText | 16 | 512 | small | 1e-2 |
| mixed50 | mixed EN/JA | 32 | 128 | small | 1e-3 |
| mix65 | JA65/EN35 | 32 | 128 | small | 1e-3 |
| mix35 | JA35/EN65 | 32 | 128 | small | 1e-3 |
| JA-only | JA Wiki | 32 | 128 | small | 1e-3 |
| 旧 H_in | EN WikiText | 8 | 64 | none | 1e-3 |

RUNBOOK の記載（32/128/1e-3）と E3 の実測設定（16/512/1e-2）が不一致。E3 と比率系の差は「混合比」だけでなく**校正設定の差**も含む。

## 3. Provenance の欠落

- 旧 PPL JSON には `max_sequences`, `max_tokens`, `config_hash` が無い
- E3b には `calibration.json` が無い。E3 の preconditioner を使った可能性が高いが、記録なし
- candidate model dir は `/tmp` 消滅で大半が無い
- `logits.json` は 10 prompt / positions が全条件で一致。ここは問題なし
- seed は単発。bootstrap / 信頼区間なし

## 4. 正しい比較の作り方

### 固定する評価
- `max_sequences=100`
- `max_tokens=128`
- corpus: `eval_en_wikitext2.txt` / `eval_ja_wikipedia.txt`
- BF16 baseline: EN 37.181 / JA 42.166（100 chunks）

### 固定する校正
- `samples=32`
- `max_tokens=128`
- `output_block=small`, `limit=2048`
- `damping=1e-3`

### 再実行が必要
| 条件 | 理由 |
|---|---|
| mixed50 | 100 chunks で再校正・再評価（実行中） |
| RTN | 100 chunks で再評価（モデルあり） |
| E3 | 32/128/1e-3 で再校正・再変換・100 chunks eval |
| E3b | E3 と同じ校正 + emb4 + 100 chunks eval |
| official 0.8B-M | 100 chunks で再評価（モデルがあれば） |
| mix65/mix35/jaonly | candidate は 100 chunks で有効。校正も 32/128/1e-3 で一致 |

## 5. 再発防止

- `eval_quantized.py` に `config_hash` / `corpus_sha256` / `max_sequences` を追加済み
- 集計時に `config_hash` が違う行を**捨てる**か**エラーにする**
- 校正側も `samples/max_tokens/damping` を `calibration.json` に必ず残す
- 実験表は「同じ config_hash の行だけ」で作る
- 旧 50 chunks の結果は `archive` 扱い。新しい表に混ぜない

## 6. 現時点の暫定値

100 chunks の比率系のみ:

| 条件 | EN | JA |
|---|---:|---:|
| BF16 | 37.18 | 42.17 |
| mix65 | 41.13 | 47.58 |
| mix35 | 40.51 | 48.47 |
| JA-only | 45.60 | 48.46 |

mixed50 は 100 chunks で再評価中。この表に旧 mixed50（43.17/41.70）を混ぜない。
