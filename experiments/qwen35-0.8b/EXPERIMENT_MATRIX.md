# 0.8B 徹底実験マトリクス

目的: JA 悪化の原因を分離し、4B/9B に転移できる校正・量子化・QAD の設計を決める。
固定プロトコル: `experiments/PROTOCOL.md`。eval/calibration は dedup 版のみ。

## Phase 0: clean baseline
| 条件 | 状態 |
|---|---|
| BF16 dedup eval 100chunks | 再評価必要 |
| RTN dedup eval 100chunks | 再評価中 |
| official 0.8B-M dedup eval | モデルがあれば |
| EN/JA eval 重複除去 | 完了（EN 17行 / JA 754行） |

## Phase 1: 校正言語（本命）
| # | 条件 | 内容 |
|---|---|---|
| 1 | EN-only | `calibration_en_wikitext2_dedup` |
| 2 | JA-only | `calibration_ja_wikipedia_dedup` |
| 3 | mixed50 | dedup mixed |
| 4 | mix65 | JA65/EN35 dedup |
| 5 | mix35 | JA35/EN65 dedup |
| 6 | **balanced H 50/50** | H_EN と H_JA を別々に推定 → 各 mean(diag) 正規化 → 50/50 混合 → damping |
| 7 | balanced 65/35 | 同上 |
| 8 | balanced 35/65 | 同上 |

判定: EN/JA PPL、BPC、top1/KL。balanced が mixed を上回るか。

## Phase 2: 校正ロバスト性
best language 条件に対して one-factor-at-a-time。
| 軸 | 値 |
|---|---|
| samples | 16 / 32 / 64 |
| max_tokens | 128 / 256 / 512 |
| damping | 1e-4 / 1e-3 / 1e-2 |
| H_O | none / small / all |
| seed | 0 / 1 / 2 |

## Phase 3: 量子化対象
| 軸 | 値 |
|---|---|
| Linear | RTN / YAQA |
| embedding/readout | BF16 / INT8 / INT4 |
| bits | 4 / 8 |
| group | 32 |
| RHT | block 32 |

## Phase 4: QAD
| 軸 | 値 |
|---|---|
| optimizer | AdamW / Muon |
| steps | 10 / 50 / 200 |
| lr | 2e-5 / 5e-5 / 1e-4 |
| grad clip | 1.0 |
| validation | 固定 2 chunks、5stepごと |
| checkpoint | 2step、keep 2 |

## Phase 5: 評価
- EN/JA PPL
- BPC = log2(PPL) × fertility（tokenizer 実測）
- top1 / top5 / KL / cosine（100 chunks）
- 生成サンプル EN/JA 各20
- 拒否・人格は Huihui 0.8B で別途

## 実行順
1. base clean suite（enonly / mixed50 / mix65 / mix35 / jaonly）
2. balanced H 50/50
3. balanced 65/35, 35/65
4. Phase 2（samples / seq / damping）
5. embedding/readout bits
6. QAD AdamW 10/50
7. BPC + 生成評価

## ルール
- 1条件 = 1ディレクトリ。config_hash を残す
- 結果は clean のみコミット
- 失敗条件は `FAILED.md` に理由を残す
- 0.8B の結論を 4B/9B に直接コピーしない
