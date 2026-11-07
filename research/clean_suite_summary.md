# 0.8B clean スイート 現状まとめ（2026-11-07）

## 完了（dedup / 100 chunks / 同一コーパス）

| 条件 | EN BF16 | EN INT4 | EN BPC Δ | JA BF16 | JA INT4 | JA BPC Δ |
|---|---:|---:|---:|---:|---:|---:|
| clean enonly | 35.38 | 39.39 | +3.02% | 43.42 | 62.52 | +9.67% |
| clean mixed50 | 35.38 | 39.87 | +3.36% | 43.42 | 49.93 | +3.70% |
| clean mix65 | 35.38 | 40.52 | +3.81% | 43.42 | 49.59 | +3.52% |

- 3 条件とも BF16 baseline は EN 35.38 / JA 43.42 で一致（比較可能）
- mixed50 が相対劣化をほぼ 1:1 に揃える（EN 3.36 / JA 3.70）
- mix65 は EN を少し犠牲にして JA を少し改善（想定どおりのトレードオフ）
- 生データ: `experiments/qwen35-0.8b/clean-{enonly,mixed50,mix65}-int4-g32/`

## 保留（実行待ち）
| レーン | 担当 | 何が分かるか |
|---|---|---|
| clean mix35 / jaonly | Muse CPU | 混合比の残り 2 点。曲線が閉じる |
| balanced-h（50/65/35） | Muse CPU | コーパス混合ではなく **Hessian 混合**の効果 |
| robustness 27 条件 | owner 3 + Muse 3（5 シャード） | samples / seq / damping の感度 |
| seed 分散（3 seed） | owner CPU | **3% の差がノイズか本物か** |
| damping 1e-3 vs 1e-2 | owner CPU | int4 調査の未証明部分の決着 |
| group size 32/64/128 | owner CPU | 4B/9B に 128 を勧めてよいか |
| 層別ビット（uniform4 vs down/out6） | owner CPU | 配分仮説と `--layer-bits-json` |
| cpu-tuning（calib bf16 / RTN） | 空き次第 | CPU 校正の高速化と YAQA の効果量 |
| QAD 10 step（freeze scales） | 空き次第 | QAD の詰まりの解消 |

## 判定の物差し
- 指標は **BPC の相対 Δ**（同一言語内）。PPL の絶対値は言語間で比較しない
- seed 分散が出るまで、3% 前後の差は「有意」と主張しない
- 全条件 `chunks: 100`、`eval_dtype` を記録（CPU は float32、T4 は不可）

## 次の判断
1. seed 分散の幅を見て、mixed50 vs mix65 の差が有意かを決める
2. damping / group / bits の結果で変換設定を確定
3. balanced-h の結果で「コーパス混合」と「Hessian 混合」のどちらを採るか決める
4. そこで 4B/9B に進む（Modal）
