# BPC / fertility 監査（0.8B clean enonly）

目的: PPL の言語間比較をやめ、文字あたりビット（BPC）と tokenizer の fertility で評価を正規化する。

## 式

```
BPC = log2(PPL) * scored_tokens / chars
```

- `scored_tokens` は `eval_quantized.py` の `tokens` と同じ定義。各 chunk は `len(chunk) - 1` 個の予測を持つ
- `chars` は採点対象の token 列をまとめて decode し、各 chunk 先頭（文脈のみ）の文字を除いた数
- chunk 境界で UTF-8 が割れた場合、1 chunk あたり最大 1 文字ずれる。誤差は 0.01% 未満

実装: `scripts/compute_bpc.py`、`--strict` で `chunks` / `tokens` の再現性を検査する。

## fertility（公開 tokenizer のみ、重み不要）

| corpus | chars | tokens | tok/char |
|---|---:|---:|---:|
| calibration_en_wikitext2_dedup.txt | 500,687 | 113,151 | 0.2260 |
| calibration_ja_wikipedia_dedup.txt | 521,460 | 316,881 | 0.6077 |
| calibration_mixed_en_ja_dedup.txt | 333,307 | 102,652 | 0.3080 |
| calibration_mixed_ja35_en65_dedup.txt | 371,548 | 107,938 | 0.2905 |
| calibration_mixed_ja65_en35_dedup.txt | 278,716 | 106,700 | 0.3828 |
| eval_en_wikitext2_dedup.txt | 501,486 | 117,964 | 0.2352 |
| eval_ja_wikipedia_dedup.txt | 506,028 | 318,087 | 0.6286 |

JA は EN の約 2.6 倍の token を消費する。同じ文字数でも JA の PPL は構造的に高く出る。

`make_mixed_corpus.py` の `--ja-ratio` は **UTF-8 バイト比**。JA は 1 文字 3 バイトなので、`ja65` の文字比は約 41% になる。mixed の tok/char が線形に伸びないのはこのため。

## 0.8B clean enonly / mixed50 の結果

条件: base `Qwen/Qwen3.5-0.8B`、RHT block32 + asymmetric INT4 group32、calibration 32 samples / 128 tokens / output_block small / damping 1e-3、eval = dedup 100 chunks。

| calibration | corpus | BF16 PPL | INT4 PPL | BF16 BPC | INT4 BPC | ΔBPC | 相対 |
|---|---|---:|---:|---:|---:|---:|---:|
| enonly | eval_en_wikitext2_dedup | 35.38 | 39.39 | 1.2428 | 1.2803 | +0.0375 | +3.02% |
| enonly | eval_ja_wikipedia_dedup | 43.42 | 62.52 | 3.4484 | 3.7817 | +0.3333 | +9.67% |
| mixed50 | eval_en_wikitext2_dedup | 35.38 | 39.87 | 1.2428 | 1.2845 | +0.0417 | **+3.36%** |
| mixed50 | eval_ja_wikipedia_dedup | 43.42 | 49.93 | 3.4484 | 3.5761 | +0.1277 | **+3.70%** |

生データ:
- `experiments/qwen35-0.8b/clean-enonly-int4-g32/bpc/report.json`
- `experiments/qwen35-0.8b/clean-mixed50-int4-g32/bpc/report.json`

### 読み方

- PPL だけ見ると enonly は「EN +11%、JA +44%」で JA が極端に悪く見える
- BPC では enonly が EN +3.0% / JA +9.7%
- **mixed50 で EN +3.36% / JA +3.70%**。相対劣化がほぼ 1:1 に揃う
- 合計は 12.7% → 7.1%。EN を犠牲にして JA を救ったのではなく、**両方が改善**した
- したがって `ja_ratio 0.5` は「等分点」の有力候補。mix65 と balanced H で上下を確認する
- BPC は文字あたりの情報量が言語で違うため、**EN と JA の絶対値は比較できない**。比較してよいのは同一言語内の Δ

## 注意

- JA の eval は件数・文字数が EN とほぼ同じになるよう作ってある（50万文字前後）。chunk 数と token 数は 100 / 12,700 で共通ではない（JA は token が多いので同じ 100 chunk でも被覆文字数が小さい）
- この BPC は「評価コーパスの文字」基準。モデルの生成品質は測らない
- 旧 eval（重複入り）の数値は使わない。ここは dedup 済みのみ

## 再現

```bash
python scripts/compute_bpc.py \
  --ppl-json experiments/qwen35-0.8b/clean-enonly-int4-g32/ppl_en.json \
  --ppl-json experiments/qwen35-0.8b/clean-enonly-int4-g32/ppl_ja.json \
  --tokenizer models/huihui-0.8b-abliterated-bf16-lalamo \
  --strict \
  --json experiments/qwen35-0.8b/clean-enonly-int4-g32/bpc/report.json
```

出力例:

```
| run | chunks | tokens | chars | tok/char | PPL | BPC |
| clean-enonly-int4-g32 | 100 | 12700 | 52573 | 0.2416 | 39.39 | 1.2803 |
| clean-enonly-int4-g32 | 100 | 12700 | 20036 | 0.6339 | 62.52 | 3.7817 |
```
