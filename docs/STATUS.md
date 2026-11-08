# 現状

最終更新: 2026-11-08 / main `53d1a1b0` / テスト 435 passed

数値・実行状態・未確定事項の現在値を 1 枚にまとめる。経緯は `docs/SESSION_SUMMARY_2026-11-08.md`（前日は `_2026-11-07.md`）、規約は `docs/ENVIRONMENT.md`。

### 実行場所の規約（2026-11-08 追加）

重い処理は**ローカルの Mac で回さない**。手元の Mac は uzu 推論の実測専用。

| 処理 | 実行場所 |
|---|---|
| 変換・校正・評価（0.8B） | **Kaggle CPU**（1 アカウント 5 並列、週次クォータ無し） |
| 変換・校正（4B / 9B） | Modal A100（CPU 不可、T4 は bf16 不可） |
| CUDA 必須の実験（QAD の GPU 検証） | Kaggle GPU（T4×2） |
| **uzu の t/s 実測** | ローカル Mac（Apple silicon 専用、代替不可） |

ローカル変換は 400% CPU で 10 分以上回り、発熱とファン騒音の実害がある。
変換結果は Kaggle 出力から回収して `/tmp` を汚さない。

---

## 0. 2026-11-08 の知見（4点）

1. **damping 1e-2 は 1e-3 より有意に良い**
   mixed50（100 chunks）で EN +3.36% → **+2.16%**、JA +3.70% → **+2.73%**。
   enonly では JA +9.67% → +6.52%。改善幅は seed 幅（EN 0.89 / JA 0.23）を超える。
   4B/9B の config はまだ 1e-3。`d1e-1` を追加して最適点を確認中。

2. **QAD の GPU OOM 犯人は checkpoint（orbax）**
   T4×2 で TP 訓練は 0.39 s/step で完走（val_kl 0.342 → 0.318）。
   訓練後のピーク RSS 24.8GB。`--skip-checkpoint` を付けると export まで到達（`QAD GPU OK`）。
   → 対策は checkpoint の保存形式（opt_state を落とす / step を減らす / 別ボリューム）。

3. **`--layer-bits-json` は無効だった**
   lalamo のサイトパスは `.mlp/.down_projection`（ドット付き）で、規則 `mlp/down_projection` が
   一致せず全層 4bit のまま。uniform4 と down-out6 の PPL が完全一致（39.68 / 50.20）。
   `975fe2a6` でパス正規化を追加。再計測中。

4. **q512 の改善は damping の寄与が大きい可能性**
   q512 probe は `d1e-2` で実行しており、既存の q128 基準は `d1e-3`。
   seq の効果を言うには `s16_q128_d1e-2`（Muse shard3 の1番目）との比較が必要。
   それまでは「q512 が良い」と主張しない。

## 1. 数値

### 0.8B clean スイート（dedup / 100 chunks / 同一コーパス）

| 条件 | EN BF16 | EN INT4 | EN BPC Δ | JA BF16 | JA INT4 | JA BPC Δ |
|---|---:|---:|---:|---:|---:|---:|
| clean enonly | 35.38 | 39.39 | +3.02% | 43.42 | 62.52 | **+9.67%** |
| clean mixed50 | 35.38 | 39.87 | +3.36% | 43.42 | 49.93 | +3.70% |
| clean mix65 | 35.38 | 40.52 | +3.81% | 43.42 | 49.59 | +3.52% |

- BF16 baseline が 3 条件とも EN 35.38 / JA 43.42 で一致 → run 間比較は有効
- mixed50 が EN/JA の劣化をほぼ 1:1 に揃える（3.36 / 3.70）
- mix65 は EN を少し犠牲に JA を少し改善（想定内のトレードオフ）
- 生データ: `experiments/qwen35-0.8b/clean-{enonly,mixed50,mix65}-int4-g32/`

### 4B / 9B RTN（Modal A100 / 20 chunks / **暫定**）

| model | EN Δ | JA Δ | 備考 |
|---|---:|---:|---|
| 4B | +0.80% | +1.65% | YAQA 未適用 |
| 9B | +1.46% | +1.32% | YAQA 未適用 |

RTN は動作確認。本番は YAQA calibration 後。

### seed 分散（mixed50 / 100 chunks / seed 0-2）【実測】

| seed | EN PPL | EN BPC Δ | JA PPL | JA BPC Δ |
|---:|---:|---:|---:|---:|
| 0 | 39.196 | +2.89% | 50.399 | +3.95% |
| 1 | 40.462 | +3.78% | 50.618 | +4.07% |
| 2 | 39.260 | +2.93% | 50.180 | +3.84% |
| 幅 | 1.27 | **0.89pp** | 0.44 | **0.23pp** |

BF16 は 3 seed とも EN 35.364 / JA 43.422（完全一致）。

**判定**: clean スイートの mixed50 vs mix65（EN 3.36 vs 3.81、JA 3.70 vs 3.52）の差は、
- EN 0.45pp < seed 幅 0.89pp
- JA 0.18pp < seed 幅 0.23pp

→ **seed ノイズ内。混合比 50 と 65 の差は有意でない。** 変換設定の比較は同一 seed の範囲で行う。


### q512 on T4×2（TP=2 / float32 / 2026-11-07）【実測】

`s16_q512_d1e-2`（samples 16 / 512 tokens / damping 1e-2）、100 chunks。

| 段階 | 秒 |
|---|---:|
| calibration | 418 |
| convert | 289 |
| eval en | 440 |
| eval ja | 443 |
| **合計** | **1637（27分）** |

| 言語 | BF16 PPL | INT4 PPL | BPC Δ |
|---|---:|---:|---:|
| EN | 35.364 | 37.934 | **+1.97%** |
| JA | 43.430 | 47.764 | **+2.52%** |

- CPU の 1 条件 ~107 分に対し **約 3.9 倍速**
- mixed50 の seed 幅（EN 2.89-3.78 / JA 3.84-4.07）より良いが、**samples 16・damping 1e-2・seq 512 が同時に違う**
- seq だけの効果は `s16_q128_d1e-2`（Muse shard3）と比較して初めて言える
- T4 で必要な処理はすべて自動（xla / f32 activations / mesh）


### 6bit は lalamo で export できない（2026-11-08 実測）

`lalamo/compressed/quant_params.py:70-80` は量子化パラメータの幅を **4/8/16/32** のみ許す。
6bit で圧縮自体は通るが、export で落ちる。

```
ValueError: unsupported quantization parameter width: 6
```

- 影響: bit-alloc の「down/out を 6bit」は不可能。**8bit に置換**して再計測
- 影響: `research/9b_bit_allocation.md` の「4bit 59層 + 6bit 69層 = 5.0GB」も実装不能。
  8bit に置き換えるとサイズが増えるため、配分の再計算が必要
- 4bit / 8bit / bf16 の3択で設計し直す


### 4B は CPU で校正できない（2026-11-08 実測）

4B を bf16 でロードした時点で **9.9GB**。1 層・1 サンプル・32 tokens でピーク **26.9GB**。
4 層・8 サンプル・256 tokens では **SIGKILL (-9)**（サンプル 1 個目）。

- 内訳: 重み bf16 8GB + 勾配 + import 時の中間で ~27GB。層フィルタでは削れない
- **4B / 9B は Modal A100 が必須**。CPU で回す余地はない
- 0.8B だけが CPU 対象（bf16 校正で q512 まで可）

### group size の次元制約（2026-11-07 実測）

`rht_int4_convert.py` は **in/out 両方**が `max(block_size, group_size)` で割り切れることを要求する。

| model | Linear 数 | 不適合になる dim | g16 | g32 | g64 | g128 |
|---|---:|---|---:|---:|---:|---:|
| 0.8B | 97 | `in_proj` out=**8224** ×18 | OK | OK | NG | NG |
| 4B | 129 | `in_proj` out=**12352** ×24 | OK | OK | OK | NG |
| 9B | 130 | `in_proj` out=**12352** ×24 | OK | OK | OK | NG |

- 0.8B で group 64/128 は原理的に不可（`lanes/group-size-ablation-08b.json` は g16/g32 に縮小）
- 4B/9B は **g64 まで可、g128 は不可**。group size の判断は 4B で行う
- 出典: 各モデルの `model.safetensors` ヘッダ（2D Linear のみ、`.conv.` を除外）

### T4 (cc 7.5) の制約（2026-11-07 実測）

| 項目 | 事実 | 対策 |
|---|---|---|
| BF16 matmul | **不可**。実形状の attention で `ALG_DOT_BF16_BF16_F32` が UNIMPLEMENTED | `--calib-dtype float32` / `--eval-dtype float32` 固定 |
| Pallas-Triton | 不可（Triton は cc>=8.0） | `UZU_ATTENTION_IMPL=xla`。未設定なら cc から自動選択（`5e61da84`） |
| preflight | 小形状 probe は T4 でも pass していた | 実形状（512×1024 dot / 1×12×512×64 sdpa）に変更 |
| bf16 指定 | 起動時に拒否 | `eval_quantized.py` / `yaqa_calibration.py` が cc<8.0 で明示エラー |
| TP=2 | `--parallel tensor` で 2 枚を 1 メモリプールに | 0.8B q512 を実測中 |

`attention=xla` は Triton だけを回避する。BF16 の問題とは独立。

---













### GPU でトークンを増やしても割に合わない（2026-11-08 / 100 chunks / GPU）

| 条件 | 総トークン | EN | JA | 校正 |
|---|---:|---:|---:|---:|
| s64_q512_d1e-2 | 32k | 1.59% | 1.98% | 1,170 秒 |
| s128_q512_d1e-2 | 65k | 1.42% | 1.53% | 2,451 秒 |

- 改善は seed 幅（EN 0.30 / JA 0.53）以内。**有意でない**
- 校正時間は 2 倍。65k tokens は効率が悪い
- **0.8B の実用最良は CPU の wide8 + d1e-1（1.13 / 1.24）**。GPU の大トークン構成より良い
- `s32_q1024` は calibration が code 1 で失敗（原因未特定。seq 上限の検証は保留）

### 8bit の範囲拡大が効く（2026-11-08 / 100 chunks / CPU / d1e-1）

| 配分 | EN | JA |
|---|---:|---:|
| uniform4 | 1.85%（3 seed 平均） | 2.40% |
| down + out（down-out8） | 1.39% | 1.70% |
| **down + out + qkvg + out_projection（wide8）** | **1.13%** | **1.24%** |

- down-out8 と uniform4 の差: EN 0.46pp / JA 0.70pp。**両方とも seed 幅（0.44 / 0.25）を超える**
- wide8 と down-out8 の差: EN 0.26pp（幅内・未確定）/ JA 0.46pp（**幅超え・有意**）
- 9B の推奨配分（out_proj + qkvg + out_projection = 5.24GB）の根拠が 0.8B で揃った

### seed 幅（d1e-1 / s32 / q128 / uniform4）

| seed | EN | JA |
|---:|---:|---:|
| 0 | 1.99% | 2.26% |
| 1 | 1.56% | 2.43% |
| 2 | 2.00% | 2.51% |
| 幅 | **0.44pp** | **0.25pp** |

seed0 は clean レーンの mixed50-d1e-1 と完全一致（決定性の確認）。

### eval は両ディレクトリとも eval copy が必要

`eval_quantized.py` は `config.json` に `token_codec_config.response_format` を要求する。
公開 BF16 データセットは lalamo 素の config なので、**bf16 側も** `make_lalamo_eval_copy.py` を通す。
`run_lane` は自動でやるが、直接呼ぶときは手動で必要。




### 公式 0.8B-M の品質を同一条件で測った（2026-11-08 / 100 chunks / float32 / CPU 比較）

| モデル | EN | JA |
|---|---:|---:|
| 公式 `Qwen3.5-0.8B-M`（4bit g32 + emb + QAD） | +1.21% | **+3.83%** |
| **我々 wide8 + d1e-1 + emb4** | **+1.13%** | **+1.24%** |
| 我々 down-out8 + d1e0 + emb4 | +1.27% | +1.49% |
| 我々 uniform4 + d1e-1 + emb4 | +1.99% | +2.26% |

- **EN は互角、JA は我々が 2.6pp 上**（seed 幅 0.25pp の 10 倍）
- 公式は英語中心の校正・QAD と見られ、日本語で崩れる
- 「少し遅いが品質は上」は **JA で成立、EN は同等**
- 注意: wide8 側は asymmetric int8 の値（§8 の symmetric 再測が前提）


### RHT block 64/128 は構造的に不可能（2026-11-08 実測）

変換が `incompatible Linear shapes` で落ちる。

```
.decoder/.transformer/.layers/[14]/.mixer/.in_proj: (8224, 1024)
```

- `8224 = 32 × 257`（257 は素数）→ **64 で割れない**
- 9B も `in_proj` が 12352 = 32 × 386 で同様
- RHT のブロックは全次元を割り切る必要があるため、**block は 32 で固定**
- エンジン側の対応（`block_size == 32` のみ受け付ける）以前に、変換が成立しない

→ block 64/128 の品質プローブは無意味。エンジン拡張も不要。
### 8bit は symmetric でないと uzu で動かない（2026-11-08 実測）

wide8 / down-out8 の変換結果を `cli bench` に渡すと、カーネル未定義で落ちる。

```
Function _D4Gemm...ScaleZeroPointDequantS1V8S2V32S13VFullPrecisionS1V0
was not found in the library
```

原因は 2 つが重なっている。

1. `gemm.metal` の制約: `B_PROLOGUE != ScaleZeroPointDequant || BITS == 4`
   → **zero-point dequant は 4bit 専用**。8bit は symmetric 前提
2. `rht_int4_convert.py` が全ビット幅で `is_symmetric=False` を固定していた
   → 8bit 層が「8bit + zero point」というコンパイルされない組み合わせになっていた

修正: `--symmetric-bits`（既定 `8`）を追加し、そのビット幅の層を symmetric で書き出す。
`IntAffineParameters.from_weights` は symmetric 時に `zero_points=None` を返すので、
8bit 層の zero_point テンソルは消え、ファイルも小さくなる。

**W8 symmetric の制約（`gemm/selection.rs`）**: group 32 or 64、k は 256 の倍数、
decode は `Tile8x32x128_Simdgroups1x2x4`。0.8B の次元は満たす。

**含意**: 過去に測った wide8 / down-out8 の品質は **asymmetric int8** の値。
symmetric に変えると品質はわずかに動く（要再測）。速度も未実測だった（動かなかったため）。

### 総トークンが支配的で、seq 長の効果は小さい（2026-11-08 / 100 chunks）

| 条件 | 総トークン | EN | JA |
|---|---:|---:|---:|
| s64_q512 d1e-3 | 32,768 | 1.55% | 1.99% |
| s256_q128 d1e-3 | 32,768 | 1.71% | 2.02% |
| s32_q512 d1e-2 | 16,384 | 1.99% | 2.26% |
| s128_q128 d1e-3 | 16,384 | 2.17% | 2.07% |

- 同トークンなら q512 と q128 はほぼ同等（EN 0.16pp、JA 0.03pp）
- **CPU の q128 で最良条件が作れる**。GPU の q512 は不要
- 校正時間はトークン数に比例（CPU の s256_q128 は 6h かかった）

### 校正 dtype は f32 > bf16（2026-11-08 / s32 / q128 / d1e-2）

| 校正環境 | EN | JA |
|---|---:|---:|
| **CPU f32** | **2.16%** | **2.73%** |
| GPU TP=2 f32 | 2.51% | 3.04% |
| CPU bf16 | 2.79% | 2.94% |

- bf16 は EN で +0.63pp 悪化。メモリ節約と引き換えのコスト
- GPU は CPU より +0.35pp 悪い（TP の数値差）
- **校正は CPU f32 が第一選択**。bf16 は q512 を CPU で回すときの妥協

### 校正デバイスで 0.3pp 変わる（2026-11-08 実測）

同一条件 `s32 / q128 / d1e-2 / mixed50 / calib float32` を CPU と GPU で比較。

| デバイス | EN | JA |
|---|---:|---:|
| CPU（1 デバイス） | **2.16%** | **2.73%** |
| GPU T4×2（TP=2） | 2.51% | 3.04% |
| 差 | +0.35pp | +0.31pp |

- JA の差は seed 幅（0.23pp）を超える → **デバイス効果は本物**
- 原因候補は TP=2 の sharded dot（丸め・演算順序）
- 帰結: **校正は CPU で行い、GPU の数値とは混ぜない**（eval は 0.02% 一致で混在可）
- q512 の CPU bf16 1.34% vs GPU f32 1.97% の差のうち、約 0.3pp がデバイス由来と説明できる

### damping の最適値は samples に依存（2026-11-08 実測）

| 条件 | d1e-2 | d1e-1 | d1e0 |
|---|---:|---:|---:|
| mixed50 / s32 / q128（EN） | 2.16% | 1.99% | **1.58%** |
| mixed50 / s32 / q128（JA） | 2.73% | 2.26% | **2.04%** |
| enonly / s32 / q128（JA） | 6.52% | 4.20% | **3.15%** |
| s64 / q512（EN/JA） | 1.59 / 1.98 | 1.67 / 1.94 | 1.95 / 2.22（悪化） |

- サンプルが少ない（s32）ほど強い damping が効く。s64 では 1.0 は行き過ぎ
- 「damping は推定ノイズの補正」という解釈と一致
- **s32 q128 の最良は d1e0**（EN 1.58 / JA 2.04）。ただし EN は seed 幅内

### seed 幅（s64 / q256 / d1e-2、GPU 3 seed）

| 言語 | 値 |
|---|---:|
| EN | 1.72 / 2.02 / 1.97%（幅 0.30pp） |
| JA | 2.03 / 2.56 / 2.45%（幅 0.53pp） |

### CPU bf16 で s64 が回る

`s64_q256_d1e-2` を CPU bf16 で実行 → EN 1.86% / JA 2.42%。GPU の seed 平均（1.90 / 2.35）と同等。
**CPU だけで最良設定が作れる**。

### 8bit 配分 × damping 1e-1 が現最良（2026-11-08 / 100 chunks / CPU）

| 条件 | EN | JA | 置き場所 |
|---|---:|---:|---|
| **down-out8 + d1e-1（q128/s32）** | **1.39%** | **1.70%** | CPU |
| uniform4 + d1e-1（= mixed50 d1e-1） | 1.99% | 2.26% | CPU |
| s64_q256_d1e-1 | 1.48% | 1.98% | GPU |
| s64_q512_d1e-1 | 1.67% | 1.94% | GPU |
| mix35 d1e-1 | 1.65% | 2.30% | CPU |

- 8bit 配分の寄与は **EN −0.60pp / JA −0.56pp**（Muse の -0.81/-0.98 は誤り）
- `uniform4-d1e-1` と `mixed50-d1e-1` は同一値（37.992 / 47.286）→ レーン間の再現性 OK
- **CPU の q128/s32 が GPU の s64/q512 を上回る**（1.39/1.70 vs 1.67/1.94）
- mix35 は d1e-1 では mixed50 と同等（優位性は消える）

### bf16 校正で CPU の q512 が通る（2026-11-08 実測）

| 条件 | 校正デバイス / dtype | EN | JA | 校正時間 |
|---|---|---:|---:|---:|
| s16_q512_d1e-2 | CPU / **bf16** | **1.34%** | 2.54% | 1,163 秒 |
| s16_q512_d1e-2 | GPU / f32 (TP) | 1.97% | 2.52% | 418 秒 |

- f32 では CPU 30GB で -9。**bf16 なら完走する**（メモリ削減が本物）
- EN は bf16 校正の方が良いが、**デバイスも違う**（CPU 1枚 vs GPU TP=2）ので分離できない
- JA は差なし。**bf16 が品質を落とす証拠はない**
- 次の確認: CPU bf16 で `s64_q256_d1e-2`（通れば GPU 不要で EN 1.6 / JA 2.2）

### s64 では q256 と q512 がほぼ同等（2026-11-08 / 100 chunks / GPU）

| 条件 | EN | JA | 校正（秒） |
|---|---:|---:|---:|
| s64_q512_d1e-3 | 1.55% | 1.99% | 1,184 |
| s64_q512_d1e-2 | 1.59% | 1.98% | 1,170 |
| **s64_q256_d1e-2** | 1.62% | 2.16% | ~650 |
| s64_q256_d1e-3 | 1.63% | 2.39% | ~650 |
| s64_q256_d1e-4 | 1.74% | 2.42% | ~650 |

- EN は全条件 1.5〜1.7% で差なし。JA は q512 が 0.18pp 良い（seed 幅 0.23 以内）
- **s64 なら q256 で十分**。校正が 45% 短い（1 条件 ~20 分）
- q256 の d1e-2 vs d1e-3 は JA 0.23pp（ノイズ境界）。d1e-2 を採る

### damping と samples は代替関係（2026-11-08 / 100 chunks）

| 条件 | EN | JA |
|---|---:|---:|
| s64_q512 d1e-3 | **1.55%** | 1.99% |
| s64_q512 d1e-1 | 1.66% | **1.94%** |
| s64_q512 d1e-0 | 1.95% | 2.22% |
| s32_q128 d1e-2 | 2.16% | 2.73% |
| s32_q128 d1e-1 | 1.99% | **2.26%** |

- **s32/q128 では d1e-1 が JA を 0.47pp 改善**（seed 幅 0.23 を超える）
- **s64/q512 では d1e-1 の効果が 0.05pp**（有意でない）
- 解釈: damping は「推定ノイズの補正」。samples が増えてノイズが減ると、damping の効きも消える
- **damping は 1e-2 で十分**。1e-1 は samples が少ないときの保険。1e0 は行き過ぎ
- レーン間の baseline は 0.06% ずれる（GPU 35.364 / CPU 35.386）ので、EN Δ の比較は ±0.06pp の誤差込みで見る

### 併用は加算しない（2026-11-08 / 100 chunks / GPU）

Hessian 混合（balanced-h）の merge damping を 1e-2 にした結果。

| 条件 | EN | JA |
|---|---:|---:|
| balanced-35 | **1.75%** | 2.90% |
| balanced-50 | 2.12% | 2.63% |
| balanced-65 | 2.29% | **2.41%** |
| 参考: damping 1e-1 のみ | 1.99% | 2.26% |
| 参考: balanced-h d1e-3 | 2.32% | 2.72% |

- 加算しない。ratio で EN/JA のトレードオフが動くだけ
- **レバーは1つに絞る**。damping は無料なので damping を採用し、Hessian 混合は不採用
- 同じ機構（信頼できない固有値方向の抑制）を叩いているため、と解釈

### 再現性（2026-11-08）

`s32_q128_d1e-2` を別レーン（shard2 / clean）で 2 回測定 → どちらも EN 2.16% / JA 2.73%。
レーン間の再現性は良好。

### damping は 1e-1 まで改善（2026-11-08 / 100 chunks / CPU）

| damping | mixed50 EN | mixed50 JA | enonly EN | enonly JA |
|---:|---:|---:|---:|---:|
| 1e-3 | 3.36% | 3.70% | 3.02% | 9.67% |
| 1e-2 | 2.16% | 2.73% | 1.91% | 6.52% |
| **1e-1** | **1.99%** | **2.26%** | 2.26% | **4.20%** |

- mixed50 は 1e-1 でも改善が続く（1e0 を未検証）
- enonly は EN が 1e-2 で最良、JA は 1e-1 でも改善 → **EN/JA で最適点が違う**
- CPU（q128 / s32）で GPU の s64_q512（1.55 / 1.99）に迫る

### 層別ビット配分（2026-11-08 / 100 chunks / CPU）

| 条件 | EN | JA |
|---|---:|---:|
| uniform4 | 3.21% | 3.84% |
| **down/out 8bit** | **2.15%** | **2.59%** |

- 6bit が使えないため 8bit で代替。それでも約 1pp の改善
- サイズ増とのトレードオフ（配分設計は 4/8/bf16 で再計算が必要）

### Hessian 混合 vs コーパス混合（2026-11-08 / 100 chunks / d1e-3）

| 条件 | EN BPC Δ | JA BPC Δ |
|---|---:|---:|
| clean mixed50（コーパス混合） | +3.36% | +3.70% |
| balanced-50（Hessian 混合） | +2.33% | +3.12% |
| **balanced-65** | **+2.32%** | **+2.72%** |
| balanced-35 | +2.54% | +2.82% |

- EN で約 1.0pp、JA で 0.6〜1.0pp 改善。seed 幅を超える
- JA は balanced-65 が最良
- damping 1e-2 + コーパス混合（EN 2.16 / JA 2.73）と同程度 → **2つは独立レバーの可能性**
- 次の候補は併用（balanced-h を d1e-2 で）

### damping 1e-2 は 1e-3 より明確に良い（2026-11-08 / 100 chunks / CPU）

| 条件 | EN BPC Δ | JA BPC Δ |
|---|---:|---:|
| enonly d1e-3（既存） | +3.02% | +9.67% |
| **enonly d1e-2** | **+1.91%** | **+6.52%** |
| mixed50 d1e-3（既存） | +3.36% | +3.70% |
| **mixed50 d1e-2** | **+2.16%** | **+2.73%** |

- 改善幅: EN 1.1〜1.2pp、JA 1.0〜3.2pp
- seed 幅（EN 0.89 / JA 0.23）を超える → **有意**
- 含意: q512 probe（d1e-2）の良さは seq 長より damping の効果である可能性が高い。seq の効果は `s16_q128_d1e-2` との比較で分離する

### bit-alloc（down/out 6bit）は未検証

uniform4 と down-out6 の PPL が完全一致（39.68 / 50.20）→ `--layer-bits-json` の規則が
lalamo のドット付きパス（`.mlp/.down_projection`）と一致せず無効だった。`975fe2a6` で修正。
再計測が必要。

## 2. インフラ

### 今日の 3 修正（main `b4e4c828` / 186 tests）

| 修正 | 何を防ぐか |
|---|---|
| `env_fingerprint` | PyPI `lalamo==0.17.0` と git `e401e177` が両方 `0.17.0`。判定は **RHT 5 ファイルの sha256** |
| `publish_kit` | `kaggle datasets version` の既定 `--dir-mode skip` で `scripts/` `data/` が落ちる。公開後に listing と manifest を読み戻す |
| 検証 bug 修正 | 既定 20 件で `scripts/` が次ページ / zip mode は `scripts.zip` 名 |

### kit

- 最新: `b2b6920d`（HF `yamahei21/uzu-quant-kit` と Kaggle `yamahei21python/uzu-quant-kit-public` の commit 一致）
- `kit_manifest.json` に `capabilities` を追加

```
rht_int4_convert.layer_filter, eval_quantized.dtype,
preflight, preflight.sdpa_xla, env_fingerprint, qad.freeze_scales
```

外部カーネルは **commit hash ではなく capabilities を assert する**。

### lalamo の判定条件

| 項目 | 期待値 |
|---|---|
| `lalamo_commit` | `e401e17761f2fcd337b0cf20218d67d656e6ecba` |
| `probes.quant_roundtrip` | `pass` |
| `environment_fingerprint.lalamo_hash_complete` | `true` |

1 つでも欠けたら owner の数値と比較しない。

---

## 3. 実行状態（2026-11-08 20:45）

```
owner cpu 0/5   gpu 0/2
```

| kernel | lane | 状態 | 結果 |
|---|---|---|---|
| `uzu-qad-best` v5 | QAD 50 step（CPU / muon） | **ERROR**（64 分後に死亡、内部例外なし） | CPU も不安定 |
| **QAD v6（Muse 依頼）** | 最良 student（wide8 + emb4 / seq64 / muon） | **依頼済み**（Issue #31 コメント 6721748454） | GPU 枠が空いている側で |
| `uzu-t4-probe` | T4 で draft 学習が回るかの実測 | RUNNING | — |
| `uzu-sym8-eval` | symmetric wide8 の品質 | RUNNING | — |
| `uzu-wide-d1e0` | wide8 + damping 1.0 | RUNNING | — |
| `uzu-official-eval` | 公式 0.8B-M の品質 | COMPLETE | EN +1.21% / JA +3.83% |
| `uzu-alloc-downout8-d1e0` | damping 1e0 | COMPLETE | EN +1.27% / JA +1.49% |
| `uzu-sym8-build` | symmetric int8 のビルド | COMPLETE | wide 124.4 / downout 128.4 t/s |

### QAD v2 の失敗 = GPU OOM（v3 で修正）

```
RESOURCE_EXHAUSTED: Out of memory ... Free 34MB / Total 15.6GB
jit_train_step
```

`.bak` の誤判定は解消済み。残った問題は 2 つ。

1. **step 0 で 15.6GB を消費**。teacher(1.5GB) + master(fp32) + AdamW(m,v) + 勾配で足りない
2. **lr 2e-5 は 1 int4 step（5.8e-4）の 3.4%**。スクリプト自身が警告していた

v3 の変更: `--parallel fsdp` / `--freeze-scales` / `--skip-checkpoint` /
`--master-dtype float32` / `--lr 5e-4` / `--grad-clip 1.0`。

> QAD は v1（export 欠陥）、v2（GPU OOM）、v3（FSDP 不可）と 3 回失敗している。

**FSDP は 0.8B では使えない**（v3 の失敗理由）。

```
ValueError: Sharding spec ('fsdp',) implies that array axis 0 is partitioned 2 times,
but does not evenly divide the dimension size 1. Got shape: (1,)
```

FSDP は全配列を 2 分割するため、サイズ 1 の norm scale などで落ちる。
v4 は TP に戻し、メモリを `--freeze-scales` と `--skip-checkpoint` で削る → それでも step 0 で OOM
（AdamW の m,v が 6GB）。v5 は Muon を試そうとしたが **GPU の週次クォータ 30h を使い切った**ため、
**CPU に切り替えた**（0.8B は CPU で回る）。

### T4 で draft 学習が回るか（`uzu-t4-probe`）

測るのは matmul の実効 TFLOP/s（fp16/bf16/fp32）、SDPA、flash-attn の install/import、
`vllm-project/speculators` の install。DFlash 学習は flash-attn を要求する可能性が高く、
T4 は sm75 なので sm80 以上要求なら**その時点で不可**。

| 結果 | 判断 |
|---|---|
| flash-attn 不可 または bf16 が激遅 | Kaggle T4 は不可 → Modal A100 |
| SDPA で実用速度 | Kaggle で fine-tune（30h/週を分割） |

## 4. 結果回収

Issue #31 の決定どおり。**各自の `uzu-results` に上げ、owner が pull する。**

```bash
# Muse（カーネル内）
python scripts/run_lane.py --spec lanes/clean-suite-08b.json \
  --hf-results kentasatomuse/uzu-results

# owner（回収）
python scripts/pull_results.py --repo kentasatomuse/uzu-results --out results/muse --strict
```

- `calibration.npz` は上げない（1.4GB）。`result.json` / `ppl_*.json` / `manifest.json` で足りる
- `--hf-results` を付けずに走った分は `publish_results.py` で事後アップロード
- `publish_results.py` は `manifest.json` の `problems` が空でなければ **upload を拒否**
- `kentasatomuse/uzu-results` の 403 は **Kaggle dataset** の話。HF は public で owner から読める

```bash
python scripts/publish_results.py --out out --as muse --repo kentasatomuse/uzu-results
```

---

## 5. 未確定

| 保留 | 何で決まる | 状態 |
|---|---|---|
| mixed50 vs mix65 が有意か | seed 分散 | 差は seed 幅内で有意でない（暫定） |
| 変換設定（damping / group / bits） | damping + bit-alloc | **damping 1e0 を採用候補**。wide8 で未検証 |
| コーパス混合 vs Hessian 混合 | balanced-h（Muse） | 加算しない。混合比でトレードオフ |
| 4B/9B の emb/readout INT4 維持 | emb4 の値段 | 0.8B で EN +0.64 / JA +0.83pp。**emb8 を既定にすべき** |
| wide8 が EN で有意か | seed 追加 2〜3 本 | 平均 +1.05% vs down-out8 +1.39%（差 0.34pp < seed 幅 0.44pp） |
| **8bit 配分を採るか** | symmetric の実測 | **symmetric は +0.60 EN / +0.86 JA 悪化**。速度 −15% と釣り合うか要判断 |
| T4 で draft 学習 | `uzu-t4-probe` | **不可**（flash-attn タイムアウト、bf16 0.25 TFLOP/s） |
| QAD が効くか | QAD before/after ΔBPC | **効く（最弱 student で EN −1.46 / JA −0.88pp）**。強い student では逓減の見込み |
| 混合比の曲線（mix35 / jaonly） | Muse clean-suite | 実行中 |

### 判定の物差し

- 指標は **同一言語内の BPC 相対 Δ**。PPL の絶対値を言語間で比べない
- seed 分散が出るまで、3% 前後の差を「有意」と主張しない
- T4 の BF16 は未確定。`dot_f32_acc` / `sdpa_xla` が pass した実績があるため「T4 BF16 不可」を確定事項にしない

---

## 6. 次の一手

**優先度順のタスク一覧は `docs/NEXT_TASKS.md` に集約した。**

要約:

| 優先 | 内容 |
|---|---|
| P0（結果待ち） | asym8-f32 の品質 / QAD v2（Muse）/ MXU sk=4 / F32 ベンチ |
| P1（週明け） | 4B を 1 条件 → 9B 本体（Modal A100） |
| P2（速度の本命） | プロンプトから応答生成 → draft fine-tune → Weaver |
| P3（整備） | kit 再公開 / 0.8B 最終表 / run_lane のテスト |

## 8. 4B の実行設定（Modal / 2026-11-08 決定）

| 項目 | 値 |
|---|---|
| 置き場所 | **Modal A100-80GB**（Kaggle CPU は SIGKILL、T4 は bf16 不可） |
| calibration | `--calib-dtype bfloat16` / samples 8 / max_tokens 256 / damping 1e-2 / output_block none / 4層グループ |
| convert | bits 4 / group 64 / block 32 / RHT |
| 8bit 配分 | `out_proj` + `qkvg_projection` + `out_projection` → 約 **2.52GB** |
| eval | 100 chunks。A100 は bf16 で可（レーン間 baseline の 0.06% ずれに注意） |

**注意**: 校正を GPU で行うと CPU より約 **0.3pp 悪い**（実測）。4B の数値は
「GPU 校正由来」として扱い、0.8B の CPU 校正値と直接比較しない。

### 未確定（4B）

- 8bit の範囲（qkvg / out_projection は 0.8B で検証中）
- group 64 の品質（0.8B は g32 上限のため未検証）
- samples 8 で足りるか（0.8B では samples が効いた）

## 7. 参照

| ファイル | 内容 |
|---|---|
| `docs/SESSION_SUMMARY_2026-11-08.md` | 2026-11-08 の結果・バグ・方針変更 |
| `docs/SESSION_SUMMARY_2026-11-07.md` | 前日の経緯・決定事項 |
| `docs/ENVIRONMENT.md` | 環境一致と provenance の規約 |
| `docs/TASKS.md` | 全 33 タスク（`scripts/task_board.py --write` で再生成） |
| `docs/KAGGLE_SLOTS.md` | 枠運用と handoff |
| `research/clean_suite_summary.md` | 0.8B の結果表 |
| `docs/LOCAL_POLICY.md` | ローカルで重い処理を回さない規約 |
