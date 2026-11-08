# 現状

最終更新: 2026-11-07 / main `b4e4c828` / テスト 186 passed

数値・実行状態・未確定事項の現在値を 1 枚にまとめる。経緯は `docs/SESSION_SUMMARY_2026-11-07.md`、規約は `docs/ENVIRONMENT.md`。

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

## 3. 実行状態

```
owner cpu 4/5   gpu 0/2
```

| kernel | lane | 状態 |
|---|---|---|
| `uzu-lane-runner` | seed-variance-08b | running |
| `uzu-lane-cpu-b2` | damping + bit-alloc | running |
| `uzu-lane-sweep-2` | robustness shard2of5 | running |
| `uzu-smoke-all` | all-lane smoke | running（`problems` 未確認） |
| Muse 5 本 | robust s3 / s4 / s5、cpu-clean、cpu-balanced | running |

### 失敗

- `uzu-lane-sweep-1` — robustness shard1of5（原因未調査）

### `uzu-qad-cpu` の原因（確定）

`val_kl` が step 0 と step 5 で `0.262012` の 6 桁完全一致 → early stop。

- validation は更新後 `params` で測っていた（5% 変えると logits は変わる）。バグではない
- 原因は **master_weights が bf16 で、AdamW の更新が 1 ulp 未満**。重み 0.008 の bf16 ulp は 6.1e-5、lr 2e-5 の更新は毎 step 丸め落ちる。累積もしない（毎回 bf16 の値から丸める）。よって 4-bit forward は bit 単位で不変
- 実測: 全 master weight を ±1 ulp 動かしても logits 差分は **0.000e+00**。5% 動かすと 0.8 変わる
- 修正: `--master-dtype float32`（既定）、`changed` / `max_delta` を毎 step 記録、`changed==0` で警告、val 完全一致は bad eval に数えない、final step で early stop を例外にしない
- **QAD の master lr は 1e-3 前後**。2e-5 は finetuning の値で、4-bit の量子化刻みに対して 2 桁小さい

---

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
| mixed50 vs mix65 が有意か | seed 分散 | 実行中 |
| 変換設定（damping / group / bits） | damping + bit-alloc | 実行中 |
| コーパス混合 vs Hessian 混合 | balanced-h（Muse） | 実行中 |
| 4B/9B の emb/readout INT4 維持 | 未検証 | 未着手 |
| QAD が効くか | QAD CPU val KL | 0.8B で lr 1e-3 の再投入待ち（2e-5 は no-op と判明） |
| 混合比の曲線（mix35 / jaonly） | Muse clean-suite | 実行中 |

### 判定の物差し

- 指標は **同一言語内の BPC 相対 Δ**。PPL の絶対値を言語間で比べない
- seed 分散が出るまで、3% 前後の差を「有意」と主張しない
- T4 の BF16 は未確定。`dot_f32_acc` / `sdpa_xla` が pass した実績があるため「T4 BF16 不可」を確定事項にしない

---

## 6. 次の一手

1. `uzu-smoke-all` の `problems` が空か確認（全レーン健全性の前提）
2. 失敗 2 本のログを原因分類（環境 / 引数 / OOM）
3. seed 分散 → 3% 差の有意性判定（**最初の分岐**）
4. damping / group / bits → 変換設定の確定
5. Muse の mix35 / jaonly / balanced-h を回収して混合比の曲線を閉じる
6. QAD を `--master-dtype float32 --lr 1e-3` で再投入し val KL の推移を見る
7. Modal 4B を YAQA calibration へ
8. `uzu-ja-corpora` 削除（走行ジョブ終了待ち）

---

## 7. 参照

| ファイル | 内容 |
|---|---|
| `docs/SESSION_SUMMARY_2026-11-07.md` | 経緯・決定事項 |
| `docs/ENVIRONMENT.md` | 環境一致と provenance の規約 |
| `docs/TASKS.md` | 全 33 タスク（`scripts/task_board.py --write` で再生成） |
| `docs/KAGGLE_SLOTS.md` | 枠運用と handoff |
| `research/clean_suite_summary.md` | 0.8B の結果表 |
| `docs/LOCAL_POLICY.md` | ローカルで重い処理を回さない規約 |
