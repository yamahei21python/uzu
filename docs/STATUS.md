# 現状

最終更新: 2026-11-07 / main `b4e4c828` / テスト 186 passed

数値・実行状態・未確定事項の現在値を 1 枚にまとめる。経緯は `docs/SESSION_SUMMARY_2026-11-07.md`、規約は `docs/ENVIRONMENT.md`。

---

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

---

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

### 失敗（原因未調査）

- `uzu-lane-sweep-1` — robustness shard1of5
- `uzu-qad-cpu` — QAD 10 step (freeze scales)

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
| QAD が効くか | QAD CPU val KL | 失敗 → 再投入待ち |
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
6. QAD CPU を再投入し val KL の推移を見る
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
