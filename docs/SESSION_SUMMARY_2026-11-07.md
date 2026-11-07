# セッションまとめ（2026-11-07）

対象: Huihui Qwen3.5（0.8B / 4B / 9B）の RHT + INT4 変換と評価。
main = `517cee37`、テスト **153 passed**。

---

## 1. 確定した結果（0.8B clean）

条件: dedup コーパス / 100 chunks / RHT block32 + asymmetric INT4 group32。
BF16 baseline は全条件で **EN 35.38 / JA 43.42**（比較可能）。

| calibration | EN INT4 | EN BPC Δ | JA INT4 | JA BPC Δ |
|---|---:|---:|---:|---:|
| enonly | 39.39 | +3.02% | 62.52 | +9.67% |
| mixed50 | 39.87 | +3.36% | 49.93 | +3.70% |
| mix65 | 40.52 | +3.81% | 49.59 | +3.52% |

- 判断は **同一言語内の BPC Δ**。PPL の絶対値は言語間比較しない
- mixed50 が相対劣化をほぼ 1:1 に揃える（EN 3.36 / JA 3.70）
- mix65 は EN を少し犠牲に JA を少し改善（想定どおりのトレードオフ）
- 生データ: `experiments/qwen35-0.8b/clean-{enonly,mixed50,mix65}-int4-g32/`
- まとめ: `research/clean_suite_summary.md` / `research/bpc_fertility_audit.md`

### 無効化した結果
- 4B RTN eval v3 / emb8 v4: `rglob` が `uzu-ja-corpora` の古い eval script を拾っていた → 破棄（`research/emb8_v4_invalid.md`）
- 旧 50 chunks / 非 dedup の 10 件 → `experiments/qwen35-0.8b/OBSOLETE.md`

---

## 2. 調査の結論

### T4 は使えない
- compute capability 7.5。素の BF16 matmul は通るが、**F32 累積の dot** と **`jax.nn.dot_product_attention(implementation="xla")`** が ALG_DOT_BF16_BF16_F32 で落ちる
- 0.8B は **CPU**、4B/9B は **Modal（A100）**
- `scripts/preflight.py` が 6 つの probe で事前判定（dot / dot_f32_acc / sdpa_xla / tokenizer / quant_roundtrip / qad_grad）

### 量子化の支配項
- eval 時間は **dequant が 80〜99%**（unpack 37〜48%、dequant 35〜55%、matmul 0.1〜2.5%）
- 4B/9B の calibration が破綻するのは **output_block (H_O)**。`output_block: none` で 50〜80% 削減（Modal は既に none）
- 0.8B の条件数は中央値 1.85e5 / 最大 2.51e6（実測）。damping 1e-2 の妥当性は未証明 → ablation 待ち

### QAD の疑い
- `lalamo/compressed/int.py:146` が `stop_gradient(scales)`。scale の勾配は dequant 経路のみで、整数コード（±8）が係数になる
- `--freeze-scales` を実装。10 step の実測は QAD CPU スモーク待ち
- 合成再現の数値は**再現しなかった**ため破棄（`research/qad_convergence_diagnosis.md`）

### 配分
- 9B 5.0GB 案: 4bit 59 層 / 6bit 69 層、代理誤差 -72%（推定）
- 代理誤差と PPL の相関は未検証 → `lanes/bit-alloc-ablation-08b.json` と検証設計チケット

---

## 3. 公開物

| 対象 | 状態 |
|---|---|
| HF 4B / 9B INT4 | public。`license: apache-2.0` + LICENSE + abliterated 告知 |
| HF `uzu-quant-kit` | public。dual license（scripts MIT / data CC BY-SA 4.0）+ 出所 |
| HF `uzu-results` | 運用先（Muse が上げる） |
| Kaggle `uzu-quant-kit-public` / `uzu-bf16-08b-lalamo` | public |
| Kaggle `uzu-ja-corpora` | **CC0 宣言が誤り**。インプレース変更不可 → 走行ジョブ終了後に削除 |

詳細: `research/license_actions_2026-11-07.md`、`research/license_audit.md`、`research/public_corpus_audit.md`（17 行の交差をローカル再現で一致確認）、`research/eval_prompts_*`（400 件検証済み）。

---

## 4. 実装したもの

| 種別 | 追加 |
|---|---|
| 実行 | `run_lane.py`（spec 駆動・resume・preflight・shard・RTN・eval dtype/parallel override） |
| 生成 | `make_sweep_spec.py`（27 条件→3/5 シャード）、`make_smoke_specs.py`（全レーンの極小版） |
| 検証 | `preflight.py`、`check_docs.py`、`combine_lane_manifests.py`、`smoke_model_config.py` |
| 変換 | `rht_int4_convert.py`: `--layer-bits-json` / `--layer-filter`（未実装・未宣言を修正）、provenance に layer_bits |
| QAD | `make_qad_dataset.py`（clean student 検証 + Kaggle upload 自動化）、`--freeze-scales` / `--warmup-steps` |
| 評価 | `eval_quantized.py --dtype`、BPC/chars を出力、`eval_generation.py`、`compute_bpc.py` |
| 運用 | `task_board.py`、`kaggle_slots.py`、`kaggle_usage.py`、`watch_kernels.py`、`cleanup_local.py`、`ag_fetch/ag_queue`（再試行・ログ復元・容量の自動掃除） |
| Modal | `modal_pipeline.py` をサイズ非依存に（0.8b/4b/9b、eval、qad） |

### スモークが見つけたバグ
1. kit が古いまま配られる（2 回）
2. `--layer-filter` が宣言だけで無効
3. `--layer-filter` の argparse 宣言漏れ（smoke-all 8/8 失敗）
4. smoke spec の出力ディレクトリ未作成
5. T4 の BF16 判定が甘い（素の matmul は通る）
6. preflight の probe が浅い（SDPA を見ていなかった）
7. Modal クライアントが古い
8. ローカルディスク満杯（アングラ連続失敗）
9. アングラ環境の容量掃除が全保護で 0 件

---

## 5. 現在の実行状態

| 資源 | 使用 | 内容 |
|---|---|---|
| owner Kaggle CPU | 5〜6 | seed / damping+bits / sweep1 / sweep2 / smoke-all v3 / QAD |
| owner Kaggle GPU | 1/2 | smoke-datasets（attach + resume） |
| Muse Kaggle CPU | 5 | shard3-5 / clean-suite / balanced-h |
| Modal | 1 | 4B → 9B smoke（import → rtn → eval 20 chunks） |
| アングラ | 2 + 待ち | judge-protocol / bit-alloc-validation、次に t4-attention / qad-gradient / model-cards |
| ローカル | アイドル | 監視のみ |

タスク一覧: `docs/TASKS.md`（`python scripts/task_board.py --write docs/TASKS.md` で更新）

---

## 6. 次にやること（順序）

1. **smoke-all v3 の結果**を確認（`problems` 空か）。落ちたら本番前に修正
2. **seed 分散**（owner）→ 3% 差が有意かを判定。ここが最初の分岐
3. **damping / group / bits**（owner）→ 変換設定の確定
4. **clean mix35 / jaonly + balanced-h**（Muse）→ 混合比の曲線を閉じる
5. **QAD CPU スモーク** → freeze-scales で val KL が下がるか
6. **4B/9B Modal スモーク** → メモリと所要の実測。通れば 4B YAQA calibration へ
7. **license 残件**: `uzu-ja-corpora` 削除（走行ジョブ終了後）
8. **bit-alloc 検証**（代理誤差 vs PPL）

### 運用ルール（再掲）
- ローカルで重い処理をしない（`docs/LOCAL_POLICY.md`）
- Kaggle push 前に `python scripts/kaggle_slots.py`（CPU 5 / GPU 2）
- 予備枠は owner が持つ（owner は 4 作業 + 1 予備）
- main にマージしたら **kit を再ビルドしてアップロード**（`build_kit.py --check` で commit 照合）
