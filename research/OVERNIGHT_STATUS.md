# Overnight Status (2026-11-07)

## Running
| job | id | 内容 |
|---|---|---|
| ratio suite | `sh_1b203882` | mixed50 → mix65 → mix35 → JA-only。各 EN/JA PPL/KL/top1 |
| overnight chain | `sh_1b2082c4` | suite後: local QAD 2step wiring → 09:05以降 Kaggle QAD 50step |
| antigravity | 4 | low_bit再試行 / uzu_model_loading再試行 / crosslingual / qad_export |

## Automatic actions
- 各条件の `calibration.npz` は変換後に削除（容量対策）
- QAD 配線: 2step/seq32 → export → `target/release/cli` generate
- Kaggle: `qad-08b-gpu` v3（dataset path 修正済）を quota reset 後に push → output を `/tmp/qad_kaggle_out`

## Deferred (judgment needed)
1. **QAD teacher**: 同一BF16 vs 9B distillation。両方可能だが GPU/時間予算の配分判断。
2. **混合比の採用**: 結果が僅差なら単一比を決めず、全条件を保持して朝判断。
3. **Kaggle quota 配分**: QAD 50step / 4B YAQA / 9B の優先順位。
4. **旧artifact削除**: E3/E3b の model/eval を消すか（再評価コストとのトレードオフ）。

## Morning checklist
- [ ] ratio suite 4条件の PPL 表を確認
- [ ] `/tmp/overnight_qad.log` の local QAD rc / uzu generate 成否
- [ ] Kaggle QAD output の有無と loss
- [ ] antigravity 4本の成果物検証（サイズ/見出し/表/ネスト）
- [ ] mixed の結果から採用混合比を決定

## Result: mixed 50/50 (03:15)
| calib | EN PPL | JA PPL | EN ΔNLL | JA ΔNLL | EN top1 | JA top1 | EN KL | JA KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EN-only (E3) | 43.81 | 43.05 | 0.071 | 0.163 | - | - | - | 0.170 |
| **mixed 50/50** | **43.17** | **41.70** | 0.056 | 0.131 | 0.823 | 0.801 | 0.093 | 0.136 |
| RTN | 45.29 | 42.47 | 0.104 | 0.149 | - | 0.779 | - | 0.151 |
| BF16 | 40.81 | 36.58 | - | - | - | - | - | - |
| official 0.8B-M | 41.97 | 42.14 | - | - | - | - | - | - |

判定: 混合校正は EN/JA 両方改善。JA は RTN と公式を上回る。

## Research collected (03:15)
- `crosslingual_quant_eval.md` 11,659B / 9 H2 / 表10 / arXiv 5件実在確認。fertility は推定、BPC式と10セル設計。
- `mixed_corpus_calibration.nested.md` 11,792B は別版。root版を残し、こちらは退避保持（削除判断は朝）。
- `qad_export_wiring.md` は未回収（アングラ実行中）。

---

## 2026-11-07 09:00 再起動後

- 夜間ジョブは server restart で全停止。`/tmp` 消滅。
- Kaggle quota は 30h に回復（週次リセット）。
- Kaggle script kernel は server 側 nbconvert バグで全滅 → notebook 形式へ変換して v4 を再投入。
- ローカル復旧:
  - lalamo pinned commit を `.deps/lalamo-e401` に再取得
  - `.deps/venv-lalamo` に jax[cpu]/equinox/optax を導入
  - `Qwen/Qwen3.5-0.8B` から BF16 + RTN を再生成（`/tmp/qwen35_0.8b_*`）
  - BF16 eval copy を再作成
- 比率スイート v2 を再起動: mix65 → mix35 → JA-only（mixed50 は既存結果を保持）
- QAD 50step v4 を Kaggle で実行中。完了後に output を取得し、uzu で generate 検証。

## 2026-11-07 09:15 追加結果

- 9B RTN on Modal: 成功。`/out/huihui-9b/int4-rtn` = 5.19GB（embedding 量子化込み）。ローカルへ fetch 中。
- QAD 0.8B v4 (TP): `optax.contrib.muon` が `Contracting dimensions are sharded` で失敗 → v5 は `--parallel replicated` で再実行中。
- 4B YAQA v4: calibration が SIGKILL (OOM 推定)。QAD 完了後に `samples=8 / max_tokens=256 / output-block=none` で再試行予定。
- 研究委任 3本: uzu logits parity / QAD TP optimizer / 9B mixed precision allocation。

## 4B SIGKILL 診断
- `--output-block small` は H_O のみ制限。H_I は全 Linear 分を保持。
- 4B hidden 2560 / MLP 9216。down_projection の H_I = 9216^2 × 4B = 340MB/層。
- 32層で H_I だけで ~11GB。BF16 model 8.5GB と合わさり Kaggle RAM で OOM。
- 対策: 9B と同じ layer-grouped calibration（4層ずつ、npz merge）か、Modal CPU 64GB へ格上げ。
- 4B は Kaggle 再投入しない。QAD 完了後に Modal 4B を検討。

## 9B RTN uzu 実測（2026-11-07 09:17）
- model: `models/huihui-9b-int4-rtn`（5.19GB、embedding 量子化込み）
- EN: "Hello" / TTFT 0.26s / prefill 68.76 t/s / gen 6.28 t/s / 4.95GB
- JA: 富士山 3,776m と回答。TTFT 0.27s / prefill 63.03 t/s / gen 11.47 t/s / 4.96GB
- 備考: JA 回答に「2004年に3776→3776へ微調整」という事実誤認あり。9B の知識限界。

## 2026-11-07 09:40 4B 結果
- 4B RTN on Kaggle: 成功。`models/huihui-4b-int4-rtn` 2.40GB。
  - EN: "Hi" / TTFT 0.19s / prefill 92.5 t/s / gen 14.0 t/s / 2.40GB
  - JA: 富士山 3,776m / TTFT 0.15s / prefill 114.3 t/s / gen 18.6 t/s / 2.40GB
- 4B calibration probe (layer 0-3 only, output-block none): SIGKILL。
  - H_I を絞っても 4B BF16 の load/forward で Kaggle RAM が足りない。
  - 4B YAQA calibration は Kaggle 不可。Modal CPU 64GB / A100 へ格上げ確定。

## 2026-11-07 09:45 Kaggle 有効活用
- `uzu-ja-corpora` dataset 作成（JA calibration/eval + mixed corpora）
- 0.8B emb8 ablation を Kaggle T4x2 に移行: `uzu-emb8-08b-gpu`
  - BF16 import → E3 calibration → emb8 変換 → EN/JA eval
  - ローカル emb8 は停止（重複回避）
- 4B RTN は `models/huihui-4b-int4-rtn` に回収済み。EN/JA generate 確認済み。
- 4B calibration probe は SIGKILL。Kaggle 不可を確定。
- 現在の Kaggle GPU: QAD 50step + emb8 の2本。

## 2026-11-07 10:05 ratio suite 結果
- mix65 (JA65/EN35): EN 41.13 / JA 47.58。EN は mixed50 より改善、JA は大幅悪化。
- mix35 / JA-only: `rht_int4_convert.py` の `--embedding-bits` 二重定義で変換失敗。修正済み、校正 npz から変換を再実行中。
- 暫定: mixed50（EN 43.17 / JA 41.70）が JA 最良。mix65 は EN 寄り。

## 2026-11-07 10:40 QAD 50step 判定
- 50step 完走。loss: step1 0.156 → step10 0.742 → step50 1.741。改善せず。
- ランダムchunkのtraining lossは比較不能。固定validation lossが必要。
- model download は 73MB/800MB で切断。partial は削除。再DLしない。
- QAD に `--val-chunks` / `--val-every` を追加。10step v3 で step_s と固定val lossを測る。
- 10step v2 は旧スクリプト。v3 は Kaggle GPU 2枠が空いてから。
