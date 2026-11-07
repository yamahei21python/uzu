# 実験の実行デバイス一覧

最終更新: 2026-11-07

方針: **CPU で回せるものは CPU。GPU は CPU で不可能な処理だけ**（`docs/KAGGLE_GPU.md`）。

デバイスは新しい実行から `eval_dtype` と `device` を JSON に記録する（`fee58813` 以降）。
それ以前の結果は「記録なし」とし、ログ・日時から推定した場合のみ推定と明記する。

## recorded（ログで確認できる）

| 実験 | デバイス | 条件 | 根拠 |
|---|---|---|---|
| seed 分散 mixed50 | **Kaggle CPU** | q128 / s32 / d1e-3 / 100 chunks / seed 0-2 | `uzu-lane-runner` ログ（`JAX_PLATFORM_NAME=cpu`） |
| smoke-all（8 spec） | **Kaggle CPU** | 各 spec の極小版 | `uzu-smoke-all` ログ |
| QAD early-stop smoke | **Kaggle CPU** | 2 step / seq 64 / freeze | `uzu-qad-smoke`（`SMOKE OK`） |
| QAD 10 step（旧） | **Kaggle CPU** | 10 step / seq 128 / freeze | `uzu-qad-cpu`（early-stop で ERROR → 修正） |
| **q512 probe** | **Kaggle GPU T4×2** | s16 / q512 / d1e-2 / 100 chunks | `uzu-lane-q512-t4`（27 分） |
| eval parity | **Kaggle GPU T4×2** | BF16 自身 / 100 chunks / f32 | `uzu-eval-parity-final` |
| 4B / 9B RTN eval | **Modal A100-80GB** | RTN / 20 chunks / bf16 | `modal_pipeline` eval ログ |
| QAD GPU smoke | **Kaggle GPU T4×2** | 10 step / seq 128 / replicated→OOM | `uzu-qad-gpu-smoke-owner`（TP 版は未検証） |

## unrecorded（当時 device を書いていない）

| 実験 | 推定デバイス | 条件 | 備考 |
|---|---|---|---|
| clean suite enonly / mixed50 / mix65 | ローカル M3（CPU） | s32 / q128 / d1e-3 / 100 chunks | 当時はローカル実行が常態。`docs/LOCAL_POLICY.md` 制定前 |
| yaqa-hin / hin-hout 系（多数） | ローカル M3（CPU） | s16-32 / q128-512 / d1e-3〜1e-2 | `experiments/qwen35-0.8b/yaqa-*` |
| rtn-int4-g32 | ローカル M3（CPU） | RTN | `experiments/qwen35-0.8b/rtn-int4-g32` |
| official 0.8B-M | ローカル M3（CPU） | 公式モデルの eval のみ | `experiments/qwen35-0.8b/official-0.8b-m` |
| balanced-raw / balanced-50 | ローカル M3（CPU）※中断 | q128 / d1e-3 | 2026-11-07 に停止（アドホック script） |
| 4B / 9B 変換（RTN） | **Modal A100** | RHT+INT4 | モデルは `models/huihui-{4b,9b}-int4-rtn` |

**推定の根拠**: `docs/SESSION_SUMMARY_2026-11-07.md` の経緯と、`experiments/` の生成時刻。
当時は `device` を JSON に残していないため、断定はできない。

## 現在走っているもの

| 実験 | デバイス | 状態 |
|---|---|---|
| robustness shard1 / shard2（q512 除外） | Kaggle CPU | running |
| damping + bit-alloc | Kaggle CPU | running |
| balanced-h | Kaggle CPU | running |
| Muse shard3/4/5, clean, cpu-tuning | Muse CPU | running |


## 設定の詳細

### 環境

| 環境 | スペック |
|---|---|
| Kaggle CPU | 4 vCPU / 30GB RAM / 12h（週次クォータ無し、5 セッション） |
| Kaggle GPU | 2 × Tesla T4（各 15.6GB）/ 12h / 30h 週 / 2 セッション |
| Modal | A100-80GB |

### 量子化（0.8B 共通）

| 項目 | 値 |
|---|---|
| bits | 4 |
| group size | 32（0.8B は 64/128 不可。`in_proj` out_dim 8224 のため） |
| block size | 32（RHT） |
| symmetric | false |
| layout | output_input |

### 校正（calibration）

| 実験 | samples | max_tokens | damping | output_block | eval chunks |
|---|---:|---:|---:|---|---:|
| clean enonly / mixed50 / mix65 | 32 | 128 | 1e-3 | small (2048) | **100** |
| seed 分散（mixed50, 3 seed） | 32 | 128 | 1e-3 | small | **100** |
| q512 probe（GPU） | 16 | 512 | 1e-2 | small | **100** |
| yaqa-hin-hout mixed / mix35 / mix65 / jaonly | 32 | 128 | 1e-3 | small | 50 |
| yaqa-hin-hout（初期） | 16 | 512 | 1e-2 | small | 50 |
| yaqa-hin（初期） | 8 | 64 | 1e-3 | none | 50 |
| robustness shard1-5 | 16-64 | 128/256/512 | 1e-4〜1e-2 | small | 100 |
| balanced-h（50/65/35） | 32 | 128 | 1e-3 | small | 100 |
| 4B | 8 | 256 | 1e-3 | none | 100（RTN は 20） |
| 9B | 8 | 512 | 1e-3 | none | 100（RTN は 20） |

### 評価

| 項目 | 値 |
|---|---|
| max_tokens | 128 |
| min_tokens | 16 |
| chunks | 100（clean / seed / q512 / robustness / balanced）、50（yaqa-hin 系・RTN の旧） |
| dtype | 0.8B = float32、4B/9B = bfloat16（A100） |
| 判定 | 同一言語内の BPC 相対 Δ。デバイス間の eval 差は 0.02% 以内 |

### QAD

| 項目 | 値 |
|---|---|
| steps | 10 |
| seq | 128（0.8B）/ 256（4B/9B の config） |
| lr / scale_lr | 2e-5 / 1e-6（freeze-scales 時） |
| warmup / val-every / val-chunks | 5 / 5 / 2 |
| early_stop_patience | 2 |
| optimizer | adamw |
| parallel | replicated（CPU）/ tensor（GPU は 2 枚必要） |

## 再現するときの目安

| 条件 | 置き場所 | 理由 |
|---|---|---|
| q128 / q256 | CPU | メモリが足りる |
| q512 | GPU（T4×2、TP=2） | CPU 30GB で SIGKILL |
| 4B calibration | Modal A100 | T4×2 の float32 では重みが載らない |
| 9B calibration | Modal A100 | 同上 |
| eval（全サイズ） | CPU で可。4B/9B は Modal | 数値はデバイス間で 0.02% 以内で一致 |


## 再計測の予定（保留 2026-11-08）

50 chunks の旧結果は**同じ chunks 数の中だけで有効**。100 chunks の結果と混ぜない。
全面再計測はしない。判断に使うものだけ揃える。

| 対象 | 状態 | 理由 | 目安 |
|---|---|---|---|
| `rtn-int4-g32` → 100 chunks | **明日** | clean suite の Δ の基準にする | 約 60 分 |
| `official-0.8b-m` eval → 100 chunks | **明日** | 報告・model card で引用する | 約 50 分 |
| emb4 / emb8 → 100 chunks | 4B 設計の直前 | 4B 設定が `embedding_bits: 4` のため | 各約 100 分 |
| yaqa-hin-hout mixed/mix35/mix65/jaonly | **捨てる** | clean suite が 100 chunks で上位互換 | — |
| yaqa-hin 初期（q64/s8） | **捨てる** | 設計が古い | — |
| yaqa-hin-hout 初期（q512/d1e-2） | **捨てる** | q512 probe が GPU で 100 chunks 済み | — |

保留理由: owner / Muse とも CPU 5 枠が埋まっている（2026-11-07 夜）。空き次第、上から順に投入する。

## ギャップ

- `calibration.json` には device が入っていない（eval のみ `device` を記録）
- 校正のデバイス間一致は未検証（校正は CPU 固定なので不要と判断）
