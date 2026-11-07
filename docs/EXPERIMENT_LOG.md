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

## 再現するときの目安

| 条件 | 置き場所 | 理由 |
|---|---|---|
| q128 / q256 | CPU | メモリが足りる |
| q512 | GPU（T4×2、TP=2） | CPU 30GB で SIGKILL |
| 4B calibration | Modal A100 | T4×2 の float32 では重みが載らない |
| 9B calibration | Modal A100 | 同上 |
| eval（全サイズ） | CPU で可。4B/9B は Modal | 数値はデバイス間で 0.02% 以内で一致 |

## ギャップ

- `calibration.json` には device が入っていない（eval のみ `device` を記録）
- 校正のデバイス間一致は未検証（校正は CPU 固定なので不要と判断）
