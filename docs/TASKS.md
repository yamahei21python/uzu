# タスクボード

全 52 件

## 実行中（3）
| source | name | lane | status |
|---|---|---|---|
| planned | modal | 0.8B GPU smoke (A100) | running |
| owner | uzu-lane-budget-q128 | token budget q128 | running |
| owner | uzu-alloc-downout8-d1e0 | alloc down-out8 x d1e0 | running |

## 待ち（外部・手動）（1）
| source | name | lane | status |
|---|---|---|---|
| antigravity | vram-budget | 4B YAQA OOM の主因特定 + T4 で通る枝 | manual |

## 未投入（2）
| source | name | lane | status |
|---|---|---|---|
| planned | uzu-lane-cpu-tuning | cpu-tuning (calib bf16 / RTN) | planned |
| planned | modal | 4B / 9B eval + QAD | planned |

## 失敗（要対応）（2）
| source | name | lane | status |
|---|---|---|---|
| owner | uzu-qad-cpu | QAD 10 step (freeze scales) | failed |
| owner | uzu-qad-gpu-smoke | QAD GPU smoke | failed |

## 完了（44）
| source | name | lane | status |
|---|---|---|---|
| owner | uzu-lane-runner | seed-variance-08b | done |
| owner | uzu-lane-cpu-b2 | damping + bit-alloc | done |
| owner | uzu-lane-sweep-1 | robustness shard1of5 | done |
| owner | uzu-lane-sweep-2 | robustness shard2of5 | done |
| owner | uzu-smoke-all | all-lane smoke | done |
| owner | uzu-gpu-preflight | T4 capability | done |
| owner | uzu-smoke-datasets | attach + resume smoke | done |
| muse | uzu-robust-s3 | robustness shard3of5 | done |
| muse | uzu-robust-s4 | robustness shard4of5 | done |
| muse | uzu-robust-s5 | robustness shard5of5 | done |
| muse | uzu-cpu-clean | clean-suite (mix65/mix35/jaonly) | done |
| muse | uzu-cpu-balanced | balanced-h 50/65/35 | done |
| owner | uzu-qad-smoke | QAD early-stop smoke | done |
| owner | uzu-lane-balanced-h | balanced-h 50/65/35 | done |
| owner | uzu-gpu-eval-parity | T4 eval parity | done |
| owner | uzu-lane-q512-t4 | q512 on T4x2 (TP) | done |
| owner | uzu-lane-bit-alloc | bit-alloc (down/out 6bit) rerun | done |
| owner | uzu-lane-damping-d1e-1 | damping 1e-1 | done |
| owner | uzu-q512-remaining-a | q512 remaining A | done |
| owner | uzu-q512-remaining-b | q512 remaining B | done |
| owner | uzu-lane-damping-d1e0 | damping 1e0 | done |
| owner | uzu-q512-bf16-calib | q512 bf16 calibration | done |
| owner | uzu-balanced-h-d1e-2 | stacking test (Hessian x d1e-2) | done |
| owner | uzu-q256-s64 | s64 q256 cells | done |
| owner | calib-device-cpu-bf16 | calib device dtype separation (CPU bf16) | done |
| owner | calib-device-gpu-f32 | calib device dtype separation (GPU f32) | done |
| owner | uzu-q256-s64-bf16 | s64 q256 bf16 on CPU | done |
| antigravity | t4-probe-mismatch |  | done |
| antigravity | license-audit | HF/Kaggle 配布物のライセンス・model card | done |
| antigravity | eval-prompt-set | P3: EN/JA 400件 + rubric | done |
| antigravity | t4-attention-bf16 | T4 で BF16 dot になる原因と非BF16経路 | done |
| antigravity | bit-alloc-9b | 4/6/8bit の層別配分と根拠 | done |
| antigravity | qad-gradient-audit | scale 勾配の導出と freeze の妥当性 | done |
| antigravity | judge-protocol | judge 手順 + キャリブレーション40件 | done |
| antigravity | int4-hyperparams | group/damping/block の根拠と判断表 | done |
| antigravity | public-corpus-audit | verified locally: all 17 cross rows identical (PR #41) | done |
| antigravity | qad-convergence | sandbox fs failure; numbers were NOT reproducible, code-reading finding kept (lalamo int.py:146 stop_gradient) | done |
| antigravity | tokenizer-audit | JA fertility と BPC の解釈 | done |
| antigravity | bit-alloc-validation | 代理誤差 vs PPL の検証設計 | done |
| antigravity | quant-hotspots | 量子化パスの CPU 実測 | done |
| antigravity | repro-audit | RUNBOOK を上から実行して検証 | done |
| antigravity | kaggle-schedule | クォータと枠の配分計画 | done |
| antigravity | model-cards | 公開 card の最終稿 | done |
| antigravity | docs-drift | docs と実装の乖離監査 | done |

