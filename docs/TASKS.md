# タスクボード

全 33 件

## 実行中（15）
| source | name | lane | status |
|---|---|---|---|
| owner | uzu-lane-runner | seed-variance-08b | running |
| owner | uzu-lane-cpu-b2 | damping + bit-alloc | running |
| owner | uzu-lane-sweep-1 | robustness shard1of5 | running |
| owner | uzu-lane-sweep-2 | robustness shard2of5 | running |
| owner | uzu-smoke-all | all-lane smoke | running |
| owner | uzu-smoke-datasets | attach + resume smoke | running |
| muse | uzu-robust-s3 | robustness shard3of5 | reported |
| muse | uzu-robust-s4 | robustness shard4of5 | reported |
| muse | uzu-robust-s5 | robustness shard5of5 | reported |
| muse | uzu-cpu-clean | clean-suite (mix65/mix35/jaonly) | reported |
| muse | uzu-cpu-balanced | balanced-h 50/65/35 | reported |
| owner | uzu-qad-cpu | QAD 10 step (freeze scales) | running |
| planned | modal | 0.8B GPU smoke (A100) | running |
| antigravity | t4-attention-bf16 | T4 で BF16 dot になる原因と非BF16経路 | running |
| antigravity | qad-gradient-audit | scale 勾配の導出と freeze の妥当性 | running |

## 待ち（キュー投入済み）（3）
| source | name | lane | status |
|---|---|---|---|
| antigravity | judge-protocol | judge 手順 + キャリブレーション40件 | queued |
| antigravity | bit-alloc-validation | 代理誤差 vs PPL の検証設計 | queued |
| antigravity | model-cards | 公開 card の最終稿 | queued |

## 待ち（外部・手動）（1）
| source | name | lane | status |
|---|---|---|---|
| antigravity | vram-budget | 4B YAQA OOM の主因特定 + T4 で通る枝 | manual |

## 未投入（2）
| source | name | lane | status |
|---|---|---|---|
| planned | uzu-lane-cpu-tuning | cpu-tuning (calib bf16 / RTN) | planned |
| planned | modal | 4B / 9B eval + QAD | planned |

## 完了（12）
| source | name | lane | status |
|---|---|---|---|
| owner | uzu-gpu-preflight | T4 capability | done |
| antigravity | license-audit | HF/Kaggle 配布物のライセンス・model card | done |
| antigravity | eval-prompt-set | P3: EN/JA 400件 + rubric | done |
| antigravity | bit-alloc-9b | 4/6/8bit の層別配分と根拠 | done |
| antigravity | int4-hyperparams | group/damping/block の根拠と判断表 | done |
| antigravity | public-corpus-audit | verified locally: all 17 cross rows identical (PR #41) | done |
| antigravity | qad-convergence | sandbox fs failure; numbers were NOT reproducible, code-reading finding kept (lalamo int.py:146 stop_gradient) | done |
| antigravity | tokenizer-audit | JA fertility と BPC の解釈 | done |
| antigravity | quant-hotspots | 量子化パスの CPU 実測 | done |
| antigravity | repro-audit | RUNBOOK を上から実行して検証 | done |
| antigravity | kaggle-schedule | クォータと枠の配分計画 | done |
| antigravity | docs-drift | docs と実装の乖離監査 | done |

