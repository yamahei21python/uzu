# CPU レーン（GPU クォータを使わない）

0.8B は CPU でも現実的。T4 の BF16 制約（`docs/KAGGLE_SLOTS.md`）を回避でき、
**GPU クォータを消費しない**。Kaggle の CPU カーネルは 4 vCPU / 30GB RAM / 12h。

## 同じ spec を CPU で回す

spec は GPU 用のままでよい。CLI で上書きする。

```bash
python kit/scripts/run_lane.py \
  --spec kit/lanes/seed-variance-08b.json \
  --jax-platform cpu --eval-dtype float32 --parallel replicated \
  --work /kaggle/working
```

- `--jax-platform cpu`: `JAX_PLATFORM_NAME=cpu`
- `--eval-dtype float32`: T4 と同じ dtype に揃える（bf16 と混ぜない）
- `--parallel replicated`: CPU 1 デバイスでは tensor parallel にしない
- `--work`: 出力先。再実行すると `result.json` がある条件は skip される

## 何を回すか

| カーネル | spec | 条件数 | 目安 |
|---|---|---:|---:|
| A | `seed-variance-08b` + `group-size-ablation-08b` | 6 | 6〜9h |
| B | `damping-ablation-08b` + `bit-alloc-ablation-08b` | 4 | 4〜6h |
| C1〜C5 | `robustness-08b-shard{1..5}of5` | 6/6/5/5/5 | 各 5〜8h |

12h 制限に収まるよう 2 本に分けている。片方が落ちても、もう片方は独立。

## 枠の割り当て（CPU は 1 アカウント 5 本まで）

| アカウント | 枠 | レーン |
|---|---|---|
| owner | 1 | `seed-variance-08b` |
| owner | 2 | `damping-ablation-08b` + `bit-alloc-ablation-08b` |
| owner | 3 | `robustness-08b-shard1of5` |
| owner | 4 | `robustness-08b-shard2of5` |
| owner | 5 | **予備**（常に空ける。再実行・検証・新しい実験用） |
| Muse | 1 | `robustness-08b-shard3of5` |
| Muse | 2 | `robustness-08b-shard4of5` |
| Muse | 3 | `robustness-08b-shard5of5` |
| Muse | 4 | `clean-suite-08b`（mix65 / mix35 / jaonly） |
| Muse | 5 | `balanced-h-08b` |

合計 10 セッション。1 つあたり 12h 以内に収まる。
**予備は owner 側に置く**（owner は 4 本だけ作業に使い、5 本目を空ける）。

## 前提

- kit: `hf download yamahei21/uzu-quant-kit --repo-type dataset --local-dir kit`
- `python kit/scripts/build_kit.py --check kit` が `KIT OK`
- 出力は `out/<lane>/<condition>/result.json`（`ppl_*.json` と calibration を含む）

## 注意

- CPU の bf16 は遅い。`--eval-dtype float32` を必ず付ける（数値も T4 と揃う）
- 12h を超えそうなら `--shard` で分ける。`result.json` がある条件は再実行で skip
- 速度はローカル M3 の実測より遅い。**最初の 1 条件の所要を記録**して、残りの見積もりを更新する
