# Kaggle GPU（T4×2）の設定と使い方

最終更新: 2026-11-07 / 検証: `s16_q512_d1e-2` 完走（27 分）

## どちらで回すか（判断）

**GPU は有限（30h/週・同時2セッション）、CPU は枠だけ（5セッション・週次クォータ無し）。**
CPU で回せるものは CPU で回す。

| 処理 | 置き場所 | 理由 |
|---|---|---|
| q128 / q256 の calibration・eval | **CPU** | メモリが足りる。GPU を消費しない |
| seed 反復、cpu-tuning | **CPU** | 同上 |
| **q512 の calibration** | **GPU のみ** | CPU 30GB で SIGKILL。条件次第で稀に通るが再現しない |
| 4B / 9B の calibration | Modal（A100） | T4×2 の f32 では重みが載らない（4B f32 = 16GB/枚超） |
| 緊急の再現・切り分け | GPU | 時間が正義のときだけ |

GPU に逃がす前に「CPU で OOM するか」を確かめる。OOM しないなら CPU。

## 実行コマンド

CPU と GPU の差分は 3 つだけ。

```bash
# CPU（既定）
python kit/scripts/run_lane.py --spec kit/lanes/<spec>.json \
  --jax-platform cpu --eval-dtype float32 --parallel replicated \
  --work /kaggle/working --hf-results <user>/uzu-results

# GPU（T4×2）
python kit/scripts/run_lane.py --spec kit/lanes/<spec>.json \
  --jax-platform gpu --eval-dtype float32 --parallel tensor \
  --work /kaggle/working --hf-results <user>/uzu-results
```

- `--parallel tensor` は必須。`replicated` は**各 GPU にモデルを複製**するので 15GB×2 を両方使い切って OOM する
- `--eval-dtype float32` は必須。bf16 は T4 で動かない（下記）
- `calib_dtype` は既定 float32 のままでよい

カーネル側は GPU を有効化するだけ。

```json
{
  "enable_gpu": true,
  "dataset_sources": ["yamahei21python/uzu-bf16-08b-lalamo"]
}
```

## 事前確認

```python
import subprocess, sys
print(subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout)
print(subprocess.run([sys.executable, "-c",
    "import jax; print(len(jax.devices()), [d.device_kind for d in jax.devices()])"],
    capture_output=True, text=True).stdout)
```

期待値:

```
Tesla T4, 7.5, 15360 MiB     ×2
2 ['Tesla T4', 'Tesla T4']
```

`1` なら machine shape が 1 枚。`--parallel tensor` は失敗するので、その場合は replicated + CPU へ戻す。

## T4 の制約と自動化（すべて実装済み）

| 症状 | 原因 | 対策 | 実装 |
|---|---|---|---|
| `Triton support is only enabled for Ampere` | Pallas-Triton は cc>=8.0 | XLA attention | `UZU_ATTENTION_IMPL` 未設定なら cc から自動で `xla` |
| `ALG_DOT_BF16_BF16_F32` UNIMPLEMENTED | `Model.load(dtype=float32)` は重みだけ。embedding の `activation_dtype` 既定が bf16 | activation を f32 に強制 | `force_float32_activations()` |
| `Length of device assignment 1 ... mesh 2` | TP の mesh context 未設定 | `jax.set_mesh` | `enter_mesh()` |
| OOM on device 1 | replicated でモデルを二重に持つ | TP | `--parallel tensor` |
| preflight が T4 を PASS | probe 形状が小さすぎた | 実形状 probe | `preflight.real_shapes` |

自動化されているので、**env を手で設定する必要はない**。`--calib-dtype float32` / `--eval-dtype float32` だけ守る。

## 実測（0.8B、100 chunks）

| 処理 | CPU（4 vCPU / 30GB） | GPU（T4×2） |
|---|---:|---:|
| q128 calibration | 1,112〜1,229 秒 | 未測定（CPU で足りるので不要） |
| q256 calibration | 可（damping 条件で稀に OOM） | 未測定 |
| **q512 calibration** | **SIGKILL（-9）** | **418 秒** |
| convert | 1,782〜1,983 秒 | 289 秒 |
| eval en / ja | 1,429〜1,620 秒 / 1,461〜1,608 秒 | 440 / 443 秒 |
| 1 条件合計 | 約 107 分 | **約 27 分** |


## 数値はデバイス間で一致する（2026-11-07 実測）

BF16 自身を candidate にして 100 chunks を eval（CPU f32 vs T4×2 f32）。

| 言語 | GPU | CPU | Δ |
|---|---:|---:|---:|
| EN | 35.3641 | 35.3645 | -0.0013% |
| JA | 43.4297 | 43.4225 | +0.0165% |

- 許容は **0.05%**。これを超えたらデバイスごとに baseline を取り直す
- 実測の条件差（1pp 前後）より 2 桁小さいので、**CPU と GPU の結果を同じ表に載せてよい**
- 校正（calibration）の一致は未検証。校正は CPU に置く方針なので不要

## 失敗したときの切り分け

1. `jax devices` が 1 → machine shape。replicated に落とすか CPU
2. `ALG_DOT_BF16` → kit が古い（`force_float32_activations` が無い）。`build_kit --check`
3. `Triton` → kit が古い（`gpu_capabilities` が無い）
4. `mesh 2` → kit が古い（`enter_mesh` が無い）
5. OOM（device 1、数十 MiB 不足）→ `--parallel tensor` を忘れている
6. 実行中はログが取れない（API は `hasLog=true, log=""`）。完了後に `kaggle kernels output` か signed URL

## 必要な capability

```
attention.auto_xla
attention.f32_activations
attention.mesh_context
preflight.real_shapes
lane.exclude
lane.continue_on_failure
lane.resume_hf_results
```

古い kit は `build_kit.py --check` で落ちる。


## GPU の実測スペックと設定（2026-11-08）

### デバイス

| 項目 | Kaggle T4×2 | Modal A100-80GB |
|---|---|---|
| compute capability | **7.5（Turing）** | 8.0（Ampere） |
| メモリ | 16GB × 2（**1 枚あたり**） | 80GB |
| bf16 matmul | **不可**（`ALG_DOT_BF16_BF16_F32` が無い） | 可 |
| fp16 | 可 | 可 |
| flash-attn 2 | **不可**（sm80 以上要求） | 可 |
| 課金 | 無料（30h/週・同時 2 セッション） | 従量 |
| 1 セッション | 12 時間 | 無し |

**T4 の 16GB は「2 枚合わせて」ではない。** 1 プロセスから見えるのは 16GB で、
TP=2 にすると各デバイスに分割される。

### 並列モードの可否（0.8B で実測）

| モード | 0.8B | 理由 |
|---|---|---|
| `replicated` | ✅ | CPU でも可。GPU では 1 枚に全部載る |
| `tensor`（TP=2） | ✅ | 実績あり（QAD 0.39 s/step） |
| `fsdp` | **❌** | サイズ 1 の配列（norm scale）を 2 分割できず即死 |
| `data` | ✅ | ただし重みは複製されるのでメモリは減らない |

FSDP のエラー（再現用）:

```
Sharding spec ('fsdp',) implies that array axis 0 is partitioned 2 times,
but does not evenly divide the dimension size 1. Got shape: (1,)
```

### メモリの見積もり（0.8B QAD の実例）

| 項目 | サイズ |
|---|---:|
| teacher（bf16） | 1.5GB |
| student master（fp32） | 3GB |
| 勾配（fp32） | 3GB |
| AdamW m + v | 6GB |
| 活性化ほか | 2〜3GB |
| **合計** | **15.5GB 超 → T4 で OOM** |

削り方（効果が大きい順）:

| 手段 | 削減 | 副作用 |
|---|---:|---|
| `--optimizer muon` | **−3GB** | lr の意味が変わる（`--muon-lr`） |
| `--seq 64` | −1〜2GB | 学習信号が減る |
| `--freeze-scales` | 小 | scales が動かない |
| `--skip-checkpoint` | 中（orbax のピーク） | resume 不可 |
| `--master-dtype bfloat16` | −1.5GB | **更新が 1 ulp 未満で丸め落ちる（不可）** |

### GPU を使う前のチェックリスト

1. `kaggle kernels status` ではなく **クォータ**を確認（`rtk python3 scripts/kaggle_usage.py`）
2. 12 時間で切れる前提で checkpoint を設計（出力は `/kaggle/working`、サイズ上限 20GB）
3. bf16 を使わない（T4）。`--calib-dtype float32` / `--dtype float32`
4. TP=2 を使うなら `--parallel tensor`。FSDP は試さない
5. 失敗したらまず **メモリ**を疑う。`Free memory/Total` がログに出る

### クォータの実際

- 30h/週は**セッション時間**で消費される（待ち時間も含む）
- 2026-11-08 に QAD の 4 連敗（うち 3 本が OOM）で**使い切った**
- 使い切ると `Maximum weekly GPU quota of 30.00 hours reached.` で **push 自体が失敗**する
- 回復まで GPU は使えない → CPU で回せる設計にしておく

## クォータ

- GPU: 1 アカウント 2 セッション同時、30h/週
- CPU: 1 アカウント 5 セッション同時、週次クォータ無し、1 セッション 12h
- GPU セッションは 2 枚セットで 1 カウント
