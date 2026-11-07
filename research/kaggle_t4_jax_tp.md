# Kaggle T4 x2 で JAX Tensor Parallel（YAQA calibration 4B/9B）

対象: lalamo `ShardingConfig.tensor_parallel()`（mesh 2 device、MATRIX 軸 shard）。
検証前提: 2 CPU device で `--parallel tensor` の calibration 動作確認済【実測】。実 T4 x2 は未検証。

数値ラベル: 【実測】=本環境で観測 /【推定】=計算・外挿 /【引用】=出典あり。表の「出典」列がある行は同列の出典に従う。

## 0. 結論

| 項目 | 判定 |
| :--- | :--- |
| JAX 0.12 + CUDA 12 で T4 (sm75) | 動く。CUDA 12 は sm5.2 以上、CUDA 13 は sm7.5 以上対応【引用】 |
| 2 device 認識 | `jax.devices()` が `CudaDevice(id=0/1)` を返せば OK。`machine_shape` 指定だけで保証はされない（要ログ確認） |
| 4B TP=2 | 余裕。重み 4.0GB/卡 + 状態で ~7GB/卡【推定】 |
| 9B TP=2 | 重み 9.0GB/卡 は載るが、全層 Hessian 同時保持は超過リスク。層フィルタ併用前提【推定】 |
| 主リスク | LD_LIBRARY_PATH 由来の CUDA 混線、75% preallocate で collective 用領域不足、Kaggle の GPU 形状名 |

## 1. 環境前提（公式値との突合）

| 項目 | 値 | 出典・注記 |
| :--- | :--- | :--- |
| VRAM | 16GB/卡（実効 ~15.6GiB）×2 | T4 datasheet【引用】 |
| 接続 | PCIe Gen3 x16、**NVLink 無し** | 仮想化下は host SHM 経由【引用】 |
| FP16 | 65 TFLOPS（Tensor Core）/ FP32 8.1 TFLOPS | T4 datasheet【引用】 |
| BF16 | **Tensor Core 非対応**（sm80 以降） | 実効は FP32 エミュレーション。FP16 推奨【引用】 |
| RAM / 制限 | 29〜32GB / 12h/回、30h/週 | Kaggle 仕様【引用】 |
| Python | 3.13（現行 Kaggle イメージ） | Dockerfile.tmpl `python3.13`【引用】 |
| LD_LIBRARY_PATH | イメージ既定で `/usr/local/nvidia/lib64:/usr/local/cuda/lib64` 設定済 | JAX は未設定を要求 → 明示 unset 必須【引用】 |

## 2. jax[cuda12] 導入手順

JAX 0.12.0.1（PyPI 最新、`requires_python >=3.12`）【引用】。CUDA 12 wheel は CUDA 12.3 でビルド、互換 CUDA>=12.1 / cuDNN 9.10 / NCCL 2.19、driver>=525【引用】。

```bash
# 1) 既存 torch 同梱 CUDA との混線を切る（同セル冒頭）
unset LD_LIBRARY_PATH
# 2) 追加 wheel リポジトリ指定は不要（extras で完結）
pip install -q -U "jax[cuda12]" 2>&1 | tail -5
# 3) バージョン整合の確認
python -c "import jax; print(jax.__version__, jax.devices())"
```

| 落とし穴 | 症状 | 対処 |
| :--- | :--- | :--- |
| torch 同梱の nvidia-* と衝突 | cuInit 失敗 / `CUDA error 303` 系（303 は標準 enum 外、報告ベース） | `unset LD_LIBRARY_PATH`、同一プロセスで torch を import しない |
| `JAX_PLATFORMS` 未設定 | CPU backend で起動し device 1 枚扱い | `export JAX_PLATFORMS=cuda`（明示） |
| 既存 jaxlib 残存 | `jaxlib` と plugin の版ずれ | 先に `pip uninstall -y jax jaxlib jax-cuda12-plugin` |
| pip 解決が古い jax を保持 | `jax.__version__` が 0.4 系 | `-U` 必須、`pip cache purge` |

推奨環境変数（script.py 冒頭で os.environ 設定。**import jax 前**）:

```python
import os
os.environ.pop("LD_LIBRARY_PATH", None)
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.90"   # 旧名 XLA_PYTHON_CLIENT_MEM_FRACTION
os.environ["NCCL_P2P_DISABLE"] = "1"             # PCIe/SHM 固定（VM で有効なことが多い）
os.environ["NCCL_DEBUG"] = "WARN"
```

`XLA_CLIENT_MEM_FRACTION` は「固定上限 + preallocate 量」を決める。既定は 75% preallocate・100% まで growth だが、**collective 用バッファは初期領域に制限**される【引用】。TP では 0.85〜0.92 を明示しないと NCCL 側で OOM になりやすい。

## 3. T4 既知エラー → 回避

| エラー | 原因 | 回避 |
| :--- | :--- | :--- |
| `CUDA_ERROR_NO_DEVICE` / cuInit | GPU 未割当、`enable_gpu` 未指定 | metadata に `"enable_gpu": "true"` + GPU 形状指定【引用】 |
| `cudaErrorInsufficientDriver` | driver < 525（CUDA12）/ < 580（CUDA13） | CUDA12 を使う（Kaggle driver は 525+ 帯）【引用】 |
| `NCCL error: unhandled system error` | preallocate 不足、P2P 不成立 | fraction 0.85〜0.92、`NCCL_P2P_DISABLE=1` |
| 通信ハング（初回 collective） | T4 に NVLink 無し、VM で P2P 遮断 | 上記 + `NCCL_DEBUG=INFO` で transport 確認 |
| BF16 で極端に遅い | sm75 に BF16 Tensor Core 無し | dtype を float16 に固定 |
| `RESOURCE_EXHAUSTED: Out of memory` | 全層同時ロード | 層単位でロード/解放（§6） |

## 4. 2 device mesh と tensor_parallel 構成

```python
import jax, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devs = jax.devices()                       # 2 枚必須
assert len(devs) == 2, f"devices={devs}"   # 1 枚なら GPU 形状/環境変数を疑う
assert jax.device_count() == jax.local_device_count() == 2  # 単一ノード
mesh = jax.make_mesh((1, 2), ("data", "tensor"), devices=np.array(devs))
sharding = NamedSharding(mesh, P(None, "tensor"))  # MATRIX 軸のみ shard
```

| 確認点 | 期待値 | 備考 |
| :--- | :--- | :--- |
| `jax.devices()` | `[CudaDevice(id=0), CudaDevice(id=1)]` | 1 枚なら CPU fallback か `CUDA_VISIBLE_DEVICES=0` |
| `device_count()` vs `local_device_count()` | 共に 2 | マルチノード時のみ差が出る【引用】 |
| mesh shape | `(1, 2)` | 第2軸 `tensor` を `P(None,"tensor")` で使用 |
| 実装側 | `ShardingConfig.tensor_parallel()` が mesh を保持 | 呼び出し前に device 数 assert を入れる |

`XLA_FLAGS` は必須ではない。入れるなら `--xla_gpu_enable_latency_hiding_scheduler=true` のみ（TP の all-reduce 隠蔽）。`--xla_gpu_triton_gemm_any` 等は T4 で効果未確認。

## 5. メモリ見積り（FP16・seq 2048・batch 1）

構成: per-GPU 重み = 総重み/2、Hessian は **層ごとに解放**する前提で 1 層分のみ計上。固定費は XLA 実行計画 + cuBLAS workspace + NCCL バッファ。

| モデル | 総重み | TP=2 重み/卡 | Hessian 1層 | 活性化 | 固定費 | 合計/卡 | 15.6GiB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.8B | 1.6GB | 0.8GB | 0.004GB | 0.3GB | 1.5GB | **~2.6GB**【推定】 | 余裕 |
| 4B | 8.0GB | 4.0GB | 0.026GB | 0.8GB | 1.5GB | **~6.3GB**【推定】 | 余裕 |
| 9B | 18.0GB | 9.0GB | 0.067GB | 1.2GB | 1.5GB | **~11.8GB**【推定】 | 余裕（層解放時） |
| 9B・全層保持 | 18.0GB | 9.0GB | 2.7GB | 1.2GB | 1.5GB | **~14.4GB**【推定】 | 危険（断片化で OOM） |

- Hessian は `hidden² × 4B`（fp32）。4096² → 67MB/層。
- 全層保持 + XLA の重複バッファで 15.6GiB を超える。**層フィルタで 1〜2 層ずつ確定 → 解放**が安全策。
- 単一卡（replicated）では 9B は重み 18GB で即 OOM。

通信（TP=2, PCIe Gen3 x16）:

| 指標 | 値 |
| :--- | :--- |
| all-reduce 1 回のサイズ | 4096 × 2048 × 2B = 16.8MB【推定】 |
| 層あたり | 2 回（attn out / mlp down）= 33.6MB【推定】 |
| 実効帯域（SHM 経由） | 5〜7GB/s【推定】 |
| 転送時間/層 | 5〜7ms【推定】 |
| 40 層合計 | 0.2〜0.3s/pass【推定】 |
| TP=2 スケール効率（prefill 主体） | 60〜75%【推定】 |

decode（batch 1）では通信が支配的になり効率はさらに落ちる。calibration は prefill 主体なので実害は小さい。

## 6. フォールバック

| 方式 | 9B 可否 | 速度 | 実装コスト |
| :--- | :--- | :--- | :--- |
| TP=2（本命） | 可（層解放前提） | 1.0x 基準【推定】 | 低（既存 `--parallel tensor`） |
| replicated 1卡 + layer filter | 4B のみ可 | 0.5〜0.7x【推定】 | 低（外側ループ追加） |
| Kaggle CPU（RAM 32GB） | 4B 可 / 9B は重み 18GB で不可 | 0.05x 未満【推定】 | 最低（環境変数のみ） |
| Modal（A10G 24GB / A100 40GB） | 可 | 3〜5x【推定】 | 中（別 script） |

```bash
# FB-1: 層フィルタ（既存 CLI をそのまま分割実行）
for s in 0 8 16 24 32; do
  python -m lalamo.calibrate --model 9b --parallel none \
    --layer-start $s --layer-end $((s+8)) --out layers_$s.npz
done
# FB-2: CPU
JAX_PLATFORMS=cpu python -m lalamo.calibrate --model 4b --parallel none
# FB-3: Modal（A10G 24GB）
modal run calibrate_modal.py --gpu A10G --model 9b
```

## 7. Kaggle 実行ファイル

`kernel-metadata.json`（T4 を要求。形状名は UI の "GPU T4 x2" と一致するか要確認）:

```json
{
  "id": "USER/yaqa-t4x2",
  "title": "yaqa-t4x2",
  "code_file": "script.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4"
}
```

`script.py`（環境変数 → device 確認 → mesh → calibration。`run_calibration(...)` の引数は呼び出し例であり実装の署名に合わせること）:

```python
import os
os.environ.pop("LD_LIBRARY_PATH", None)
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"

import jax, numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

print("JAX", jax.__version__)
devs = jax.devices()
print("devices", devs)
assert jax.default_backend() == "gpu", jax.default_backend()
assert len(devs) == 2, f"need 2 GPUs, got {len(devs)}"

mesh = jax.make_mesh((1, 2), ("data", "tensor"), devices=np.array(devs))
print("mesh", mesh)
sharding = NamedSharding(mesh, P(None, "tensor"))

# lalamo 側は mesh を渡して実行（本番コード変更なし）
from lalamo.calibrate import run_calibration
run_calibration(model="4b", parallel="tensor", mesh=mesh, sharding=sharding)
print("DONE")
```

`kaggle kernels push -p .` 実行ログの典型:

```text
JAX 0.12.0.1
devices [CudaDevice(id=0), CudaDevice(id=1)]
mesh Mesh(device_ids=[[0, 1]], axis_names=('data', 'tensor'))

# 失敗例1: 1 枚しか見えていない
devices [CudaDevice(id=0)]
AssertionError: need 2 GPUs, got 1

# 失敗例2: preallocate 不足で NCCL が落ちる
jax.errors.JaxRuntimeError: INTERNAL: NCCL error: unhandled system error

# 失敗例3: 9B 全層保持
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 4294967296 bytes
```

## 8. 未確認事項

1. Kaggle の実機 driver バージョン（570 系という報告はあるが公式保証なし）。
2. `machine_shape` に T4 x2 を明示する値が存在するか（`NvidiaTeslaT4` が 1 枚か 2 枚か）。
3. 実 T4 x2 での NCCL transport（SHM / P2P）と実効帯域の実測。
4. `NCCL_P2P_DISABLE=1` が T4 x2 で有利か（P2P 有効の方が速い可能性）。
5. sm75 での BF16 エミュレーション速度低下率、FP16 との calibration 精度差。
6. `ShardingConfig.tensor_parallel()` が `jax.make_mesh` をどう呼ぶか（axis_types 既定 Auto のままで良いか）。
7. 9B 全層 Hessian 保持時の実測ピーク VRAM（断片化含む）。

## References

- JAX Installation（CUDA 版要件・driver・LD_LIBRARY_PATH） https://docs.jax.dev/en/latest/installation.html
- JAX GPU memory allocation（75% preallocate / collective 領域） https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- jax on PyPI（0.12.0.1, requires-python>=3.12） https://pypi.org/project/jax/
- jax.make_mesh https://docs.jax.dev/en/latest/_autosummary/jax.make_mesh.html
- Multi-controller JAX（device_count/local_device_count） https://docs.jax.dev/en/latest/501/multiprocess.html
- Kaggle GPU 利用（12h/30h） https://www.kaggle.com/docs/efficient-gpu-usage
- Kaggle docker-python（Python 3.13 / LD_LIBRARY_PATH 既定） https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl
- kaggle-api（machine_shape / retired accelerator） https://github.com/Kaggle/kaggle-api/blob/main/src/kaggle/api/kaggle_api_extended.py
- NVIDIA T4 datasheet（16GB, PCIe, NVLink 無し, FP16 65TFLOPS） https://www.nvidia.com/en-us/data-center/tesla-t4/
- NCCL 環境変数（P2P_DISABLE 等） https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- Modal GPU ガイド https://modal.com/docs/guide/gpu
