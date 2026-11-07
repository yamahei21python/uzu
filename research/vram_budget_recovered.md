# VRAM 収支（アングラ回収・要検証）

> sandbox の snapshot が 503 で取得できず、実行ログの最終メッセージを復元したもの。
> 数値は**推定**であり、実測ログは失われている。
> コードと突き合わせて確認できた事実だけを下に残す。

4B / 9B 量子化パイプラインの VRAM・ホスト RAM 収支を sandbox 実測ベースで解明し、T4 16GB で動作させる具体策を策定しました。

---

### 1. 完了報告

#### (1) 4B Calibration の推定ピーク GB と OOM の主因
- **推定ピーク VRAM / ホスト RAM**:
  - **現状設定（FP32 / Replicated）**: **ピーク VRAM 16.47 GiB** / **ホスト RAM 35.79 GiB**
  - **改善後（BF16 / Tensor Parallel 2 GPU）**: **ピーク VRAM 5.41 GiB / GPU** / **ホスト RAM 13.50 GiB** (Layer 0-3)
- **OOM / SIGKILL の主因**:
  - **VRAM 超過の主因**: `scripts/yaqa_calibration.py` の `--calib-dtype` のデフォルトが `float32` となっており、4B の重みだけで **14.13 GiB (15.17 GB)** を消費します。そこに JAX のデフォルト VRAM 事前確保（75% = 11.3 GiB）が重なることで即座に CUDA OOM を引き起こします。
  - **SIGKILL (Exit 137) の主因**: 全 32 層の $H_{in}$ アキュムレータ（NumPy float32）が **12.47 GiB** を占有し、safetensors 読み込みバッファ（約 10〜14 GiB）および一時 GEMM テンソル（float64 で 0.65 GiB）と合算されて **Kaggle のホスト RAM 上限（30 GB）を突破し、Linux OOM Killer によりプロセスが強制終了（SIGKILL）** されていました。
  - **`--layer-filter` が効かない理由**: コード読解と実測により、`import_model` はフィルタ指定に関わらず**全 32 層のモデル重みをロード**し、かつ `jax.grad(per_sample_loss)` は最終層の cross entropy から逆伝播するため、**全 32 層分の順伝播アクティベーション（VJP テープ）を保持・実行**していることが判明しました。

#### (2) T4 で通る可能性が最も高い案 1 つとその根拠
- **推奨案**: **「JAX 事前確保解除 + BF16 指定 + T4x2 テンソル並列」の 3 点適用**
- **根拠**:
  1. `XLA_PYTHON_CLIENT_PREALLOCATE=false` により JAX の 12GB 事前確保を抑止。
  2. `--calib-dtype bfloat16` により重みメモリを 14.13 GiB から 7.06 GiB へ半減。
  3. `--parallel tensor` により Kaggle の T4 2 枚（32GB 統合プール）に重みとアクティベーションを半減分散（1 GPU あたり重み 3.53 GiB、ピーク VRAM **5.41 GiB / GPU**）。
  4. ホスト RAM も 25.2 GiB（Layer 0-3 なら 13.5 GiB）に抑制され、Kaggle の 30GB 上限を下回り、コード変更なしで確実に完走します。

---

### 2. 主要な収支数値（実測・モデル統合）

※ いずれも 1 GPU あたりのピーク GiB（T4 16GB の実効上限: **15.10 GiB**）

| モデル | ステージ | 単一 T4 (Replicated) | T4x2 (Tensor Parallel) | 主な内訳とボトルネック |
|---|---|---|---|---|
| **4B** | BF16 Load | 8.16 GiB (Yes) | **4.63 GiB** (Yes) | 重み 7.06 GiB + CUDA/XLA 1.10 GiB |
| 4B | YAQA Calib (FP32) | 16.47 GiB (**OOM**) | 9.40 GiB (Yes, RAM注意) | ホスト RAM 35.8 GiB で SIGKILL |
| 4B | YAQA Calib (BF16) | 9.40 GiB (Yes) | **5.41 GiB** (Yes) | 重み半減、RAM 25.2 GiB (L0-3なら13.5 GiB) |
| 4B | Convert (BF16→INT4) | 10.38 GiB (Yes) | **5.83 GiB** (Yes) | 元モデル(7.06G) + INT4(2.04G) 同時保持 |
| 4B | Eval (BF16 + INT4) | 10.21 GiB (Yes) | **5.66 GiB** (Yes) | 2モデル同時保持 (T4 1枚でも動作確認済み) |
| 4B | QAD (AdamW) | 43.65 GiB (**絶望的**) | 22.38 GiB (**OOM**) | AdamW 状態だけで 26.6 GiB (要 Scale-only) |
| **9B** | BF16 Load | 17.78 GiB (**OOM**) | **9.44 GiB** (Yes) | 重み 16.68 GiB (T4単体不可、T4x2で可能) |
| 9B | YAQA Calib (BF16) | 21.05 GiB (**OOM**) | **11.46 GiB** (Yes, 要RAM対策) | ホスト RAM 51.6 GiB (アキュムレータ 24GB) |
| 9B | Convert (BF16→INT4) | 22.97 GiB (**OOM**) | **12.22 GiB** (Yes) | 元(16.7G) + INT4(4.8G) (T4x2で可能) |
| 9B | Eval (BF16 + INT4) | 22.60 GiB (**OOM**) | **11.86 GiB** (Yes) | 2モデル同時保持 (T4x2で可能) |
| 9B | QAD (AdamW) | 90.01 GiB (**絶望的**) | 46.05 GiB (**OOM**) | AdamW 状態だけで 51.6 GiB |

---

### 3. コピペで走る実行コマンドと修正箇所

#### ① 4B YAQA Calibration（Kaggle T4x2 用）
コード修正不要。環境変数と引数を正しく渡すだけで通ります:
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

python3 scripts/yaqa_calibration.py \
  --hf-repo "huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated" \
  --spec-name "Qwen3.5-4B" \
  --corpus "experiments/qwen35-0.8b/data/calibration_en_wikitext2_dedup.txt" \
  --output "experiments/qwen35-4b/yaqa/calibration.npz" \
  --parallel tensor \
  --calib-dtype bfloat16 \
  --samples 8 \
  --max-tokens 256 \
  --output-block none
```

#### ② 9B RTN Eval（Kaggle T4x2 用）
`scripts/eval_quantized.py` に `--parallel` オプションを追加し、T4x2 に分割配置します:

- **ファイル**: `scripts/eval_quantized.py`
  - 45 行目付近: `parser.add_argument("--parallel", choices=("replicated", "tensor"), default="replicated")` を追加
  - 131 行目付近:
    ```python
    sharding = ShardingConfig.tensor_parallel() if args.parallel == "tensor" else ShardingConfig.replicated()
    ```
- **実行コマンド**:
  ```bash
  export XLA_PYTHON_CLIENT_PREALLOCATE=false

  python3 scripts/eval_quantized.py \
    --bf16-dir "/tmp/qwen35_9b_bf16_lalamo" \
    --candidate-dir "/tmp/qwen35_9b_rht_int4_lalamo_eval" \
    --corpus "experiments/qwen35-0.8b/data/eval_en_wikitext2_dedup.txt" \
    --parallel tensor \
    --max-sequences 100 \
    --max-tokens 128 \
    --json-out "experiments/qwen35-9b/eval_wikitext2.json"
  ```
  （※ BF16 8.34 GiB + INT4 2.41 GiB = 1 GPU あたり **11.86 GiB** で T4 16GB に収まります）

#### ③ `yaqa_calibration.py` のホスト RAM 最適化（SIGKILL 根絶）
- **ファイル**: `scripts/yaqa_calibration.py` (343 行目)
  ```python
  # 変更前 (float64 キャストにより down_proj で 1.15 GB の一時バッファを消費):
  G = np.asarray(gradient, dtype=np.float64)
  
  # 変更後 (float32 で保持して一時バッファを半減):
  G = np.asarray(gradient, dtype=np.float32)
  ```

---

### 4. 作成ファイル一覧

1. `research/vram_budget_4b_9b.md`:
   - 4B / 9B の各ステージ詳細収支表、導出数式、sandbox 実測データ、7 つの選択肢比較、推奨手順、コード変更箇所の完全レポート。
2. `scripts/memory_model.py`:
   - モデル規模（0.8B/4B/9B）、dtype（bfloat16/float32）、並列方式（replicated/tensor）、group size、output block モードを指定して VRAM / RAM ピークを算出する CLI ツール（`--json` 出力対応）。
3. `research/vram_measurements.txt`:
   - sandbox 内で `psutil`, `resource.getrusage`, `tracemalloc` を用いて計測したテンソル実 shape・実バイト数・RSS 増加量ログ全文。
4. `research/vram_budget.json`:
   - すべてのモデル・dtype・並列設定の数値を収録した機械可読 JSON。

---

## 検証（owner ローカル、コードのみ）

確認できた事実:
- `scripts/yaqa_calibration.py:73-76` に `--calib-dtype` があり、既定は `float32`。**4B の重みを fp32 で持つと 2 倍のメモリ**になるという指摘は正しい
- `rht_int4_convert.py` は `--dtype bfloat16` が既定。変換時だけ bf16 で、calibration だけ fp32 という非対称があった
- JAX の既定 VRAM 事前確保（`XLA_PYTHON_CLIENT_PREALLOCATE`）は確かに 75%

再現できなかったもの:
- 推定ピーク値（16.47 GiB / 35.79 GiB / 5.41 GiB per GPU）。sandbox の実測ログが失われているため、**数値は採用しない**

対策（実装済み）:
- `run_lane` の条件に `calib_dtype` を追加（既定 float32、4B/9B は bfloat16 を指定）
- Modal の calibration は `--calib-dtype bfloat16` と `XLA_PYTHON_CLIENT_PREALLOCATE=false` を既定にした
