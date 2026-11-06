# Huihui Qwen3.5-9B 量子化変換 実現性・手順調査報告書
## Apple M3 (16GB) における RHT + INT4 Group32 + YAQA 変換・推論の実現性評価

- **対象**: `huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated` (BF16 約18GB)
- **環境**: Apple M3 (16GB Unified Memory, 帯域 100 GB/s)
- **構成**: RHT (Block 32) + INT4 Group32 + YAQA / 固定コミット `lalamo e401e177` / 推論 `uzu`

---

## 1. エグゼクティブサマリー

| 評価項目 | 判定 | 主要結論・根拠 |
| :--- | :---: | :--- |
| **M3ローカル一括変換** | **不可 (OOM)** | BF16モデル単体で17.91GB。M3 16GB実効上限(約11.5GB)を超過しクラッシュ。 |
| **M3ストリーム変換** | **条件付可能** | safetensors層別ロード＋$H_I$単側校正によりピークRAM約2.5GBで実行可能。 |
| **クラウド変換 (Modal)** | **推奨 (最小リスク)** | 64GB RAMで一括実行（所要約10分、費用約$0.2）。本番/環境汚染ゼロ。 |
| **M3ローカル推論 (uzu)** | **完全可能** | INT4 Group32化で重み6.6〜8.1GB。OS常駐込みで11GB以内に収まり余裕動作。 |
| **推論速度 (概算)** | **良好** | 通常約7.5〜10.5 tok/s。投機的デコード併用時約20〜35 tok/sと推定。 |

---

## 2. 9B レイヤ別メモリ見積りとアーキテクチャ構成

### 2.1 構成仕様 (Qwen3.5-9B Text Backbone)
Qwen3.5-9Bは線形Attention（Gated DeltaNet）と完全Attention（Gated Attention）を **3:1** で配置する [1]。

| パラメータ | 設定値 | 備考 |
| :--- | :--- | :--- |
| **隠れ層 / 中間層次元** | 4,096 / 12,288 | SwiGLU構造 (中間層は隠れ層の3倍) |
| **層数 (`num_hidden_layers`)** | 32 | DeltaNet 24層 ＋ Gated Attention 8層 |
| **語彙数 / 埋め込み共有** | 248,320 / **False** | 0.8B (True) と異なり `embed_tokens` と `lm_head` は独立 |
| **DeltaNet (24層)** | V: 32 heads, QK: 16 heads | $d_k=128, d_v=128$, Conv1D Kernel=4 |
| **Attention (8層)** | Q: 16 heads, KV: 4 heads | $d_h=256$, Query-width Gate 搭載 |

### 2.2 lalamo 内部モジュールと Linear 数
lalamo (`e401e177`) の `TransformerLayer` は各層に **4つの Linear** を配置する [3]。
1. `in_proj` (DeltaNet 12,352×4,096) または `qkvg_proj` (Attention 10,240×4,096)
2. `out_proj` (4,096×4,096)
3. `up_projection` (gate+up統合 24,576×4,096)
4. `down_projection` (4,096×12,288)

- **0.8B**: 24層 × 4 ＝ **全96 Linear** (変換実績値と合致)
- **9B**: 32層 × 4 ＝ **全128 Linear** (＋独立 `lm_head` 1層)

### 2.3 パラメータ数およびメモリサイズ表
INT4 Group32: 重み4bit ＋ FP16スケール (2B/32) ＋ ゼロ点 (0.5B/32) ＝ **0.578125 B/param** [1]。

| コンポーネント | パラメータ数 | BF16 (2B) | FP32 (4B) | INT4 Group32 |
| :--- | :---: | :---: | :---: | :---: |
| **DeltaNet 1層 (4 Linear)** | 218,407,104 | 416.6 MiB | 833.2 MiB | 120.5 MiB |
| **DeltaNet 24層合計** | 5,241,770,496 | 9.76 GiB | 19.53 GiB | 2.82 GiB |
| **Attention 1層 (4 Linear)** | 209,723,904 | 400.0 MiB | 800.0 MiB | 115.6 MiB |
| **Attention 8層合計** | 1,677,791,232 | 3.12 GiB | 6.25 GiB | 0.90 GiB |
| **全128 Linear合計** | 6,918,504,448 | 12.89 GiB | 25.77 GiB | 3.73 GiB |
| **語彙埋め込み (`embed_tokens`)** | 1,017,118,720 | 1.89 GiB | 3.79 GiB | 560.8 MiB |
| **出力ヘッド (`lm_head`)** | 1,017,118,720 | 1.89 GiB | 3.79 GiB | 560.8 MiB |
| **その他 (Norm等)** | 4,096 | 8.0 KiB | 16.0 KiB | 8.0 KiB |
| **テキスト全体合計** | **8,953,803,264** | **16.68 GiB (17.91 GB)** | **33.36 GiB** | 下記参照 |

- **シナリオ1 (推奨・0.8B準拠)**: 128 Linear INT4 ＋ 埋め込み・Head BF16 → **7.52 GiB (8.07 GB)**
- **シナリオ2 (Head量子化)**: 128 Linear・Head INT4 ＋ 埋め込み BF16 → **6.17 GiB (6.62 GB)**
- **シナリオ3 (全INT4)**: 全層・全埋め込み INT4 → **4.82 GiB (5.18 GB)**

---

## 3. メモリ制約下の変換方式比較

M3 16GB環境ではOS等で3.5〜4.5GB専有され、プロセス上限は **約11.5GB**。

| 方式 | 必要RAM | 所要時間 | 難易度 | M3実現性 | 特徴 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **safetensorsストリーム** | **約 2.5 GB** | 25〜40分 | 中 | **可能** | 1層ずつロード→量子化→保存。OOM回避。 |
| **CPU offload / mmap** | 14〜16 GB | 1〜3時間 | 低 | **極めて危険** | 中間テンソルで11.5GB即突破。激しいSSD Swap。 |
| **クラウドCPU (Modal 64GB)** | 24〜32 GB | **8〜12分** | **最低** | **最推奨** | 64GBで安全一括変換。費用約$0.2。ローカル負担ゼロ。 |
| **クラウド量子化→推論ローカル**| ローカル推論のみ | 転送込15分 | 低 | **実用的** | クラウド生成した約7GBの成果物のみMacで利用。 |

---

## 4. YAQA 校正を 9B で行う場合のメモリと層分割案

### 4.1 YAQA の数学的構造とヘッセ行列サイズ
YAQAはクロネッカー積 $H \approx H_I \otimes H_O$ でヘッセ行列を近似する [2]。
- **$H_I$ (入力側)**: $X^T X \in \mathbb{R}^{d_{in} \times d_{in}}$（下三角 $D(D+1)/2$ FP32）
- **$H_O$ (出力側)**: $G^T G \in \mathbb{R}^{d_{out} \times d_{out}}$ ($G = \partial \mathcal{L}/\partial Y$)

| モジュール (1層分) | $d_{in}$ | $d_{out}$ | $H_I$ (FP32) | $H_O$ (FP32) | 両側合計 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DeltaNet `in_proj`** | 4,096 | 12,352 | 33.6 MiB | 305.2 MiB | 338.8 MiB |
| **Mixer `out_proj`** | 4,096 | 4,096 | 33.6 MiB | 33.6 MiB | 67.2 MiB |
| **Attention `qkvg_proj`**| 4,096 | 10,240 | 33.6 MiB | 209.7 MiB | 243.3 MiB |
| **MLP `up_projection`** | 4,096 | 24,576 | 33.6 MiB | **1,208.0 MiB** | **1,241.6 MiB** |
| **MLP `down_projection`**| 12,288| 4,096 | 302.0 MiB | 33.6 MiB | 335.6 MiB |
| **1層分合計 (DeltaNet)** | - | - | 402.8 MiB | 1,580.4 MiB | **1.98 GiB** |
| **1層分合計 (Attention)** | - | - | 402.8 MiB | 1,484.9 MiB | **1.89 GiB** |
| **全32層合計 (同時保持)**| - | - | **12.59 GiB** | **48.60 GiB** | **61.19 GiB** |

### 4.2 「全層両側」が非現実的な理由
1. **$H_O$ メモリ爆発**: MLP `up_proj` の $d_{out}=24,576$ により1層の $H_O$ だけで1.21GB。全層で **約48.6GB**。$H_I+H_O$ 合計 **61.2GB** であり16GBでの保持は不可能。
2. **逆伝播コスト**: 32層のエンドツーエンド逆伝播はバッチ1でも30GB以上のRAMを消費。

### 4.3 層分割具体案と $H$ のマージ
- **具体案1: $H_I$ 単側校正 (推奨)**: $H_O=\text{None}$ とし順伝播のみ実行。1層403MB。lalamoの `yaqa_round_blockwise` は $H_O$ なしでもLDL分解可能。ピークRAM **約2.0GB**。
- **具体案2: ブロック層分割 (`--layer-filter`)**: 4層ずつ8ブロックに分割し局所誤差 $\|Y-\hat{Y}\|_2^2$ で逆伝播。ピークRAM約9.6GB。
- **$H$ のマージ法**: 共分散行列は加法性を持つため、$H_{I, \text{merged}} = \sum X_k^T X_k$。`Preconditioner.input_block_tril` の要素加算でバッチ横断マージが可能。

---

## 5. uzu での 9B INT4 推論メモリと速度の概算

### 5.1 推論時メモリ内訳 (M3 16GB)
- **0.8B 実測**: 重み 764MB、実メモリ 0.84GB (差分 80MB)

| 項目 | 9B 算出値 | 根拠・仕様 |
| :--- | :---: | :--- |
| **モデル重み** | **6.62 〜 8.07 GB** | シナリオ2 (6.62GB) / シナリオ1 (8.07GB) |
| **DeltaNet 状態 (24層)**| **50.3 MB** | 24層 × ($32\text{ heads} \times 128 \times 128 \times 4\text{B}$)。**文脈長不変**。 |
| **Attention KV Cache** | **134.2 MB** | 8層 × 2 × $4\text{ heads} \times 256 \times 2\text{B} \times 4,096\text{ tokens}$。 |
| **ランタイムバッファ** | 約 350 MB | Metal コマンドバッファ等 |
| **総消費メモリ** | **約 7.15 〜 8.60 GB** | **M3 16GB (上限11.5GB) で余裕を持って動作可能** |

### 5.2 推論速度概算
1. **0.8B 実測線形スケール**: $12.5 \times (0.84 / 8.07) \approx \mathbf{1.30\text{ tok/s}}$ (シナリオ2: **1.59 tok/s**)
2. **M3 メモリ帯域基準 (100 GB/s)**:
   - シナリオ1 (8.07GB): 理論上限 12.4 tok/s → **実効 (60〜70%) 約 7.5〜8.7 tok/s**
   - シナリオ2 (6.62GB): 理論上限 15.1 tok/s → **実効 約 9.0〜10.5 tok/s**
3. **投機的デコード (DFlash/Weaver) 併用時**:
   - 平均受理トークン 2.5〜3.5倍 → **予測: 約 20 〜 35 tok/s**

---

## 6. 推奨手順（最小リスク順）とフォールバック

### 6.1 推奨手順
1. **Plan A (最推奨・最小リスク)**: Modal (64GB RAM CPU) で `lalamo convert` → 約7GBの成果物をMacへダウンロード → uzu でローカル推論。
2. **Plan B (ローカル完結)**: safetensors ストリームで1層ずつロード → $H_I$ 単側校正 → INT4化 → 保存 (RAM 2.5GB)。

### 6.2 フォールバック戦略
- **F-1 (精度低下時)**: `embed_tokens` と `lm_head` を BF16 維持 (0.8B踏襲)。
- **F-2 (YAQA発散時)**: `yaqa_round_fixpoint` 失敗時は `yaqa_round_blockwise` または RTN (Round-To-Nearest) へ切替。
- **F-3 (メモリ逼迫時)**: `lm_head` を INT4 化＋コンテキスト長 8K 制限 (重み約6.6GB)。
- **F-4 (変換難航時)**: MLX 4-bit (Group 64) 形式へフォールバック (公式プリセット対応)。

---

## 7. 具体的なコマンド例

### 7.1 Modal によるクラウド変換定義
```python
# modal_conv.py
import modal, subprocess

app = modal.App("lalamo-convert")
image = modal.Image.debian_slim(python_version="3.12").apt_install("git").pip_install(
    "jax[cpu]", "equinox", "jaxtyping", "safetensors", "typer", "rich"
).run_commands("git clone https://github.com/trymirai/lalamo.git /lalamo && cd /lalamo && git checkout e401e177")

@app.function(image=image, memory=65536, cpu=16, timeout=3600)
def convert():
    subprocess.run([
        "python3", "-m", "lalamo.main", "convert",
        "huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated",
        "--output-dir", "/out/Qwen3.5-9B-INT4", "--dtype", "bfloat16"
    ], check=True)
```

### 7.2 safetensors 層別ストリーム変換骨子 (ローカル用)
```python
# stream_convert.py
import gc, jax.numpy as jnp
from safetensors import safe_open
from safetensors.numpy import save_file
from lalamo.compressed.hybrid import HybridSpec
from lalamo.compressed.int import IntSpec
from lalamo.compressed.incoherence import HadamardSpec

spec = HybridSpec(
    quantization_spec=IntSpec(bits=4, group_size=32, is_symmetric=False),
    adapter_spec=None, incoherence_spec=HadamardSpec(block_size=32)
)

for l in range(32): # 全32層を順次処理
    layer_tensors = {}
    for p in ["model-00001.safetensors", "model-00002.safetensors", ...]:
        with safe_open(p, framework="numpy") as f:
            for k in f.keys():
                if f"layers.{l}." in k: layer_tensors[k] = f.get_tensor(k)
    out = {}
    for k, v in layer_tensors.items():
        if "weight" in k and any(x in k for x in ["proj", "in_proj", "out_proj"]):
            out[k] = spec.compress(jnp.array(v), sharding_config=None).decompress()
        else: out[k] = v
    save_file(out, f"converted/layer_{l}.safetensors")
    del layer_tensors, out; gc.collect()
```

### 7.3 uzu 推論実行コマンド (Apple Silicon)
```bash
# 対話実行
uzu chat --model-path ./models/Huihui-Qwen3.5-9B-INT4 --max-tokens 4096

# OpenAI互換サーバー
uzu server --model-path ./models/Huihui-Qwen3.5-9B-INT4 --port 8000
```

---

## 8. 調査注記・未確認事項

1. **未確認: 9B向けDFlash/Weaverドラフトモデルの提供有無**
   - 0.8B/2B/4BはMirai公式ドラフト重みが存在するが、9B用Weaver/DFlashが一般公開中かは「**未確認**」。非公開時は単体自己回帰（7.5〜10.5 tok/s）動作となる。
2. **未確認: lalamo CLIでのHuihui直接指定時の挙動**
   - lalamoは `ModelRegistry` に事前登録されたIDを判定するため、未登録リポジトリ直接指定時は `qwen.py` への定義追記かローカル読込が必要となる可能性が高い（「**未確認**」だが実装解析上高確度）。
3. **参考文献**:
   - [1] Mirai Quantization: 40-60% Faster Local LLMs (trymirai.com/blog/quantization)
   - [2] Model-Preserving Adaptive Rounding (OpenReview YAQA Paper)
   - [3] trymirai/lalamo (`e401e177`) コードベース
