# Qwen3.5 量子化感度・Mixed Precision 設計（検証版）

**数値ラベル**: 【実測】= HF の `config.json` / safetensors ヘッダをローカル取得し集計、【引用】= 公開資料、【推定】= 計算式から算出。単位は断りなければ GB（1 GB = 10^9 B）。

---

## 1. アーキテクチャ（公式資料 + 実測）

### 1.1 確定仕様

出典: [Qwen3.5-9B config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)、[0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)、[9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)。パラメータ数は safetensors ヘッダ集計【実測】。

| 項目 | 0.8B | 9B | 根拠 |
| :--- | ---: | ---: | :--- |
| 層数 / 周期 | 24（GDN 18 + GA 6） | 32（GDN 24 + GA 8） | `layer_types`, `full_attention_interval=4`【実測】 |
| hidden / FFN | 1,024 / 3,584 | 4,096 / 12,288 | config【実測】 |
| vocab | 248,320 | 248,320 | config【実測】 |
| GDN heads (QK/V) | 16 / 16, d=128 | 16 / 32, d=128 | config【実測】 |
| GA heads (Q/KV), head_dim | 8 / 2, **256** | 16 / 4, **256** | config【実測】 |
| Embedding | **tied**（`lm_head` 無し） | **untied**（`lm_head.weight` 独立） | safetensors index【実測】 |
| 総パラメータ | 873.4M | 9,653.1M | safetensors 合計【実測】 |
| 内訳: text / visual / lm_head / MTP | 752.4M / 100.6M / 0 / 20.5M | 7,936.7M / 456.0M / 1,017.1M / 243.3M | ヘッダ prefix 集計【実測】 |

> 9B は model card 上「Causal LM with Vision Encoder」。text-only 推論は vision 0.456B を落とせる（[model card の `--language-model-only`](https://huggingface.co/Qwen/Qwen3.5-9B)）。9B は **tied ではない**。依頼前提「9B tied」は誤り。

### 1.2 DeltaNet と Gated Attention

GDN 更新（[Gated Delta Networks, arXiv:2412.06464](https://arxiv.org/abs/2412.06464)）:

$$S_t = \alpha_t S_{t-1}(I-\beta_t k_t k_t^\top)+\beta_t v_t k_t^\top,\quad o_t=\mathrm{RMSNorm}(S_t q_t)\odot\sigma(z_t)$$

- 状態は head あたり 128×128。9B は V head 32 → 24 層 × 32 × 16,384 × 2 B = **24.0 MB 固定**【推定】。文脈長に非依存。
- GA は 8 層のみ。KV は 2 × 8 × 4 × 256 × 2 B = **32 KB/token**【推定】。32k で 1.00 GB。
- GDN の write は現 key 方向で状態を上書きするため、注入誤差は平坦 plateau 後に消える（[arXiv:2609.04098](https://arxiv.org/abs/2609.04098) の機構解析）【引用】。

### 1.3 Embedding / readout

| モデル | 入力埋め込み | 出力 `lm_head` | 合計 / text 比 |
| :--- | ---: | ---: | ---: |
| 0.8B | 254.28M（tied 共有） | 同左（共有） | 254.28M / 33.8%【推定】 |
| 9B | 1,017.1M | 1,017.1M（独立） | 2,034.2M / 22.7%【推定】 |

- 9B は embedding + lm_head だけで 2.03B。INT4 0.5625 B/param でも **1.14 GB**、BF16 なら **4.07 GB**。ここが最大の配分変数。
- tied の 0.8B は入力 Gather と出力 GEMM が同一重み。AWQ 型の活性化平滑化は入力埋め込み空間を歪めるため両立困難（[PyTorch Blog](https://pytorch.org/blog/advancing-low-bit-operators-in-pytorch-and-executorch-dynamic-kernel-selection-kleidiai-and-quantized-tied-embeddings/)、[TensorRT-Edge-LLM #253](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/253)）【引用】。

### 1.4 壊れやすい部位の仮説（文献で修正済み）

1. **`lm_head`（9B は独立）**: 248k 語彙の順位を直接決める。最優先で INT8 以上。
2. **MLP `down_proj`**: SwiGLU 積後の外れ値が集中。INT8 または RHT 必須。
3. **GDN 主射影**: 最も堅牢。W4A4 NVFP4 でも BF16 同等（2609.04098）【引用】。
4. **GDN ゲート $\alpha,\beta$**: 直感に反し**低感度**。softplus/sigmoid が GEMM 誤差 ~11% を出力 ~2% に圧縮（同論文）【引用】。BF16 固定は不要。
5. **GA 射影**: 8 層のみだが大域照合。INT8 の費用対効果が高い。
6. **vision / MTP**: text-only では不要。残すと BF16 で +1.40 GB【推定】。

---

## 2. 公開研究

### 2.1 Layer-wise sensitivity

- **U 字型**: 浅層・深層が高感度、中間層は冗長。特に第 0 層と最終層が最大（[SliderQuant, arXiv:2603.25284](https://arxiv.org/abs/2603.25284)）【引用】。
- 層別ビット配分を品質制約下で最適化する手法も同様の結論（[arXiv:2608.28003](https://arxiv.org/abs/2608.28003)）【引用】。
- ただし 9B で第 0/31 層**全体**を BF16 にすると 428.1M param、+0.615 GB（INT4 比）【推定】。費用大。

### 2.2 Outlier / rotation

| 手法 | 要点 | 出典 |
| :--- | :--- | :--- |
| SmoothQuant | 活性化外れ値を重みへ移す W8A8 | [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) |
| AWQ | 上位 1% チャネルをスケーリング保護 | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| QuaRot / SpinQuant | Hadamard 回転で外れ値分散 | [2404.00456](https://arxiv.org/abs/2404.00456) / [2405.16406](https://arxiv.org/abs/2405.16406) |
| RHT Group32 | ブロック対角 Hadamard、SIMD-32 整合 | 依頼者現行方式【未確認】 |

### 2.3 部位別の感度とパラメータ比（9B text-only 8.954B 基準）

| 部位 | パラメータ | 比率 | 感度 | 根拠 |
| :--- | ---: | ---: | :---: | :--- |
| 入力 embedding | 1.017B | 11.4% | 高 | Gather。tied なら readout と共有 |
| `lm_head` | 1.017B | 11.4% | **極高** | 語彙ロジット直接 |
| MLP gate+up | 3.221B | 36.0% | 低 | 正規化直後、外れ値穏和 |
| MLP down | 1.611B | 18.0% | **高** | SwiGLU 積の外れ値 |
| GDN 射影 | 1.618B | 18.1% | 低 | 2609.04098 |
| GA 射影 | 0.470B | 5.2% | 中〜高 | 8 層の大域照合 |
| Norm | 0.0003B | 0.003% | 高 | BF16 固定（容量無視可） |

---

## 3. Mixed precision 方針と実験設計

### 3.1 配分基準

感度当たりメモリ効率 $\nabla = \Delta\text{PPL}/(\Delta\text{MB})$ を最大化する。

| 精度 | 割当候補 | 理由 |
| :--- | :--- | :--- |
| BF16 | Norm、必要なら GA 射影のみ | 容量増が小さい |
| INT8 | 9B `lm_head`、`down_proj`、GA 射影 | 高感度部位の最小コスト保護 |
| INT4 G32+RHT | MLP gate/up、GDN 射影、入力 embedding | 低感度・大容量 |
| INT4+QAD | embedding / lm_head（品質回復前提） | 依頼者情報。ローカル再現未確認 |

**重要な修正**: GDN ゲートを BF16 に固定する必要はない（2609.04098）。一方、9B の `lm_head` を INT4 のままにするのは危険。tied でないため embedding と分離して扱える。

### 3.2 Ablation 設計

1. Baseline: 全 Linear INT4 G32+RHT。One-at-a-time で 1 部位ずつ INT8/BF16 へ。
2. データ: WikiText-2 (4k), C4 val, Pile-val, GSM8K 8-shot, MATH-500, HumanEval。
3. 指標: PPL、$D_{KL}(P_{BF16}\|P_{Q})$（top-100）、top-1/top-5 一致率、長文 8k/32k の PPL 劣化。
4. 統制: 3 seed、temperature 0、同一プロンプト。量子化誤差のみを比較するため BF16 参照は同一 checkpoint。

---

## 4. 0.8B 先行 ablation

0.8B text-only = 752.4M param【実測】。全 INT4 は 752.4M × 0.5625 = **423.2 MB**【推定】。vision/MTP 込みは 491.3 MB。

| ID | 条件 | 対象 param | メモリ増（INT4 比） | 優先度・期待 |
| :---: | :--- | ---: | ---: | :--- |
| 0 | Baseline 全 INT4 | — | 0（423.2 MB） | 基準 |
| 1 | tied embedding BF16 | 254.28M | +365.5 MB | **最高**。語彙崩壊の切り分け |
| 2 | tied embedding INT8 | 254.28M | +111.2 MB | 高。QAD 無しの現実解 |
| 3 | L0+L23 全体 BF16 | 39.3M | +56.5 MB | 高。U 字端部の実測 |
| 4 | 全 24 層 `down_proj` INT8 | 88.08M | +38.5 MB | 中。外れ値対策 |
| 5 | GA 6 層 射影 INT8 | 44.05M | +19.3 MB | 中。長文照合 |
| 6 | GDN $\alpha,\beta$ BF16 | 0.59M | +0.85 MB | 低（文献上は不要） |

計算式: 増分 = param × (新 B/param − 0.5625)。INT8=1.0、BF16=2.0。EXP-3 は 21.5M (L0 GDN) + 17.8M (L23 GA) = 39.3M。

**順序**: 1 → 2 → 3 → 5 → 4。EXP-6 は文献否定の確認用に最後。

---

## 5. 9B を M3 16GB に収める

### 5.1 前提と固定コスト

text-only、vision/MTP 除外。パラメータは §1.1 の実測値。

| 項目 | 計算 | 値 |
| :--- | :--- | ---: |
| 全 INT4 重み | 8.954B × 0.5625 | 5.04 GB |
| GDN state | 24 × 32 × 128×128 × 2 B | 0.024 GB |
| KV 8k / 32k | 32 KB/token × 8,192 / 32,768 | 0.25 / 1.00 GB |
| 作業バッファ | Metal、活性化、投機 | 1.20 GB（仮定） |
| macOS | 常駐 | 2.80 GB（環境依存） |

### 5.2 推奨構成

| 部位 | A: QAD・攻め | B: 非 QAD・安全 | 計算 |
| :--- | ---: | ---: | :--- |
| 入力 embedding | 0.572 (INT4) | 0.572 (INT4) | 1.017B × 0.5625 |
| `lm_head` | 0.572 (INT4+QAD) | **1.017 (INT8)** | 1.017B × 1.0 |
| L0+L31 全体 | 0.856 (BF16) | 0.856 (BF16) | 428.1M × 2.0 |
| `down_proj` 30 層 | 0.849 (INT4) | **1.510 (INT8)** | 1.510B × 1.0 |
| gate/up 30 層 | 1.699 (INT4) | 1.699 (INT4) | 3.020B × 0.5625 |
| GDN 23 層 | 0.872 (INT4) | 0.872 (INT4) | 1.550B × 0.5625 |
| GA 7 層 | 0.231 (INT4) | **0.411 (INT8)** | 0.411B × 1.0 |
| Norm | 0.050 | 0.050 | BF16 |
| **重み計** | **5.70** | **6.99** | 合計 |
| **32k 総計** | **10.72** | **12.01** | + state 0.024 + KV 1.00 + buf 1.20 + OS 2.80 |
| 16GB 残 | 5.28 | 3.99 | 理論値 |

- B は実効上限 12.5〜13.0 GB に対し **0.5〜1.0 GB の余裕**。スワップ回避は可能だが、投機デコーディングを増やすと危険【推定】。
- A は QAD checkpoint が前提。QAD 無しで lm_head INT4 にすると PPL 劣化リスク大。
- 中間案: `lm_head` INT8 のまま edges を attention のみ BF16（126.1M、+0.181 GB）にすると **11.58 GB**、余裕 +0.43 GB。
- vision を残す場合: INT4 +0.257 GB、BF16 +0.912 GB。MTP は INT4 +0.137 GB。text-only 推奨。

### 5.3 注意

- macOS の GPU wired limit は 16GB 機で既定 ~10.7 GB の報告があり、B の 12.0 GB は超過し得る。`iogpu.wired_limit_mb` 調整または context 短縮が必要【未確認・環境依存】。
- KV 32 KB/token は GA 8 層前提。full attention を増やすと線形に増える。

---

## 6. 未確認事項

1. **M3 16GB の実効 GPU 上限**: 既定 wired limit、OS 予約 2.8 GB の実測値。B 構成の成否を左右。
2. **9B の tied 前提崩れに伴う既存実装の整合**: 現行量子化器が `lm_head.weight` を別テンソルとして扱えるか。tied 前提のコードなら要修正。
3. **QAD checkpoint の入手性・再現性**: Mirai-M の INT4 embedding + QAD の公開有無、M3 での蒸留コスト。
4. **RHT Group32 の実装詳細**: 9B の GA head_dim 256 / GDN head 128 に対する回転ブロック整合。
5. **GDN 長文ドリフト**: 32k 超での誤差蓄積。2609.04098 は 32K plateau を報告するが 64K 以上は限定的。
6. **vision encoder / MTP の量子化可否**: text-only で落とす前提の検証。MTP を残す場合の投機精度。
7. **model card の "sparse MoE" 記述**: 9B config に `num_experts` が無く不一致。別 variant の可能性。
8. **0.8B baseline 423 MB の実測**: mmap・アロケータ・スケール込みの RSS は未計測。

---

## 参考文献

1. https://huggingface.co/Qwen/Qwen3.5-9B
2. https://huggingface.co/Qwen/Qwen3.5-0.8B
3. https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json
4. https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json
5. https://qwen.ai/blog?id=qwen3.5
6. https://arxiv.org/abs/2412.06464 — Gated Delta Networks
7. https://arxiv.org/abs/2609.04098 — Why Gated DeltaNet Survives 4-Bit Quantization
8. https://arxiv.org/abs/2609.38166 — LeapQuant
9. https://arxiv.org/abs/2609.39223 — QATFactory
10. https://arxiv.org/abs/2603.25284 — SliderQuant
11. https://arxiv.org/abs/2608.28003 — Layer Bit-Width Allocation
12. https://arxiv.org/abs/2211.10438 — SmoothQuant
13. https://arxiv.org/abs/2306.00978 — AWQ
14. https://arxiv.org/abs/2404.00456 — QuaRot
15. https://arxiv.org/abs/2405.16406 — SpinQuant
16. https://pytorch.org/blog/advancing-low-bit-operators-in-pytorch-and-executorch-dynamic-kernel-selection-kleidiai-and-quantized-tied-embeddings/
17. https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/253
