# 公式 uzu モデル（Mirai M）の構成を解析

2026-11-08 / 対象: `~/.cache/mirai/models/mirai/alibaba-qwen3.5-{0.8b,9b}-mirai-mirai-m-4`

## 1. 量子化の仕様

| 項目 | 0.8B-M | 9B-M |
|---|---|---|
| ファイルサイズ | 0.409 GB | 4.833 GB |
| bits | **全て 4bit** | **全て 4bit** |
| group size | **32** | **32** |
| RHT | block 32 / input_output | 同 |
| scales の dtype | BF16 | BF16 |
| zero points | U8（4bit を 2 個詰め） | 同 |
| embedding | 量子化（tied、単一） | **input/output を別々に量子化**（untied） |
| 8bit / 6bit 配分 | **無し** | **無し** |
| speculator | 同梱なし | **同梱あり（755MB, BF16）** |

**公式は「uniform 4bit group32 + embedding 量子化」だけ**。我々の down-out8 / wide8 は公式を超える試み。

## 2. scales が意外に大きい（サイズの 11%）

9B-M の内訳（実ファイルから集計）:

| 種別 | サイズ |
|---|---:|
| 量子化重み（U8） | 4,402 MB |
| **scales（BF16）** | **534 MB（11%）** |
| その他（F32/I32） | 12.5 MB |

- 埋め込みの scales だけで **127MB**（input/output 各 248320×128×2B）
- 層の scales は group 32 で `params/32 × 2B`

### 含意: group 64 にすると −5.5%

9B の次元（12352 / 24576 / 8192 / 4096 / 9216）はすべて 64 で割り切れる。
`group_size=64` にすると scales が半分 → **約 267MB 減（4.83 → 4.57GB）**、TPS **+5.8%** 相当。
公式が 32 を選んでいるのは品質優先と推測される（未検証）。

## 3. speculator（9B のみ同梱）

| ファイル | サイズ |
|---|---:|
| `speculator/model.safetensors` | **755 MB（BF16）** |
| `speculator/config.json` | 17 KB |
| `speculator/shapes.json` | 1.4 KB |

- 9B は 4.83GB + 0.755GB = **5.59GB**。M3 16GB に余裕
- 0.8B には同梱なし（小型で既に速いため）
- 投機は**帯域の壁を越える唯一の手段**（`research/uzu_engine_tps.md`）

## 4. 我々の変換との差分

| 項目 | 公式 | 我々（従来） | 対応 |
|---|---|---|---|
| embedding 量子化 | する | **しない**（既定 off） | **必ず付ける**（速度 +63%） |
| 8bit 配分 | しない | wide8 で品質改善 | 公式超え。4B で要検証 |
| group | 32 | 32 | 4B/9B は 64 の余地（−5.5%） |
| speculator | 9B に同梱 | なし | 欲しければ draft 学習（SpecForge） |

## 5. 速度の実測（0.8B）

| モデル | サイズ | Generate |
|---|---:|---:|
| 公式 0.8B-M | 0.521GB | 142.9 t/s |
| 我々（RTN + emb4） | 0.521GB | 143.0 t/s |

**同一サイズ・同一速度**。差は embedding の量子化だけ。

## 6. そこから得られる設計指針

1. **embedding/readout は必ず量子化**する。0.8B では 34% を占め、速度差が +63%
2. **4bit group32 を基準**にする（公式と同じ）
3. **group 64 は 4B/9B の高速化オプション**（サイズ −5.5%、品質影響は未検証）
4. **8bit 配分は公式にない追加レバー**。品質は上がるがサイズ増 → 速度 −4〜12%
5. **speculator は速度の次元を変える**。9B は公式同梱の 755MB があるが、Huihui 用は自前で学習が必要

## 7. 公式公開情報（Web / 2026-11-08 調査）

### 7.1 量子化の方法（モデルカードの明記）

> "Mirai Medium uses 4-bit asymmetric integer quantization with 4-bit zero points,
> bfloat16 scales, and group size 32. Block-diagonal Random Hadamard Transforms are used
> to reduce activation and weight outliers. The checkpoint was prepared with post-training
> quantization followed by quantization-aware distillation."

出典: [trymirai/Qwen3.5-4B-M](https://huggingface.co/trymirai/Qwen3.5-4B-M)、[blog/quantization](https://trymirai.com/blog/quantization)

- **PTQ（＝YAQA 相当）+ QAD** の2段構成。我々の YAQA + QAD と同系統
- 4bit 非対称、zero point 4bit、scales bf16、group 32、RHT block 32
- Mirai-L（8bit）は **symmetric** int8、group 32、PTQ のみ

### 7.2 品質の主張

- 0.8B / 2B の M: llama.cpp の **Q4_K_M と同程度の品質で 20% 小さい**
- 4B の M: Q3_K_M と Q4_0 の間
- L（8bit）: MLX 8bit と区別できない
- KL vs サイズの Pareto 前沿にあると主張

### 7.3 速度の公称値（M5 Max 128GB）

| モデル | uzu | MLX | llama.cpp |
|---|---:|---:|---:|
| 9B-M 出力 | **264 t/s（specdec）** | 71 | 90 |
| 9B-M 入力 | 2,357 t/s | 2,129 | 2,264 |
| 9B-M 常駐 | 6.03 GB | 6.03 | 6.21 |

- 投機デコードは **2026-09-03 リリース**
- **budget 16〜32 トークン**の積極的な投機。MTP の 3〜4 より大きい
- M5 の Neural Accelerator を使う GEMM カーネル前提
- 我々の M3（82GB/s）で 143 t/s なので、M5 Max の公称値と直接比較できない

### 7.4 投機の構造（第三者検証）

[Zenn の検証記事](https://zenn.dev/amu_lab/articles/uzu-speculative-decoding-16gb-mac-benchmark)（M5 16GB, uzu 0.6.2）:

- 投機あり 85.9〜88.1 t/s → **投機を外すと 26.9 t/s**
- DFlash の draft は本体 32 層のうち `1,5,9,...,29` 層の特徴を読む（`target_layer_ids`）
- **draft はモデルごとに専用設計で、他モデルに差し替え不可**
- 常駐メモリ 6.22GB → 5.32GB（投機を外すと −0.9GB）

### 7.5 我々の位置づけ

| 項目 | 公式 | 我々 | 差 |
|---|---|---|---|
| 量子化 | 4bit g32 + PTQ + **QAD** | YAQA + (QAD 実験中) | QAD が未完了 |
| embedding | 量子化 | 既定 off → 修正済み | 速度 +63% |
| 8bit 配分 | 無し | wide8 で改善 | 公式超え（品質） |
| speculator | モデル毎に同梱 | 無し | 速度で大差 |
| 品質指標 | KL で Pareto 主張 | BPC Δ で評価 | 指標が違う |

**重要な示唆**: 公式の速度は「量子化」ではなく **speculator** で稼いでいる。
9B で 264 vs 90（llama.cpp）の差は投機由来。量子化だけなら我々は既に同等。

## 8. 追加調査（2026-11-09）

### 8.1 公開リポジトリの全体像

| リポジトリ | 中身 |
|---|---|
| `trymirai/uzu` | エンジン本体（★3,240）。我々のフォーク元 |
| `trymirai/lalamo` | JAX の量子化基盤。我々が変換に使用 |
| **`trymirai/sglang`** | **DFlash-TfM の再現実装**（論文つき） |
| **`trymirai/romakl`** | KL 評価スイート（`kl-eval` の後継。旧 `kl-eval` は archive） |
| `trymirai/miraitensors` | INT4 を扱う safetensors フォーク |
| `trymirai/mpsgraph-rs` / `xgrammar-rs` / `slang-rs` | 補助クレート |

**我々が使っている lalamo は公式の量子化基盤そのもの。** 変換の再現性はここで担保される。

### 8.2 DFlash-TfM（公式 speculator の正体）

論文: *Trees from Marginals: Autoregressive drafting with factorized priors*
（arXiv 2607.06763、2026-07、v3 2026-10）

- DFlash は 16 トークンの**周辺分布**を並列に出す（独立仮定）
- **Weaver** が「実現した draft トークン」を条件に、**top-512 候補上の残差 logits** を予測して木を作る
- フル語彙の射影を避ける（512 候補に限定）
- **Weaver は 56.7M パラメータ**（我々の 9B speculator の config と一致: 1 層 / 2048 dim / 16 heads / max_depth 15 / rounds 8 / expand 4 / candidate_pool 512）
- 効果（Qwen3.6-27B, B200）: AR 89.9 → DFlash 315.0（3.50x）→ **DFlash-TfM 392.8（4.37x）**
- **budget 64** が最速。我々が使っていた budget 8〜16 より大きい

含意:

1. **Weaver は 57M の小さなアダプタ**。ゼロから 9B を学習するより桁で安い
2. ただし**自己回帰**なので、並列 draft のように 1 forward では終わらない
3. 学習には「draft の候補集合」と「実現した系列」が必要（受理信号ではなく次トークン教師でよい）
4. **budget を 16 → 64 に増やす余地**。我々の shapes.json は 16 で固定している

### 8.3 KL 評価を公式と同じ土俵にする

`romakl` は lalamo / GGUF / EXL3 を同じ基準で比較する。
既定データセット `trymirai/kl-mixture` は**非公開**（アクセス申請が必要）。

- 我々が自前で KL を実装するより、`romakl` に lalamo モデルを渡すのが早い
- ただし NVIDIA GPU 必須（Linux）
- データセットが非公開なので、**完全に同じ数値にはならない**。相対比較のみ

### 8.4 我々の現在地との差分

| 項目 | 公式 | 我々 |
|---|---|---|
| 量子化 | PTQ + QAD | PTQ（QAD 実験中） |
| Weaver | 57M、top-512 残差 | **無し**（Argmax chain） |
| budget | 64 | 16 |
| 評価 | KL（非公開データ） | BPC（公開コーパス） |

**次にやるなら budget 64 の検証が一番安い。** shapes.json の `tree_budget` を上げるだけで試せる
（Argmax では `dflash_depth` が上限なので 16 まで。64 は Weaver 前提）。
