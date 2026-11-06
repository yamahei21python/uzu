# Qwen3.5-0.8B 量子化評価ハーネス計画書 (BF16 / RTN INT4 / YAQA INT4)

## 1. 概要と評価方針

### 1.1 目的
軽量モデル **Qwen3.5-0.8B** に対する量子化（**BF16 Reference**, **RTN INT4 Baseline**, **YAQA INT4 Target**）の劣化度合いを、日英両環境で BF16 との差分（$\Delta$）として高速・高精度に測る評価ハーネス計画を策定する。既存資産（teacher-forced PPL/logits スクリプト、WikiText-2、日本語 Wikipedia 固定コーパス）と推論エンジン [trymirai/uzu](https://github.com/trymirai/uzu)（Rust/Metal）を最大限に活用し、本番コードは一切変更しない。

### 1.2 モデル仕様とハードウェア構成
- **対象モデル**: Qwen3.5-0.8B（24層、Hidden Dim 1024、Vocab 248,320、Gated DeltaNet + Gated Attention）
- **ハードウェア**: Apple M3 (8-core CPU, 10-core GPU, 16GB Unified Memory)
- **基盤スタック**: Python 3.12 / JAX（損失・logits 計算）、`uzu`（Metal 最適化推論 / OpenAI 互換サーバー）

| モデル構成 | 表現形式 | メモリ使用量 | 評価上の位置づけ | 期待される挙動 |
| :--- | :--- | :--- | :--- | :--- |
| **BF16** | 16-bit Float | ~1.6 GB | 正解基準 (Anchor) | 全タスクの基準値（$\Delta = 0$） |
| **RTN INT4** | 4-bit (Round-To-Nearest) | ~0.55 GB | ベースライン (Naive PTQ) | 丸め誤差蓄積による大幅劣化 |
| **YAQA INT4** | 4-bit (Adaptive Rounding) | ~0.55 GB | 評価対象 (SOTA PTQ) | 出力分布維持により BF16 近傍を維持 |

---

## 2. llm-jp-eval 最新セットアップと 0.8B 向けサブセット選定

### 2.1 概要・ライセンス・セットアップ
[llm-jp/llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)（v2.1.5 dev）は `uv` と `Taskfile` に移行し、OpenAI 互換サーバー推論（`provider: vllm-openai`）に対応している。
- **ライセンス**: 本体 Apache-2.0。個別データ（JGLUE: CC BY-SA 4.0、JMMLU: MIT/CC等）。非商用データ除外は `--include_non_commercial false` で制御。

```bash
# セットアップ手順 (uv / Taskfile)
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/llm-jp/llm-jp-eval.git && cd llm-jp-eval
uv sync && uv tool install go-task-bin
cp configs/config_template.yaml configs/config.yaml && cp env.sample .env
task hf_login  # .env に HF_TOKEN 設定後
```

### 2.2 0.8B 向け現実的サブセットと実行見積もり
0.8B では推論破綻リスクと評価時間の観点から、出力が短く弁別力の高いタスクを厳選する。

| カテゴリ | タスク名 | データセット | サンプル数 | 出力形式 | 評価指標 | 所要時間 (M3/uzu)* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **常識推論** | `jcommonsenseqa` | JGLUE | 1,119 (dev) | 選択肢 (1-5) | Exact Match | 約 2.5 分 |
| **含意関係** | `jnli` | JGLUE | 2,434 (dev) | 3値分類 | Exact Match | 約 5.0 分 |
| **知識 QA** | `jmmlu` (コア10分野) | JMMLU | 1,000 (抽出) | 選択肢 (A-D) | Exact Match | 約 2.5 分 |
| **指示追従** | `m_ifeval` (ja split) | M-IFEval | 172 (ja) | 自由記述 | Rule Strict Acc | 約 3.0 分 |
| **合 計** | - | - | **4,725 件** | - | - | **約 13.0 分** |

*\*M3 Metal 上の uzu 推論（生成 ~200 tok/s、Prefill ~500 tok/s）に基づく試算。*

### 2.3 実行コマンド (uzu OpenAI 互換サーバー連携)

```bash
# データ準備
task prepare dataset=jcommonsenseqa
task prepare dataset=jnli
task prepare dataset=m_ifeval

# configs/config.yaml に online_inference_config (uzu サーバー: localhost:8000) を設定し実行
uv run scripts/evaluate_llm.py eval --config configs/config.yaml
```

---

## 3. 指示追従ベンチマーク (M-IFEval / JFBench / IFEval) 最小構成

### 3.1 ルールベース判定の選定理由
LLM-as-a-Judge は判定ブレや API コストが伴う。コードによる決定論的ルール検証に絞ることで、量子化に起因するフォーマット崩れや構文逸脱を直接検出する。

| ベンチマーク | リポジトリ / 参照 | 言語 | 判定方式 | 最小評価構成 |
| :--- | :--- | :--- | :--- | :--- |
| **IFEval** | [google-research/ifeval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | 英語 | ルールベース | 100問抽出 (Greedy, `temp=0.0`) |
| **M-IFEval** | [lightblue-tech/M-IFEval](https://github.com/lightblue-tech/M-IFEval) | 多言語 | ルールベース | 日本語 split 全量 (172問) |
| **JFBench** | [pfnet-research/jfbench](https://github.com/pfnet-research/jfbench) | 日本語 | ルール/Judge 併用 | **決定論的ルール制約のみ** (制約数1~2, 100問) |

### 3.2 実行手順

```bash
# M-IFEval (ja) 実行
git clone https://github.com/lightblue-tech/M-IFEval.git && cd M-IFEval
uv pip install -r requirements.txt
python3 -m evaluation_main \
  --input_data=data/m_ifeval_input_ja.jsonl \
  --input_response_data=outputs/response_qwen35.jsonl \
  --output_dir=results/m_ifeval --language=ja

# JFBench (ルールベース検証モード) 実行
git clone https://github.com/pfnet-research/jfbench.git && cd jfbench && uv sync
python3 src/jfbench/benchmark/run.py \
  --n-constraints=1,2 --constraint-set=test --n-benchmark-data=100 \
  --no-with-generate --with-eval --output-dir=results/jfbench
```

---

## 4. 日本語長文評価 (RULER / LongBench v2 / ABEJA) 4K/8K 最小構成

### 4.1 M3 16GB での成立性
Qwen3.5-0.8B は Gated DeltaNet（Linear Attention）を内包し、KV キャッシュ消費が極めて小さい。4K/8K は M3 16GB で余裕を持って動作する。16K 超の安定性は未確認。

| 系列長 | 重み (INT4) | KVキャッシュ/状態 | ワークスペース | 合計メモリ | 1クエリ所要時間 (Prefill+16tok) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4K (4,096)** | ~0.55 GB | ~0.08 GB | ~0.50 GB | **約 1.15 GB** | 約 1.2 秒 |
| **8K (8,192)** | ~0.55 GB | ~0.16 GB | ~0.70 GB | **約 1.41 GB** | 約 2.5 秒 |
| 16K (未確認) | ~0.55 GB | ~0.32 GB | ~1.10 GB | 約 2.00 GB | 未確認 (スループット急減懸念) |

### 4.2 ABEJA longcontext-evaluation による統合実行
[abeja-inc/longcontext-evaluation](https://github.com/abeja-inc/longcontext-evaluation)（Apache-2.0）は、RULER、LongBench v2、OpenAI-MRCR（日本語訳: `abejaOpenAI-MRCR-Translation-JPN`）を単一 CLI で実行できる。

```bash
git clone https://github.com/abeja-inc/longcontext-evaluation.git && cd longcontext-evaluation && uv sync

# データセット合成 (RULER 日本語 QA / NIAH 4K & 8K)
python3 scripts/benchmarks/RULER/synthesize_evaluation_dataset/niah/make_dataset.py \
  --config scripts/benchmarks/RULER/synthesize_evaluation_dataset/niah/config.yaml

# uzu OpenAI 互換サーバーに対して 4K/8K 評価を実行
export BASE_URL=http://localhost:8000/v1
python3 scripts/benchmarks/run.py \
  --config scripts/benchmarks/run_configs/vllm_openai_compatible.yaml \
  --context_length 4096,8192
```

---

## 5. Perplexity (PPL) 評価: WikiText-2 & 日本語 Wikipedia

### 5.1 計算手法の標準化 (Strided Sliding Window)
固定長ブロック分割による先頭トークンのコンテキスト欠損（コールドスタート）を防ぐため、**Strided Sliding Window**（Context=2048, Stride=512）を採用する。llama.cpp 方式に準拠し、チャンク後半（512トークン）のみのクロスエントロピー損失を集計する。

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_{<i}), \quad \text{PPL} = \exp(\mathcal{L})$$

### 5.2 コーパス仕様と判定基準

| コーパス | 言語 | 規模 | 取得元 | 目的 |
| :--- | :--- | :--- | :--- | :--- |
| **WikiText-2-raw-v1** | 英語 | 約 24 万 tok | Hugging Face (`wikitext`) | 英語基盤モデリング能力 |
| **JA-Wikipedia** | 日本語 | 約 25 万 tok | 既存固定コーパス | 日本語語彙・構文モデリング能力 |

| 判定基準 | $\Delta \text{PPL}$ (絶対値差) | 相対劣化率 ($\Delta / \text{PPL}_{\text{BF16}}$) | 解釈 |
| :--- | :--- | :--- | :--- |
| **合格 (Pass)** | $\le +0.35$ | $\le +3.0\%$ | 出力分布が強固に保持（YAQA INT4 目標） |
| **警告 (Warn)** | $+0.35 \sim +0.80$ | $+3.0\% \sim +8.0\%$ | 微小劣化。下流タスクでの検証要 |
| **不合格 (Fail)** | $> +0.80$ | $> +8.0\%$ | 丸め誤差崩壊（Naive RTN で発生） |

---

## 6. KL / Logits 評価: LMSYS-Chat-1M による Mirai 流手順の再現

### 6.1 数理定義と Mirai 実績値
[Mirai Labs Quantization](https://trymirai.com/blog/quantization) に倣い、実対話プロンプトに対する **BF16 Teacher 生成回答系列上** で、Teacher ($P$) と Quantized ($Q$) の Softmax 分布間 Mean KL Divergence を測定する。

$$D_{\text{KL}}(P \parallel Q) = \sum_{t \in \text{Response}} \sum_{v \in \mathcal{V}} P(v \mid x_{<t}) \log \frac{P(v \mid x_{<t})}{Q(v \mid x_{<t})}$$

- **理論的ノイズ床**: 浮動小数点演算順序差による限界値 $\approx 8 \times 10^{-4}$ (0.0008)
- **Top-1 一致率**: 各位置で $\operatorname{argmax}(P) == \operatorname{argmax}(Q)$ となる割合（健全時 $\ge 95\%$）

### 6.2 手順と JAX 再現コード案
1. [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) より第1ターンプロンプト 100 件抽出（英 50 / 日 50）。
2. BF16 モデルで Greedy 回答（`max_tokens=256`）を事前生成。
3. プロンプト＋回答系列を Teacher-Forced 順伝播し、回答部分の logits（Vocab 248,320）から KL を集計。

```python
import jax
import jax.numpy as jnp

def compute_mean_kl_and_agreement(t_logits, s_logits, mask):
    """t_logits, s_logits: [SeqLen, 248320], mask: [SeqLen] (回答部=1.0)"""
    log_p = jax.nn.log_softmax(t_logits, axis=-1)
    log_q = jax.nn.log_softmax(s_logits, axis=-1)
    p = jnp.exp(log_p)
    kl_per_tok = jnp.sum(p * (log_p - log_q), axis=-1)
    
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    mean_kl = jnp.sum(kl_per_tok * mask) / denom
    
    t_top1 = jnp.argmax(t_logits, axis=-1)
    s_top1 = jnp.argmax(s_logits, axis=-1)
    top1_match = jnp.sum((t_top1 == s_top1) * mask) / denom
    return float(mean_kl), float(top1_match)
```

---

## 7. 推奨実行順序と段階別合格目安 (4-Tier Gated Pipeline)

高速判定可能な指標から順に実行し、不合格モデルを即座に弾く構成とする。

```
[Stage 1: PPL & Logits ゲート]  --> (Pass) --> [Stage 2: 指示追従ルール判定]
   (~3分 / Teacher-Forcing)                        (~5分 / M-IFEval)
            | (Fail)                                        | (Fail)
            v                                               v
        即時破棄                                         即時破棄
            |                                               |
            +--------------> [Stage 3: 日本語コア] <---------+
                             (~10分 / JGLUE & JMMLU)
                                      | (Pass)
                                      v
                             [Stage 4: 長文文脈 4K/8K]
                             (~15分 / RULER 日本語)
```

### 段階別判定マトリクス

| 評価段階 | 評価項目 | 評価指標 | BF16 基準値 | YAQA INT4 目標 (Pass) | RTN INT4 警戒域 (Fail) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1**<br>(~3分) | LMSYS 100問 | Mean KL | 0.0000 | **$\le 0.0150$** | $> 0.0450$ |
| | LMSYS 100問 | Top-1 一致率 | 100.0% | **$\ge 95.0\%$** | $< 90.0\%$ |
| | WikiText-2 | $\Delta \text{PPL}$ | 0.00 | **$\le +0.30$** | $> +0.80$ |
| | JA-Wikipedia | $\Delta \text{PPL}$ | 0.00 | **$\le +0.35$** | $> +0.90$ |
| **Stage 2**<br>(~5分) | M-IFEval (ja) | Strict Acc | ~48%* | **$\Delta \ge -3.0\%$** | $\Delta < -8.0\%$ |
| | IFEval (en) | Strict Acc | ~58%* | **$\Delta \ge -3.0\%$** | $\Delta < -8.0\%$ |
| **Stage 3**<br>(~10分) | JCommonsenseQA | Exact Match | ~70%* | **$\Delta \ge -1.5\%$** | $\Delta < -4.0\%$ |
| | JNLI | Exact Match | ~62%* | **$\Delta \ge -2.0\%$** | $\Delta < -5.0\%$ |
| | JMMLU (10分野) | Exact Match | ~42%* | **$\Delta \ge -2.0\%$** | $\Delta < -5.0\%$ |
| **Stage 4**<br>(~15分) | RULER NIAH 4K | Retrieval Acc | > 90% | **$\Delta \ge -2.0\%$** | $\Delta < -8.0\%$ |
| | RULER NIAH 8K | Retrieval Acc | > 85% | **$\Delta \ge -3.0\%$** | $\Delta < -10.0\%$ |

*\*0.8B クラスの推定値。未確認の厳密値は初回 BF16 実行時にアンカー確定する。*

---

## 8. 自動実行スクリプト計画 (`run_eval_pipeline.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_TARGET=${1:-"yaqa_int4"}  # bf16 | rtn_int4 | yaqa_int4
OUT="research/results/${MODEL_TARGET}" && mkdir -p "${OUT}"

# Stage 1: PPL & KL (JAX 既存スクリプト)
python3 scripts/eval_ppl_kl.py --model "${MODEL_TARGET}" --out "${OUT}/stage1.json"

# Stage 2: M-IFEval ルール判定 (uzu サーバー経由)
python3 scripts/eval_ifeval.py --endpoint "http://localhost:8000/v1" --out "${OUT}/stage2.json"

# Stage 3: llm-jp-eval コアサブセット
(cd llm-jp-eval && uv run scripts/evaluate_llm.py eval --config configs/config_0.8b.yaml --output_dir "../${OUT}/stage3")

# Stage 4: ABEJA 長文評価 (4K/8K)
(cd longcontext-evaluation && python3 scripts/benchmarks/run.py --config configs/uzu_4k_8k.yaml --output_dir "../${OUT}/stage4")
```

### 留意事項
- **Linear Attention の状態発散 (未確認)**: DeltaNet が長文で発散しないかを Stage 4 で早期確認する。
- **LMSYS 利用規約**: Gated 規約未同意時は、オープンな `lmsys-chat-1m-synth` または WildChat-ja で代替する。
