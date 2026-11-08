# DFlash draft 用のプロンプト集（2026-11-08）

draft は「本体の隠れ状態 → 次 16 トークン」を写すモデルなので、**応答は我々のモデルで生成する**。
ここで用意するのは**プロンプトだけ**。

## 生成物

```
data/mtp_prompts/train.jsonl   18,132 本
data/mtp_prompts/holdout.jsonl    954 本
data/mtp_prompts/manifest.json
```

平均 305 文字。応答を平均 400 トークンとすると **約 7.6M トークン**。

再生成（`--per-source` で量を調整）:

```bash
python scripts/build_mtp_prompts.py --out data/mtp_prompts --per-source 300
```

## 出所とライセンス（すべて再配布可）

| データセット | 言語 | 区分 | ライセンス | weight |
|---|---|---|---|---:|
| Open-Orca/OpenOrca | EN | instruction | MIT | 3 |
| LDJnr/Capybara | EN | reasoning | Apache-2.0 | 2 |
| openai/gsm8k | EN | reasoning | MIT | 1 |
| openai/openai_humaneval | EN | code | MIT | 1 |
| CausalLM/GPT-4-Self-Instruct-Japanese | JA | instruction | CC-BY-4.0 | 2 |
| llm-jp/oasst1-21k-ja | JA | instruction | Apache-2.0 | 3 |

**ロールプレイは入れない**（2026-11-08 方針）。Puffin と Aratako は外した。

- 性的内容もロールプレイも含めない。実務（指示・推論・コード）に寄せる
- Aratako 系は SFW でも persona なので外した

## 分布（train）

| | JA | EN |
|---|---:|---:|
| instruction | 6,955 | 5,402 |
| reasoning | — | 5,618 |
| code | — | 157 |

JA が 38%、EN が 62%。日本語を増やしたい場合は `llm-jp` の weight を上げる。

取得は HF datasets-server のレート制限（429）に当たるため、ページ間 1.5 秒・
リトライ 5→60 秒のバックオフを入れてある。

## holdout の二重ガード

- 評価用の `research/eval_prompts_{ja,en}.jsonl`（400 本）は**学習に入れない**
- `build_mtp_prompts.py` が SHA256 で突き合わせ、一致したプロンプトを捨てる
- さらに train/holdout を 95:5 で分割（seed 固定）

## 次の手順（未実施）

1. このプロンプトを**我々の 9B** に通して応答を生成（vLLM のバッチで A100）
2. 生成データを teacher-forced 形式に変換（`messages` のまま）
3. `vllm-project/speculators` の DFlash レシピで draft を fine-tune
4. `lalamo speculator convert` → `quantize_speculator.py` → uzu で実測

## 注意

- プロンプトのライセンスはすべて許諾的（MIT / Apache-2.0）だが、**生成した応答**は
  我々のモデル由来。配布する場合はモデルカードに生成データであることを明記する
- `openai/openai_humaneval` は 164 本と少ない（データセット全体が小さい）
