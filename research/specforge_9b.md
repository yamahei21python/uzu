# SpecForge → DFlash 9B (Huihui) 再学習手順

対象: Huihui 9B 用 DFlash draft を SpecForge で学習 → lalamo 変換 → uzu。本番変更なし。

前提【実測】: `config.json` = `model_dim=4096`、32 層（DeltaNetConfig 24 + AttentionConfig 8、注意層 `[3,7,...,31]`）。公式 `trymirai/Qwen3.5-9B-M` speculator = 755,228,072 B、3 ファイル（config/shapes/weights）【実測: HF API】。

---

## 1. SpecForge 概要・環境・9B 対応

repo: https://github.com/sgl-project/SpecForge （`main`）【実測】
論文: "Trees from Marginals: Autoregressive drafting with factorized priors" arXiv 2607.06763【引用: https://huggingface.co/papers/2607.06763】

| 依存（`pyproject.toml` pin） | 指定 | 9B への意味 |
|---|---|---|
| python | `>=3.11` | 3.11 で venv |
| torch | `==2.13.0` | `flex_attention` 前提 |
| transformers | `==5.12.1` | Qwen3.5 hybrid config 解釈 |
| sglang | `==0.5.18` | hidden state 抽出（必須） |
| liger-kernel | `>=0.8.3,<0.9`（linux のみ） | macOS 不可 = CUDA 学習 |

```bash
git clone https://github.com/sgl-project/SpecForge.git && cd SpecForge
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[liger]"
```

CLI 3 つ【実測: `specforge/cli.py`】: `train -c <yaml> [--plan] [overrides]`（`section.field=value`）/ `export --to hf|sglang --checkpoint P --draft-config J --output-dir D [--embedding-source T]` / `benchmark`。

### Qwen3.5 / GDN hybrid 対応

| 項目 | 事実 | 根拠 |
|---|---|---|
| 公式 config | `qwen3.5-4b-dflash{,-2}.json`, `qwen3.5-35b-a3b-dflash.json` | 【実測】repo tree |
| 9B config | **無い**。0.8B/4B/35B-A3B のみ | 【実測】repo tree |
| draft 構造 | 標準 GQA 5 層（`full_attention`×5）。GDN 無し → hybrid 改修不要 | 【実測】`configs/qwen3.5-4b-dflash.json` |
| capture 層 | `dflash_config.target_layer_ids` をそのまま使用 | 【実測】`specforge/algorithms/model_providers.py:235` |
| hybrid embedding | `embedding_key: model.language_model.embed_tokens.weight` 必須 | 【実測】`examples/configs/offline/colocated/qwen3.5-4b-dflash-amd.yaml` |
| capture 実装 | SGLang。Radix Cache 回避に `--sglang-disable-radix-cache` | 【引用】`scripts/prepare_hidden_states.py` 周辺 docstring |

9B config は 4B 雛形 + `hidden_size=4096`、`target_layer_ids=[1,5,9,13,17,21,25,29]`、`num_target_layers=32`、`vocab_size=248320`、`mask_token_id=248077`（公式 9B-M と同値）【推定】。

注意: 9B の capture 層 `[1,5,...,29]` は **全て DeltaNet 層**（公式 9B-M も同じ）【実測】。attention 層を混ぜると公式 draft と非互換。

---

## 2. データ形式

### 2.1 rollout（target 自身で再生成）

```bash
python scripts/regenerate_train_data.py --model huihui-ai/Huihui-Qwen3.5-9B-abliterated \
  --concurrency 64 --temperature 0.7 --max-tokens 4096 \
  --input-file-path ./cache/dataset/sharegpt.jsonl \
  --output-file-path ./cache/dataset/regen.jsonl --server-address localhost:30000
```
既定: temperature 0.7 / max-tokens 4096 / concurrency 64【実測】。

### 2.2 hidden state 抽出

```bash
python scripts/prepare_hidden_states.py --target-model-path huihui-ai/Huihui-Qwen3.5-9B-abliterated \
  --strategy dflash --draft-model-config configs/qwen3.5-9b-dflash.json \
  --data-path ./cache/dataset/regen.jsonl --max-length 3072 --chat-template qwen3.5 \
  --tp-size 2 --batch-size 8 --cache-dir ./cache --hidden-state-transfer async --file-group-size 2000
```
既定: max-length 2048 / chat-template llama3 / batch-size 32 / transfer async / group 2000 / compression-level 6【実測: 同ファイル】。

保存キー【実測: `specforge/algorithms/common/hidden_states_data.py`】:

| キー | dtype | shape | 内容 |
|---|---|---|---|
| `input_ids` | int64 | `[1,seq]` | token id |
| `loss_mask` | int32 | `[1,seq]` | 応答トークンのみ 1 |
| `hidden_states` | bf16 | `[1,seq,8*4096]` | capture 8 層を連結 |
| `target_last_hidden_states` | bf16 | `[1,seq,4096]` | 最終層。teacher metrics 有効時のみ |

- 最終層は DFlash loss に不要。`dflash_teacher_metrics: false` で保存自体を止める【実測: `dflash/providers.py` `last_hidden_feature=None`】。
- `vocab_mapping.pt` は **EAGLE3 専用**（DFlash では作られない）【実測: `specforge/training/vocab_mapping.py` docstring】。
- top-K marginals / `selector_top_k` は **DFlash2** の設定。`strategy: dflash` には無い【実測: `dflash/providers.py:94`】。

保存先は `hidden_states_path/rows_<a>-<b>/data_*.ckpt` 分割【実測】。

---

## 3. 9B 抽出メモリ

BF16 重み 18.4GB（9.2B × 2B）【推定】。1 サンプル 3072 token:

- 8 層: `3072×8×4096×2 = 201,326,592 B = 192 MiB`【実測計算】
- 5 層: 120 MiB。1,600 サンプル = 300 GiB / 187.5 GiB、gzip で約 6 割【推定】

| 構成 | 重み | 活性+KV(seq3072,bs8) | 判定 | 制約 |
|---|---|---|---|---|
| T4 ×2 16GB | 9.2GB/GPU (TP2) | 2〜3GB | 条件付可（遅） | sm75 = BF16/flash-attn 非対応 → FP16 と backend 調整【推定】 |
| A100 40GB | 18.4GB | 8〜12GB | 推奨 | BF16 + flex_attention 可【推定】 |
| CPU 32GB | 18.4GB | 20GB+ | 不可 | SGLang 0.5.18 は CUDA 前提【推定】 |

削減: `target_layer_ids` 5 層化（37.5% 減・公式非互換）、`--file-group-size 500`【推定】。
学習: draft 377M params（755MB/2B）【実測計算】。`bs 2 × accum 4` / seq 3072 で A100 40GB 想定。抽出と学習は分離（offline）前提【推定】。

---

## 4. 既存 DFlash から fine-tune（1.6K サンプル）

`model.draft_checkpoint_path` = 重みのみ読込（optimizer 復元なし）。再開は `training.resume_from`【実測: `specforge/config/schema.py:278,1073`】。

```yaml
model:
  target_model_path: "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
  draft_model_config: "configs/qwen3.5-9b-dflash.json"
  draft_checkpoint_path: "z-lab/Qwen3.5-9B-DFlash"
  target_backend: "sglang"
  embedding_key: "model.language_model.embed_tokens.weight"
  torch_dtype: "bfloat16"
  use_liger_kernel: true
data:
  hidden_states_path: "./cache/hidden_states/huihui-9b"
  max_length: 3072
  chat_template: "qwen3.5"
training:
  strategy: "dflash"
  num_epochs: 4
  max_steps: 800          # 1.6K×4ep ÷ (bs2×accum4) ≒ 800
  batch_size: 2
  accumulation_steps: 4
  learning_rate: 1.0e-4   # 公式 6e-4 の 1/6
  warmup_ratio: 0.04
  num_anchors: 512
  loss_decay_gamma: 7
  attention_backend: "flex_attention"
output_dir: "./outputs/huihui-9b-dflash"
```

```bash
specforge train -c configs/fine_tune_9b_huihui.yaml --plan
specforge train -c configs/fine_tune_9b_huihui.yaml training.max_steps=200
```

`z-lab/Qwen3.5-9B-DFlash` は HTTP 200 で存在確認【実測】。層数・`target_layer_ids` の一致は未取得【未確認】。不一致時は公式 9B-M の `speculator/model.safetensors` を warm start に使い、接頭辞 `speculator.draft_model.` を除去【推定】。

1.6K は分布較正用。公式 6 epoch / 800K の再現ではなく lr を下げる設計【推定】。

---

## 5. 出力 → lalamo → uzu

```bash
specforge export --to hf \
  --checkpoint ./outputs/huihui-9b-dflash/checkpoint-800 \
  --draft-config configs/qwen3.5-9b-dflash.json \
  --embedding-source huihui-ai/Huihui-Qwen3.5-9B-abliterated \
  --output-dir ./hf_draft

lalamo speculator convert ./hf_draft \
  --output-dir models/Huihui-9B/speculator --overwrite --dtype bf16
# Weaver 併用: --weaver <repo（*.pth 1 個）>
```

lalamo 実装事実【実測: `lalamo/commands.py`, `lalamo/main.py`】: `convert_speculator(repo_id, output_dir, weaver_repo_id, dtype)` が `snapshot_download(allow_patterns=["config.json","*.safetensors"])` を実行 → ローカル dir も渡せる。`--overwrite` 無しで既存 dir は対話確認（非対話終了）。Weaver は `*.pth` 1 個のみ許容（0/複数で `ValueError`）。

重み対応【実測: `dflash_loader.py`。qkv 融合のみ【推定】】:

| HF (SpecForge export) | lalamo 内部 | uzu suffix |
|---|---|---|
| `fc.weight` | `encoder.fc.weight` | `.weights` |
| `hidden_norm.weight` | `encoder.output_norm_enc.weight` | `.scales` |
| `norm.weight` | `output_norm` | `.scales` |
| `layers.{i}.self_attn.{q,k,v}_proj.weight` | `layers.{i}.mixer.qkvg_projection` | `.weights` |
| `layers.{i}.post_attention_layernorm.weight` | `layers.{i}.pre_mlp_norm` | `.scales` |

必須: `config.json` の `model_type` は **`"qwen3"`**（`"qwen3_5"` 不可、`Literal["qwen3"]`）【実測】。`model_dim` 4096 一致、`num_target_layers=32`、`max(target_layer_ids)<32`。

### 変換後の必須: `shapes.json`

uzu は `speculator/shapes.json` を必ず読む（無いと IO エラー）。デバイス名キー → 無ければ `"default"`。`construction_method` は `Argmax` か `Weaver{...prune_sigma=1.5}`【実測: `crates/uzu-engine/src/speculators/dflash_tfm.rs:73-104`】。

lalamo は `shapes.json` を生成しない【実測: lalamo 全 `.py` に該当なし】。手書き:

```json
{"default": {"tree_budget": 8, "max_tree_depth": 8, "dflash_depth_override": null, "construction_method": {"type": "Argmax"}},
 "Apple M5": {"tree_budget": 16, "max_tree_depth": 16, "dflash_depth_override": null, "construction_method": {"type": "Weaver", "rounds": 16, "expand_per_round": 4, "expand_width": 4}}}
```
Weaver 無しは `Argmax` のみで可。公式 9B-M は全キー Weaver【実測: HF `speculator/shapes.json`】。

---

## 6. acceptance 計測

ローカル実体は `scripts/bench_mtp.sh`（`target/release/cli`、timeout 300s）。speculator を rename して 2 条件比較し、ログから抽出【実測】。

```bash
bash scripts/bench_mtp.sh models/Huihui-9B "富士山について3文で説明して"
```

`tokens per forward pass = num_tokens_accepted / num_forward_passes`【実測: `research/apple_silicon_benchmarking.md:113`】。

| 指標 | 合格 | 境界 | 再学習 |
|---|---|---|---|
| tokens per forward pass | ≥ 3.0 | 2.5〜3.0 | < 2.5 |
| tok/s（有/無） | ≥ 1.5× | 1.2〜1.5× | < 1.2× |
| 日本語・創作の破綻 | なし | 軽微 | あり |

作法【推定】: 20 プロンプト以上（説明/コード/数学/創作/要約）、warmup 1 + 3 回中央値、`--reasoning-effort disabled` 固定。自由対話は低く出るため用途別閾値。

---

## 7. フォールバック

| 順 | 手段 | 手順 | 期待 | 欠点 |
|---|---|---|---|---|
| 0 | 公式 9B-M 流用 | `python scripts/attach_speculator.py --model-dir <uzu dir> --repo trymirai/Qwen3.5-9B-M` | 即時。dim/layer 検証あり【実測】 | abliteration で acceptance 低下しうる |
| 1 | fine-tune（§4） | 1.6K、A100 1〜3h | 分布差吸収 | 抽出 300GiB・A100 必須 |
| 2 | Weaver なし | `--weaver` 省略 + `shapes.json` を `Argmax` | 実装単純、メモリ減 | 候補減で tok/p 低下【推定】 |
| 3 | MTP なし | `speculator/` を退避 | 確実動作、755MB 解放 | 加速なし |

`attach_speculator.py` 検証: `target.model_dim == draft.model_dim`、`max(target_layer_ids) < 層数`、`num_target_layers == 層数`【実測】。vocab_size 検査は**無い**。

---

## 8. 未確認事項

| # | 項目 | 理由 | 確認方法 |
|---|---|---|---|
| 1 | SpecForge 9B 学習事例 | config は 0.8B/4B/35B-A3B のみ【実測】 | `--plan` → 1 step 実行、OOM/NaN 監視 |
| 2 | SGLang 0.5.18 で 9B capture が通るか | 4B 実績のみ | A100 で `--num-samples 4` |
| 3 | `z-lab/Qwen3.5-9B-DFlash` の層構成 | repo 存在のみ確認【実測: HTTP 200】 | `config.json` を取得 |
| 4 | export キー名 == lalamo 期待 | lalamo 側のみ確認、実 export 未実施 | safetensors キー diff |
| 5 | 9B 抽出ピーク VRAM | 公開ログなし | `nvidia-smi` + batch sweep |
| 6 | Huihui + 公式 draft の実測 acceptance | 未計測 | §6 を 20 本で実行 |
| 7 | Weaver 学習コード公開状況 / T4×2 capture | 推論・変換のみ確認。sm75 非対応推定 | trymirai Issue 検索、Colab T4 で 1 サンプル |

参照: https://github.com/sgl-project/SpecForge ／ https://sgl-project.github.io/SpecForge/ ／ https://huggingface.co/trymirai/Qwen3.5-9B-M ／ https://huggingface.co/papers/2607.06763
