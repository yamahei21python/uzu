# 9B 本番計画（2026-11-08 決定）

## 決定

| 項目 | 決定 |
|---|---|
| 主戦場 | **9B**（本番） |
| 4B | **軽く**（設定を確認するだけ。seed も振らない） |
| KL 評価 | **やらない**（BPC と実測 t/s で判断） |
| 9B の速度 | **speculator まで含めて取る**（Phase 3B） |

## 前提: 公式 speculator の中身（実ファイルから判明）

`speculator/config.json` = `SpeculatorModelConfig` + `DFlashSpeculatorConfig` + `Weaver`。

| 要素 | 値 |
|---|---|
| draft 本体 | `model_dim 4096` / `hidden 12288` / **8 層** / block_size 16 |
| 入力 | 本体の `target_layer_ids = [1,5,9,13,17,21,25,29]` の隠れ状態 |
| mask token | 248077（未確定トークンをマスクして 16 トークンまとめて予測） |
| Weaver | 1 層 cross-attn、候補木を作る（rounds 8 / expand 4） |
| 木の幅 | `tree_budget` 8（既定）/ 16（M5） |
| サイズ | 755MB BF16 ≈ 377M パラメータ |

**uzu 側は推論のみ実装**（`crates/uzu-engine/src/speculators/dflash_tfm.rs`）。
学習コードは無い。公式の draft はモデル毎に専用学習されている。

## 順序

0.8B の残件 → 4B を 1 条件 → **9B 本体** → speculator。9B は最後。

## 段取り

### Step 1: 公式 draft の流用テスト（最安・最速）

我々の 9B に公式 9B の `speculator/` をそのまま付けて動くか。

- モデル次元は同一（同じ Qwen3.5-9B 由来）なので**ロードは通る見込み**
- 重みが QAD 済みの公式と、RTN の我々では隠れ状態の分布が違う → **受理率が落ちる**
- 測るもの: 動くか / 受理率 / 実効 t/s

**判定**: 受理率 50% 以上なら「流用で実用」。それ未満なら Step 2。

### Step 2: 自前 draft の学習

Step 1 が不十分な場合のみ。

| 項目 | 設計 |
|---|---|
| 教師 | 我々の 9B（量子化後ではなく BF16 で教師を作る） |
| 入力 | 層 1,5,...,29 の隠れ状態（4 層おき） |
| 目的関数 | block 16 の masked token 予測 |
| Weaver | 最初は省略（chain のみ）= tree_budget 1 |
| データ | 校正コーパス + コード/agentic（受理率はドメイン依存） |
| 計算 | Modal A100-80GB。draft 377M パラメータ |
| 実装 | 新規（JAX か PyTorch）。uzu の config 形式に合わせて export |

**難所**: uzu の speculator export 形式（`model.safetensors` + `shapes.json` + `config.json`）に合わせる必要がある。形式は `config/speculator/dflash.rs` で定義されているので、そこから逆算する。

### Step 3: 9B 本体の確定

speculator と独立に、本体は先に固める。

| 項目 | 値 |
|---|---|
| 校正 | Modal A100 bf16（CPU 不可）、s32_q128 相当、damping 1e0 |
| 変換 | bits4 / **group 64** / RHT 32 / wide8 / **emb8** |
| eval | 100 chunks EN/JA |

group 64 は 9B で −267MB（−5.5%）= TPS +5.8%。公式は g32 なので、ここは我々のアドバンテージ。
