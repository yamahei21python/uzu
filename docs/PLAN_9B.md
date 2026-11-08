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

**自前学習は必須。** 理由: draft は「本体の隠れ状態 → 次の 16 トークン」を写すモデルなので、
教師（本体）が変われば draft も作り直し。我々の 9B は無検閲の蒸留モデルで、
公式 draft が学習した（検閲済み・QAD 済み）隠れ状態とは分布が違う。
公式 draft の流用は**診断として数分試すだけ**で、採用前提にしない。

### Step 1: 9B 本体を確定（Modal A100）

| 項目 | 値 |
|---|---|
| 校正 | bf16 / s32_q128 相当 / damping 1e0 |
| 変換 | bits4 / **group 64** / RHT 32 / wide8 / **emb8** |
| eval | 100 chunks EN/JA |

group 64 は −267MB（−5.5%）= TPS +5.8%。公式は g32 なのでここが我々の取り分。

### Step 2: draft 学習（Modal A100）

既存の実装を使う。**一から書かない。**

| 候補 | 内容 |
|---|---|
| [z-lab/dflash](https://github.com/z-lab/dflash) | 原論文の公式実装。block diffusion draft の学習 |
| [vllm-project/speculators](https://github.com/vllm-project/speculators) | DFlash / DFlash2 / DSpark の学習レシピ。HF 形式で export |
| [NVIDIA/ModelOpt](https://github.com/NVIDIA/Model-Optimizer) | Qwen3.5 の DFlash 学習手順と注意点（頭の次元・M-RoPE） |

**重要**: 公開済みの `z-lab/Qwen3.5-9B-DFlash` が存在する（ModelOpt の例に記載）。
これを**初期値**にして我々の 9B で fine-tune すれば、ゼロ学習より桁で安い。
まず存在と形式を確認する。

#### 学習の設計（公式 config から合わせる）

| 項目 | 値 | 根拠 |
|---|---|---|
| block_size | **16**（8 も比較） | 公式 9B draft が 16 |
| target_layer_ids | **1,5,9,13,17,21,25,29** | 公式 config と一致させる |
| mask_token_id | 248077 | 公式 config |
| draft 層数 | 8（公式）/ 5（論文推奨） | 受け入れ率と速度のトレードオフ |
| model_dim / hidden | 4096 / 12288 | 公式 config |
| embed / lm_head | 本体と共有・凍結 | DFlash の仕様 |
| 学習対象 | draft stack + context_projection + norms | 同上 |
| データ | 我々の 9B で生成した応答（teacher-forced） | 学習と推論の分布を揃える |
| 目的関数 | block CE（位置ごとに指数減衰） | 受理率に直結 |


### Step 2.5: Kaggle T4 で 100 step 実測（9B 本体ビルド後）

draft 学習を Kaggle GPU で回せるかを**数字で決める**ための短時間テスト。
9B 本体ができてから実施する（教師は配布する int4 モデル）。

#### 構成

| 要素 | 値 |
|---|---|
| 教師 | **我々の 9B int4**（凍結、5GB） |
| draft | 377M / fp16 / 学習対象 |
| optimizer | AdamW（fp32 state） |
| seq / block | 2048 / 16 |
| バッチ | 8（T4 16GB ×2、TP=2 or FSDP） |
| データ | 校正コーパス + 生成応答 |

#### 測るもの

1. **int4 教師の forward が T4 で回るか**（bf16 matmul 不可のため f32 dequant になる）
2. **1 step の秒数**（forward + backward + optimizer）
3. 100 step の loss 低下（学習が成立しているか）
4. ピークメモリ
5. 参考: BF16 教師との loss 低下の差

#### 判定

```
総 step 数 = 目標トークン数 / (batch × seq)
総時間     = 総 step 数 × 1 step 秒
```

| 条件 | 判断 |
|---|---|
| 30h 以内（Kaggle 週次枠） | **Kaggle で完走** |
| 30h 超 | Modal A100（概算 1/8 の時間） |
| int4 forward が動かない/遅すぎる | BF16 教師に変更（= Modal 必須） |

**注意**: 週次 30h を学習で使い切ると評価レーンが止まる。学習は週の前半に寄せる。
### Step 3: export → uzu

lalamo に変換コマンドがある。

```bash
lalamo speculator convert <dflash_repo_or_path> [--weaver <weaver_repo>] \
  --output-dir <9Bモデル>/speculator
```

- uzu の形式（`model.safetensors` + `config.json` + `shapes.json`）は lalamo が書き出す
- Weaver（候補木を作る 1 層）は最初は省略可。公式は tree_budget 8（M5 は 16）
- 公式 speculator の中身: 205 テンソル / 755MB / 4bit 量子化済み（scales BF16）

### Step 4: 実測

M3 で `cli bench`。現状 9B は 15.3 t/s（投機なし）。目標 40 t/s 以上。
