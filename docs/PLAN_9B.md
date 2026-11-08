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


## Step 0（実施済み）: 公開 draft の convert 検証（2026-11-08）

`lalamo speculator convert guglxni/Qwen3.5-9B-abliterated-DFlash` は**通った**。

| 項目 | 結果 |
|---|---|
| 出力 | `config.json` / `model.safetensors`（44 テンソル, 2.18GB bf16） |
| 層数 | 5（公式 speculator は 8） |
| block_size | 16 |
| target_layer_ids | [1, 8, 15, 22, 29] |
| mask_token_id | 248070（公式は 248077） |
| weaver | **なし**（config の `weaver_config` は null） |
| shapes.json | **生成されない → 手で作る必要あり** |

### 欠けていたもの: shapes.json

uzu は `speculator/shapes.json` を必須で読む（`dflash_tfm.rs`）。
lalamo の convert は書き出さないので、以下の最小構成を手で置く。

```json
{"default": {"tree_budget": 16, "max_tree_depth": 16,
             "dflash_depth_override": null,
             "construction_method": {"type": "Argmax"}}}
```

- `Argmax` は**一直線の chain**（木を作らない）。weaver 重みが無い場合はこれ
- `tree_budget` は `dflash_depth`（既定 block_size=16）以下であること
- weaver を使うなら `{"type":"Weaver","rounds":8,"expand_per_round":4,"expand_width":4}`

### 4bit 化ツール（実装済み: `scripts/quantize_speculator.py`）

`lalamo speculator convert` は float しか出せないため、変換後の draft を
4bit に圧縮する専用スクリプトを追加した。

```bash
python scripts/quantize_speculator.py \
  --source /tmp/spec_guglxni --output /tmp/spec_guglxni_int4 \
  --bits 4 --group-size 32
```

- 本体と同じ **HybridSpec（RHT block 32 + 非対称 int4 g32）** で全 Linear を圧縮
- 22 層を 22 秒で処理、2.18GB → **632MB**
- `shapes.json`（Argmax chain / budget 16）を自動生成
- lalamo の入れ子 HybridSpec を uzu のフラット形式へ**自動 rewrite**
  （これを忘れると `AnyWeightMatrixSpec` でロード失敗する）

### 実測（公式 9B-M を target に流用）

| 構成 | メモリ | Generate | 倍率 |
|---|---:|---:|---:|
| 公式 9B-M（投機なし） | 5.00GB | 16.2 t/s | 1.0x |
| 公式 9B-M + guglxni draft（5層 / Argmax / **bf16 2.18GB**） | 7.05GB | 41.2 t/s | 2.5x |
| 公式 9B-M + guglxni draft（5層 / Argmax / **4bit 632MB**） | 5.60GB | **49.0 t/s** | **3.0x** |
| 公式 9B-M + 公式 draft（8層 / Weaver / 4bit 755MB） | 5.74GB | 63.4 t/s | 3.9x |

**draft を 4bit 化するだけで +19%（41.2 → 49.0）とメモリ −1.45GB。**
draft は 1 ブロック 1 回しか走らないが、bf16 の帯域消費が効いていた。

**読み取れること**:

1. 別モデル（lukey03 版 abliterated）で学習した draft でも **2.5x** 出る
   → abliterated 同士なら分布が近い。ゼロから学習する必要はない
2. 公式との差（2.5x vs 3.9x）の要因は 3 つ
   - **Weaver の木**（Argmax は chain のみ）
   - 8 層 vs 5 層
   - 4bit 量子化（755MB）による帯域削減
3. したがって我々の作業は「公開 draft を Huihui 9B で fine-tune」+「Weaver 追加」+「4bit 化」

> 注意: target は公式 9B-M（QAD 済み・4bit）。我々の変換モデルでの数値ではない。
> 我々の 9B は 4-8bit 配分なので hidden state が少しずれる。fine-tune で吸収する。

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




## 公開 draft の棚卸し（2026-11-08 調査）

HF の DFlash 系を全件走査し、Qwen3.5-9B 用を config で比較した。

| repo | 層 | block | target_layer_ids | weaver |
|---|---:|---:|---|---|
| **guglxni/Qwen3.5-9B-abliterated-DFlash** | **5** | 16 | 1,8,15,22,29 | なし |
| unieai/Qwen3.5-9B-DFlash | 5 | 16 | 1,8,15,22,29 | なし |
| blaj/Qwen3.5-9B-abliterated-DFlash-{bf16,int8,int4}-ov | 5 | 16 | 1,8,15,22,29 | なし |
| z-lab/Qwen3.5-9B-DFlash | **6** | 16 | 1,5,9,...,29（8 本） | なし |
| modal-labs/Qwen3.5-9B-DFlash | 6 | 16 | 1,5,9,...,29（8 本） | なし |

**9B 用の Weaver は公開されていない。** `trymirai/weaver` は存在するが中身は
`weaver/qwen36_27b_weaver.pth`（27B 用）だけで、9B には使えない。
公式 9B の 8 層 + Weaver は非公開の自社学習。

### 6 層の方が良いか → **No**

公式 9B-M を target に、両方 4bit 化して実測（M3、同一タスク）。

| draft | 層 | メモリ | Generate |
|---|---:|---:|---:|
| guglxni | 5 | 5.60GB | **49.0 t/s** |
| z-lab | 6 | 5.74GB | 46.2 t/s |

6 層は target 特徴が 8 本に増えるぶん 1 パスが重く、受理率の向上で償却できていない
（±4 t/s の測定ノイズ込みでも 5 層に勝てない）。

### 結論

**公開 draft の中では guglxni（5 層）が最良**。これ以上の改善は
「我々のモデルでの fine-tune」と「Weaver の自前学習」しかない。
Weaver は 9B 用の重みが無いため、ゼロから学習するしかない（`--weaver` に渡す HF repo が要る）。

## draft の投資順序（2026-11-08 判断）

| 段階 | 構成 | 追加コスト | 期待 |
|---|---|---|---|
| **①** | 5層 / Argmax / bf16（公開 draft を fine-tune） | 1〜2 日（Modal） | 2.5〜3.0x |
| **②** | ① + draft を 4bit 化 | 半日（量子化ツール） | 同速 + メモリ −1.4GB |
| ③ | ② + Weaver 学習 | 1〜2 週間（受理信号のパイプライン込み） | +30〜50% |

**Weaver は「測定してから」**。判断材料は**受理長 τ（1 回の検証で採用されるトークン数）**。

- τ が 3 未満 → draft 自体が弱い。Weaver より fine-tune を伸ばす
- τ が 4 以上で頭打ち → 木を広げる価値あり（Weaver）

Weaver の利得は「top-1 の chain が外れたときに別候補が当たる」こと。
draft が十分賢いと chain がほぼ通るので、利得は縮む。

### Step 2.4: 学習をどこで回すか（T4 プローブで判定）

| 対象 | Kaggle T4×2 | 判断 |
|---|---|---|
| 4B 変換 | ❌ bf16 データセットが無い / f32 で 16GB ぴったり | Modal |
| 9B 変換 | ❌ 18GB | Modal |
| **draft 学習** | 測手中 | プローブ次第 |
| 0.8B の追加 seed | ✅ | Kaggle GPU |

draft 学習の可否を決めるのは 3 つ。

1. **flash-attn が sm75（T4）で動くか** — DFlash 学習の必須依存。sm80 以上要求なら不可
2. fp16 / bf16 matmul の実効速度（T4 は bf16 tensor core 無し）
3. `vllm-project/speculators` が T4 環境で install できるか

結果が「不可」なら Modal A100。「可」でも週 30h を分割する前提で、
fine-tune は 2〜3 セッションに分ける（checkpoint は Kaggle 出力に置く）。

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
