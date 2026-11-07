# 無検閲＋Claude蒸留モデルをuzuで使う手順

更新日: 2026-11-06
対象: `Huihui-Qwen3.5-*-Claude-4.6-Opus-abliterated` → `trymirai/uzu`

---

## 結論

**現時点で通ったのは `MLX affine 4bit → lalamo e401e177 → uzu` の流れ。**

4Bで実際に起動確認済み。

```text
mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit
  → lalamo e401e177
  → 約2.39GBのuzuモデル
  → uzu CLIで会話OK
```

計測例：

```text
generation speed: 31.77 t/s
memory used:       2.33 GB
```

**9Bは mlx-community の `4bit` モデルが実体はBF16だったため、BF16ソースをModal CPUでMLX affine 4bitへ量子化してからuzu化する流れを追加した。**

```text
huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated (BF16)
  → mlx_lm.convert (Modal CPU, affine 4bit)
  → lalamo e401e177
  → uzu
```

---

## 0. 対象モデル

### 4B — 検証済み

```text
mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit
```

- 約3GB
- MLX affine 4bit
- ハイブリッドQwen3.5
- visionあり
- MTPあり
- 無検閲
- Claude 4.6 Opus蒸留

### 9B — 実測で不可

```text
mlx-community/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-4bit
```

リポジトリ名は `4bit` だが、実体はMLXのBF16フル精度だった。

検証結果：

| 項目 | 値 |
|---|---|
| モデルファイル数 | 4 shards |
| ファイル合計 | 約17.82GB |
| safetensors dtype | BF16中心 |
| `quantization_config` | なし |
| lalamo変換後 | **17.93GB** |
| M3 16GBでの実行 | 不可 |

したがって、このモデルはuzu化パイプラインを通過してもローカルでは動かせない。

Modal Volume上の出力：

```text
Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu
```

これは「変換成功」ではなく、「巨大なBF16モデルを生成した」だけ。

### 9Bの純正MLX 8bit

```text
mlx-community/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-8bit
```

- 品質は良い可能性
- uzu Metalカーネル非対応の可能性あり
- 先に4Bを本命として使う

---

## 1. まず結論できる判断基準

### 1.1 これは「公式モデル」ではなく「独自変換」

uzu公式モデル `qwen3.5:4b` / `qwen3.5:9b` は、既にuzu向け最適化済み。

一方、Huihuiモデルは：

- 無検閲
- Claude 4.6 Opus蒸留
- オリジナルQwen3.5のfinetune
- uzu用最適化・incoherence signsなどは別工程

したがって、次のどちらかを選ぶ。

| 目的 | 推奨 |
|---|---|
| 品質と安定性 | 公式 `qwen3.5:9b` |
| 無検閲＋Claude蒸留 | Huihui MLX 4bitをuzuへ変換 |
| 27B級 | 公式または別ハードウェア |

---

## 2. 現在の環境

| 項目 | 内容 |
|---|---|
| Mac | MacBook Air M3 16GB |
| OS | 現行 |
| uzu | `/Users/kohei/Myproject/uzu` / `efa1d7d` / 2026-11-06 |
| lalamo | `/tmp/lalamo` |
| uzu用lalamo | `e401e177` / 2026-11-02 |
| Modal Volume | `uzu-models` |
| uzu CLI | `./target/release/cli` |
| 公式モデル保存先 | `~/.cache/mirai` |
| 独自モデル保存先 | `models/` |

`uzu` を最新に更新してあることが前提。

```bash
cd /Users/kohei/Myproject/uzu
git fetch origin
git log -1 --format='%h %ad %s' --date=short
```

---

## 3. なぜ直接は動かないか

### 3.1 uzuが読むのはHFのconfigではない

HFモデルの `config.json` には、次のフィールドが混在する。

```text
model_type: qwen3_5
architectures: Qwen3_5ForConditionalGeneration
text_config
vision_config
```

uzuが必要とするのは：

```text
{
  "type": "LanguageModelConfig",
  "decoder_config": ...,
  "token_codec_config": ...,
  "generation_config": ...
}
```

そのままでは動かない。

### 3.2 量子化方式が合わないとMetalが失敗する

実際に遭遇したエラー：

```text
Linear error: Matmul error: Backend error: Cannot create function:
_D4GemmS6VbfloatS6VbfloatS6VbfloatS26VTile32x32x32_Simdgroups2x2...
```

これはconfigの問題ではなく、**Metalバックエンドの対応カーネルがない**ことによる実行エラー。

### 3.3 GPTQは今回使えない

`Huihui-Qwen3.5-4B` をGPTQ化したところ、uzuへ変換しようとして失敗した。

```text
ValueError: Expected in_proj, in_proj_qkvz/in_proj_ba,
or in_proj_qkv/in_proj_z/in_proj_b/in_proj_a weights for DeltaNet.
```

lalamoはこのモデル構造に対して、GPTQの `qweight`/`g_idx`/`scales` を直接展開できない。

### 3.4 AutoAWQも使えない

```text
TypeError: qwen3_5 isn't supported yet.
```

### 3.5 llm-compressorはAWQを作れるが形式が違う

`vllm-project/llm-compressor` は、Qwen3.5を処理できる。

ただし出力は：

```json
"quant_method": "compressed-tensors"
```

`quant_method: awq` は生成されない。

uzmuが読む独自形式ではないため、そのままでは不可。

### 3.6 MLX affine 4bitのみ実用

今回の推奨ルート：

```text
MLX affine 4bit
→ lalamo e401e177
→ uzu
```

過去の失敗とは別物。

---

## 4. 全体フロー

```text
HF上のMLX 4bitモデル
        │
        ▼
lalamo e401e177でuzu形式へ変換
        │
        ▼
config.json修正
        │
        ▼
encoding.json追加
        │
        ▼
ローカルへ同期
        │
        ▼
uzu CLIで会話テスト
```

実際にこの流れで動作した。

### 4.1 やらないこと

- GPTQをuzuへ直接入れない
- AutoAWQで変換しようとしない
- `compressed-tensors` をそのまま入れない
- MLX 8bitをいきなり試さない
- Uzu更新前に変換しない
- lalamo最新版へ無確認で進めない

---

## 5. 互換性の要点

### 5.1 lalamoは `e401e177` 固定

現在のuzuは：

```text
efa1d7d（2026-11-06以降）
residual_config: なし
```

新しいlalamoはコンフィグへ `residual_config` を出力する。

uzuは未知フィールドを受け付けない。

そのため、必ず：

```text
e401e177
```

を使う。

### 5.2 量子化形式の対応表

| 形式 | 結果 |
|---|---|
| MLX affine 4bit | ✅ 実際に動作確認済み |
| MLX affine 8bit | 未検証／Metal未対応の疑い |
| GPTQ W4A16 | ❌ lalamoで変換失敗 |
| AutoAWQ W4A16 | ❌ `qwen3_5 isn't supported yet` |
| llm-compressor | △ `compressed-tensors`のみ |
| `compressed-tensors` | ❌ uzu未対応 |
| `auto-round` | ❌ 未対応 |
| `mxfp4` | ❌ 未対応 |
| BF16／FP16 | 可能性あり、要検証 |

### 5.3 モデル構造

Qwen3.5はハイブリッド構成。

```text
linear_attention 24層
full_attention 8層
vision encoder（マルチモーダル）
MTP（Multi-Token Prediction）
```

`delta_net_attention`、`q/k/v/o projection`、`MLP` などの再配置が必要。

BF16でも動くかは未検証で、メモリを多く消費する。

---

## 6. 9B化の前に、9Bの選択

### 6.1 9Bは「4bit」表記を信じない

`mlx-community/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-4bit` は、実際にはBF16フル精度だった。

変換前に必ず確認する：

```bash
curl -sL "https://huggingface.co/<REPO>/resolve/main/config.json" | python3 -m json.tool
```

チェック項目：

```text
quantization_config        → 必須
mode: affine               → 必須
bits                       → 4または8
group_size                 → 確認
ファイル合計サイズ          → 4bitなら約5GB級
```

17〜20GBある場合は、量子化されていない。

### 6.2 無検閲＋Claude蒸留の9Bでローカル実行できる既製モデルは見つかっていない

現在の選択肢：

| 候補 | 判定 |
|---|---|
| mlx-community Huihui 9B `4bit` | ❌ 実体はBF16、17.93GB |
| huihui-ai純正 BF16 | ❌ 約19GB（そのままでは不可） |
| BF16 → MLX affine 4bit | ✅ Modal CPUで量子化可能 |
| GGUF版 | ❌ lalamo非対応 |
| NVFP4 | ❌ compressed-tensors系 |
| GPTQ/AWQ | ❌ DeltaNetまたはuzu非対応 |
| llm-compressor AWQ | △ 生成可能だがcompressed-tensors |
| 公式 `qwen3.5:9b` | ✅ 動作済み、ただし無検閲ではない |

したがって、無検閲＋Claude蒸留の9Bを使うには、BF16を自分でMLX affine 4bitへ量子化する必要がある。

### 6.3 9Bを本気で使うなら：Modal CPUでMLX affine 4bitへ量子化

`mlx-community/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-4bit` は偽4bitだった。

代わりに、**元のBF16リポから直接MLX affine 4bitを作る**。

```text
huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated (BF16 ~19GB)
        │
        ▼
Modal CPU + mlx[cpu] + mlx-lm
        │
        ▼
mlx_lm.convert --q-bits 4 --q-group-size 64 --dtype float16
        │
        ▼
Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-mlx-affine-4bit-g64 (~5GB)
        │
        ▼
lalamo e401e177でuzu化
        │
        ▼
Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu
```

MLX affine 4bitはGPTQのようなキャリブレーション不要。重みを丸めるだけなので、Modal CPUで十分。

#### 6.3.1 量子化コマンド

```bash
python3 -m mlx_lm.convert \
  --hf-path huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated \
  --mlx-path Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-mlx-affine-4bit-g64 \
  -q \
  --q-bits 4 \
  --q-group-size 64 \
  --dtype float16
```

#### 6.3.2 Modal CPUスペック

```text
cpu: 16
memory: 65536  # 64GB
```

9B BF16は約18〜19GB。量子化中に倍程度必要なので、32GBではギリギリ。64GB推奨。

#### 6.3.3 量子化後のuzu化

量子化済みMLXモデルに対して、lalamo e401e177を実行する。

```python
LanguageModelSpec(
    vendor="Huihui",
    family="Qwen3.5",
    name="Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated",
    size="9B",
    origin=HuggingFaceOrigin(repo="/models/...-mlx-affine-4bit-g64"),
    ...
)
```

#### 6.3.4 注意：mlx_lm.convert の出力に `generation_config.json` が含まれない

実際に確認した `mlx_lm.convert` 出力ファイル：

```text
README.md
chat_template.jinja
config.json
model.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
```

**含まれない**: `generation_config.json`

lalamoは変換時に `generation_config.json` を読むため、別途補う必要がある。

#### 補完方法

元モデル `huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated` に `generation_config.json` があればそこから取得。無い場合は同ファミリ公式 `Qwen/Qwen3.5-27B` から取得する。

```python
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

mlx_dir = "/models/Huihui-...-mlx-affine-4bit-g64"
dest = Path(mlx_dir) / "generation_config.json"

# 優先: 元のBF16リポ
source_repo = "huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated"
try:
    downloaded = hf_hub_download(
        repo_id=source_repo,
        filename="generation_config.json",
        local_dir_use_symlinks=False,
    )
    shutil.copy(downloaded, dest)
except Exception:
    # フォールバック: 公式Qwen3.5-27B
    downloaded = hf_hub_download(
        repo_id="Qwen/Qwen3.5-27B",
        filename="generation_config.json",
        local_dir_use_symlinks=False,
    )
    shutil.copy(downloaded, dest)
```

#### 他に不足する可能性があるファイル

| ファイル | 状態 | 対処 |
|---|---|---|
| `config.json` | ✅ mlx_lm.convertで出力 | 不要 |
| `tokenizer.json` | ✅ mlx_lm.convertで出力 | 不要 |
| `tokenizer_config.json` | ✅ mlx_lm.convertで出力 | 不要 |
| `chat_template.jinja` | ✅ mlx_lm.convertで出力 | 不要 |
| `generation_config.json` | ❌ 不足 | 上記で補完 |
| `preprocessor_config.json` | ⚠️ 無視可 | lalamoは通常不要 |

lalamo実行前に、 MLX ディレクトリ内に `generation_config.json` があることを確認すること。

#### 6.3.5 注意：MLX affine 4bitの精度

MLX affine 4bitは単純な重み丸め。GPTQ/AWQのようにキャリブレーションデータを通さない。

- 速度：速い
- 品質：Claude蒸留の恩恵は大部分保持される見込み
- 9Bなので、4Bより明らかに高品質の可能性

品質比較は必ず同じプロンプトで実施する。

#### 6.3.6 実行ファイル

統合スクリプト：

```text
/tmp/convert_9b_bf16_to_uzu_modal.py
```

これで量子化とuzu化を一連で行い、中間ファイルも削除する。

### 6.4 8bitを選ぶ場合のリスク

- Metalカーネルが足りない可能性
- 9Bモデルのメモリプレッシャーが大きい
- そもそも公式モデルのように最適化されていない

4B 8bitも、使う前にMetal対応を検証する必要がある。

### 6.5 BF16を選ぶ場合

```text
huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated
```

- 無検閲
- Claude蒸留を最大限保持
- ただし約18〜20GB
- Mac 16GBでは実行困難

本機では無理。

---

## 7. Modal変換フロー

### 7.1 方針

量子化はModal、変換もModal。

```text
HF MLX 4bit
   ↓ Hugging Faceから直接取得
lalamo e401e177
   ↓
/models/<OUT_NAME>
   ↓ Modal Volume
MacへDL
```

CPU Functionで実行する。GPUは不要。

### 7.2 変換スクリプトの要点

重要なのは、lalamoを `e401e177` 固定にすること。

```python
LALAMO_COMMIT = "e401e177"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "git clone https://github.com/trymirai/lalamo.git /lalamo",
        f"cd /lalamo && git checkout {LALAMO_COMMIT}",
        'cd /lalamo && uv pip install --system -e ".[cpu]"',
    )
)
```

モデルID：

```python
REPO = "mlx-community/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-4bit"
OUT_NAME = "Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu"
```

9Bの場合、4B用スクリプトのモデルID・OUT_NAMEだけを差し替える。

### 7.3 モデル仕様

変換に必要な設定：

```python
LanguageModelSpec(
    vendor="Huihui",
    family="Qwen3.5",
    name="Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated",
    size="9B",
    origin=HuggingFaceOrigin(repo=REPO),
    config_type=HFQwen35Config,
    response_format=QWEN_RESPONSE_FORMAT,
    reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    configs=ConfigMap(
        tokenizer=FileSpec("tokenizer.json"),
        tokenizer_config=FileSpec("tokenizer_config.json"),
        chat_template=FileSpec("chat_template.jinja"),
        generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
    ),
)
```

注意：

- `chat_template.jinja` を明示する
- `generation_config.json` は元モデルに無いことがある
- 同ファミリの公式リポから補う

### 7.4 9B BF16から直接uzu化する統合スクリプト

```text
/tmp/convert_9b_bf16_to_uzu_modal.py
```

処理：

```text
huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated (BF16)
→ mlx_lm.convert (affine 4bit, group_size=64, float16)
→ lalamo e401e177 (uzu形式)
→ Modal Volume /models/Huihui-...-MLX-4bit-e401e177-uzu
```

実行：

```bash
modal run /tmp/convert_9b_bf16_to_uzu_modal.py
```

環境変数で上書き可能：

```bash
export SOURCE_REPO=huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated
export MLX_OUT_NAME=Huihui-9B-mlx-affine-4bit-g64
export UZU_OUT_NAME=Huihui-9B-uzu-4bit
export MODAL_VOLUME=uzu-models
export LALAMO_COMMIT=e401e177

modal run /tmp/convert_9b_bf16_to_uzu_modal.py
```

---

## 8. 変換後のconfig修正

HFの形を、そのままuzuへ入れない。

uzmuが読む形へ変更する。

### 8.1 uzuが要求する形

```json
{
  "type": "LanguageModelConfig",
  "decoder_config": {...},
  "token_codec_config": {...},
  "generation_config": {...}
}
```

変換器が出力したconfigに、次の未知キーが残ることがある。

```text
response_format
extra_fields
output_parser_regex
```

### 8.2 実施済みの修正

```python
# token_codec_config
- response_format を削除
+ output_parser_regex    = null
+ end_of_thinking_tag    = null
+ bos_token              = null   # 不足時のみ
```

### 8.3 9Bでも同じ修正をする

4Bと同じでOK。

```python
uzu_config.json["token_codec_config"].pop("response_format", None)
uzu_config.json["token_codec_config"].setdefault("output_parser_regex", None)
uzu_config.json["token_codec_config"].setdefault("end_of_thinking_tag", None)
```

zu側のベンチ用関数と違い、uzu本体は触らない。

---

## 9. encoding.json の設置

変換しただけでは、uzuはどのチャット形式で動かすか分からない。

同じディレクトリへ：

```text
Huihui-...-MLX-4bit-e401e177-uzu/
├── config.json
├── model.safetensors
├── tokenizer.json
└── encoding.json
```

`encoding.json` の中身：

```json
{
  "type": "hanashi",
  "name": "qwen3.5"
}
```

または配列：

```json
[
  {
    "type": "hanashi",
    "name": "qwen3.5"
  }
]
```

`qwen3.5` は、HuihuiのQwen3.5系に対応するHanashi設定。

---

## 10. ローカルへ取得

変換が終わったら、Modal Volumeから取得。

```bash
modal volume get --force uzu-models \
  Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  /Users/kohei/Myproject/uzu/models/
```

保存先：

```text
/Users/kohei/Myproject/uzu/models/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu/
```

サイズ：

```text
約2.4GB（4B MLX 4bit）
```

---

## 11. uzuで起動する

### 11.1 基本コマンド

```bash
cd /Users/kohei/Myproject/uzu

MODEL="/Users/kohei/Myproject/uzu/models/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu"

./target/release/cli \
  --model "$MODEL" \
  --message "富士山について3文で説明して"
```

### 11.2 実測値

```text
generation speed: 31.77 t/s
memory used: 2.33 GB
duration: 15.46 s
```

正常に会話できた。

### 11.3 9Bで同じことをする場合

パスだけ差し替え。

```bash
MODEL="/Users/kohei/Myproject/uzu/models/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu"
```

---

## 12. ベンチマーク

### 12.1 比較対象

同じプロンプト「富士山について3文で説明して」で比較。

| モデル | prefill | 生成速度 | メモリ | TTFB | 備考 |
|---|---|---|---|---|---|
| 公式 `qwen3.5:0.8b` | — | 145.68 t/s | 0.50 GB | — | 軽量 |
| 公式 `qwen3.5:9b` | 53.21 t/s | 43.69 t/s | 5.69 GB | 0.34 s | 標準（別日測定） |
| Huihui 4B MLX 4bit | — | 31.77 t/s | 2.33 GB | — | 無検閲＋Claude蒸留 |
| Huihui 9B MLX 4bit | 76.41 t/s | 17.09 t/s | 4.82 GB | 0.24 s | 無検閲＋Claude蒸留（別日測定） |

### 12.2 実際の出力例

2026-11-06に同一プロンプトで測定したが、実行条件は完全には揃っていない。以下は参考値。

| モデル | prefill | generation | memory | TTFB | 測定条件 |
|---|---|---|---|---|---|
| 公式 `qwen3.5:9b` | 61.25 t/s | 35.00 t/s | 5.75 GB | 0.29 s | uzu CLI, `--reasoning-effort medium`, 同一プロンプト |
| Huihui 9B → uzu | 27.41 t/s | 9.87 t/s | 4.82 GB | 0.66 s | uzu CLI, `--reasoning-effort medium`, 上記と同一プロンプト |
| Huihui 9B → MLX 4bit native | 1.382 t/s | 2.221 t/s | 5.214 GB | — | mlx_lm 0.32.2, `--max-tokens 200`, thinkingトークンを含む固定長生成 |

注意事項:

- Huihui 9B uzuのgeneration speedは変動大。前回（別日・別条件）は17.09 t/s、今回は9.87 t/s。値はreasoningプロセスの長さに大きく依存する。
- MLX 4bit nativeは `--max-tokens 200` で200トークンを強制生成。うち大半がthinkingプロセスのため、表示応答トークンの実効速度はさらに低い。
- 公式9Bは `--reasoning-effort disabled` にすると16.08 t/sに低下（通常とは逆の挙動を確認）。

上記は暫定比較であり、正確な速度判定には **再計測が必須** 。

再計測条件案（必須を ★、推奨を ○ とする）:

| # | 条件 | 内容 | 優先度 |
|---|---|---|---|
| 1 | reasoning設定統一 | すべて同じ `--reasoning-effort`、または全モデルで `disabled` | ★ |
| 2 | MLX nativeの生成長調整 | `--max-tokens 50`程度に短縮し、表示応答トークン数を他モデルと揃える | ★ |
| 3 | 繰り返し測定 | 各モデル・各条件で3回以上計測し、分散を確認 | ★ |
| 4 | 指標の分離記録 | prefill / generation / memory / TTFB / duration / 表示応答トークン数 / 総生成トークン数（thinking含む） | ★ |
| 5 | プロンプト・デコード設定固定 | 同一プロンプト、同一temperature等 | ○ |

現状の値は参考値として扱う。速度比較の根拠にはしないこと。

### 12.3 比較考察（前回測定値ベース）

#### 公式 qwen3.5:9b

```text
富士山は日本国内で最も高い山であり、標高 3776 メートルです。
日本を象徴する美しい山として知られ、ユネスコの世界遺産にも登録されています。
過去に噴火した火山ですが、現在は活動していない休火山です。
```

#### Huihui 9B MLX 4bit

```text
富士山は日本に位置する標高 3776 メートルの山です。
日本国内で最も高く、美しい円錐形をしています。
日本を象徴する名山として世界に有名です。
```

両方とも正確な3文。

### 12.4 9B 3モデル暫定比較（条件不揃い、再計測必須）

**速度**

- 公式9B: 43.69 t/s（短い応答では 60.45 t/s）
- Huihui 9B MLX 4bit: 17.09 t/s（短い応答でも 17.04 t/s）

Huihui 9Bは公式9Bの約28〜39%の生成速度。これは主に量子化方式と重みレイアウトの違い。

**推論プロセスの影響**

`--reasoning-effort disabled` で推論をOFFにしても、Huihui 9Bのgeneration speedは17.00 t/sのまま。したがって、推論トークンが表示上のTPSを下げているわけではない。低速はモデル形式そのものの特性。

**durationの違いについて**

- 「富士山」プロンプト: 96.20 s（reasoning ONで長い思考プロセスを含む）
- 「こんにちは」プロンプト: 4.84 s（reasoning OFF、短い応答）

durationは総実行時間であり、推論トークン数に大きく依存する。generation speedは「表示応答トークン / 生成時間」では測定されている可能性があるため、推論ON時は実際の生成トークン数が多くdurationが伸びる。今後は以下を分けて記録する：

- モデルロード時間
- prefill時間
- generation speed
- 表示応答トークン数
- 総生成トークン数（thinking含む）
- duration

**メモリ**

- 公式9B: 5.69 GB
- Huihui 9B MLX 4bit: 4.82 GB

Huihui 9Bは約0.87 GB少ない。

**重み形式の違い（テンソル個数ではなく総バイト数で比較）**

`model.safetensors` のdtype別総バイト数：

| モデル | U8 | I32 | F32 | BF16 | 合計 |
|---|---|---|---|---|---|
| 公式9B | 4.616 GB | 0.009 GB | 0.004 GB | 0.560 GB | 5.189 GB |
| Huihui 9B | 4.476 GB | 0 GB | 0.004 GB | 0.560 GB | 5.040 GB |

重要な発見：

- BF16の総バイト数は両方で**同じ 0.560 GB**
- U8量子化の総バイト数もほぼ同じ（4.616 vs 4.476 GB）
- 違いは **テンソルの分割・スケール保持の形式**
  - 公式9B: U8 260個 + I32 258個（スケール・バイアスを別テンソルで保持）
  - Huihui 9B: U8 130個 + I32 0個（スケール情報を埋め込み？または非対称量子化なし？）
- Linear層のBF16残存数：公式130個、Huihui 260個

したがって、「HuihuiはBF16が多い」というより、**量子化されたLinear層の数が少なく、テンソルレイアウトがuzuの高速カーネルに適していない**可能性が高い。

これはH1（量子化形式がuzu高速カーネルに適合しない）とH3（レイアウトがGEMVに適していない）の両方を支持する。

**MTPの有無**

両方の `config.json` で `ple_model_config` はNone。MTPは有効化されていない。速度差の主因はMTPではない。

**品質**

両方とも「3文で説明する」制約を守った。

- 公式9B: 事実正確、簡潔
- Huihui 9B: 事実正確、Claude蒸留の影響で丁寧な内省（Thinking Process）が見られる

**総合**

| 観点 | 勝者 |
|---|---|
| 速度 | 公式9B |
| メモリ効率 | Huihui 9B |
| 無検閲 | Huihui 9B |
| Claude蒸留推論 | Huihui 9B |
| 安定性 | 公式9B |

用途で選ぶ：

- 速度・安定性優先 → 公式9B
- 無検閲＋Claude蒸留品質優先 → Huihui 9B（速度は許容範囲）

### 12.5 推論の質

Claude蒸留は、推論能力を大幅に高めている可能性が高い。

同じプロンプトで比較すること。

---

## 13. 失敗ポイントと対処

### 13.1 `_D4GemmS...FullPrecision`

原因：

- MLX 8bit／Metal未対応
- 量子化形式がMetal幅に対応していない

対処：

- MLX 4bitへ変更
- それでもダメなら、configだけでなく重みの**layout違い**を疑う

### 13.2 `ValueError: No chat template specified.`

原因：

- tokenizer設定にchat templateが無い
- GPTQの保存でchat_template.jinjaが破損

対処：

```python
chat_template=FileSpec("chat_template.jinja")
```

を `ConfigMap` に追加し、元リポから取得する。

### 13.3 `Expected in_proj_qkv ...`

GPTQ出力はDeltaNetに読めない。

原因：

- GPTQはqweight/g_idxであり、Lalamoの期待する配置と違う
- DeltaNetはMLX affine形式を期待する

対処：

MLX affine 4bitを使う。

### 13.4 `data did not match any variant...`

対処：

```text
token_codec_config.response_format を削除
```

### 13.5 `can not get encoding config`

対処：

`encoding.json` を追加。

```json
{
  "type": "hanashi",
  "name": "qwen3.5"
}
```

---

## 14. Modalで実行した場合の費用

MLX変換はCPUのみ。

CPU Functionの目安：

| 条件 | 目安 |
|---|---|
| Modal 16 vCPU | 変換10〜30分前後 |
| GPTQ 4B（GPU） | 5〜20分 |
| GPTQ 9B（GPU） | 30〜90分 |

CPU変換の方が安上がり。

Huihui 9B MLX 4bitも、同様にCPUでいける可能性が高い。

---

## 15. チェックリスト（9B版）

### 15.1 準備

- [ ] `uzu` を最新へfetchする
- [ ] `lalamo` を `e401e177` に固定する
- [ ] 公式Qwen3.5-9Bのベンチと同じ測定方法を決める
- [ ] `models/` に空きが10GB以上あるか確認

### 15.2 変換

- [ ] 対象モデルが存在することを確認する
- [ ] MLX 4bitを選択する
- [ ] HFリポIDを確認する
- [ ] Modalでlalamo変換を実行する
- [ ] 出力が3ファイルのみであることを確認する

### 15.3 config修正

- [ ] `token_codec_config.response_format` を削除
- [ ] `output_parser_regex` を追加／`null`
- [ ] `end_of_thinking_tag` を追加／`null`

### 15.4 encoding追加

- [ ] `encoding.json` を追加する
- [ ] `hanashi/qwen3.5` を指定する

### 15.5 起動確認

- [ ] `./target/release/cli --model ... --message ...` を実行する
- [ ] 出力速度とメモリを記録する
- [ ] 公式9Bとの比較を残す

---

## 16. 付録：失敗実績

### GPTQ路線

```text
Huihui Qwen3.5 4B
→ GPTQ 4bit化に成功
→ lalamo e401e177で変換
→ ValueError: Expected in_proj_qkv...
→ 変換不能
```

### AutoAWQ路線

```text
TypeError: qwen3_5 isn't supported yet.
→ AutoAWQ利用不可
```

### llm-compressor路線

```text
Qwen3.5に対応
→ ただし compressed-tensors で出力
→ uzu互換ではない
→ 追加の専用変換器が必要
```

### 結論

- GPTQはlalamoでUno
- AWQはAutoAWQで不可
- llm-compressorはuzuそのまま不可
- **MLX affine 4bitが正解ルート**

---

## 17. 速度低下の原因分析と検証計画

### 17.1 現時点での仮説

実験B（4B）の結果により、仮説を更新。

| 仮説 | 内容 | 支持度 | 根拠 |
|---|---|---|---|
| H1 | 量子化形式がuzu高速カーネルに適合しない | 中 | uzuの方がMLXネイティブより速いため、uzu自体はそこまで悪くない |
| H2 | BF16フォールバックが多い | 低 | BF16総バイトは公式と同じ 0.560 GB |
| H3 | 量子化後のテンソルレイアウトがGEMVに適していない | 高 | 公式とテンソル分割が異なる。U8 260+I32 258 vs U8 130+I32 0 |
| H4 | MLX affine 4bit自体が遅い | 高 | MLXネイティブでも4Bで26.89 t/sに留まる |
| H5 | lalamo変換が非効率な形式を生成している | 中〜高 | lalamo e401e177がMLX affine 4bitをそのままuzuにマッピング。最適化されていない可能性 |

### 17.2 検証実験計画

| 実験 | 内容 | 確認できること |
|---|---|---|
| A | Huihui 9B BF16をuzuで実行 | uzuがHuihui構造自体を遅くしているか |
| B | Huihui 9B MLX 4bitをMLXネイティブで実行 | MLX 4bit自体の速度 |
| C | Huihui 9B MLX 4bitをuzuで実行 | 済（17.09 t/s） |
| D | Huihuiを公式と同じ量子化形式に変換してuzu実行 | 形式変更で速度が上がるか |

### 17.3 実験の優先順位

1. **実験B**（最も簡単）: 4B/9B MLX 4bitをMLXネイティブで測定
2. **実験A**: BF16をuzu化して実行（M3 16GBではメモリ不足の可能性あり）
3. **実験D**: lalamoの新しいコミットまたはMirai量子化方式を試す

### 17.4 実験Bの結果

MLX 4bitネイティブとuzu MLX 4bitを比較。ネイティブ側は `mlx_lm.generate`、uzu側は `uzu CLI` で計測。条件は以下のとおり。

#### 4B

同一プロンプト「富士山について3文で説明して」:

| 形式 | prefill | generation | memory | 測定条件 |
|---|---|---|---|---|
| MLX 4bit ネイティブ | 5.82 t/s | 26.89 t/s | 2.53 GB | mlx_lm.generate, 通常デコード |
| uzu MLX 4bit | 129.71 t/s | 32.78 t/s | 2.33 GB | uzu CLI, reasoning medium |

短いプロンプト「こんにちは」:

| 形式 | generation | memory | 測定条件 |
|---|---|---|---|
| MLX 4bit ネイティブ | 17.75 t/s | 2.47 GB | mlx_lm.generate, 通常デコード |
| uzu MLX 4bit | 31.99 t/s | 2.33 GB | uzu CLI, reasoning medium |

4Bではuzuの方がMLXネイティブより速い。

#### 9B

2026-11-06にBF16ソースからMLX affine 4bitを生成し、ローカルで測定：

| 形式 | prefill | generation | memory | 測定条件 |
|---|---|---|---|---|
| MLX 4bit ネイティブ | 1.382 t/s | 2.221 t/s | 5.214 GB | mlx_lm 0.32.2, `--max-tokens 200`, thinkingトークンを含む固定長生成 |
| uzu MLX 4bit | 27.41 t/s | 9.87 t/s | 4.82 GB | uzu CLI, `--reasoning-effort medium`, 同プロンプト（12.4と同一計測） |

9BでもuzuはMLXネイティブより速い（約4.4倍）。ただし、MLXネイティブは `--max-tokens 200` の固定長生成であり、thinkingトークンを含む。表示応答トークン数が少ないため、実効速度は計測値より低い可能性あり。

なお、9B uzuの本計測は12.4の暫定3モデル比較にも採用済み。こちらも **再計測が必須** 。

### 17.5 実験Bの解釈

4Bと9Bの両方で、uzu変換後の方がMLXネイティブより速い。

これは仮説H1/H3（uzu変換が非効率）を弱める。低速の主因は「MLX affine 4bit自体の性能」または「lalamo e401e177が生成する重みレイアウト」にある可能性が高い。

ただし、MLXネイティブの測定条件がuzuと異なるため、決定的な比較ではない。再計測が必要。

## 18. 現在の状況と次の目標

### 18.1 9B MLX 4bit量子化 → uzu化（成功）

mlx-communityの偽4bitモデルではなく、元のBF16ソースからMLX affine 4bitを生成してuzu化した。

```text
huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated
→ mlx_lm.convert (Modal CPU, affine 4bit, group_size=64, float16)
→ lalamo e401e177
→ Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu
```

結果：

| 項目 | 値 |
|---|---|
| 量子化後 model.safetensors | 4.80 GB |
| bits per weight | 4.501 |
| uzu変換後合計 | 5.06 GB |
| ファイル数 | 3 (config.json, model.safetensors, tokenizer.json) |
| ローカル実行 | ✅ M3 16GB MacBook Airで動作 |

実測パフォーマンス：

```text
prefill speed:    47.12 t/s
generation speed: 17.07 t/s
memory used:       4.82 GB
time to first token: 0.38 s
```

無検閲＋Claude 4.6 Opus蒸留の9Bモデルを、M3 16GBで実用速度で動かすことに成功した。

### 18.2 uzu CLIで速度計測

結果：

```text
generation speed: 17.09 t/s
memory used: 4.82 GB
```

目標達成。

### 18.3 公式Qwen3.5 9Bと比較（済）

同じプロンプト「富士山について3文で説明して」で実施。

| 観点 | 公式9B | Huihui 9B |
|---|---|---|
| generation speed | 43.69 t/s | 17.09 t/s |
| memory | 5.69 GB | 4.82 GB |
| 無検閲 | 不明/制限あり | ✅ |
| Claude蒸留 | ❌ | ✅ |
| 事実正確性 | ✅ | ✅ |

### 18.4 無検閲＋蒸留モデルを1モデルとして本採用するか決める

選ぶ場合は、上記4項目を同じプロンプトで比較する。

現時点での判断材料：

- 速度が許容できるなら **Huihui 9B** を採用
- 速度・安定性最重要なら **公式9B** のまま

---

## 付録A：実際に動作したファイル構成

```text
Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu/
├── config.json
├── model.safetensors
├── tokenizer.json
└── encoding.json
```

### config.jsonの要修正箇所

```text
token_codec_config.response_format を削除
output_parser_regex を追加／null
end_of_thinking_tag を追加／null
```

### encoding.json

```json
{
  "type": "hanashi",
  "name": "qwen3.5"
}
```

---

## 付録B：9B変換に必要なコマンド

Modal Imageの方針：

```python
modal.Image.debian_slim(python_version="3.12")
.apt_install("git")
.pip_install("uv")
.run_commands(
  "git clone https://github.com/trymirai/lalamo.git /lalamo",
  "cd /lalamo && git checkout e401e177",
  'cd /lalamo && uv pip install --system -e ".[cpu]"',
)
```

変換スクリプト（要点）：

```python
REPO = "mlx-community/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-4bit"
OUT_NAME = "Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu"

LanguageModelSpec(
    vendor="Huihui",
    family="Qwen3.5",
    name="Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated",
    size="9B",
    origin=HuggingFaceOrigin(repo=REPO),
    config_type=HFQwen35Config,
    response_format=QWEN_RESPONSE_FORMAT,
    reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    configs=ConfigMap(
        tokenizer=FileSpec("tokenizer.json"),
        tokenizer_config=FileSpec("tokenizer_config.json"),
        chat_template=FileSpec("chat_template.jinja"),
        generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
    ),
)
```

---

## 付録C：結果

### 4B

```text
MLX 4bit → uzu: 成功
./target/release/cli: 会話OK
generation speed: 31.77 t/s
memory used: 2.33 GB
```

### 9B

```text
mlx-community Huihui 9B "4bit": 実体はBF16 (17.93GB) → ローカル実行不可
BF16ソースからMLX affine 4bit量子化: 成功 (4.80GB)
→ lalamo e401e177 uzu化: 成功 (5.06GB)
→ M3 16GB MacBook Airで会話OK

Huihui 9B MLX 4bit:
  prefill speed:    76.41 t/s
  generation speed: 17.09 t/s
  memory used:       4.82 GB
  time to first token: 0.24 s

公式 qwen3.5:9b:
  prefill speed:    53.21 t/s
  generation speed: 43.69 t/s
  memory used:       5.69 GB
  time to first token: 0.34 s
```

同じプロンプトで比較した結果、Huihui 9Bは速度で劣るがメモリ効率で勝ち、無検閲＋Claude蒸留の特性を保持している。
