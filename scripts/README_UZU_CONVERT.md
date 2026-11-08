# uzu_convert.py — 無検閲 Claude 蒸留モデルパイプライン

**対象**: `Huihui-Qwen3.5-*-Claude-4.6-Opus-abliterated` → `trymirai/uzu`
**更新日**: 2026-11-06

---

## 結論

**現時点で通った唯一のルート: `MLX affine 4bit → lalamo e401e177 → uzu`**

```
mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit
  → lalamo e401e177
  → 約2.39GBのuzuモデル
  → uzu CLIで会話OK (31.77 t/s, 2.33GBメモリ)
```

**9Bは mlx-community の `4bit` モデルが実体はBF16だったため、BF16ソースをModal CPUでMLX affine 4bitへ量子化してからuzu化した。**

```
huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated (BF16)
  → mlx_lm.convert (Modal CPU, affine 4bit)
  → lalamo e401e177
  → 約5.06GBのuzuモデル
  → uzu CLIで会話OK (17.07 t/s, 4.82GBメモリ)
```

---

## 前提条件

| 項目 | 要件 |
|---|---|
| Python | 3.10+ |
| git | PATH 通っていること |
| modal CLI | `pip install modal` |
| uzu | 最新ビルド済み (`./target/release/cli` 存在) |
| Modal Volume | `uzu-models` が作成済み |
| HF トークン | `HF_TOKEN` 環境変数 (プライベートリポの場合) |

---

## インストール

```bash
# uv を使う場合
cd /Users/kohei/Myproject/uzu
uv add modal huggingface-hub

# pip の場合
pip install modal huggingface-hub
```

`.env.example` をコピーして環境変数を設定:

```bash
cp scripts/.env.example .env
# .env を編集して実際の値をセット
source .env
```

---

## 使用法

```bash
# 基本フォーマット
python3 scripts/uzu_convert.py \
  --repo <HF_REPO_ID> \
  --out <MODEL_NAME> \
  <SUBCOMMAND>
```

### 共通オプション

| オプション | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--repo` | はい | — | HFリポジトリID (例: `mlx-community/Huihui-Qwen3.5-4B-...`) |
| `--out` | はい | — | models/以下の出力ディレクトリ名 |
| `--lalamo-commit` | いいえ | `e401e177` | lalamoのコミットハッシュ |
| `--model-size` | いいえ | `4B` | `4B` または `9B` |
| `--modal-volume` | いいえ | `uzu-models` | Modal Volume名 |
| `--uzu-bin` | いいえ | `./target/release/cli` | uzu CLIバイナリパス |
| `--base-dir` | いいえ | スクリプトの親dir | uzuレポのルート |
| `--source-kind` | いいえ | `hf-mlx-4bit` | `hf-mlx-4bit` / `hf-bf16` / `modal-mlx-4bit` |
| `--safe` | いいえ | False | 対話確認モード (convert/get-localで確認要求) |
| `--message` | いいえ | `テストメッセージ` | uzuテスト用メッセージ |
| `--run-after` | いいえ | False | full実行後にrunステップを実行 |

---

## サブコマンド

### `prereq` — 環境前提確認

git, modal CLI, uzuバイナリ, Modal Volumeの存在を確認.

```bash
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out test-model \
  prereq
```

### `dry-run` — 入力検証 + 実行計画

量子化方式の判定、計画の表示. 何も実行しない.

```bash
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out test-model \
  dry-run
```

### `convert` — Modal で lalamo 変換

Modal CPU Function で lalamo e401e177 を使い変換. 結果をModal Volumeに保存.

```bash
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  convert
```

`--safe` 付きで対話確認.

#### BF16ソースをMLX 4bit量子化してからuzu化

```bash
python3 scripts/uzu_convert.py \
  --repo huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated \
  --out Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  --model-size 9B \
  --source-kind hf-bf16 \
  convert
```

これはModal上で以下を実行する:

1. `mlx_lm.convert` で BF16 → MLX affine 4bit
2. 元のBF16リポ、または `Qwen/Qwen3.5-27B` から `generation_config.json` を取得
3. lalamo e401e177 で uzu 形式へ変換
4. 中間MLXディレクトリを削除

注意: `mlx_lm.convert` の出力に `generation_config.json` は含まれない。lalamoがこれを必要とするため、フォールバックリポから取得する。

### `fix-config` — config.json 修正

lalamo出力のconfig.jsonをuzu互換に修正.

**修正内容**:
- `token_codec_config.response_format` を削除
- `token_codec_config.output_parser_regex = null` を追加
- `token_codec_config.end_of_thinking_tag = null` を追加
- `bos_token`, `eos_token` なければnull追加
- `decoder_config` から不要なキーを削除

```bash
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  fix-config
```

### `add-encoding` — encoding.json 設置

```json
{"type": "hanashi", "name": "qwen3.5"}
```

を設置.

```bash
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  add-encoding
```

### `get-local` — Modal Volume からローカル取得

```bash
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  get-local
```

### `sanity` — モデル構造検証

config.json, encoding.json, model.safetensors, tokenizer.json の存在・中身を検証.

```bash
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  sanity
```

### `run` — uzu CLI で会話テスト

```bash
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  run \
  --message "富士山について3文で説明して"
```

### `full` — 全工程実行

```
prereq → dry-run → convert → fix-config → add-encoding → get-local → sanity → (run)
```

```bash
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  full \
  --message "テスト" \
  --run-after
```

BF16ソースの場合:

```bash
python3 scripts/uzu_convert.py \
  --repo huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated \
  --out Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  --model-size 9B \
  --source-kind hf-bf16 \
  full \
  --message "富士山について3文で説明して" \
  --run-after
```

`--safe` で各ステップで対話確認.

---

## 4B Smoke Test

```bash
# 1. 環境確認
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  prereq

# 2. 検証
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  dry-run

# 3. 検証済みのモデルがある場合は sanity + run で即テスト
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  sanity

python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  run \
  --message "こんにちは"
```

---

## 9B への変更

9B は同一手順で `--repo` と `--out` を変更.

```bash
python3 scripts/uzu_convert.py \
  --repo mlx-community/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-4bit \
  --out Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu \
  --model-size 9B \
  full \
  --safe
```

**目標**:
- 生成速度: ≥ 20 t/s
- メモリ使用: ≤ 6 GB

---

## 量子化方式の互換性

| 量子化方式 | 結果 | 原因 |
|---|---|---|
| **MLX affine 4bit** | ✅ 動作確認済み | Metalカーネル対応 |
| **MLX affine 8bit** | ❌ Metal未対応の疑い | `_D4GemmS...FullPrecision` エラー |
| **GPTQ W4A16** | ❌ DeltaNet非互換 | `ValueError: Expected in_proj_qkv...` |
| **AutoAWQ W4A16** | ❌ 未対応 | `TypeError: qwen3_5 isn't supported yet` |
| **compressed-tensors** | ❌ uzu未対応 | uzuが読む独自形式ではない |

**pipeline は unsupported 量子化を入力として検出すると即エラー停止. 黙って置換しない.**

---

## 環境変数

| 変数 | 必須 | 説明 |
|---|---|---|
| `HF_TOKEN` | 条件 | Hugging Face トークン (プライベートリポ/クオータ超過時) |
| `MODAL_TOKEN_ID` | convert 時 | Modal 認証 |
| `MODAL_TOKEN_SECRET` | convert 時 | Modal 認証 |
| `UZU_BASE_DIR` | いいえ | uzu レポのルート (デフォルト: スクリプトの親dir) |

---

## config.json 修正の詳細

lalamo が出力する config.json は uzu が要求する形と異なる.

### lalamo 出力 (例)

```json
{
  "type": "LanguageModelConfig",
  "decoder_config": {
    "embedding_config": {...},
    "transformer_config": {...},
    "vocab_size": 151936,
    "extra_field": "lalamoのみが出力"
  },
  "token_codec_config": {
    "type": "...",
    "response_format": {...},    ← uzuは未知 → 削除
    "output_parser_regex": null  ← lalamoは未出力 → 追加
  }
}
```

### uzu 要求形

```json
{
  "type": "LanguageModelConfig",
  "decoder_config": {
    "embedding_config": {...},
    "transformer_config": {...},
    "vocab_size": 151936,
    "ple_model_config": {...},
    "embedding_norm_config": {...}
  },
  "token_codec_config": {
    "type": "...",
    "prompt_template": "...",
    "system_role_name": "...",
    "user_role_name": "...",
    "assistant_role_name": "...",
    "eos_token": null,
    "bos_token": null,
    "default_system_prompt": "...",
    "reasoning_config": {...},
    "output_parser_regex": null,
    "end_of_thinking_tag": null
  },
  "generation_config": {...}
}
```

### fix-config が行う変更

1. `token_codec_config.response_format` → **削除**
2. `token_codec_config.output_parser_regex` → **null追加** (なければ)
3. `token_codec_config.end_of_thinking_tag` → **null追加** (なければ)
4. `token_codec_config.bos_token` → **null追加** (なければ)
5. `token_codec_config.eos_token` → **null追加** (なければ)
6. `decoder_config` の不要キー → **削除**

---

## encoding.json の仕様

uzu はチャット形式が不明だと起動できない.

```json
{
  "type": "hanashi",
  "name": "qwen3.5"
}
```

- `type: "hanashi"` — uzu のチャット形式
- `name: "qwen3.5"` — Huihui Qwen3.5 系の Hanashi 設定

---

## 失敗パターンと対処

### `_D4GemmS...FullPrecision`

**原因**: MLX 8bit / Metalカーネル未対応
**対処**: MLX 4bit に変更

### `ValueError: Expected in_proj_qkv...`

**原因**: GPTQ 出力は lalamo DeltaNet 配置と互換なし
**対処**: MLX affine 4bit を使用

### `data did not match any variant...`

**原因**: `token_codec_config.response_format` 残留
**対処**: `fix-config` を実行

### `can not get encoding config`

**原因**: `encoding.json` 未設置
**対処**: `add-encoding` を実行

---

## 既知の制約

- lalamo commit `e401e177` 固定 (residual_config 非出力版)
- 新 lalamo は residual_config を出力 → uzu が未知フィールドとして拒否
- Modal Volume `uzu-models` が必要 (変更は `--modal-volume` で対応可能)
- Modal 変換は CPU Function のみ. GPU は不要.
- 変換は10〜30分前後 (CPU 16 vCPU 基準)
- BF16/FP16 は未検証. メモリ大量消費の可能性.
- 既存Uzu本体コードは一切編集しない.

---

## ファイル構成 (成功例)

```
models/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-MLX-4bit-e401e177-uzu/
├── config.json       (92.8K, LanguageModelConfig 形式)
├── encoding.json     ({"type":"hanashi","name":"qwen3.5"})
├── model.safetensors (2259.4M)
└── tokenizer.json    (19.1M)
```

---

## 結果

```text
MLX 4bit → uzu: 成功
./target/release/cli: 会話OK
generation speed: 31.77 t/s
memory used: 2.33 GB
```