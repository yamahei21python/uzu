# uzu 向けモデル変換・運用マニュアル

〜 Hugging Face モデル選定 → lalamo 変換 → Uzu 実行までの完全手順 〜

## 目次

1. モデル選定基準
   1.1 必須条件 (lalamo 変換可能か)
   1.1.1 量子化形式の確認 (ここが一番ハマる)
   1.2 推奨条件 (M3 16GB で快適)
   1.3 モデル評価スコアシート
2. 事前準備
   2.1 共通環境変数
   2.2 Modal 用 (クラウド変換時)
   2.3 変換済みモデル保存用 Volume
3. lalamo ModelSpec 登録
   3.1 登録ファイル場所
   3.2 追加テンプレート (qwen.py 例)
   3.3 登録確認
   3.4 よく使う ConfigType マッピング
4. ローカル変換 (M3 16GB 等)
   4.1 基本コマンド
   4.2 メモリ不足時の対策
   4.3 実行例: DeepSeek-V4-Pro heretic MLX 8bit
   4.4 成功確認
5. Modal クラウド変換 (大容量・高速)
   5.1 いつ使うか
   5.2 Modal スクリプト テンプレート
   5.2.1 ハマりどころ (実測で踏んだもの)
   5.2.2 lalamo と uzu のバージョンを合わせる (最重要)
   5.3 実行・結果取得
   5.4 コスト目安
6. Uzu 統合・動作確認
   6.1 ビルド
   6.2 簡易チャットテスト
   6.3 ベンチマーク測定
   6.4 エイリアス登録 (日常運用)
7. 投機デコード対応 (DFlash/Weaver) [blocked]
   7.1 現状の選択肢
   7.2 公式 DFlash 使用 (推奨)
   7.3 Weaver 自作手順 (上級者向け)
8. トラブルシューティング
   8.1 変換エラー
   8.2 Uzu 実行エラー
   8.3 よくあるハマりどころ
9. テンプレート集
   9.1 ModelSpec 追加ワンライナー
   9.2 ローカル変換ワンライナー
   9.3 Modal 実行ワンライナー
   9.4 ベンチマークワンライナー
   9.5 完全自動化スクリプト (model_convert.sh)

---

## 1. モデル選定基準

### 1.1 必須条件 (lalamo 変換可能か)

| 条件 | 確認方法 | NG 例 |
| --- | --- | --- |
| safetensors 形式 | HF リポの Files タブで `.safetensors` 存在 | GGUF のみ、`.bin` のみ |
| 対応アーキテクチャ | `lalamo list-models` で同ファミリ存在 | 新アーキ (未対応) |
| config.json 正常 | `model_type` が既知 (qwen3_5, llama, gemma 等) | カスタムタイプ |
| **量子化形式が対応済み** | 下の 1.1.1 を参照 | `compressed-tensors` |
| 公開リポ | Gated でない、またはトークン所持 | プライベート/ゲート無しアクセス不可 |

#### 1.1.1 量子化形式の確認 (ここが一番ハマる)

lalamo が実装している量子化は **3種類だけ**。`quant_method` がこれ以外だと、
config は読めても量子化情報が黙って落ちて (`quantization: None`)、重みのロードで死ぬ。

| `quant_method` | lalamo 対応 | テンソル名 | 備考 |
| --- | --- | --- | --- |
| `awq` | ✅ | `qweight` / `qzeros` / `scales` | AutoAWQ 形式のみ |
| `gptq` | ✅ | `qweight` / `qzeros` / `scales` | |
| `mlx` (affine) | ✅ | `weight` / `scales` / `biases` | 4bit / 8bit |
| `mlx` (mxfp4) | ❌ | `weight` / `scales` | **affine として誤読される** |
| `compressed-tensors` | ❌ | `weight_packed` / `weight_scale` / `weight_zero_point` / `weight_shape` | 実装なし |
| (なし = BF16/FP16) | ✅ | `weight` | フル精度 |

**判定コマンド** (変換前に必ずやる):

```bash
REPO="<HFリポID>"
curl -s "https://huggingface.co/$REPO/raw/main/config.json" \
  | python3 -c "import json,sys; c=json.load(sys.stdin); print(c.get('quantization_config'))"
```

`quant_method` が `compressed-tensors` なら**そのモデルは見送る**。回避策はない。
量子化層に `.weight` が存在しないので、フル精度フォールバックも効かない
(`load_full_precision` が `KeyError` になる)。

`mode: mxfp4` の MLX モデルも見送る。lalamo の `MLXQuantizationConfig` は
`group_size` と `bits` しか読まず、`mode` を無視する。affine 前提でアンパックするので
数値が壊れる。**`mode: affine` の MLX モデルだけが安全**。

### 1.2 推奨条件 (M3 16GB で快適)

| 指標 | 目安 | 理由 |
| --- | --- | --- |
| 量子化済み | AWQ W4A16 / MLX **affine** 4-8bit / GPTQ 4bit | 変換メモリ < 16GB、推論 VRAM < 8GB |
| サイズ | 9B 以下 (量子化後 4-6GB) | 16GB で余裕 |
| 投機対応 | 公式 DFlash あり、または同ファミリで作成済み | 速度 2-4倍 |

> MLX は `mode: affine` のものだけ選ぶ。`mxfp4` は 1.1.1 のとおり使えない。

### 1.3 モデル評価スコアシート

| 項目 | 重み | スコア (1-5) | 重み付け |
| --- | --- | --- | --- |
| safetensors 公開 | 必須 | 5 | - |
| 量子化済み (4bit) | 高 | 5 | 5×3=15 |
| DeepSeek/Heretic 等蒸留 | 中 | 4 | 4×2=8 |
| 無検閲/アブリテート | 中 | 4 | 4×2=8 |
| lalamo 対応アーキ | 高 | 5 | 5×3=15 |
| 投機モデル存在 | 低 | 3 | 3×1=3 |
| **合計** | | | **49/55** |

閾値: 40 以上 → 即採用 / 30-39 → 条件付き / 30 未満 → 見送り

---

## 2. 事前準備

### 2.1 共通環境

```bash
# 作業ディレクトリ
export WORK_DIR="/Users/kohei/Myproject/uzu"
export MODELS_DIR="$WORK_DIR/models"
mkdir -p "$MODELS_DIR"

# lalamo クローン・セットアップ
cd /tmp
git clone https://github.com/trymirai/lalamo.git
cd lalamo
uv sync --frozen  # 初回のみ 30-60秒
```

### 2.2 Modal 用 (クラウド変換時)

```bash
# Modal CLI インストール・認証
pip install --upgrade modal
modal token new  # ブラウザで認証

# 設定固定 (バージョン互換用)
mkdir -p ~/.config/modal
cat > ~/.config/modal.toml << 'EOF'
[image_builder]
version = "2025.06"
EOF
```

### 2.3 変換済みモデル保存用 Volume (Modal)

```bash
# 初回のみ作成
modal volume create uzu-models
```

---

## 3. lalamo ModelSpec 登録

### 3.1 登録ファイル場所

```
/tmp/lalamo/lalamo/model_import/model_specs/
├── qwen.py      # Qwen 系
├── llama.py     # Llama 系
├── gemma.py     # Gemma 系
├── mistral.py   # Mistral 系
└── ...          # その他
```

### 3.2 追加テンプレート (qwen.py 例)

```python
# 対象ファミリのリスト末尾に追加 (QWEN35 リスト等)

LanguageModelSpec(
    vendor="<ベンダー名>",           # 例: "Sohailhosseini"
    family="<ファミリ名>",           # 例: "Qwen3.5"
    name="<モデル名>",               # 例: "DeepSeek-V4-Pro-Qwen3.5-9B-AWQ-W4A16"
    size="<サイズ>",                 # 例: "9B"
    origin=HuggingFaceOrigin(repo="<HFリポID>"),
    config_type=HFQwen35Config,     # アーキに合わせる (HFQwen35Config, HFQwen3Config, HFLlamaConfig 等)
    response_format=QWEN_RESPONSE_FORMAT,  # ファミリ標準
    reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    # tokenizer等が非標準なら configs で指定
    # configs=ConfigMap(
    #     tokenizer=FileSpec("tokenizer.json", "<ベースリポ>"),
    #     tokenizer_config=FileSpec("tokenizer_config.json", "<ベースリポ>"),
    #     generation_config=FileSpec("generation_config.json", "<ベースリポ>"),
    # ),
)
```

### 3.3 登録確認

```bash
cd /tmp/lalamo
uv run lalamo list-models | grep -i "<キーワード>"
# 表示されれば OK
```

### 3.4 よく使う ConfigType マッピング

| アーキテクチャ | ConfigType | ResponseFormat |
| --- | --- | --- |
| Qwen3.5 | `HFQwen35Config` | `QWEN_RESPONSE_FORMAT` |
| Qwen3 | `HFQwen3Config` | `THINKING_RESPONSE_FORMAT` |
| Qwen2.5 | `HFQwen2Config` | `THINKING_RESPONSE_FORMAT` |
| Llama 3.x | `HFLlamaConfig` | `LLAMA_RESPONSE_FORMAT` |
| Gemma 3 | `HFGemma3Config` | `GEMMA_RESPONSE_FORMAT` |
| Mistral | `HFMistralConfig` | `MISTAL_RESPONSE_FORMAT` |

> この表が正。9.5 のスクリプトはこの表を `case` 文で持っている。ファミリを追加したら両方に足す。
> `MISTAL_RESPONSE_FORMAT` は lalamo 側の綴り (MISTRAL ではない)。そのまま書く。

---

## 4. ローカル変換 (M3 16GB 等)

### 4.1 基本コマンド

```bash
cd /tmp/lalamo

JAX_DISABLE_JIT=1 uv run lalamo convert \
  <HFリポID> \
  --output-dir <出力先> \
  --dtype bfloat16
```

オプション:

| オプション | 説明 | 推奨値 |
| --- | --- | --- |
| `JAX_DISABLE_JIT=1` | JIT コンパイル無効化 (メモリ節約) | 必須 (16GB機) |
| `--dtype` | 活性化・非量子化重みの dtype | `bfloat16` (標準) |
| `--output-dir` | 出力ディレクトリ | `$MODELS_DIR/<モデル名>` |

### 4.2 メモリ不足時の対策

1. **swap 追加 (macOS)**

```bash
sudo sysctl vm.swapfile_size=8589934592  # 8GB 追加
```

2. **Modal クラウド変換推奨** (バッチサイズ調整困難なため)
3. **段階的変換** (未対応)

### 4.3 実行例: DeepSeek-V4-Pro heretic MLX 8bit

> 旧版の例は `Sohailhosseini/DeepSeek-V4-Pro-Qwen3.5-9B-AWQ-W4A16` だったが、
> これは `compressed-tensors` 形式で lalamo が読めない (1.1.1 参照)。
> 同じ DeepSeek-V4-Pro 派生の MLX affine 8bit に差し替えた。

```bash
cd /tmp/lalamo

JAX_DISABLE_JIT=1 uv run lalamo convert \
  mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-8bit \
  --output-dir /Users/kohei/Myproject/uzu/models/DeepSeek-V4-Pro-Qwen3.5-9B-8bit
```

ただしこのリポジトリには `generation_config.json` が無いので、CLI 経由だと
404 で落ちる。`ConfigMap` で補う必要があるため、**このモデルは 5.2 の
Modal スクリプト (または同等に spec を自前構築するコード) で変換する**。

- 所要時間: M3 16GB で 30-45分 / Modal 16vCPU で 1分28秒
- ピークメモリ: 12-14 GB
- 出力: 9.54 GB

### 4.4 成功確認

```bash
ls -la /Users/kohei/Myproject/uzu/models/<出力名>/
# 出力はこの 3 ファイルだけ (lalamo の保存形式)
# config.json          ← lalamo 形式に変換済み ("type": "LanguageModelConfig")
# model.safetensors    ← 重み
# tokenizer.json
```

`tokenizer_config.json` と `generation_config.json` は**入力**であって出力ではない。
変換時に読まれて config.json に畳み込まれる。出力に無くても正常。

config.json が lalamo 形式になっているか確認:

```bash
head -c 100 models/<出力名>/config.json
# {"type": "LanguageModelConfig", "token_codec_config": ... なら OK
```

---

## 5. Modal クラウド変換 (大容量・高速)

### 5.1 いつ使うか

- ローカルメモリ不足 (OOM)
- BF16 全精度 (18GB+) 変換
- より高速に終わらせたい
- 夜間バッチで回したい

### 5.2 Modal スクリプト テンプレート

**実測で動いた版**。2026-11-06 に `mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-8bit`
(9.54GB) を 1分28秒で変換。`/tmp/convert_modal.py` にそのまま置いてある。

設計方針 — 元のテンプレートから変えた点:

- **ビルド時に Python を実行しない**。`run_commands` 内の `python3 << PYEOF`
  ヒアドキュメントは Modal のビルド環境で `SyntaxError` / `EOL while scanning
  string literal` になる。ビルドは `git clone` と `uv pip install` だけ。
- **lalamo のソースを書き換えない**。`lalamo.commands.convert()` は ModelSpec
  オブジェクトを直接受け取れるので、`model_specs/*.py` への追記も
  `ModelRegistry` への登録も CLI 起動も要らない。
- **`app.image_builder_version` は書かない**。Modal 1.2.6 の `App` にそんな属性は
  なく、代入しても黙って無視される。バージョン固定は環境変数
  `MODAL_IMAGE_BUILDER_VERSION=2025.06` で行う (5.3 参照)。

```python
"""lalamo 変換を Modal で実行する。

用法:
  MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run convert_modal.py
  MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run convert_modal.py \
    --repo <HFリポID> --out-name <出力名>
"""

import modal

REPO = "<HFリポID>"
OUT_NAME = "<出力名>"

app = modal.App("lalamo-convert")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "git clone https://github.com/trymirai/lalamo.git /lalamo",
        # jax[cpu] ごとシステム Python に入れる。これで convert() を
        # プロセス内で直接呼べる (uv run でサブプロセス起動する必要がない)。
        'cd /lalamo && uv pip install --system -e ".[cpu]"',
    )
)

volume = modal.Volume.from_name("uzu-models", create_if_missing=True)


@app.function(
    image=image,
    cpu=16,
    memory=65536,      # 64 GB
    timeout=3600 * 4,  # 4時間
    volumes={"/models": volume},
)
def convert(repo: str = REPO, out_name: str = OUT_NAME) -> dict:
    import os
    import sys
    import traceback
    from pathlib import Path

    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    sys.path.insert(0, "/lalamo")

    from lalamo.commands import DType, convert as lalamo_convert
    from lalamo.model_import.model_configs import HFQwen35Config
    from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
    from lalamo.model_import.model_specs.qwen import QWEN_RESPONSE_FORMAT
    from lalamo.model_import.model_specs.reasoning_configs import (
        BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    )
    from lalamo.model_import.origins import HuggingFaceOrigin

    output_dir = Path("/models") / out_name
    if output_dir.exists():
        return {"status": "skipped", "reason": f"{output_dir} already exists"}

    spec = LanguageModelSpec(
        vendor=repo.split("/")[0],
        family="Qwen3.5",
        name=out_name,
        size="9B",
        origin=HuggingFaceOrigin(repo=repo),
        config_type=HFQwen35Config,
        response_format=QWEN_RESPONSE_FORMAT,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
        # リポジトリに generation_config.json が無い場合、同ファミリの
        # 公式リポから補う (lalamo の既存 spec と同じやり方)。
        configs=ConfigMap(
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
        ),
    )
    print(f"[spec] {spec.vendor}/{spec.name} <- {spec.origin.description}", flush=True)

    try:
        lalamo_convert(spec, output_dir, DType.BFLOAT16)
    except Exception:
        traceback.print_exc()
        return {"status": "failed", "traceback": traceback.format_exc()[-4000:]}

    files = sorted(p.name for p in output_dir.iterdir())
    sizes = {p.name: p.stat().st_size for p in output_dir.iterdir() if p.is_file()}
    total = sum(sizes.values())
    print(f"[done] {len(files)} files, {total / 1e9:.2f} GB", flush=True)
    for name in files:
        print(f"  {name:40s} {sizes.get(name, 0) / 1e6:10.1f} MB", flush=True)

    volume.commit()
    return {
        "status": "success",
        "repo": repo,
        "out_name": out_name,
        "files": files,
        "total_bytes": total,
    }


@app.local_entrypoint()
def main(repo: str = REPO, out_name: str = OUT_NAME) -> None:
    print(f"converting {repo} -> /models/{out_name}")
    print(convert.remote(repo=repo, out_name=out_name))
```

#### 5.2.1 ハマりどころ (実測で踏んだもの)

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| `VersionError: ... You are using '2026.10'` | Modal クライアント既定の image builder がクライアントより新しい | `MODAL_IMAGE_BUILDER_VERSION=2025.06` を付ける |
| ビルドで `SyntaxError` / `EOL while scanning string literal` | `run_commands` 内のヒアドキュメント | ビルドで Python を実行しない。変換関数内でやる |
| `404 ... generation_config.json` | リポジトリに同ファイルが無い | `ConfigMap(generation_config=FileSpec(..., "Qwen/Qwen3.5-27B"))` で補う |
| `quantization: None` で重みロードが `KeyError` | `compressed-tensors` 形式 | 1.1.1 参照。そのモデルは見送る |
| `ModuleNotFoundError: jax` | `uv pip install -e .` だけでは jax が入らない | `-e ".[cpu]"` にする |

#### 5.2.2 lalamo と uzu のバージョンを合わせる (最重要)

**変換が成功しても uzu で読めないことがある。** 原因は lalamo と uzu の
config スキーマの不一致。2026-11-06 に実際に踏んだ。

症状:

```
Error: Unable to get models: Unable to resolve specialization for <model path>:
Unable to deserialize model configuration:
data did not match any variant of untagged enum AnyModelConfig
```

原因:

- lalamo の `5d7e33b6` (2026-11-03, Transformer Layer Refactor #397) で
  `TransformerConfig` に `residual_config` が追加された
- uzu 側の `TransformerConfig` にはこのフィールドが無い
  (2026-10-27 が最終コミット。origin/main も同じ)
- よって lalamo が出力した config.json を uzu がデシリアライズできない

`decoder_config` のキーは全部揃っていても、`transformer_config` の中に
余分なキーが1個あるだけでこのエラーになる。エラーメッセージからは
どのフィールドか分からないので、以下の手順で突き合わせる。

**手順 1: uzu 側の期待フィールドを確認**

```bash
cd /Users/kohei/Myproject/uzu
grep -A 10 'pub struct TransformerConfig' crates/uzu-engine/src/config/transformer.rs
```

**手順 2: lalamo 側の出力フィールドを確認**

```bash
cd /tmp/lalamo
grep -A 8 'class TransformerConfig' lalamo/modules/transformer.py
```

**手順 3: 差分があれば lalamo を uzu に合うコミットに固定**

```bash
# uzu の最終コミット日を確認
git -C /Users/kohei/Myproject/uzu log -1 --format='%h %ad %s' --date=short

# lalamo 側でその日以前のコミットを探す
git -C /tmp/lalamo log --format='%h %ad %s' --date=short --before='2026-11-03' -5

# 該当コミットの TransformerConfig が uzu と一致するか確認
git -C /tmp/lalamo show <commit>:lalamo/modules/transformer.py | grep -A 8 'class TransformerConfig'
```

Modal で変換する場合は、スクリプト内でコミットを固定する:

```python
LALAMO_COMMIT = "e401e177"  # 2026-11-02。uzu の TransformerConfig と一致

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

**実測の対応表** (2026-11-06 時点):

| lalamo | uzu | 相性 |
| --- | --- | --- |
| `e401e177` (2026-11-02) 以前 | `13b138e` (2026-10-27) | ✅ 一致 |
| `5d7e33b6` (2026-11-03) 以降 | 同上 | ❌ `residual_config` で失敗 |

> uzu を更新すれば解決する場合もある。ただし uzu の origin/main に
> `residual_config` 対応が入っているかを必ず確認する。入っていなければ
> lalamo 側を下げるしかない。

### 5.3 実行・結果取得

```bash
# 実行。MODAL_IMAGE_BUILDER_VERSION は必須 (5.2.1 参照)
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run convert_modal.py

# 引数を変えて回す場合
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run convert_modal.py \
  --repo <HFリポID> --out-name <出力名>

# 完了後、ローカルに同期
modal volume get uzu-models <出力名> /Users/kohei/Myproject/uzu/models/
```

`modal volume get` は**ディレクトリ名を渡す**。`<出力名>/` と末尾スラッシュを
付けても動くが、宛先を省略すると挙動が変わるので明示する。

#### 実測値 (2026-11-06)

`mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-8bit` (9.54GB, MLX 8bit affine):

| 項目 | 値 |
| --- | --- |
| Modal 実行時間 | **1分28秒** (起動 2.06s 含む) |
| コンテナ起動 → `[done]` | 69秒 |
| 出力サイズ | 9.54GB (3ファイル) |
| 内訳 | config.json 0.1MB / model.safetensors 9516.6MB / tokenizer.json 20.0MB |

所要時間の大半は HF からのダウンロード。今回は非認証で約 146MB/s。
`HF_TOKEN` を環境変数に入れるとレート制限が緩む。

**ローカル変換との比較**: M3 16GB で同じモデルを BF16 変換すると 30-45分。
Modal は 30倍速い。ただしこれは**すでに量子化済みのモデル**を読んだ場合で、
BF16 フル精度 (18GB+) ならダウンロードも変換も倍以上かかる。

#### ローカルへ落とす前のディスク確認

```bash
df -h /Users/kohei/Myproject/uzu | tail -1
```

出力サイズ + 2GB 程度の余裕を見る。足りない場合の候補:

- `target/` を消す (`cargo build` キャッシュ。再ビルドに 5-10分)
- より小さい量子化版に切り替える (MLX affine 4bit なら 5GB 前後)
- ローカルには落とさず Volume に置いたままにする

### 5.4 コスト目安

| 構成 | 時間 | 概算コスト |
| --- | --- | --- |
| 16 vCPU, 64GB (実測 9.5GB モデル) | **1分28秒** | 約 $0.01 |
| 16 vCPU, 64GB (BF16 18GB+ 想定) | 5-10分 | $0.10-0.20 |
| 32 vCPU, 128GB | 上記の 6-7割 | 同程度 |

$30 クレジットなら、量子化済みモデルの変換は数百回回せる計算。
実測後もクレジット残高は $29.95 (1回の変換で $0.05 前後)。

---

## 6. Uzu 統合・動作確認

### 6.1 ビルド

```bash
cd /Users/kohei/Myproject/uzu
cargo build --release -p uzu-engine  # 初回 5-10分、以降 数秒
```

### 6.2 簡易チャットテスト

`cargo run --example chat` は**使えない**。`crates/legacy/uzu/examples/chat.rs` は
モデルIDをハードコードしていて、`--model-path` も `--prompt` も受け取らない。
正しい入口は CLI。

```bash
cd /Users/kohei/Myproject/uzu
cargo build --release -p cli   # 初回 5-10分

MODEL_DIR="/Users/kohei/Myproject/uzu/models/DeepSeek-V4-Pro-Qwen3.5-9B-8bit"

./target/release/cli \
  --model "$MODEL_DIR" \
  --message "富士山について3文で説明して"
```

`--model` はモデルIDでも HF リポIDでも**ローカルディレクトリ**でも受け付ける
(`engine.model_by_path()` 経由)。

オプション:

| フラグ | 説明 |
| --- | --- |
| `--model <PATH>` | モデルID / HFリポID / ローカルディレクトリ |
| `-m, --message <TEXT>` | 1回だけ送って終了。省略すると対話モード |
| `--reasoning-effort <EFFORT>` | disabled / default / low / medium / high |
| `--seed <SEED>` | サンプリングシード |
| `--no-tools` | 組み込みツールを無効化 |

その他のサブコマンド: `bench`, `list-checkpoints`, `list-models`, `server`, `storage`。

### 6.3 ベンチマーク測定

```bash
#!/bin/bash
# bench.sh
MODEL_DIR="/Users/kohei/Myproject/uzu/models/DeepSeek-V4-Pro-Qwen3.5-9B-8bit"
PROMPTS=(
  "Rustで非同期処理のベストプラクティスを5つ挙げて"
  "量子コンピューティングの基本原理を簡潔に説明"
  "短編小説の冒頭200文字を書いて"
)

for p in "${PROMPTS[@]}"; do
  echo "=== Prompt: $p ==="
  /usr/bin/time -l ./target/release/cli \
    --model "$MODEL_DIR" \
    --message "$p" 2>&1 | grep -E "tokens/sec|t/s|peak memory|Generated"
done
```

合格ライン (M3 16GB, 投機なし):

| 指標 | 目標 |
| --- | --- |
| 生成速度 | ≥ 25 t/s |
| ピークメモリ | ≤ 10 GB |
| 初期化時間 | ≤ 10秒 |

### 6.4 エイリアス登録 (日常運用)

`~/.zshrc` または `~/.bashrc`:

```bash
uzu_chat() {
  /Users/kohei/Myproject/uzu/target/release/cli \
    --model "/Users/kohei/Myproject/uzu/models/DeepSeek-V4-Pro-Qwen3.5-9B-8bit" \
    "$@"
}

# 使い方
uzu_chat --message "質問"
uzu_chat  # 対話モード
```

---

## 7. 投機デコード対応 (DFlash/Weaver) [blocked]

### 7.1 現状の選択肢

| 方式 | 入手性 | 速度向上 | 実装難易度 |
| --- | --- | --- | --- |
| 公式 DFlash | 配布済みのみ | 2-4倍 (62 t/s) | 低 (設定のみ) |
| コミュニティ DFlash | 稀 | 不明 | 中 (互換検証) |
| Weaver 自作 | 可能 | 1.2-1.5倍 | 中 (週末PJ) |
| DFlash 自作 | 極めて困難 | 2-4倍 | 極高 (研究級) |

### 7.2 公式 DFlash 使用 (推奨)

```bash
# 公式モデルには同梱済み
uzu run trymirai/Qwen3.5-9B-M  # 投機あり 62 t/s
```

### 7.3 Weaver 自作手順 (上級者向け)

1. 隠れ状態収集 (lalamo フックで実装必要)
2. MLP 訓練 (MLX/JAX/PyTorch)
3. lalamo SpeculatorModel 形式でエクスポート
4. uzu 側で読み込み対応実装

詳細は別途「Weaver自作ガイド」参照。

---

## 8. トラブルシューティング

### 8.1 変換エラー

| エラー | 原因 | 対処 |
| --- | --- | --- |
| `Model "xxx" not found` | ModelSpec 未登録 | qwen.py 追加確認、`list-models` で確認 |
| OOM / MemoryError | メモリ不足 | `JAX_DISABLE_JIT=1`、swap増設、Modal使用 |
| `config.json: model_type unknown` | 非対応アーキ | ConfigType 追加、または見送り |
| `tokenizer.json not found` | トークナイザ別リポ | configs でベースリポ指定 |
| `safetensors: tensor not found` | シャード欠落 | HF から全シャード再DL |

### 8.2 Uzu 実行エラー

| エラー | 原因 | 対処 |
| --- | --- | --- |
| ParameterLoader error | safetensors 破損/不整合 | 変換し直し、ファイルサイズ確認 |
| Tokenizer error | tokenizer.json 形式不正 | HF からオリジナル取得 |
| Metal: out of memory | VRAM 不足 | 量子化版使用、コンテキスト短縮 |
| Speculator error | 投機モデル不整合 | 投機なしで起動、speculator 除外 |

### 8.3 よくあるハマりどころ

- **uv キャッシュ問題**

```bash
uv cache clean
uv sync --frozen
```

- **lalamo モジュールリロード失敗** → プロセス再起動 (Modal なら別関数呼び出し)
- **JAX プラットフォーム自動選択 (CPU/GPU)**

```bash
export JAX_PLATFORM_NAME=cpu  # 強制 CPU
```

- **Hugging Face レート制限**

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1  # 高速DL
```

---

## 9. テンプレート集

### 9.1 ModelSpec 追加ワンライナー

qwen.py に追加:

```bash
cat >> /tmp/lalamo/lalamo/model_import/model_specs/qwen.py << 'EOF'

LanguageModelSpec(
    vendor="<ベンダー>",
    family="Qwen3.5",
    name="<モデル名>",
    size="9B",
    origin=HuggingFaceOrigin(repo="<HFリポID>"),
    config_type=HFQwen35Config,
    response_format=QWEN_RESPONSE_FORMAT,
    reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
),
EOF
```

### 9.2 ローカル変換ワンライナー

```bash
cd /tmp/lalamo
JAX_DISABLE_JIT=1 uv run lalamo convert \
  <HFリポID> \
  --output-dir /Users/kohei/Myproject/uzu/models/<出力名> \
  --dtype bfloat16
```

### 9.3 Modal 実行ワンライナー

```bash
# スクリプト作成後
MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run convert_modal.py

# 結果取得
modal volume get uzu-models <出力名>/ /Users/kohei/Myproject/uzu/models/
```

### 9.4 ベンチマークワンライナー

```bash
MODEL="/Users/kohei/Myproject/uzu/models/<出力名>"
for p in "Rustの非同期処理例" "量子コンピューティング説明" "小説冒頭200字"; do
  echo "=== $p ===" && ./target/release/cli \
    --model "$MODEL" --message "$p" 2>&1 | tail -3
done
```

### 9.5 完全自動化スクリプト (model_convert.sh)

`model_convert.sh`:

```bash
#!/bin/bash
# 用法: ./model_convert.sh <HFリポID> <ベンダー> <ファミリ> [出力名] [local|modal] [サイズ]
#
# 出力名を省略すると HF リポID のリポ名をそのまま使う。
#   例: mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-8bit
#       → name="DeepSeek-V4-Pro-Qwen3.5-9B-8bit"
#       → models/DeepSeek-V4-Pro-Qwen3.5-9B-8bit/
# ディレクトリ名だけ短くしたい場合は第4引数で上書きする (name は HF リポ名のまま)。

set -euo pipefail

HF_REPO="$1"
VENDOR="$2"
FAMILY="$3"
OUT_NAME="${4:-${HF_REPO##*/}}"   # ModelSpec の name。既定は HF リポ名
MODE="${5:-local}"
SIZE="${6:-9B}"

# 出力ディレクトリ名。既定は name と同じ
DIR_NAME="${OUT_DIR_NAME:-$OUT_NAME}"

LALAMO_DIR="/tmp/lalamo"
MODELS_DIR="/Users/kohei/Myproject/uzu/models"
OUT_DIR="$MODELS_DIR/$DIR_NAME"

# ファミリ → (spec モジュール, ConfigType, ResponseFormat)
# 3.4 のマッピング表と一致させること。新ファミリはここに 1 行追加する。
# 文字列から機械的に組み立てない (Qwen3.5 → qwen3.5.py / QWEN3.5_RESPONSE_FORMAT になり壊れる)。
case "$FAMILY" in
  "Qwen3.5") SPEC_MODULE="qwen";    CONFIG_TYPE="HFQwen35Config";  RESPONSE_FORMAT="QWEN_RESPONSE_FORMAT" ;;
  "Qwen3")   SPEC_MODULE="qwen";    CONFIG_TYPE="HFQwen3Config";   RESPONSE_FORMAT="THINKING_RESPONSE_FORMAT" ;;
  "Qwen2.5") SPEC_MODULE="qwen";    CONFIG_TYPE="HFQwen2Config";   RESPONSE_FORMAT="THINKING_RESPONSE_FORMAT" ;;
  "Llama3")  SPEC_MODULE="llama";   CONFIG_TYPE="HFLlamaConfig";   RESPONSE_FORMAT="LLAMA_RESPONSE_FORMAT" ;;
  "Gemma3")  SPEC_MODULE="gemma";   CONFIG_TYPE="HFGemma3Config";  RESPONSE_FORMAT="GEMMA_RESPONSE_FORMAT" ;;
  "Mistral") SPEC_MODULE="mistral"; CONFIG_TYPE="HFMistralConfig"; RESPONSE_FORMAT="MISTAL_RESPONSE_FORMAT" ;;
  *) echo "未対応ファミリ: $FAMILY — この case に 1 行追加してください" >&2; exit 1 ;;
esac

SPEC_FILE="$LALAMO_DIR/lalamo/model_import/model_specs/${SPEC_MODULE}.py"

# ModelSpec 本体 (local / modal で共用)
MODELSPEC=$(cat << EOF

LanguageModelSpec(
    vendor="$VENDOR",
    family="$FAMILY",
    name="$OUT_NAME",
    size="$SIZE",
    origin=HuggingFaceOrigin(repo="$HF_REPO"),
    config_type=$CONFIG_TYPE,
    response_format=$RESPONSE_FORMAT,
    reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
),
EOF
)

# 1. ModelSpec 追加 (ローカル lalamo ツリー。二重登録はスキップ)
# 注: repo= で判定する。name を変えて再実行しても二重登録しない。
if grep -q "repo=\"$HF_REPO\"" "$SPEC_FILE" 2>/dev/null; then
    echo "ModelSpec 登録済み: $HF_REPO (スキップ)"
else
    printf '%s\n' "$MODELSPEC" >> "$SPEC_FILE"
fi

# 2. 変換
if [[ "$MODE" == "modal" ]]; then
    # Modal コンテナは lalamo を新規 clone するので、spec は base64 で持ち込んで書き込む
    MODELSPEC_B64=$(printf '%s' "$MODELSPEC" | base64)

    cat > /tmp/convert_$$.py << PYEOF
import base64, os, subprocess, sys
import modal

app = modal.App("convert-$OUT_NAME")
app.image_builder_version = "2025.06"
image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("git").pip_install("uv")
    .run_commands("git clone https://github.com/trymirai/lalamo.git /lalamo",
                  "cd /lalamo && uv sync --frozen"))
vol = modal.Volume.from_name("uzu-models", create_if_missing=True)

SPEC_PATH = "/lalamo/lalamo/model_import/model_specs/${SPEC_MODULE}.py"
SPEC_B64 = "$MODELSPEC_B64"

@app.function(image=image, cpu=16, memory=65536, timeout=14400, volumes={"/models": vol})
def convert():
    sys.path.insert(0, "/lalamo")
    os.chdir("/lalamo")

    # 1. ModelSpec をコンテナ内の lalamo ツリーに書き込む
    spec = base64.b64decode(SPEC_B64).decode("utf-8")
    with open(SPEC_PATH, "a") as f:
        f.write(spec)

    # 2. リロードして登録を確認
    import importlib
    importlib.reload(importlib.import_module("lalamo.model_import.model_specs.${SPEC_MODULE}"))
    mr = importlib.reload(importlib.import_module("lalamo.model_registry"))
    registry = mr.ModelRegistry.init()
    assert "$HF_REPO" in registry.repo_to_model, "ModelSpec not registered!"

    # 3. 変換実行
    cmd = ["uv", "run", "lalamo", "convert", "$HF_REPO",
           "--output-dir", "/models/$DIR_NAME", "--dtype", "bfloat16"]
    env = os.environ.copy()
    env["JAX_DISABLE_JIT"] = "1"
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed: {result.stderr}")

    vol.commit()
    return {"status": "success", "files": os.listdir("/models/$DIR_NAME")}

@app.local_entrypoint()
def main():
    print(convert.remote())
PYEOF

    MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run /tmp/convert_$$.py
    modal volume get uzu-models "$DIR_NAME/" "$MODELS_DIR/"
else
    cd "$LALAMO_DIR"
    JAX_DISABLE_JIT=1 uv run lalamo convert "$HF_REPO" --output-dir "$OUT_DIR" --dtype bfloat16
fi

# 3. 確認
ls -la "$OUT_DIR/"
echo "Done: $OUT_DIR"

# 実行例
chmod +x model_convert.sh

# name = DeepSeek-V4-Pro-Qwen3.5-9B-8bit (HF リポ名がそのまま入る)
./model_convert.sh mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-8bit mlx-community Qwen3.5

# ディレクトリだけ短くしたい場合
OUT_DIR_NAME=deepseek-v4-9b \
  ./model_convert.sh mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-8bit mlx-community Qwen3.5

# Modal で回す (第5引数)
./model_convert.sh mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-8bit mlx-community Qwen3.5 "" modal
```

> このスクリプトは `generation_config.json` を補わないので、それを持たない
> リポジトリ (例: `mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-8bit`) では 404 で落ちる。
> その場合は 5.2 のスクリプトを使う。

---

## 付録: よく使うモデル一覧 (2026年11月時点)

`lalamo` 列は **実際に変換できるか**。✅ は実測済み、❌ は不可と確定。

### 無検閲・量子化済み・Qwen3.5-9B 対応

| モデル | 量子化 | サイズ | ベース | lalamo | 備考 |
| --- | --- | --- | --- | --- | --- |
| `mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-8bit` | MLX **affine** 8bit | 9.54GB | heretic | ✅ **実測済** | 1分28秒で変換成功 |
| `Sohailhosseini/DeepSeek-V4-Pro-Qwen3.5-9B-AWQ-W4A16` | `compressed-tensors` | ~8.6GB | heretic | ❌ **不可** | 名前は AWQ だが中身は compressed-tensors |
| `TheCluster/Qwen3.5-9B-Heretic-MLX-mxfp4` | MLX **mxfp4** | ~5.2GB | heretic | ❌ **不可** | affine として誤読される |
| `TheCluster/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-MLX-mxfp4` | MLX **mxfp4** | ~5GB | HauhauCS | ❌ **不可** | 同上 |
| `TheUnderscore/Qwen3.5-9B-The-Defiant-Fable-...-W4A16-AWQ` | 未検証 | ~9GB | DavidAU NEO | ? | 変換前に 1.1.1 で判定すること |

**この表で一番大事な行**: `Sohailhosseini/...-AWQ-W4A16` はリポ名に `AWQ` と
書いてあるが、`config.json` の `quantization_config.quant_method` は
`compressed-tensors`。名前で判断するとハマる。必ず 1.1.1 のコマンドで確認する。

### 公式・投機あり (検閲あり)

| モデル | 投機 | 速度 | 用途 |
| --- | --- | --- | --- |
| `trymirai/Qwen3.5-9B-M` | ✅ DFlash | 62 t/s | メイン運用 |

---

## 更新履歴

- **2026-11-06**: Modal 変換を実測で検証。`mlx-community/DeepSeek-V4-Pro-Qwen3.5-9B-8bit`
  を 1分28秒で変換。1.1.1 (量子化形式の判定) を追加、5.2 を実測版に差し替え、
  4.4 の出力ファイル一覧を修正 (lalamo の出力は 3 ファイル)、付録に可否を明記。

---

このマニュアルは随時更新してください。新しいモデル・手法が出たらテンプレートに追加し、チーム共有しましょう。
