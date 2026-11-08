#!/usr/bin/env python3
"""
uzu_convert.py — 無検閲 Claude 蒸留モデルを uzu へ変換する再利用可能パイプライン.

成功ルート: MLX affine 4bit → lalamo e401e177 → uzu
失敗ルート: GPTQ (DeltaNet非互換) / AutoAWQ (qwen3_5未対応) / compressed-tensors (uzu未対応)

Usage:
    uzu_convert.py --repo <HF_REPO> --out <OUT_NAME> <subcommand>

Subcommands:
    prereq      環境前提確認
    dry-run     入力検証 + 実行計画表示
    convert     Modal で lalamo 変換 (CPU function)
    fix-config  ローカル config.json 修正
    add-encoding encoding.json 設置
    get-local   Modal Volume からローカル取得
    sanity      モデル構造検証
    run         uzu CLI で会話テスト
    full        全工程実行
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LALAMO_COMMIT = "e401e177"
DEFAULT_MODAL_VOLUME = "uzu-models"
DEFAULT_MODEL_SIZE = "4B"
SUCCESSED_QUANTIZATION = [
    "mlx",
    "4bit",
]
DISALLOWED_QUANTIZATION: dict[str, str] = {
    "gptq": "GPTQはDeltaNet構造と互換性がありません.\n"
            "lalamoがqweight/g_idx/scalesをDeltaNet用配置に展開できません.\n"
            "対策: MLX affine 4bit を使用してください.",
    "awq": "AutoAWQ は qwen3_5 モデル構造に対応していません.\n"
           "エラー: TypeError: qwen3_5 isn't supported yet.\n"
           "対策: MLX affine 4bit を使用してください.",
    "compressed-tensors": "llm-compressor が出力する compressed-tensors 形式は uzu で読み込めません.\n"
                          "uzmu が読む独自形式ではないため、追加の専用変換が必要です.\n"
                          "対策: MLX affine 4bit を使用してください.",
}
DEPRECATED_QUANTIZATION: dict[str, str] = {
    "8bit": "MLX 8bit は Metal カーネル非対応の可能性があります.\n"
            "対策: MLX affine 4bit を使用してください.",
}
SUPPORTED_FAMILY_KEYWORDS = ["qwen3.5", "qwen3-5", "qwen3_5"]
FALLEN_BACK_GENERATION_CONFIG_REPO = "Qwen/Qwen3.5-27B"

# ---------------------------------------------------------------------------
# Quantization detection
# ---------------------------------------------------------------------------

def _fetch_hf_json(url: str) -> Optional[dict]:
    """Fetch a public Hugging Face JSON file. Return None on network/parse error."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _sum_safetensors_bytes(files: list[dict]) -> int:
    total = 0
    for item in files:
        if item.get("type") != "file":
            continue
        path = item.get("path", "")
        if path.endswith(".safetensors"):
            lfs = item.get("lfs") or {}
            total += int(lfs.get("size") or item.get("size") or 0)
    return total


def _has_mlx_quant_config(config: Optional[dict]) -> bool:
    if not config:
        return False
    # MLX quantization config typically contains quant_model, quantization_config,
    # or per-layer config in text_config.
    if config.get("quantization_config"):
        return True
    if config.get("quant_model"):
        return True
    text_config = config.get("text_config", {})
    if text_config.get("quantization"):
        return True
    if text_config.get("quantization_config"):
        return True
    return False


def verify_repo_quantization(repo_id: str, declared_mode: str) -> tuple[bool, Optional[str], Optional[int]]:
    """Verify that the repo actually contains quantized weights.

    Returns:
        (passed, message, total_bytes)
    """
    lower = repo_id.lower()
    is_mlx_4bit = "mlx" in lower and "4bit" in lower
    if not is_mlx_4bit:
        return True, None, None

    files = _fetch_hf_json(f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true")
    config = _fetch_hf_json(f"https://huggingface.co/{repo_id}/resolve/main/config.json")
    if not files or not isinstance(files, list):
        return False, f"Hugging Faceのファイル一覧を取得できません: {repo_id}", None

    total_bytes = _sum_safetensors_bytes(files)

    # A genuine MLX affine 4-bit Qwen3.5 model should be far smaller than BF16.
    # 4B ~ 2.4GB, 9B ~ 5GB. Anything > 12GB is almost certainly full BF16.
    bf16_threshold = 12 * 1024**3
    has_quant_config = _has_mlx_quant_config(config)
    if total_bytes >= bf16_threshold and not has_quant_config:
        return False, (
            f"{repo_id} は '4bit' と名乗っていますが、実体はBF16フル精度です。\n"
            f"safetensors合計: {total_bytes / 1e9:.2f} GB\n"
            f"quantization_config: なし\n"
            "M3 16GBでは実行不可。先にMLX affine 4bitへ量子化する必要があります。"
        ), total_bytes

    if total_bytes > 0:
        return True, None, total_bytes

    return False, f"{repo_id} に .safetensors ファイルがありません", None


def detect_quantization(repo_id: str) -> tuple[bool, Optional[str], Optional[str]]:
    """repo_id から量子化方式を判定.

    Returns:
        (supported, mode_string, error_message)
        supported=True なら使用可能.
        supported=False なら error_message に原因.
    """
    lower = repo_id.lower()

    # Check supported
    has_mlx = "mlx" in lower
    has_4bit = "4bit" in lower

    if has_mlx and has_4bit:
        return True, "mlx-affine-4bit", None

    # Check disallowed
    for kw, err in DISALLOWED_QUANTIZATION.items():
        if kw in lower:
            return False, kw, err

    # Check deprecated
    if "8bit" in lower and not has_4bit:
        return False, "8bit", DEPRECATED_QUANTIZATION["8bit"]

    # BF16 / FP16
    if any(kw in lower for kw in ["bf16", "fp16", "bfloat"]):
        return True, "bf16", (
            "BF16/FP16 は未検証です. メモリ使用量が大きくなる可能性があります.\n"
            "4B: 約 8GB, 9B: 約 18GB の見込み."
        )

    # Unknown
    return True, "unknown", (
        f"量子化方式 '{repo_id}' が不明です. サポート済み: MLX affine 4bit.\n"
        "安全のため MLX 4bit モデルを再確認してください."
    )


# ---------------------------------------------------------------------------
# Model name / size extraction
# ---------------------------------------------------------------------------

def extract_model_name(repo_id: str) -> str:
    repo_id.strip()
    parts = repo_id.split("/")
    name = parts[-1] if parts else repo_id
    # Strip common suffixes for clean name
    for suffix in ("-4bit", "-8bit", "-MLX", "-uzu", "-e401e177"):
        name = name.replace(suffix, "")
    return name


def extract_size(repo_id: str) -> str:
    repo_id_stripped = repo_id.strip()
    parts = repo_id_stripped.split("/")
    name = parts[-1] if parts else repo_id_stripped
    size_match = re.search(r"(\d+[MB])", name)
    if size_match:
        return size_match.group(1)
    # Infer from --model-size
    return DEFAULT_MODEL_SIZE


# ---------------------------------------------------------------------------
# Config fix
# ---------------------------------------------------------------------------

def fix_config(config_path: str) -> bool:
    """lalamo 出力の config.json を uzu 互換に修正.

    Returns:
        True if file was modified.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")

    with open(path) as f:
        cfg = json.load(f)

    changed = False
    tc = cfg.get("token_codec_config", {})

    # Remove unknown fields
    if "response_format" in tc:
        del tc["response_format"]
        changed = True

    # Ensure required null fields
    if "output_parser_regex" not in tc:
        tc["output_parser_regex"] = None
        changed = True
    if "end_of_thinking_tag" not in tc:
        tc["end_of_thinking_tag"] = None
        changed = True
    if "bos_token" not in tc:
        tc["bos_token"] = None
        changed = True
    if "eos_token" not in tc:
        tc["eos_token"] = None
        changed = True

    # Clean decoder_config
    dc = cfg.get("decoder_config", {})
    allowed_decoder_keys = {
        "embedding_config",
        "transformer_config",
        "vocab_size",
        "ple_model_config",
        "embedding_norm_config",
    }
    for key in list(dc.keys()):
        if key not in allowed_decoder_keys:
            del dc[key]
            changed = True

    if changed:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"FIXED {config_path}")
    else:
        print(f"OK    {config_path}")

    return changed


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

ENCODE_JSON_DEFAULT = {
    "type": "hanashi",
    "name": "qwen3.5",
}


def add_encoding(model_dir: str, encoding_data: Optional[dict] = None) -> Path:
    """encoding.json を設置."""
    encoding = encoding_data or ENCODE_JSON_DEFAULT
    path = Path(model_dir) / "encoding.json"
    with open(path, "w") as f:
        json.dump(encoding, f, ensure_ascii=False, indent=2)
    print(f"CREATED {path}")
    return path


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def run_sanity(model_dir: str) -> List[str]:
    """モデルディレクトリの構造を検証.

    Returns:
        エラーメッセージのリスト. 空なら OK.
    """
    errors = []
    p = Path(model_dir)

    # 1. config.json
    cfg_path = p / "config.json"
    if not cfg_path.exists():
        errors.append(f"config.json not found: {cfg_path}")
    else:
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"config.json parse error: {e}")
            cfg = None

        if cfg:
            # Check top-level keys
            for key in ("type", "decoder_config", "token_codec_config", "generation_config"):
                if key not in cfg:
                    errors.append(f"config.json missing key: {key}")

            # Check decoder_config
            dc = cfg.get("decoder_config", {})
            for dk in ("embedding_config", "transformer_config"):
                if dk not in dc:
                    errors.append(f"decoder_config missing: {dk}")

            # Check token_codec_config
            tc = cfg.get("token_codec_config", {})
            if tc.get("type") is None:
                errors.append("token_codec_config.type is None")

    # 2. encoding.json
    enc_path = p / "encoding.json"
    if not enc_path.exists():
        errors.append(f"encoding.json not found: {enc_path}")
    else:
        try:
            with open(enc_path) as f:
                enc = json.load(f)
            if isinstance(enc, list):
                enc = enc[0] if enc else {}
            if enc.get("type") != "hanashi":
                errors.append(f"encoding.json type must be 'hanashi', got: {enc.get('type')}")
            if not enc.get("name"):
                errors.append("encoding.json name is empty")
        except json.JSONDecodeError as e:
            errors.append(f"encoding.json parse error: {e}")

    # 3. model.safetensors
    st_path = p / "model.safetensors"
    if not st_path.exists():
        errors.append(f"model.safetensors not found: {st_path}")
    elif st_path.stat().st_size == 0:
        errors.append(f"model.safetensors is empty: {st_path}")
    else:
        size_gb = st_path.stat().st_size / (1024 ** 3)
        print(f"  model.safetensors: {size_gb:.2f} GB")
        # Rough sanity for 9B: should be < 12GB for 4bit
        if "9B" in model_dir and size_gb > 12:
            errors.append(
                f"9B 4bit model should be < 12GB, got {size_gb:.2f} GB. "
                "Possibly BF16 full precision."
            )

    # 4. tokenizer.json
    tok_path = p / "tokenizer.json"
    if not tok_path.exists():
        errors.append(f"tokenizer.json not found: {tok_path}")

    return errors


# ---------------------------------------------------------------------------
# Prereq check
# ---------------------------------------------------------------------------

def check_prereqs(
    base_dir: str,
    uzu_bin: str,
    modal_volume: str,
) -> List[str]:
    """環境前提を確認.

    Returns:
        不足している前提のリスト. 空なら OK.
    """
    missing = []

    # Python version — check the current interpreter
    if sys.version_info < (3, 10):
        missing.append(f"Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}\n"
                       f"  Use: uv run uzu_convert.py ...  or  python3.12 uzu_convert.py ...")

    # Git
    if shutil.which("git") is None:
        missing.append("git not found in PATH")

    # Modal CLI
    if shutil.which("modal") is None:
        missing.append("modal CLI not found. Install: pip install modal")

    # uzu binary
    uzu_path = Path(base_dir) / uzu_bin
    if not uzu_path.exists():
        missing.append(f"uzu binary not found: {uzu_path}")

    # Modal Volume
    if shutil.which("modal"):
        try:
            result = subprocess.run(
                ["modal", "volume", "list"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0 or modal_volume not in result.stdout:
                missing.append(f"Modal Volume '{modal_volume}' not found.\n"
                               f"  Create: modal volume create {modal_volume}")
        except subprocess.TimeoutExpired:
            missing.append(f"Modal volume check timed out: {modal_volume}")

    return missing


# ---------------------------------------------------------------------------
# Modal convert script generator
# ---------------------------------------------------------------------------

def generate_modal_script(
    repo_id: str,
    out_name: str,
    lalamo_commit: str,
    modal_volume: str,
    model_name: str,
    model_size: str,
) -> str:
    """Modal 上で実行する変換スクリプトを生成 (proper Modal App 構造)."""

    script = f'''#!/usr/bin/env python3
"""Auto-generated Modal conversion script for uzu pipeline."""
import os
import sys
import json
import shutil

import modal

APP = modal.App("uzu-convert")

LALAMO_COMMIT = "{lalamo_commit}"
REPO = "{repo_id}"
OUT_NAME = "{out_name}"
OUT_DIR = f"/models/{{OUT_NAME}}"
MODAL_VOLUME = "{modal_volume}"
MODEL_NAME = "{model_name}"
MODEL_SIZE = "{model_size}"
FALLEN_BACK_REPO = "{FALLEN_BACK_GENERATION_CONFIG_REPO}"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv", "huggingface-hub")
    .run_commands(
        "git clone https://github.com/trymirai/lalamo.git /lalamo",
        f"cd /lalamo && git checkout {{LALAMO_COMMIT}}",
        'cd /lalamo && uv pip install --system -e ".[cpu]"',
    )
)


@APP.function(
    image=image,
    volumes={"/models": modal.Volume.from_name("{modal_volume}")},
    cpu=16,
    memory=32768,
)
def convert_task():
    """Run lalamo conversion on Modal CPU."""
    print("Setting up lalamo...")
    if os.path.exists("/lalamo"):
        shutil.rmtree("/lalamo")
    os.system(f"git clone https://github.com/trymirai/lalamo.git /lalamo")
    os.chdir("/lalamo")
    os.system(f"git checkout {{LALAMO_COMMIT}}")
    print(f"Cloned lalamo at commit {{LALAMO_COMMIT}}")

    print(f"Downloading model from {{REPO}}...")
    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download(REPO, cache_dir="/hf_cache")
    print(f"Downloaded to {{cache_dir}}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Running lalamo conversion...")
    print(f"  repo: {{REPO}}")
    print(f"  out:  {{OUT_DIR}}")
    print(f"  commit: {{LALAMO_COMMIT}}")

    # Try lalamo CLI first
    result = os.system(
        f"cd /lalamo && python3 -m lalamo convert "
        f"--repo {{REPO}} "
        f"--out {{OUT_DIR}} "
        f"--lalamo-commit {{LALAMO_COMMIT}} "
        f"--fallback-generation-config-repo {{FALLEN_BACK_REPO}}"
    )

    if result != 0:
        print(f"lalamo CLI conversion failed with exit code {{result}}")
        print("Falling back to Python API...")
        try:
            _run_python_api_conversion(OUT_DIR)
        except Exception as e:
            print(f"Python API conversion also failed: {{e}}")
            sys.exit(1)

    print(f"Conversion complete. Files in {{OUT_DIR}}:")
    for f in os.listdir(OUT_DIR):
        fpath = os.path.join(OUT_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {{f}}: {{size_mb:.1f}} MB")
    print("DONE")


def _run_python_api_conversion(out_dir: str) -> None:
    """Fall back to lalamo Python API."""
    from lalamo.registry import convert_model
    from lalamo.configs import HFQwen35Config
    from lalamo.specs import (
        LanguageModelSpec,
        HuggingFaceOrigin,
        QWEN_RESPONSE_FORMAT,
        BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    )
    from lalamo.specs.file_spec import FileSpec, ConfigMap

    spec = LanguageModelSpec(
        vendor="Huihui",
        family="Qwen3.5",
        name=MODEL_NAME,
        size=MODEL_SIZE,
        origin=HuggingFaceOrigin(repo=REPO),
        config_type=HFQwen35Config,
        response_format=QWEN_RESPONSE_FORMAT,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json"),
            tokenizer_config=FileSpec("tokenizer_config.json"),
            chat_template=FileSpec("chat_template.jinja"),
            generation_config=FileSpec(
                "generation_config.json",
                FALLEN_BACK_GENERATION_CONFIG_REPO,
            ),
        ),
    )

    result = convert_model(spec)
    print(f"Python API result: {{result}}")


if __name__ == "__main__":
    APP.run()
'''
    return script


def generate_bf16_modal_script(
    repo_id: str,
    mlx_out_name: str,
    uzu_out_name: str,
    lalamo_commit: str,
    modal_volume: str,
    model_name: str,
    model_size: str,
) -> str:
    """Modal 上で BF16 -> MLX affine 4bit -> uzu 変換を一括実行するスクリプト."""

    script = f'''#!/usr/bin/env python3
"""Auto-generated Modal script: BF16 -> MLX affine 4bit -> uzu."""
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import modal

APP = modal.App("uzu-bf16-to-uzu")

SOURCE_REPO = "{repo_id}"
MLX_OUT_NAME = "{mlx_out_name}"
UZU_OUT_NAME = "{uzu_out_name}"
LALAMO_COMMIT = "{lalamo_commit}"
MODAL_VOLUME = "{modal_volume}"
MODEL_NAME = "{model_name}"
MODEL_SIZE = "{model_size}"

volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv", "mlx-lm", "mlx[cpu]", "huggingface-hub")
    .run_commands(
        "git clone https://github.com/trymirai/lalamo.git /lalamo",
        f"cd /lalamo && git checkout {{LALAMO_COMMIT}}",
        'cd /lalamo && uv pip install --system -e ".[cpu]"',
    )
)


@APP.function(
    image=image,
    volumes={{"/models": volume}},
    cpu=16,
    memory=65536,
    timeout=3600 * 2,
)
def convert() -> dict:
    mlx_dir = f"/models/{{MLX_OUT_NAME}}"
    uzu_dir = f"/models/{{UZU_OUT_NAME}}"

    for d in (mlx_dir, uzu_dir):
        print(f"Removing if exists: {{d}}")
        shutil.rmtree(d, ignore_errors=True)

    print(f"[1/2] Quantizing {{SOURCE_REPO}} -> {{mlx_dir}}")
    quant_cmd = [
        "python3", "-m", "mlx_lm", "convert",
        "--hf-path", SOURCE_REPO,
        "--mlx-path", mlx_dir,
        "-q",
        "--q-bits", "4",
        "--q-group-size", "64",
        "--dtype", "float16",
    ]
    result = subprocess.run(quant_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"mlx_lm.convert failed: {{result.returncode}}")

    print("[1/2] Quantized output:")
    for f in sorted(os.listdir(mlx_dir)):
        fpath = os.path.join(mlx_dir, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {{f}}: {{size_mb:.1f}} MB")

    fallback_generation_config = Path(mlx_dir) / "generation_config.json"
    if not fallback_generation_config.exists():
        print("Downloading generation_config.json from Qwen/Qwen3.5-27B...")
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="Qwen/Qwen3.5-27B",
            filename="generation_config.json",
            local_dir_use_symlinks=False,
        )
        shutil.copy(downloaded, fallback_generation_config)
        print(f"Downloaded: {{fallback_generation_config}}")

    print(f"[2/2] Converting {{mlx_dir}} -> {{uzu_dir}}")
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    sys.path.insert(0, "/lalamo")

    from lalamo.commands import DType, convert as lalamo_convert
    from lalamo.model_import.model_configs import HFQwen35Config
    from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
    from lalamo.model_import.model_specs.qwen import QWEN_RESPONSE_FORMAT
    from lalamo.model_import.model_specs.reasoning_configs import (
        BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    )
    from lalamo.model_import.origins import LocalOrigin

    output_dir = Path(uzu_dir)
    spec = LanguageModelSpec(
        vendor="Huihui",
        family="Qwen3.5",
        name=MODEL_NAME,
        size=MODEL_SIZE,
        origin=LocalOrigin(
            root=mlx_dir,
            weight_files=("model.safetensors",),
        ),
        config_type=HFQwen35Config,
        response_format=QWEN_RESPONSE_FORMAT,
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
        configs=ConfigMap(
            tokenizer=FileSpec("tokenizer.json"),
            tokenizer_config=FileSpec("tokenizer_config.json"),
            generation_config=FileSpec("generation_config.json", "Qwen/Qwen3.5-27B"),
            chat_template=FileSpec("chat_template.jinja"),
        ),
    )

    print(f"[spec] {{spec.vendor}}/{{spec.name}} <- {{spec.origin.description}}", flush=True)

    try:
        lalamo_convert(spec, output_dir, DType.BFLOAT16)
    except Exception:
        traceback.print_exc()
        return {{"status": "failed", "traceback": traceback.format_exc()[-4000:]}}

    volume.commit()

    print("Cleaning up intermediate MLX directory...")
    shutil.rmtree(mlx_dir, ignore_errors=True)
    volume.commit()

    files = sorted(p.name for p in output_dir.iterdir())
    sizes = {{p.name: p.stat().st_size for p in output_dir.iterdir() if p.is_file()}}
    total = sum(sizes.values())
    print(f"[done] {{len(files)}} files, {{total / 1e9:.2f}} GB", flush=True)
    for name in files:
        print(f"  {{name:40s}} {{sizes.get(name, 0) / 1e6:10.1f}} MB", flush=True)

    return {{
        "status": "success",
        "output_dir": str(output_dir),
        "files": files,
        "total_bytes": total,
    }}


@APP.local_entrypoint()
def main() -> None:
    print(convert.remote(), flush=True)
'''
    return script


# ---------------------------------------------------------------------------
# CLI subcommand implementations
# ---------------------------------------------------------------------------

def cmd_prereq(args: argparse.Namespace) -> int:
    """環境前提確認."""
    print("=== Prerequisite Check ===")
    missing = check_prereqs(args.base_dir, args.uzu_bin, args.modal_volume)
    if missing:
        print("\nFAIL — missing prerequisites:")
        for m in missing:
            print(f"  - {m}")
        return 1
    print("\nOK — all prerequisites met.")
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    """入力検証 + 実行計画表示."""
    print("=== Dry Run ===")
    print(f"  source_kind:    {args.source_kind}")

    if args.source_kind == "hf-bf16":
        print("\nPLAN: BF16 source -> MLX affine 4bit quantization -> lalamo -> uzu")
        print(f"  repo:           {args.repo}")
        print(f"  out_dir:        models/{args.out}")
        print(f"  model_size:     {args.model_size}")
        print(f"  lalamo_commit:  {args.lalamo_commit}")
        print(f"  modal_volume:   {args.modal_volume}")
        print(f"\nSteps: convert (quantize+uzu) → fix-config → add-encoding → get-local → sanity")
        return 0

    if args.source_kind == "modal-mlx-4bit":
        print("\nPLAN: Modal Volume source -> lalamo -> uzu")
        print(f"  repo:           {args.repo}")
        print(f"  out_dir:        models/{args.out}")
        print(f"\nSteps: convert → fix-config → add-encoding → get-local → sanity")
        return 0

    # Quantization check from repo id
    supported, mode, msg = detect_quantization(args.repo)
    if not supported:
        print(f"\nQUANTIZATION DENIED ({mode}):")
        print(msg)
        return 1
    if msg:
        print(f"\nWARNING ({mode}): {msg}")

    # Deep verification against Hugging Face metadata
    ok, verify_msg, total_bytes = verify_repo_quantization(args.repo, mode)
    if not ok:
        print(f"\nREPO VERIFICATION DENIED:")
        print(verify_msg)
        return 1
    if total_bytes:
        print(f"\nREPO VERIFIED: safetensors total {total_bytes / 1e9:.2f} GB")

    print(f"\nPlan:")
    print(f"  repo:           {args.repo}")
    print(f"  out_dir:        models/{args.out}")
    print(f"  model_size:     {args.model_size}")
    print(f"  lalamo_commit:  {args.lalamo_commit}")
    print(f"  modal_volume:   {args.modal_volume}")
    print(f"  quantum_mode:   {mode}")
    print(f"\nSteps: convert → fix-config → add-encoding → get-local → sanity")
    return 0


def _run_modal_script(script: str, args: argparse.Namespace, description: str) -> int:
    """Write a Modal script to a temp file and run `modal run`."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(script)
        script_path = f.name

    print(f"=== {description} ===")
    print(f"  repo:           {args.repo}")
    print(f"  out:            {args.out}")
    print(f"  lalamo_commit:  {args.lalamo_commit}")
    print(f"  modal_volume:   {args.modal_volume}")
    print(f"  script:         {script_path}")

    if args.safe:
        confirm = input(f"Deploy to Modal? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("Aborted by user.")
            return 0

    try:
        result = subprocess.run(
            ["modal", "run", script_path],
            check=False, capture_output=False,
        )
        if result.returncode != 0:
            print(f"\n{description} FAILED with exit code {result.returncode}")
            return 1
        print(f"\n{description} completed.")
        return 0
    except FileNotFoundError:
        print("\nmodal CLI not found. Install: pip install modal")
        return 1
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def cmd_convert(args: argparse.Namespace) -> int:
    """Modal で lalamo 変換を実行."""
    repo_id = args.repo
    out_name = args.out
    lalamo_commit = args.lalamo_commit
    modal_volume = args.modal_volume
    model_name = extract_model_name(repo_id)
    model_size = extract_size(args.repo)

    # BF16 source uses the combined quantize + convert pipeline
    if args.source_kind == "hf-bf16":
        mlx_out_name = f"{out_name}-mlx-affine-4bit-g64"
        modal_script = generate_bf16_modal_script(
            repo_id, mlx_out_name, out_name, lalamo_commit,
            modal_volume, model_name, model_size,
        )
        return _run_modal_script(modal_script, args, "Convert BF16 -> MLX 4bit -> uzu (Modal)")

    # Re-run repo verification before spending Modal time/credits
    _, mode, _ = detect_quantization(repo_id)
    ok, verify_msg, total_bytes = verify_repo_quantization(repo_id, mode)
    if not ok:
        print("=== Convert ===")
        print(f"\nREPO VERIFICATION DENIED:")
        print(verify_msg)
        return 1
    if total_bytes:
        print(f"REPO VERIFIED: safetensors total {total_bytes / 1e9:.2f} GB")

    # Generate Modal script
    modal_script = generate_modal_script(
        repo_id, out_name, lalamo_commit, modal_volume,
        model_name, model_size,
    )
    return _run_modal_script(modal_script, args, "Convert (Modal)")


def cmd_fix_config(args: argparse.Namespace) -> int:
    """ローカル config.json 修正."""
    model_dir = Path(args.base_dir) / "models" / args.out
    config_path = model_dir / "config.json"

    print(f"=== Fix Config ===")
    print(f"  path: {config_path}")

    try:
        changed = fix_config(str(config_path))
        if not changed:
            print("  No changes needed.")
        return 0
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}")
        return 1


def cmd_add_encoding(args: argparse.Namespace) -> int:
    """encoding.json 設置."""
    model_dir = Path(args.base_dir) / "models" / args.out

    print(f"=== Add Encoding ===")
    print(f"  dir: {model_dir}")

    try:
        add_encoding(str(model_dir))
        return 0
    except OSError as e:
        print(f"ERROR: {e}")
        return 1


def _check_disk_space(target: Path, needed_gb: float) -> tuple[bool, str]:
    """Check if target directory has enough free space (rough)."""
    try:
        stat = os.statvfs(target)
        free_gb = stat.f_bavail * stat.f_frsize / (1024 ** 3)
        if free_gb < needed_gb:
            return False, (
                f"Disk space insufficient: {free_gb:.1f} GB free, "
                f"need ~{needed_gb:.1f} GB"
            )
        return True, f"{free_gb:.1f} GB free"
    except OSError as e:
        return False, f"Cannot check disk space: {e}"


def cmd_get_local(args: argparse.Namespace) -> int:
    """Modal Volume からローカル取得."""
    modal_volume = args.modal_volume
    out_name = args.out
    target = Path(args.base_dir) / "models"

    print(f"=== Get Local ===")
    print(f"  volume: {modal_volume}")
    print(f"  name:   {out_name}")
    print(f"  target: {target}")

    # Rough size estimate by model name
    size_gb = 2.5 if "4B" in out_name else 6.0
    if "9B" in out_name:
        size_gb = 6.0
    ok, msg = _check_disk_space(target, size_gb)
    if not ok:
        print(f"\nFAILED: {msg}")
        return 1
    print(f"  disk:   {msg}")

    if args.safe:
        confirm = input(f"Get from Modal Volume? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("Aborted by user.")
            return 0

    try:
        result = subprocess.run(
            [
                "modal", "volume", "get", "--force",
                modal_volume, out_name, str(target),
            ],
            check=True, capture_output=False,
        )
        print("\nGet local completed.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nGet local FAILED: {e}")
        return 1
    except FileNotFoundError:
        print("\nmodal CLI not found.")
        return 1


def cmd_sanity(args: argparse.Namespace) -> int:
    """モデル構造検証."""
    model_dir = Path(args.base_dir) / "models" / args.out

    print(f"=== Sanity Check ===")
    print(f"  dir: {model_dir}")

    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}")
        return 1

    errors = run_sanity(str(model_dir))
    if errors:
        print("\nFAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nOK — sanity check passed.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """uzu CLI で会話テスト."""
    model_dir = Path(args.base_dir) / "models" / args.out
    uzu_bin = Path(args.base_dir) / args.uzu_bin
    message = args.message or "テストメッセージ"

    print(f"=== Run (uzu CLI) ===")
    print(f"  model: {model_dir}")
    print(f"  bin:   { uzu_bin}")
    print(f"  msg:   {message}")

    if not uzu_bin.exists():
        print(f"ERROR: uzu binary not found: {uzu_bin}")
        return 1
    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}")
        return 1

    try:
        result = subprocess.run(
            [str(uzu_bin), "--model", str(model_dir), "--message", message],
            check=False, capture_output=True, text=True, timeout=120,
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}", file=sys.stderr)

        if result.returncode == 0:
            print("\nRun OK.")
            return 0
        else:
            print(f"\nRun FAILED with exit code {result.returncode}")
            return 1
    except subprocess.TimeoutExpired:
        print("\nRun TIMEOUT (120s)")
        return 1
    except FileNotFoundError:
        print(f"\nuzu binary not found: {uzu_bin}")
        return 1


def cmd_full(args: argparse.Namespace) -> int:
    """全工程実行."""
    if args.safe:
        confirm = input("Run full pipeline? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("Aborted by user.")
            return 0

    # Steps that run without confirmation (dry-run always OK to run)
    steps = [
        ("prereq", cmd_prereq),
        ("dry-run", cmd_dry_run),
        ("convert", cmd_convert),
        ("fix-config", cmd_fix_config),
        ("add-encoding", cmd_add_encoding),
        ("get-local", cmd_get_local),
        ("sanity", cmd_sanity),
    ]

    for name, fn in steps:
        print(f"\n{'='*40}")
        print(f"  STEP: {name}")
        print(f"{'='*40}")
        ret = fn(args)
        if ret != 0:
            print(f"\nFull pipeline FAILED at step: {name}")
            return ret

    # Optionally run uzu CLI
    if args.run_after:
        print(f"\n{'='*40}")
        print(f"  STEP: run")
        print(f"{'='*40}")
        ret = cmd_run(args)
        if ret != 0:
            print("\nFull pipeline completed with run failure.")
            return ret

    print("\n" + "=" * 40)
    print("  Full pipeline OK")
    print("=" * 40)
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uzu_convert.py",
        description="無検閲 Claude 蒸留モデルを uzu へ変換するパイプライン.",
    )

    # Global flags
    parser.add_argument("--repo", required=True, help="Hugging Face repo ID or Modal Volume path")
    parser.add_argument("--out", required=True, help="Output directory name (under models/)")
    parser.add_argument("--aliases", default="", help="Aliases (comma-separated)")
    parser.add_argument("--lalamo-commit", default=DEFAULT_LALAMO_COMMIT,
                        help=f"lalamo commit (default: {DEFAULT_LALAMO_COMMIT})")
    parser.add_argument("--model-size", default=DEFAULT_MODEL_SIZE, choices=["4B", "9B"],
                        help=f"Model size (default: {DEFAULT_MODEL_SIZE})")
    parser.add_argument("--modal-volume", default=DEFAULT_MODAL_VOLUME,
                        help=f"Modal volume name (default: {DEFAULT_MODAL_VOLUME})")
    parser.add_argument("--uzu-bin", default="./target/release/cli",
                        help="uzu CLI binary path")
    parser.add_argument("--base-dir", default=None,
                        help="Base directory (default: script's parent)")
    parser.add_argument("--safe", action="store_true",
                        help="Interactive confirmation for destructive steps")
    parser.add_argument("--message", default=None,
                        help="Message to test with uzu CLI (for run step)")
    parser.add_argument("--run-after", action="store_true",
                        help="Run uzu CLI after full pipeline")
    parser.add_argument("--source-kind", default="hf-mlx-4bit",
                        choices=["hf-mlx-4bit", "hf-bf16", "modal-mlx-4bit"],
                        help="Source model kind (default: hf-mlx-4bit)")

    return parser


def main() -> int:
    base_dir = os.environ.get("UZU_BASE_DIR", None)

    parser = build_parser()

    # Detect subcommand from raw argv
    subcommands = {
        "prereq": cmd_prereq,
        "dry-run": cmd_dry_run,
        "convert": cmd_convert,
        "fix-config": cmd_fix_config,
        "add-encoding": cmd_add_encoding,
        "get-local": cmd_get_local,
        "sanity": cmd_sanity,
        "run": cmd_run,
        "full": cmd_full,
    }

    # Flags that consume the next token as a value
    flag_with_value = {
        "--repo", "--out", "--aliases", "--lalamo-commit",
        "--model-size", "--modal-volume", "--uzu-bin",
        "--base-dir", "--message", "--source-kind",
    }

    subcmd = None
    subcmd_idx = None
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in flag_with_value:
            i += 2  # skip flag and its value
        elif arg.startswith("-"):
            i += 1  # skip flag (no value, e.g. --safe, --help)
        elif subcmd is None:
            subcmd = arg
            subcmd_idx = i
            break
        else:
            i += 1  # extra positional, skip

    if subcmd and subcmd in subcommands:
        # Remove only the subcommand token
        remaining = sys.argv[:subcmd_idx] + sys.argv[subcmd_idx + 1:]
        args, _ = parser.parse_known_args(remaining)
    elif len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        parser.print_help()
        print(f"\nError: unknown subcommand '{sys.argv[1]}'")
        print(f"Available: {', '.join(subcommands.keys())}")
        return 2
    else:
        parser.print_help()
        return 0

    if base_dir:
        args.base_dir = base_dir
    else:
        args.base_dir = str(Path(__file__).resolve().parent.parent)

    args.base_dir = str(Path(args.base_dir).resolve())

    if subcmd is None:
        parser.print_help()
        return 0

    return subcommands[subcmd](args)


if __name__ == "__main__":
    sys.exit(main())