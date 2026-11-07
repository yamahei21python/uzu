#!/usr/bin/env python3
"""Modal pipeline for 9B RHT + INT4 + YAQA conversion.

Local work stays local. Only scripts and a small calibration corpus are baked
into the image; the 18GB BF16 source is downloaded inside Modal from Hugging
Face, and the INT4 output is written to a Modal Volume.

Setup:
  modal secret create huggingface HF_TOKEN=...
  modal volume create uzu-quant

Commands:
  modal run scripts/modal_pipeline.py --command import-9b
  modal run scripts/modal_pipeline.py --command rtn-9b --quantize-embedding
  modal run scripts/modal_pipeline.py --command calibrate-9b --samples 8 --max-tokens 512
  modal run scripts/modal_pipeline.py --command yaqa-9b --quantize-embedding
  modal run scripts/modal_pipeline.py --command fetch-9b
"""

from __future__ import annotations

import argparse
import os
import subprocess

import modal

APP_NAME = "uzu-quant-pipeline"
HF_REPO_9B = "huihui-ai/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated"
SPEC_9B = "Qwen3.5-9B"

source_volume = modal.Volume.from_name("uzu-models", create_if_missing=True)
output_volume = modal.Volume.from_name("uzu-quant", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/trymirai/lalamo /opt/lalamo",
        "cd /opt/lalamo && git checkout e401e17761f2fcd337b0cf20218d67d656e6ecba",
        "pip install -e '/opt/lalamo[cuda]'",
    )
    .env({"HF_HUB_ENABLE_XET": "0", "PYTHONUNBUFFERED": "1", "JAX_PLATFORM_NAME": "cpu"})
    .add_local_file("scripts/rht_int4_convert.py", "/workspace/scripts/rht_int4_convert.py")
    .add_local_file("scripts/rewrite_lalamo_hybrid_spec.py", "/workspace/scripts/rewrite_lalamo_hybrid_spec.py")
    .add_local_file("scripts/normalize_lalamo_config.py", "/workspace/scripts/normalize_lalamo_config.py")
    .add_local_file("scripts/yaqa_calibration.py", "/workspace/scripts/yaqa_calibration.py")
    .add_local_file("scripts/make_lalamo_eval_copy.py", "/workspace/scripts/make_lalamo_eval_copy.py")
    .add_local_file("scripts/merge_calibration_groups.py", "/workspace/scripts/merge_calibration_groups.py")
    .add_local_file("scripts/corpus_guard.py", "/workspace/scripts/corpus_guard.py")
    .add_local_file("scripts/compare_logits.py", "/workspace/scripts/compare_logits.py")
    .add_local_file(
        "experiments/qwen35-0.8b/data/calibration_en_wikitext2_dedup.txt",
        "/workspace/data/calibration_en_dedup.txt",
    )
    .add_local_file(
        "experiments/qwen35-0.8b/data/eval_en_wikitext2.txt",
        "/workspace/data/eval_en.txt",
    )
    .add_local_file(
        "experiments/qwen35-0.8b/data/eval_ja_wikipedia.txt",
        "/workspace/data/eval_ja.txt",
    )
)

app = modal.App(APP_NAME)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ensure_bf16_eval(run_name: str) -> str:
    """Return a lalamo-loadable copy of the import_9b output.

    import_9b normalizes config.json for uzu, so lalamo cannot load that
    directory directly. make_lalamo_eval_copy restores config.lalamo.json and
    hardlinks the 18GB weights.
    """
    source = f"/out/{run_name}/bf16"
    dest = f"/out/{run_name}/bf16-lalamo"
    if not os.path.exists(f"{dest}/config.json"):
        run(
            [
                "python",
                "/workspace/scripts/make_lalamo_eval_copy.py",
                "--source",
                source,
                "--dest",
                dest,
            ]
        )
        output_volume.commit()
    return dest


@app.function(image=image, cpu=2, memory=4096, timeout=900)
def check_env() -> str:
    import jax
    import lalamo

    return f"lalamo={lalamo.__file__} jax={jax.__version__} devices={jax.devices()}"


@app.function(
    image=image,
    volumes={"/models": source_volume, "/out": output_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    cpu=16,
    memory=65536,
    timeout=7200,
)
def import_9b(run_name: str) -> str:
    """Download HF 9B and save a BF16 lalamo model to /out/<run>/bf16."""
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if os.path.exists(f"/out/{run_name}/bf16/COMPLETE"):
        print(f"skip import: /out/{run_name}/bf16/COMPLETE exists", flush=True)
        return f"/out/{run_name}/bf16"
    run(
        [
            "python",
            "/workspace/scripts/rht_int4_convert.py",
            "--spec-name",
            SPEC_9B,
            "--hf-repo",
            HF_REPO_9B,
            "--lalamo-path",
            "/opt/lalamo",
            "--bf16-dir",
            f"/out/{run_name}/bf16",
            "--output-dir",
            f"/out/{run_name}/bf16-import",
            "--dry-run",
        ]
    )
    output_volume.commit()
    return f"/out/{run_name}/bf16"


@app.function(
    image=image,
    volumes={"/models": source_volume, "/out": output_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    cpu=16,
    memory=65536,
    timeout=7200,
)
def rtn_9b(run_name: str, quantize_embedding: bool) -> str:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if os.path.exists(f"/out/{run_name}/int4-rtn/COMPLETE"):
        print(f"skip rtn: /out/{run_name}/int4-rtn/COMPLETE exists", flush=True)
        return f"/out/{run_name}/int4-rtn"
    bf16 = ensure_bf16_eval(run_name)
    command = [
        "python",
        "/workspace/scripts/rht_int4_convert.py",
        "--load-from",
        bf16,
        "--lalamo-path",
        "/opt/lalamo",
        "--output-dir",
        f"/out/{run_name}/int4-rtn",
    ]
    if quantize_embedding:
        command.append("--quantize-embedding")
    run(command)
    output_volume.commit()
    return f"/out/{run_name}/int4-rtn"


@app.function(
    image=image,
    volumes={"/models": source_volume, "/out": output_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=14400,
)
def calibrate_9b(
    run_name: str,
    samples: int,
    max_tokens: int,
    group_size: int,
) -> str:
    """Layer-grouped Sketch B calibration (H_I only for 9B memory limits)."""
    os.environ["JAX_PLATFORM_NAME"] = "cuda"
    merged = f"/out/{run_name}/calibration/calibration.npz"
    if os.path.exists(merged):
        print(f"skip calibrate & merge: {merged} exists", flush=True)
        return merged

    bf16 = ensure_bf16_eval(run_name)
    paths = []
    for start in range(0, 32, group_size):
        end = min(start + group_size, 32)
        filters = ",".join(f"/[{index}]/" for index in range(start, end))
        output = f"/out/{run_name}/calibration/group-{start}-{end}.npz"
        if os.path.exists(output):
            print(f"skip calibration: {output} exists", flush=True)
            paths.append(output)
            continue
        print(f"calibrating layers {start}..{end - 1}", flush=True)
        run(
            [
                "python",
                "/workspace/scripts/yaqa_calibration.py",
                "--model-dir",
                bf16,
                "--corpus",
                "/workspace/data/calibration_en_dedup.txt",
                "--eval-corpus",
                "/workspace/data/eval_en.txt",
                "--eval-corpus",
                "/workspace/data/eval_ja.txt",
                "--output",
                output,
                "--json-out",
                f"/out/{run_name}/calibration/group-{start}-{end}.json",
                "--progress-jsonl",
                f"/out/{run_name}/calibration/group-{start}-{end}.progress.jsonl",
                "--samples",
                str(samples),
                "--max-tokens",
                str(max_tokens),
                "--min-tokens",
                str(max(8, max_tokens // 8)),
                "--output-block",
                "none",
                "--layer-filter",
                filters,
                "--damping",
                "1e-2",
                "--checkpoint-every",
                "2",
                "--resume",
                "auto",
            ]
        )
        paths.append(output)
        output_volume.commit()

    # Merge per-layer NPZ files (keys are disjoint).
    if os.path.exists(merged):
        print(f"skip merge: {merged} exists", flush=True)
        return merged
    # If group npz files were skipped because they already existed, gather all expected group paths
    if not paths:
        paths = [
            f"/out/{run_name}/calibration/group-{start}-{min(start + group_size, 32)}.npz"
            for start in range(0, 32, group_size)
        ]
    merge = [
        "python",
        "/workspace/scripts/merge_calibration_groups.py",
        "--output",
        merged,
        "--layers",
        "32",
        "--group-size",
        str(group_size),
        *paths,
    ]
    run(merge)
    output_volume.commit()
    return merged


@app.function(
    image=image,
    volumes={"/models": source_volume, "/out": output_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    cpu=16,
    memory=65536,
    timeout=7200,
)
def yaqa_9b(run_name: str, quantize_embedding: bool) -> str:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if os.path.exists(f"/out/{run_name}/int4-yaqa/COMPLETE"):
        print(f"skip yaqa: /out/{run_name}/int4-yaqa/COMPLETE exists", flush=True)
        return f"/out/{run_name}/int4-yaqa"
    bf16 = ensure_bf16_eval(run_name)
    command = [
        "python",
        "/workspace/scripts/rht_int4_convert.py",
        "--load-from",
        bf16,
        "--lalamo-path",
        "/opt/lalamo",
        "--output-dir",
        f"/out/{run_name}/int4-yaqa",
        "--preconditioners",
        f"/out/{run_name}/calibration/calibration.npz",
    ]
    if quantize_embedding:
        command.append("--quantize-embedding")
    run(command)
    output_volume.commit()
    return f"/out/{run_name}/int4-yaqa"


def download(remote_path: str, local_path: str) -> None:
    os.makedirs(local_path, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "uzu-quant", remote_path, local_path],
        check=True,
    )


@app.local_entrypoint()
def main(
    command: str = "check",
    run_name: str = "huihui-9b",
    samples: int = 8,
    max_tokens: int = 512,
    group_size: int = 4,
    quantize_embedding: bool = False,
    download_dir: str = "downloads/huihui-9b",
) -> None:
    if command == "check":
        print(check_env.remote())
    elif command == "import-9b":
        print(import_9b.remote(run_name))
    elif command == "rtn-9b":
        print(rtn_9b.remote(run_name, quantize_embedding))
    elif command == "calibrate-9b":
        print(calibrate_9b.remote(run_name, samples, max_tokens, group_size))
    elif command == "yaqa-9b":
        print(yaqa_9b.remote(run_name, quantize_embedding))
    elif command == "fetch-9b":
        download(f"{run_name}/int4-yaqa", download_dir)
    else:
        raise SystemExit(f"unknown command: {command}")
