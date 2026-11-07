#!/usr/bin/env python3
"""Modal pipeline for 4B / 9B RHT + INT4 quantization.

T4 (16GB) cannot hold 4B or 9B BF16 during calibration, and 9B BF16 (18GB) does
not even fit for evaluation. Modal A100 is the escape hatch.

Local work stays local: only scripts and the dedup corpora are baked into the
image. The BF16 source is downloaded inside Modal and the INT4 output is written
to the ``uzu-quant`` volume.

Commands (``--size`` is 4b or 9b):
  modal run scripts/modal_pipeline.py --command import    --size 9b
  modal run scripts/modal_pipeline.py --command rtn       --size 9b --quantize-embedding
  modal run scripts/modal_pipeline.py --command calibrate --size 9b --samples 8 --max-tokens 512
  modal run scripts/modal_pipeline.py --command yaqa      --size 9b --quantize-embedding
  modal run scripts/modal_pipeline.py --command eval      --size 9b
  modal run scripts/modal_pipeline.py --command qad       --size 9b --steps 10 --freeze-scales
  modal run scripts/modal_pipeline.py --command fetch     --size 9b --remote int4-yaqa

Every function is idempotent: a COMPLETE marker or an existing output skips the
step, so a timed-out run resumes.
"""

from __future__ import annotations

import json
import os
import subprocess

import modal

from modal_targets import COMMANDS, default_run_name, spec

APP_NAME = "uzu-quant-pipeline"

source_volume = modal.Volume.from_name("uzu-models", create_if_missing=True)
output_volume = modal.Volume.from_name("uzu-quant", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/trymirai/lalamo /opt/lalamo",
        "cd /opt/lalamo && git checkout e401e17761f2fcd337b0cf20218d67d656e6ecba",
        "pip install -e '/opt/lalamo[cuda]'",
        "pip install tokenizers",
    )
    .env({"HF_HUB_ENABLE_XET": "0", "PYTHONUNBUFFERED": "1", "JAX_PLATFORM_NAME": "cpu"})
    .add_local_file("scripts/rht_int4_convert.py", "/workspace/scripts/rht_int4_convert.py")
    .add_local_file("scripts/rewrite_lalamo_hybrid_spec.py", "/workspace/scripts/rewrite_lalamo_hybrid_spec.py")
    .add_local_file("scripts/normalize_lalamo_config.py", "/workspace/scripts/normalize_lalamo_config.py")
    .add_local_file("scripts/yaqa_calibration.py", "/workspace/scripts/yaqa_calibration.py")
    .add_local_file("scripts/make_lalamo_eval_copy.py", "/workspace/scripts/make_lalamo_eval_copy.py")
    .add_local_file("scripts/merge_calibration_groups.py", "/workspace/scripts/merge_calibration_groups.py")
    .add_local_file("scripts/corpus_guard.py", "/workspace/scripts/corpus_guard.py")
    .add_local_file("scripts/eval_quantized.py", "/workspace/scripts/eval_quantized.py")
    .add_local_file("scripts/qad_finetune.py", "/workspace/scripts/qad_finetune.py")
    .add_local_file("scripts/lalamo_forward.py", "/workspace/scripts/lalamo_forward.py")
    # imported by this entrypoint at startup; must sit beside it, not in /workspace/scripts
    .add_local_file("scripts/modal_targets.py", "/root/modal_targets.py")
    .add_local_file(
        "experiments/qwen35-0.8b/data/calibration_en_wikitext2_dedup.txt",
        "/workspace/data/calibration_en_dedup.txt",
    )
    .add_local_file(
        "experiments/qwen35-0.8b/data/calibration_ja_wikipedia_dedup.txt",
        "/workspace/data/calibration_ja_dedup.txt",
    )
    .add_local_file(
        "experiments/qwen35-0.8b/data/eval_en_wikitext2_dedup.txt",
        "/workspace/data/eval_en_dedup.txt",
    )
    .add_local_file(
        "experiments/qwen35-0.8b/data/eval_ja_wikipedia_dedup.txt",
        "/workspace/data/eval_ja_dedup.txt",
    )
)

app = modal.App(APP_NAME)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ensure_bf16_eval(run_name: str) -> str:
    """Return a lalamo-loadable BF16 directory.

    Import normalizes config.json for uzu, so lalamo cannot load that directory.
    make_lalamo_eval_copy restores the lalamo form and hardlinks the weights.
    """
    source = f"/out/{run_name}/bf16"
    dest = f"/out/{run_name}/bf16-lalamo"
    if not os.path.exists(f"{source}/model.safetensors"):
        raise SystemExit(f"missing BF16 at {source}; run --command import first")
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
def import_model(size: str, run_name: str) -> str:
    """Download the HF model and save BF16 in lalamo format."""
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    model = spec(size)
    if os.path.exists(f"/out/{run_name}/bf16/COMPLETE"):
        print(f"skip import: /out/{run_name}/bf16/COMPLETE exists", flush=True)
        return f"/out/{run_name}/bf16"
    run(
        [
            "python",
            "/workspace/scripts/rht_int4_convert.py",
            "--spec-name",
            model["spec_name"],
            "--hf-repo",
            model["repo"],
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
def rtn(size: str, run_name: str, quantize_embedding: bool) -> str:
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
def calibrate(size: str, run_name: str, samples: int, max_tokens: int, group_size: int, corpus: str) -> str:
    """Calibrate in layer groups so the Hessian accumulators fit in memory."""
    os.environ["JAX_PLATFORM_NAME"] = "cuda"
    # JAX otherwise reserves 75% of the device and OOMs before the Hessian fits
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    model = spec(size)
    layers = int(model["layers"])
    merged = f"/out/{run_name}/calibration/calibration.npz"
    if os.path.exists(merged):
        print(f"skip calibrate & merge: {merged} exists", flush=True)
        return merged

    bf16 = ensure_bf16_eval(run_name)
    corpus_path = f"/workspace/data/{corpus}"
    paths = []
    for start in range(0, layers, group_size):
        end = min(start + group_size, layers)
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
                corpus_path,
                "--eval-corpus",
                "/workspace/data/eval_en_dedup.txt",
                "--eval-corpus",
                "/workspace/data/eval_ja_dedup.txt",
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
                "--calib-dtype",
                "bfloat16",
                "--checkpoint-every",
                "2",
                "--resume",
                "auto",
            ]
        )
        paths.append(output)
        output_volume.commit()

    if not paths:
        paths = [
            f"/out/{run_name}/calibration/group-{start}-{min(start + group_size, layers)}.npz"
            for start in range(0, layers, group_size)
        ]
    run(
        [
            "python",
            "/workspace/scripts/merge_calibration_groups.py",
            "--output",
            merged,
            "--layers",
            str(layers),
            "--group-size",
            str(group_size),
            *paths,
        ]
    )
    summary = {
        "overlap_checked": True,
        "corpus": corpus_path,
        "samples": samples,
        "max_tokens": max_tokens,
        "output_block": "none",
        "damping": 1e-2,
        "calib_dtype": "bfloat16",
        "calibration_corpora": [corpus_path],
        "eval_corpora": ["/workspace/data/eval_en_dedup.txt", "/workspace/data/eval_ja_dedup.txt"],
        "groups": paths,
    }
    with open(f"/out/{run_name}/calibration/calibration.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
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
def yaqa(size: str, run_name: str, quantize_embedding: bool) -> str:
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
        "--calibration-json",
        f"/out/{run_name}/calibration/calibration.json",
    ]
    if quantize_embedding:
        command.append("--quantize-embedding")
    run(command)
    output_volume.commit()
    return f"/out/{run_name}/int4-yaqa"


@app.function(
    image=image,
    volumes={"/models": source_volume, "/out": output_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=14400,
)
def evaluate(size: str, run_name: str, candidate: str, max_sequences: int) -> str:
    """PPL / BPC for EN and JA against the dedup eval corpora."""
    os.environ["JAX_PLATFORM_NAME"] = "cuda"
    bf16 = ensure_bf16_eval(run_name)
    candidate_dir = f"/out/{run_name}/{candidate}"
    if not os.path.exists(f"{candidate_dir}/model.safetensors"):
        raise SystemExit(f"missing candidate at {candidate_dir}")
    eval_dir = f"/out/{run_name}/eval-{candidate}"
    if not os.path.exists(f"{eval_dir}/config.json"):
        run(
            [
                "python",
                "/workspace/scripts/make_lalamo_eval_copy.py",
                "--source",
                candidate_dir,
                "--dest",
                eval_dir,
            ]
        )
        output_volume.commit()

    outputs = []
    for lang in ("en", "ja"):
        json_out = f"/out/{run_name}/eval-{candidate}-{lang}.json"
        if os.path.exists(json_out):
            print(f"skip eval {lang}: {json_out} exists", flush=True)
            outputs.append(json_out)
            continue
        run(
            [
                "python",
                "/workspace/scripts/eval_quantized.py",
                "--bf16-dir",
                bf16,
                "--candidate-dir",
                eval_dir,
                "--corpus",
                f"/workspace/data/eval_{lang}_dedup.txt",
                "--max-sequences",
                str(max_sequences),
                "--max-tokens",
                "128",
                "--json-out",
                json_out,
            ]
        )
        output_volume.commit()
        outputs.append(json_out)
    for path in outputs:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        print(
            path,
            "chunks",
            payload.get("chunks"),
            "bf16",
            round(payload.get("bf16_ppl", 0), 2),
            "int4",
            round(payload.get("candidate_ppl", 0), 2),
            "bpc",
            payload.get("bpc_candidate"),
        )
    return ",".join(outputs)


@app.function(
    image=image,
    volumes={"/models": source_volume, "/out": output_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=14400,
)
def qad(
    size: str,
    run_name: str,
    student: str,
    steps: int,
    optimizer: str,
    lr: float,
    scale_lr: float,
    freeze_scales: bool,
    warmup_steps: int,
    save_every: int,
    corpus: str,
) -> str:
    """Distill the teacher into the quantized student.

    ``student`` is a directory under /out/<run> containing a raw lalamo config,
    provenance.json, and the tokenizer. Build it with make_qad_dataset.py.
    """
    os.environ["JAX_PLATFORM_NAME"] = "cuda"
    bf16 = ensure_bf16_eval(run_name)
    student_dir = f"/out/{run_name}/{student}"
    if not os.path.exists(f"{student_dir}/provenance.json"):
        raise SystemExit(f"missing student provenance at {student_dir}")
    out = f"/out/{run_name}/qad"
    command = [
        "python",
        "/workspace/scripts/qad_finetune.py",
        "--teacher-dir",
        bf16,
        "--student-dir",
        student_dir,
        "--corpus",
        f"/workspace/data/{corpus}",
        "--eval-corpus",
        "/workspace/data/eval_en_dedup.txt",
        "--eval-corpus",
        "/workspace/data/eval_ja_dedup.txt",
        "--output-dir",
        out,
        "--steps",
        str(steps),
        "--optimizer",
        optimizer,
        "--lr",
        str(lr),
        "--scale-lr",
        str(scale_lr),
        "--warmup-steps",
        str(warmup_steps),
        "--save-every",
        str(save_every),
        "--resume",
        "auto",
    ]
    if freeze_scales:
        command.append("--freeze-scales")
    run(command)
    output_volume.commit()
    return out


def download(remote_path: str, local_path: str) -> None:
    os.makedirs(local_path, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "uzu-quant", remote_path, local_path],
        check=True,
    )


@app.local_entrypoint()
def main(
    command: str = "check",
    size: str = "9b",
    run_name: str = "",
    samples: int = 8,
    max_tokens: int = 512,
    group_size: int = 4,
    corpus: str = "calibration_en_dedup.txt",
    quantize_embedding: bool = False,
    candidate: str = "int4-yaqa",
    max_sequences: int = 100,
    student: str = "student",
    steps: int = 10,
    optimizer: str = "adamw",
    lr: float = 2e-5,
    scale_lr: float = 1e-6,
    freeze_scales: bool = False,
    warmup_steps: int = 0,
    save_every: int = 10,
    remote: str = "int4-yaqa",
    download_dir: str = "",
) -> None:
    if command not in COMMANDS:
        raise SystemExit(f"unknown command: {command}; choose {COMMANDS}")
    run_name = run_name or default_run_name(size)

    if command == "check":
        print(check_env.remote())
    elif command == "import":
        print(import_model.remote(size, run_name))
    elif command == "rtn":
        print(rtn.remote(size, run_name, quantize_embedding))
    elif command == "calibrate":
        print(calibrate.remote(size, run_name, samples, max_tokens, group_size, corpus))
    elif command == "yaqa":
        print(yaqa.remote(size, run_name, quantize_embedding))
    elif command == "eval":
        print(evaluate.remote(size, run_name, candidate, max_sequences))
    elif command == "qad":
        print(
            qad.remote(
                size,
                run_name,
                student,
                steps,
                optimizer,
                lr,
                scale_lr,
                freeze_scales,
                warmup_steps,
                save_every,
                corpus,
            )
        )
    elif command == "fetch":
        download(f"{run_name}/{remote}", download_dir or f"downloads/{run_name}-{remote}")
    else:
        raise SystemExit(f"unreachable command: {command}")
