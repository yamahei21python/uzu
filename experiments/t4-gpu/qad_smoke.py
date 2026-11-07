import json, os, subprocess, sys, time
from pathlib import Path

os.environ["JAX_PLATFORM_NAME"] = "gpu"
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/trymirai/lalamo@e401e17761f2fcd337b0cf20218d67d656e6ecba",
    "jax[cuda12]", "huggingface_hub", "tokenizers"], check=True)
subprocess.run(["hf", "download", "yamahei21/uzu-quant-kit", "--repo-type", "dataset",
    "--local-dir", "/kaggle/working/kit"], check=True)
subprocess.run([sys.executable, "/kaggle/working/kit/scripts/build_kit.py",
    "--check", "/kaggle/working/kit"], check=True)

# the kit must have the T4 fixes: auto xla and float32 activations
guard = Path("/kaggle/working/kit/scripts/lalamo_forward.py").read_text()
for needle in ("def gpu_capabilities(", "def force_float32_activations("):
    if needle not in guard:
        raise SystemExit(f"kit is stale: missing {needle}")

print(subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout, flush=True)
devices = subprocess.run([sys.executable, "-c",
    "import jax; print(len(jax.devices()), [d.device_kind for d in jax.devices()])"],
    capture_output=True, text=True).stdout.strip()
print("jax devices:", devices, flush=True)
if not devices.startswith("2 "):
    raise SystemExit(f"need 2 GPUs, got: {devices}")

candidates = [p.parent for p in Path("/kaggle/input").rglob("model.safetensors")
              if (p.parent / "tokenizer.json").exists()]
if not candidates:
    raise SystemExit("no BF16 model under /kaggle/input")
bf16 = candidates[0]
kit = Path("/kaggle/working/kit")

started = time.time()
subprocess.run([sys.executable, str(kit / "scripts/make_qad_dataset.py"),
    "--bf16", str(bf16), "--out", "/kaggle/working/qad",
    "--corpus", str(kit / "data/calibration_mixed_en_ja_dedup.txt"),
    "--eval-corpus", str(kit / "data/eval_en_wikitext2_dedup.txt"),
    "--eval-corpus", str(kit / "data/eval_ja_wikipedia_dedup.txt"),
    "--samples", "8", "--max-tokens", "128", "--output-block", "small",
    "--damping", "1e-3", "--parallel", "tensor"], check=True)
print(f"dataset+convert {time.time()-started:.0f}s", flush=True)

started = time.time()
subprocess.run([sys.executable, str(kit / "scripts/qad_finetune.py"),
    "--teacher-dir", str(bf16), "--student-dir", "/kaggle/working/qad/student",
    "--corpus", str(kit / "data/calibration_mixed_en_ja_dedup.txt"),
    "--eval-corpus", str(kit / "data/eval_en_wikitext2_dedup.txt"),
    "--eval-corpus", str(kit / "data/eval_ja_wikipedia_dedup.txt"),
    "--output-dir", "/kaggle/working/qad/run",
    "--steps", "10", "--seq", "128", "--optimizer", "adamw",
    "--lr", "2e-5", "--scale-lr", "1e-6", "--freeze-scales", "--warmup-steps", "5",
    "--val-every", "5", "--val-chunks", "2", "--save-every", "0",
    "--parallel", "tensor"], check=True)
print(f"qad 10 steps {time.time()-started:.0f}s", flush=True)

run = Path("/kaggle/working/qad/run")
if not (run / "model.safetensors").exists():
    raise SystemExit("no exported model")
checkpoints = sorted(p.name for p in (run / "checkpoints").glob("step_*"))
print("checkpoints:", checkpoints, flush=True)
for line in (run / "progress.jsonl").read_text().splitlines()[-3:]:
    print("progress:", line, flush=True)
print("QAD GPU OK", flush=True)
