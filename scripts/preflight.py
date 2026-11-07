#!/usr/bin/env python3
"""Preflight a machine before a lane starts.

Two T4 lanes died 10 minutes in because Turing has no BF16 matmul and lalamo's
attention requests `ALG_DOT_BF16_BF16_F32`. Catch that in seconds instead.

    python scripts/preflight.py
    python scripts/preflight.py --strict      # exit 1 when BF16 is unavailable
    python scripts/preflight.py --json out.json

Checks: jax devices, compute capability, an actual BF16 matmul, working/RAM/disk,
and the kit manifest commit when present.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TRITON_MIN = 8.0


def parse_capabilities(output: str) -> list[float]:
    values = []
    for line in output.splitlines():
        text = line.strip().split(",")[0].strip()
        try:
            values.append(float(text))
        except ValueError:
            continue
    return values


def verdict(capabilities: list[float], probes: dict | None = None) -> dict:
    """Pure decision table so it can be tested without a GPU."""
    probes = probes or {}
    ampere = bool(capabilities) and all(value >= TRITON_MIN for value in capabilities)

    def status(name: str) -> str:
        return str(probes.get(name, "skip"))

    core = ("dot", "dot_f32_acc", "sdpa_xla", "quant_roundtrip", "qad_grad", "tokenizer")
    failures = [name for name in core if status(name) == "fail"]
    bf16 = all(status(name) == "pass" for name in ("dot", "dot_f32_acc", "sdpa_xla"))

    if status("sdpa_xla") == "fail":
        attention = "unavailable"
        note = "dot_product_attention is unsupported on this GPU: use CPU or Modal"
    elif status("sdpa_xla") == "pass":
        attention = "default"
        note = "" if bf16 else "BF16 accumulation unsupported: use float32"
    else:
        attention = "default" if ampere else "xla"
        note = "" if bf16 else "BF16 unsupported: use float32 or CPU/Modal"

    return {
        "ampere_or_newer": ampere,
        "bf16_matmul": bf16,
        "bf16_source": "measured" if probes else "capability",
        "probes": probes,
        "probe_failures": failures,
        "lane_ok": not failures,
        "eval_dtype": "bfloat16" if bf16 else "float32",
        "attention": attention,
        "note": note,
    }


PROBES = {
    # plain bf16 matmul: T4 passes this (XLA upcasts internally)
    "dot": (
        "import jax.numpy as jnp\n"
        "x = jnp.ones((32, 32), dtype=jnp.bfloat16)\n"
        "(x @ x).block_until_ready()\n"
    ),
    # bf16 inputs with f32 accumulation: this is what fails on T4
    "dot_f32_acc": (
        "import jax.numpy as jnp\n"
        "x = jnp.ones((32, 32), dtype=jnp.bfloat16)\n"
        "jnp.dot(x, x, preferred_element_type=jnp.float32).block_until_ready()\n"
    ),
    # lalamo's xla attention uses jax.nn.dot_product_attention(implementation='xla')
    "sdpa_xla": (
        "import jax, jax.numpy as jnp\n"
        "q = jnp.ones((2, 4, 8, 16), dtype=jnp.bfloat16)\n"
        "jax.nn.dot_product_attention(q, q, q, implementation='xla').block_until_ready()\n"
    ),
    # tokenizer: load and round-trip both scripts
    "tokenizer": (
        "import sys\n"
        "from tokenizers import Tokenizer\n"
        "tok = Tokenizer.from_file(sys.argv[1])\n"
        "for text in ('the quick brown fox', '日本語のテキストです'):\n"
        "    ids = tok.encode(text).ids\n"
        "    assert ids, text\n"
        "    tok.decode(ids)\n"
    ),
    # quantize / dequantize a tiny matrix through lalamo's INT4 + RHT path
    "quant_roundtrip": (
        "from lalamo.compressed.hybrid import HybridSpec\n"
        "from lalamo.compressed.int import IntSpec\n"
        "from lalamo.compressed.incoherence import HadamardSpec, IncoherenceProcessingMode\n"
        "from lalamo.utils.sharding import ShardingConfig\n"
        "from lalamo.weight_matrix import Layout\n"
        "import jax, jax.numpy as jnp, numpy as np\n"
        "spec = HybridSpec(\n"
        "    quantization_spec=IntSpec(bits=4, group_size=32, is_symmetric=False, layout=Layout.OUTPUT_INPUT),\n"
        "    adapter_spec=None,\n"
        "    incoherence_spec=HadamardSpec(block_size=32, processing_mode=IncoherenceProcessingMode.INPUT_OUTPUT),\n"
        ")\n"
        "cfg = ShardingConfig.replicated()\n"
        "w = jax.device_put(np.random.RandomState(0).randn(64, 32).astype(np.float32), cfg.resolve_sharding((None, None)))\n"
        "compressed = spec.compress(w, key=jax.random.key(0), sharding_config=cfg)\n"
        "restored = compressed.decompress()\n"
        "assert bool(jnp.isfinite(restored).all())\n"
        "assert restored.shape == w.shape\n"
    ),
    # QAD gradient path: finite grads through quantize/dequantize
    "qad_grad": (
        "from lalamo.compressed.hybrid import HybridSpec\n"
        "from lalamo.compressed.int import IntSpec\n"
        "from lalamo.utils.sharding import ShardingConfig\n"
        "from lalamo.weight_matrix import Layout\n"
        "import jax, jax.numpy as jnp, numpy as np\n"
        "spec = HybridSpec(\n"
        "    quantization_spec=IntSpec(bits=4, group_size=32, is_symmetric=False, layout=Layout.OUTPUT_INPUT),\n"
        "    adapter_spec=None, incoherence_spec=None,\n"
        ")\n"
        "sharding = ShardingConfig.replicated()\n"
        "def loss(w):\n"
        "    return jnp.sum(spec.compress(w, key=jax.random.key(0), sharding_config=sharding).decompress() ** 2)\n"
        "w = jax.device_put(np.random.RandomState(0).randn(64, 32).astype(np.float32), sharding.resolve_sharding((None, None)))\n"
        "grad = jax.grad(loss)(w)\n"
        "assert bool(jnp.isfinite(grad).all())\n"
    ),
}

# probes that need lalamo; missing lalamo is a skip, not a failure
LALAMO_PROBES = {"quant_roundtrip", "qad_grad"}


def run_probe(name: str, code: str, extra: list[str] | None = None) -> tuple[str, str | None]:
    """Returns (pass|fail|skip, error)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, *(extra or [])],
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return "fail", str(error)
    if result.returncode == 0:
        return "pass", None
    message = (result.stderr or result.stdout or "").strip()[-400:] or "unknown failure"
    if name in LALAMO_PROBES and "lalamo" in message and (
        "ModuleNotFoundError" in message or "No module named" in message
    ):
        return "skip", "lalamo not importable in this interpreter"
    return "fail", message


def measure_probes(tokenizer_path: Path | None = None) -> tuple[dict, dict]:
    results, errors = {}, {}
    for name, code in PROBES.items():
        extra = [str(tokenizer_path)] if name == "tokenizer" and tokenizer_path else None
        if name == "tokenizer" and tokenizer_path is None:
            results[name] = "skip"
            continue
        status, error = run_probe(name, code, extra)
        results[name] = status
        if error:
            errors[name] = error
    return results, errors


def disk_free_gib(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return -1.0
    return round(usage.free / 1024**3, 1)


def kit_commit(root: Path) -> str | None:
    manifest = root / "kit_manifest.json"
    if manifest.exists():
        try:
            return json.loads(manifest.read_text(encoding="utf-8")).get("built_from_commit")
        except json.JSONDecodeError:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="exit 1 if BF16 matmul is unavailable")
    parser.add_argument("--skip-bf16", action="store_true", help="do not import jax / lalamo")
    parser.add_argument("--tokenizer", type=Path, default=None, help="tokenizer.json for the round-trip probe")
    args = parser.parse_args()

    try:
        nvidia = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        nvidia_stdout = nvidia.stdout or ""
    except OSError:
        nvidia_stdout = ""
    capabilities = parse_capabilities(nvidia_stdout)
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        for candidate in Path("/kaggle/input").rglob("tokenizer.json"):
            tokenizer_path = candidate
            break
    if args.skip_bf16:
        probes, probe_errors = {}, {}
    else:
        probes, probe_errors = measure_probes(tokenizer_path)

    report = {
        "jax_platform": os.environ.get("JAX_PLATFORM_NAME"),
        "compute_capabilities": capabilities,
        "gpu_names": [line.split(",")[-1].strip() for line in nvidia_stdout.splitlines() if line.strip()],
        "probe_errors": probe_errors if not args.skip_bf16 else {},
        "disk_free_gib": disk_free_gib(Path("/kaggle/working")) if Path("/kaggle/working").exists() else disk_free_gib(Path(".")),
        "kit_commit": kit_commit(Path("kit")),
    }
    report.update(verdict(capabilities, probes))
    report["jax_version"] = None
    try:
        from importlib.metadata import version

        report["jax_version"] = version("jax")
    except Exception:  # noqa: BLE001
        pass

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.strict and not report["bf16_matmul"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
