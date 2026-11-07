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


def verdict(capabilities: list[float], bf16_ok: bool | None, probes: dict | None = None) -> dict:
    """Pure decision table so it can be tested without a GPU."""
    ampere = bool(capabilities) and all(value >= TRITON_MIN for value in capabilities)
    if bf16_ok is None:
        bf16 = ampere  # no measurement: fall back to the hardware generation
        source = "capability"
    else:
        bf16 = bf16_ok
        source = "measured"
    probes = probes or {}
    sdpa_ok = probes.get("sdpa_xla")
    if sdpa_ok is False:
        attention = "unavailable"
        note = "dot_product_attention is unsupported on this GPU: use CPU or Modal"
    elif sdpa_ok is True:
        attention = "default"
        note = "" if bf16 else "BF16 accumulation unsupported: use float32"
    else:
        attention = "default" if ampere else "xla"
        note = "" if bf16 else "BF16 unsupported: use float32 or CPU/Modal"
    return {
        "ampere_or_newer": ampere,
        "bf16_matmul": bf16,
        "bf16_source": source,
        "probes": probes,
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
}


def run_probe(code: str) -> tuple[bool, str | None]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], text=True, capture_output=True, timeout=180, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    if result.returncode == 0:
        return True, None
    return False, ((result.stderr or result.stdout or "").strip()[-300:] or "unknown failure")


def measure_probes() -> tuple[dict, dict]:
    results, errors = {}, {}
    for name, code in PROBES.items():
        ok, error = run_probe(code)
        results[name] = ok
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
    parser.add_argument("--skip-bf16", action="store_true", help="do not import jax")
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
    if args.skip_bf16:
        probes, probe_errors = {}, {}
    else:
        probes, probe_errors = measure_probes()
    bf16_ok = None if args.skip_bf16 else (bool(probes) and all(probes.values()))

    report = {
        "jax_platform": os.environ.get("JAX_PLATFORM_NAME"),
        "compute_capabilities": capabilities,
        "gpu_names": [line.split(",")[-1].strip() for line in nvidia_stdout.splitlines() if line.strip()],
        "bf16_measurement": bf16_ok,
        "probe_errors": {k: v for k, v in (probe_errors if not args.skip_bf16 else {}).items()},
        "disk_free_gib": disk_free_gib(Path("/kaggle/working")) if Path("/kaggle/working").exists() else disk_free_gib(Path(".")),
        "kit_commit": kit_commit(Path("kit")),
    }
    report.update(verdict(capabilities, bf16_ok, probes))
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
