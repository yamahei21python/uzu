#!/usr/bin/env python3
"""Assemble or check the public quantization kit (scripts + corpora).

The Kaggle/HF kit is a copy of a fixed file list. Copying by hand is how
``rewrite_lalamo_hybrid_spec.py`` went missing even though
``rht_int4_convert.py`` imports it lazily. This script is the single source of
truth for what goes in, so the kit can be rebuilt and verified.

Usage:
    python scripts/build_kit.py --output /tmp/uzu-kit
    python scripts/build_kit.py --check /tmp/uzu-kit
    python scripts/build_kit.py --list

A kit is valid when:
- every file in KIT_FILES exists
- every ``import <local_module>`` / ``from <local_module> import`` inside the
  kit resolves to a file in the kit
- every script compiles
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KIT_SCRIPTS = [
    # pipeline
    "rht_int4_convert.py",
    "rewrite_lalamo_hybrid_spec.py",
    "normalize_lalamo_config.py",
    "make_lalamo_eval_copy.py",
    "yaqa_calibration.py",
    "merge_language_hessians.py",
    "merge_calibration_groups.py",
    "eval_quantized.py",
    "qad_finetune.py",
    "compute_bpc.py",
    "run_lane.py",
    "make_sweep_spec.py",
    "make_smoke_specs.py",
    "combine_lane_manifests.py",
    "make_qad_dataset.py",
    "ag_fetch.py",
    "lalamo_forward.py",
    "kaggle_usage.py",
    "eval_generation.py",
    "pull_results.py",
    # guards / audits
    "corpus_guard.py",
    "dedup_lines.py",
    "dedup_calibration.py",
    "check_experiment.py",
    "check_docs.py",
    "preflight.py",
    "env_fingerprint.py",
    "kaggle_slots.py",
    "publish_results.py",
    # sharding / audit
    "tiny_sharding_probe.py",
    "audit_public_corpus.py",
    # operator tooling
    "build_kit.py",
    "bootstrap_kaggle.py",
]

KIT_DATA = [
    "calibration_en_wikitext2_dedup.txt",
    "calibration_ja_wikipedia_dedup.txt",
    "calibration_mixed_en_ja_dedup.txt",
    "calibration_mixed_ja35_en65_dedup.txt",
    "calibration_mixed_ja65_en35_dedup.txt",
    "eval_en_wikitext2_dedup.txt",
    "eval_ja_wikipedia_dedup.txt",
]

KIT_LANES = [
    "README.md",
    "smoke-08b.json",
    "clean-suite-08b.json",
    "balanced-h-08b.json",
    "damping-ablation-08b.json",
    "eval-4b-rtn.json",
    "seed-variance-08b.json",
    "group-size-ablation-08b.json",
    "bit-alloc-ablation-08b.json",
    "bits-down-out-6.json",
    "cpu-tuning-08b.json",
    "robustness-08b-shard1of3.json",
    "robustness-08b-shard2of3.json",
    "robustness-08b-shard3of3.json",
    "robustness-08b-shard1of5.json",
    "robustness-08b-shard2of5.json",
    "robustness-08b-shard3of5.json",
    "robustness-08b-shard4of5.json",
    "robustness-08b-shard5of5.json",
]

KIT_TESTS = [
    "test_experiment_guards.py",
    "test_compute_bpc.py",
    "test_run_lane.py",
    "test_tiny_sharding.py",
]

DATA_ROOT = ROOT / "experiments" / "qwen35-0.8b" / "data"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
        ).stdout.strip() or None
    except OSError:
        return None


def build(output: Path) -> list[str]:
    copied: list[str] = []
    layout = {
        "scripts": [(ROOT / "scripts" / name, name) for name in KIT_SCRIPTS],
        "data": [(DATA_ROOT / name, name) for name in KIT_DATA],
        "lanes": [(ROOT / "lanes" / name, name) for name in KIT_LANES],
        "tests": [(ROOT / "tests" / name, name) for name in KIT_TESTS],
        "configs": [(ROOT / "experiments" / model / "config.json", f"{model}.json") for model in ("qwen35-0.8b", "qwen35-4b", "qwen35-9b")],
    }
    for subdir, entries in layout.items():
        target_dir = output / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for source, name in entries:
            if not source.exists():
                raise SystemExit(f"missing source: {source}")
            shutil.copy2(source, target_dir / name)
            copied.append(f"{subdir}/{name}")

    MANIFEST_NAME = "kit_manifest.json"
    manifest = {
        "built_from_commit": source_commit(),
        # Downstream kernels should assert these, not an exact commit hash:
        # the kit is rebuilt often, but a capability answers the real question.
        "capabilities": detect_capabilities(output),
        "files": {
            name: sha256(output / name) for name in copied
        },
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return copied


def detect_capabilities(root: Path) -> list[str]:
    """Features a consumer may depend on, independent of the commit hash."""
    wanted = {
        "rht_int4_convert.layer_filter": ("scripts/rht_int4_convert.py", "--layer-filter"),
        "eval_quantized.dtype": ("scripts/eval_quantized.py", "--dtype"),
        "preflight": ("scripts/preflight.py", "def verdict"),
        "preflight.sdpa_xla": ("scripts/preflight.py", "sdpa_xla"),
        "preflight.real_shapes": ("scripts/preflight.py", "jnp.ones((512, 1024)"),
        "env_fingerprint": ("scripts/env_fingerprint.py", "lalamo_code_hashes"),
        "qad.freeze_scales": ("scripts/qad_finetune.py", "--freeze-scales"),
        "qad.early_stop_nonfatal": ("scripts/qad_finetune.py", "def early_stop_decision"),
        "lane.exclude": ("scripts/run_lane.py", "--exclude"),
        "lane.continue_on_failure": ("scripts/run_lane.py", "def run_conditions"),
        "lane.resume_hf_results": ("scripts/run_lane.py", "def pull"),
        "attention.auto_xla": ("scripts/lalamo_forward.py", "def gpu_capabilities"),
        "attention.f32_activations": ("scripts/lalamo_forward.py", "def force_float32_activations"),
    }
    found = []
    for name, (relative, needle) in wanted.items():
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if needle in text:
            found.append(name)
    return found


def local_modules(scripts: list[Path]) -> dict[str, set[str]]:
    """Map module name -> places where it is imported, for kit-local modules."""
    available = {path.stem for path in scripts}
    imported: dict[str, set[str]] = {}
    for path in scripts:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in available and name != path.stem:
                    imported.setdefault(name, set()).add(path.name)
    return imported


def check(kit: Path) -> list[str]:
    problems: list[str] = []
    for subdir, names in (
        ("scripts", KIT_SCRIPTS),
        ("data", KIT_DATA),
        ("lanes", KIT_LANES),
        ("tests", KIT_TESTS),
    ):
        for name in names:
            if not (kit / subdir / name).exists():
                problems.append(f"missing {subdir}/{name}")

    scripts = sorted((kit / "scripts").glob("*.py"))
    if not scripts:
        return problems + ["no scripts in kit"]
    for path in scripts:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as error:
            problems.append(f"syntax error in {path.name}: {error}")

    manifest_path = kit / "kit_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest.get("files", {}).items():
            target = kit / name
            if target.exists() and sha256(target) != expected:
                problems.append(f"hash mismatch: {name}")
        print(f"built_from_commit: {manifest.get('built_from_commit')}")

    for module, importers in local_modules(scripts).items():
        if not (kit / "scripts" / f"{module}.py").exists():
            problems.append(
                f"missing scripts/{module}.py imported by {', '.join(sorted(importers))}"
            )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="build the kit here")
    parser.add_argument("--check", type=Path, default=None, help="verify an existing kit")
    parser.add_argument("--list", action="store_true", help="print the manifest as JSON")
    args = parser.parse_args()

    if args.list:
        print(
            json.dumps(
                {"scripts": KIT_SCRIPTS, "data": KIT_DATA, "lanes": KIT_LANES, "tests": KIT_TESTS},
                indent=2,
            )
        )
        return
    if args.output:
        copied = build(args.output)
        print(f"wrote {len(copied)} files to {args.output}")
        problems = check(args.output)
    elif args.check:
        problems = check(args.check)
    else:
        parser.error("pass --output, --check, or --list")

    if problems:
        print("KIT INVALID")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("KIT OK")


if __name__ == "__main__":
    main()
