#!/usr/bin/env python3
"""Run a quantization experiment lane from a JSON spec.

The point is that the owner defines an experiment as data and the Kaggle
operator only runs it. Every step still goes through the existing scripts, so
guards and provenance stay in force:

  yaqa_calibration.py -> rht_int4_convert.py -> make_lalamo_eval_copy.py -> eval_quantized.py
  (+ merge_language_hessians.py when the spec asks for a language-balanced merge)

Usage:
    python scripts/run_lane.py --spec lanes/clean-suite-08b.json --dry-run
    python scripts/run_lane.py --spec lanes/clean-suite-08b.json
    python scripts/run_lane.py --spec lanes/clean-suite-08b.json --hf-results <user>/uzu-results

Resume is the default: a condition whose ``result.json`` exists and passes the
sanity checks is skipped, so a kernel that dies at the 12h limit can be rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_CONDITION = {
    "samples": 32,
    "calib_dtype": "float32",
    "max_tokens": 128,
    "output_block": "small",
    "output_block_limit": 2048,
    "damping": 1e-3,
    "raw_output": False,
}
DEFAULT_EVAL = {"max_sequences": 100, "max_tokens": 128, "min_tokens": 16}
TRITON_MIN_CAPABILITY = 8.0


def parse_compute_capabilities(output: str) -> list[float]:
    capabilities = []
    for line in output.splitlines():
        text = line.strip().split(",")[0].strip()
        try:
            capabilities.append(float(text))
        except ValueError:
            continue
    return capabilities


def needs_triton_off(capabilities: list[float]) -> bool:
    """XLA's Triton GEMM path only supports Ampere (cc 8.0) and newer."""
    return any(capability < TRITON_MIN_CAPABILITY for capability in capabilities)


def detect_compute_capabilities() -> list[float]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return parse_compute_capabilities(result.stdout)


def find_dataset_dir(root: Path, name: str) -> Path | None:
    """Locate a mounted dataset regardless of the /kaggle/input layout."""
    candidates = [
        root / name,
        *sorted(root.glob(f"datasets/*/{name}")),
        *sorted(root.glob(f"datasets/*/*/{name}")),
    ]
    for candidate in candidates:
        if (candidate / "model.safetensors").exists():
            return candidate
    # last resort: any directory with both weights and a tokenizer
    for weights in sorted(root.rglob("model.safetensors")):
        directory = weights.parent
        if (directory / "tokenizer.json").exists():
            return directory
    return None


class Runner:
    def __init__(self, spec: dict[str, Any], spec_path: Path, *, dry_run: bool = False) -> None:
        self.spec = spec
        self.spec_path = spec_path
        self.dry_run = dry_run
        self.commands: list[list[str]] = []

        paths = spec.get("paths", {})
        kit = Path(paths.get("kit", "kit")).expanduser()
        work = Path(paths.get("work", "/kaggle/working")).expanduser()
        self.kit = kit
        self.work = work
        self.scripts = {
            "calibrate": self._script(paths.get("calibrate", "scripts/yaqa_calibration.py")),
            "convert": self._script(paths.get("convert", "scripts/rht_int4_convert.py")),
            "merge": self._script(paths.get("merge", "scripts/merge_language_hessians.py")),
            "eval": self._script(paths.get("eval", "scripts/eval_quantized.py")),
            "copy": self._script(paths.get("copy", "scripts/make_lalamo_eval_copy.py")),
        }
        self.lane = spec.get("lane") or spec_path.stem
        capabilities = detect_compute_capabilities()
        self.triton_off = needs_triton_off(capabilities)
        # T4 (cc 7.5) has no BF16 matmul: ALG_DOT_BF16_BF16_F32 is unsupported.
        # Fall back to float32 so the lane can run at all.
        configured_dtype = (spec.get("eval") or {}).get("dtype")
        self.eval_dtype = configured_dtype or ("float32" if self.triton_off else "bfloat16")
        if self.triton_off and configured_dtype == "bfloat16":
            print("GPU cannot do BF16 matmul; overriding eval dtype to float32", flush=True)
            self.eval_dtype = "float32"
        if self.triton_off:
            print(
                f"GPU caps {capabilities}: disabling XLA Triton GEMM (needs >= {TRITON_MIN_CAPABILITY})",
                flush=True,
            )
        self.lane_dir = work / "out" / self.lane
        self.jax_platform = paths.get("jax_platform") or os.environ.get("JAX_PLATFORM_NAME") or None

    # ---------- helpers ----------

    def _script(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else self.kit / path

    def resolve(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.kit / path

    def env_provenance(self) -> dict[str, str]:
        """Environment knobs that change numerics must travel with the results."""
        prefixes = ("XLA_FLAGS", "JAX_", "UZU_")
        seen = {}
        for key, value in self.environment().items():
            if key.startswith(prefixes):
                seen[key] = value
        try:
            from importlib.metadata import version

            seen["jax_version"] = version("jax")
        except Exception:  # noqa: BLE001 - provenance must never break a run
            pass
        return seen

    def environment(self, base: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        if self.jax_platform:
            env["JAX_PLATFORM_NAME"] = self.jax_platform
        for key, value in (self.spec.get("paths", {}).get("env") or {}).items():
            env[str(key)] = str(value)
        flags = list(self.spec.get("paths", {}).get("xla_flags", []))
        if self.triton_off:
            flags.append("--xla_gpu_enable_triton_gemm=false")
            # lalamo inference defaults to the Pallas-Triton flash attention kernel,
            # which T4 cannot compile. XLA attention is the supported path there.
            env.setdefault("UZU_ATTENTION_IMPL", "xla")
        existing = env.get("XLA_FLAGS", "")
        for flag in flags:
            if flag not in existing:
                existing = (existing + " " + flag).strip()
        if existing:
            env["XLA_FLAGS"] = existing
        return env

    def run(self, step: str, command: list[str], log: Path) -> None:
        display = " ".join(str(part) for part in command)
        print(f"$ {display}", flush=True)
        self.commands.append([str(part) for part in command])
        if self.dry_run:
            return
        env = self.environment()
        started = time.time()
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== {step} {time.strftime('%H:%M:%S')} ===\n$ {display}\n")
            handle.flush()
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False)
        if result.returncode != 0:
            raise SystemExit(f"{step} failed with code {result.returncode}; see {log}")
        print(f"  {step} ok ({time.time() - started:.0f}s)", flush=True)

    # ---------- steps ----------

    def ensure_bf16(self) -> Path:
        """Return a lalamo-loadable BF16 directory.

        The import path (rht_int4_convert --dry-run) normalizes config.json for
        uzu: it drops ``response_format`` and adds ``output_parser_regex`` /
        ``end_of_thinking_tag``. Pinned lalamo refuses that schema, so
        calibration cannot load the imported directory directly. The import is
        therefore written to a raw directory and ``make_lalamo_eval_copy``
        produces the loadable copy that both calibration and eval use.
        """
        model = self.spec.get("model", {})
        target = self.resolve(model.get("bf16_dir", "bf16-lalamo"))
        assert target is not None
        if (target / "model.safetensors").exists():
            return target

        source = self.resolve(model.get("bf16_source"))
        if source and not (source / "model.safetensors").exists() and model.get("bf16_source_auto"):
            found = find_dataset_dir(Path("/kaggle/input"), source.name)
            if found is not None:
                print(f"bf16 source resolved to {found}", flush=True)
                source = found
        if source and (source / "model.safetensors").exists():
            print(f"copy bf16 from {source} -> {target}", flush=True)
            if not self.dry_run:
                shutil.copytree(source, target, dirs_exist_ok=True)
            return target

        raw = self.resolve(model.get("bf16_raw_dir", "bf16-raw"))
        assert raw is not None
        if not (raw / "model.safetensors").exists():
            command = [
                sys.executable,
                str(self.scripts["convert"]),
                "--spec-name",
                model.get("spec_name", "Qwen3.5-0.8B"),
                "--output-dir",
                str(self.work / "import"),
                "--dry-run",
                "--bf16-dir",
                str(raw),
            ]
            if model.get("hf_repo"):
                command += ["--hf-repo", model["hf_repo"]]
            if model.get("hf_dir"):
                command += ["--hf-dir", str(self.resolve(model["hf_dir"]))]
            self.run("bf16 import", command, self.lane_dir / "lane.log")

        self.run(
            "bf16 lalamo copy",
            [sys.executable, str(self.scripts["copy"]), "--source", str(raw), "--dest", str(target)],
            self.lane_dir / "lane.log",
        )
        return target

    def eval_corpora(self) -> dict[str, Path]:
        corpora = self.spec.get("eval", {}).get("corpora", {})
        if not corpora:
            raise SystemExit("spec.eval.corpora is empty")
        resolved = {}
        for lang, path in corpora.items():
            target = self.resolve(path)
            assert target is not None
            resolved[lang] = target
        return resolved

    def condition_settings(self, condition: dict[str, Any]) -> dict[str, Any]:
        settings = dict(DEFAULT_CONDITION)
        settings.update(self.spec.get("condition_defaults", {}))
        settings.update({key: value for key, value in condition.items() if key not in ("name", "calibration_corpus")})
        return settings

    def existing_result(self, name: str) -> dict[str, Any] | None:
        path = self.lane_dir / name / "result.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if payload.get("status") != "complete":
            return None
        return payload

    def evaluate(self, name: str, candidate: Path, bf16: Path) -> dict[str, Any]:
        out = self.lane_dir / name
        base = read_json(out / "calibration.json") or {}
        summary: dict[str, Any] = {
            "name": name,
            "calibration": {
                "corpus": base.get("corpus"),
                "samples": base.get("samples"),
                "max_tokens": base.get("max_tokens"),
                "output_block": base.get("output_block"),
                "damping": base.get("damping"),
                "calib_dtype": base.get("calib_dtype"),
                "raw_output": base.get("raw_output"),
                "ja_ratio": base.get("ja_ratio"),
                "overlap_checked": base.get("overlap_checked"),
            },
            "ppl": {},
        }
        for lang, corpus in self.eval_corpora().items():
            json_out = out / f"ppl_{lang}.json"
            self.run(
                f"{name} eval {lang}",
                [
                    sys.executable,
                    str(self.scripts["eval"]),
                    "--bf16-dir",
                    str(bf16),
                    "--candidate-dir",
                    str(candidate),
                    "--corpus",
                    str(corpus),
                    "--max-sequences",
                    str(self.spec.get("eval", {}).get("max_sequences", DEFAULT_EVAL["max_sequences"])),
                    "--max-tokens",
                    str(self.spec.get("eval", {}).get("max_tokens", DEFAULT_EVAL["max_tokens"])),
                    "--dtype",
                    str(self.eval_dtype),
                    "--min-tokens",
                    str(self.spec.get("eval", {}).get("min_tokens", DEFAULT_EVAL["min_tokens"])),
                    "--json-out",
                    str(json_out),
                ],
                out / "lane.log",
            )
            payload = read_json(json_out) or {}
            summary["ppl"][lang] = {
                key: payload.get(key)
                for key in (
                    "chunks",
                    "tokens",
                    "bf16_ppl",
                    "candidate_ppl",
                    "delta_nll_per_token",
                    "config_hash",
                )
            }
        return summary

    def run_condition(self, condition: dict[str, Any], bf16: Path) -> dict[str, Any]:
        name = condition["name"]
        out = self.lane_dir / name
        if not self.dry_run:
            out.mkdir(parents=True, exist_ok=True)
        settings = self.condition_settings(condition)
        if self.triton_off and str(settings.get("calib_dtype")) == "bfloat16":
            print(f"{name}: BF16 calibration is unsupported on this GPU; using float32", flush=True)
            settings["calib_dtype"] = "float32"
        is_rtn = bool(condition.get("rtn"))
        corpus = self.resolve(condition.get("calibration_corpus")) if not is_rtn else None
        if not is_rtn and corpus is None:
            raise SystemExit(f"{name}: calibration_corpus is required unless rtn=true")

        existing = self.existing_result(name)
        if existing is not None:
            print(f"skip {name}: result.json already complete", flush=True)
            return existing

        skip_convert = (not is_rtn) and (bool(condition.get("skip_convert")) or (
            settings.get("raw_output") and bool(self.spec.get("merge")) and not condition.get("force_convert")
        ))
        calib_npz = out / "calibration.npz"
        if is_rtn:
            print(f"{name}: RTN (no calibration)", flush=True)
        elif not calib_npz.exists():
            command = [
                sys.executable,
                str(self.scripts["calibrate"]),
                "--model-dir",
                str(bf16),
                "--corpus",
                str(corpus),
                "--output",
                str(calib_npz),
                "--json-out",
                str(out / "calibration.json"),
                "--samples",
                str(settings["samples"]),
                "--max-tokens",
                str(settings["max_tokens"]),
                "--output-block",
                str(settings["output_block"]),
                "--output-block-limit",
                str(settings["output_block_limit"]),
                "--damping",
                str(settings["damping"]),
            ]
            for eval_corpus in self.eval_corpora().values():
                command += ["--eval-corpus", str(eval_corpus)]
            if settings.get("calib_dtype"):
                command += ["--calib-dtype", str(settings["calib_dtype"])]
            if settings.get("seed") is not None:
                command += ["--seed", str(settings["seed"])]
            if settings.get("raw_output"):
                command.append("--raw-output")
            if self.spec.get("convert", {}).get("parallel"):
                command += ["--parallel", str(self.spec["convert"]["parallel"])]
            self.run(f"{name} calibration", command, out / "lane.log")

        if skip_convert:
            summary = {
                "name": name,
                "status": "complete",
                "skipped": "convert" if not condition.get("force_convert") else None,
                "calibration": {
                    "corpus": str(corpus),
                    "samples": settings["samples"],
                    "max_tokens": settings["max_tokens"],
                    "output_block": settings["output_block"],
                    "damping": settings["damping"],
                    "raw_output": settings["raw_output"],
                },
                "ppl": {},
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "spec": str(self.spec_path),
            }
            if not self.dry_run:
                (out / "result.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
            return summary

        int4 = self.work / f"int4-{self.lane}-{name}"
        int4_eval = self.work / f"int4-{self.lane}-{name}-eval"
        convert_command = [
            sys.executable,
            str(self.scripts["convert"]),
            "--load-from",
            str(bf16),
            "--output-dir",
            str(int4),
        ]
        if not is_rtn:
            convert_command += [
                "--preconditioners",
                str(calib_npz),
                "--calibration-json",
                str(out / "calibration.json"),
            ]
        convert_settings = dict(self.spec.get("convert", {}))
        convert_settings.update(condition.get("convert") or {})
        for key in ("bits", "group_size", "block_size", "parallel"):
            if key in convert_settings:
                convert_command += [f"--{key.replace('_', '-')}", str(convert_settings[key])]
        if condition.get("layer_bits_json"):
            bits_path = self.resolve(condition["layer_bits_json"])
            assert bits_path is not None
            convert_command += ["--layer-bits-json", str(bits_path)]
        self.run(f"{name} convert", convert_command, out / "lane.log")
        self.run(
            f"{name} eval copy",
            [sys.executable, str(self.scripts["copy"]), "--source", str(int4), "--dest", str(int4_eval)],
            out / "lane.log",
        )

        summary = self.evaluate(name, int4_eval, bf16)
        summary["status"] = "complete"
        summary["env"] = self.env_provenance()
        summary["eval_dtype"] = self.eval_dtype
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        summary["spec"] = str(self.spec_path)
        if not self.dry_run:
            (out / "result.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

        if not self.dry_run:
            shutil.rmtree(int4, ignore_errors=True)
            shutil.rmtree(int4_eval, ignore_errors=True)
            calib_npz.unlink(missing_ok=True)
        return summary

    def run_merge(self, bf16: Path) -> list[dict[str, Any]]:
        merge = self.spec.get("merge")
        if not merge:
            return []
        en_raw = self.lane_dir / merge["en"] / "calibration.npz"
        ja_raw = self.lane_dir / merge["ja"] / "calibration.npz"
        summaries = []
        for ratio in merge.get("ratios", [0.5]):
            label = f"{float(ratio) * 100:.0f}" if float(ratio) <= 1 else str(ratio)
            name = merge.get("name_template", "balanced-{ratio}").format(ratio=label)
            out = self.lane_dir / name
            if not self.dry_run:
                out.mkdir(parents=True, exist_ok=True)
            existing = self.existing_result(name)
            if existing is not None:
                print(f"skip {name}: result.json already complete", flush=True)
                summaries.append(existing)
                continue
            merged = out / "calibration.npz"
            self.run(
                f"{name} merge",
                [
                    sys.executable,
                    str(self.scripts["merge"]),
                    "--en",
                    str(en_raw),
                    "--ja",
                    str(ja_raw),
                    "--ja-ratio",
                    str(ratio),
                    "--damping",
                    str(merge.get("damping", 1e-3)),
                    "--output",
                    str(merged),
                    "--json-out",
                    str(out / "calibration.json"),
                    "--en-json",
                    str(self.lane_dir / merge["en"] / "calibration.json"),
                    "--ja-json",
                    str(self.lane_dir / merge["ja"] / "calibration.json"),
                ],
                out / "lane.log",
            )
            int4 = self.work / f"int4-{self.lane}-{name}"
            int4_eval = self.work / f"int4-{self.lane}-{name}-eval"
            convert_command = [
                sys.executable,
                str(self.scripts["convert"]),
                "--load-from",
                str(bf16),
                "--output-dir",
                str(int4),
                "--preconditioners",
                str(merged),
                "--calibration-json",
                str(out / "calibration.json"),
            ]
            for key in ("bits", "group_size", "block_size", "parallel"):
                if key in self.spec.get("convert", {}):
                    convert_command += [f"--{key.replace('_', '-')}", str(self.spec["convert"][key])]
            self.run(f"{name} convert", convert_command, out / "lane.log")
            self.run(
                f"{name} eval copy",
                [sys.executable, str(self.scripts["copy"]), "--source", str(int4), "--dest", str(int4_eval)],
                out / "lane.log",
            )
            summary = self.evaluate(name, int4_eval, bf16)
            summary["status"] = "complete"
            summary["env"] = self.env_provenance()
            summary["eval_dtype"] = self.eval_dtype
            summary["ja_ratio"] = ratio
            summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            summary["spec"] = str(self.spec_path)
            if not self.dry_run:
                (out / "result.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
            if not self.dry_run:
                shutil.rmtree(int4, ignore_errors=True)
                shutil.rmtree(int4_eval, ignore_errors=True)
                merged.unlink(missing_ok=True)
            summaries.append(summary)
        return summaries

    # ---------- validation ----------

    def validate(self, summaries: list[dict[str, Any]]) -> list[str]:
        problems = []
        expected_chunks = self.spec.get("eval", {}).get("max_sequences", DEFAULT_EVAL["max_sequences"])
        baseline: dict[str, float] = {}
        for summary in summaries:
            name = summary.get("name")
            for lang, ppl in (summary.get("ppl") or {}).items():
                if ppl.get("chunks") != expected_chunks:
                    problems.append(f"{name}/{lang}: chunks={ppl.get('chunks')} (expected {expected_chunks})")
                value = ppl.get("bf16_ppl")
                if value is None:
                    problems.append(f"{name}/{lang}: missing bf16_ppl")
                    continue
                if lang not in baseline:
                    baseline[lang] = float(value)
                elif abs(baseline[lang] - float(value)) > 0.05:
                    problems.append(
                        f"{name}/{lang}: bf16_ppl drift {float(value):.2f} vs {baseline[lang]:.2f}"
                    )
        return problems

    def write_manifest(self, summaries: list[dict[str, Any]], problems: list[str]) -> Path:
        manifest = {
            "lane": self.lane,
            "spec": str(self.spec_path),
            "model": self.spec.get("model"),
            "eval": self.spec.get("eval"),
            "env": self.env_provenance(),
            "eval_dtype": self.eval_dtype,
            "results": summaries,
            "problems": problems,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = self.lane_dir / "manifest.json"
        if self.dry_run:
            print(json.dumps(manifest, indent=2, ensure_ascii=False)[:2000])
            return path
        self.lane_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def upload(self, repo: str) -> None:
        if self.dry_run:
            print(f"[dry-run] would upload {self.lane_dir} to {repo}")
            return
        subprocess.run(
            ["hf", "upload", repo, str(self.lane_dir), self.lane, "--repo-type", "dataset"],
            check=False,
        )


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hf-results", default=None, help="upload <lane_dir> to this HF dataset repo")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--work", default=None, help="override spec.paths.work")
    parser.add_argument("--jax-platform", choices=("cpu", "gpu"), default=None, help="override spec.paths.jax_platform")
    parser.add_argument("--eval-dtype", choices=("bfloat16", "float32"), default=None, help="override the eval model dtype")
    parser.add_argument("--parallel", choices=("tensor", "replicated"), default=None, help="override convert/calibration sharding")
    parser.add_argument(
        "--shard",
        default=None,
        help="run only part of the condition list, e.g. 0/3. Merges are not allowed with sharding.",
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    conditions = spec.get("conditions", [])
    if not conditions and not spec.get("merge"):
        raise SystemExit("spec has no conditions")

    if args.shard:
        if spec.get("merge"):
            raise SystemExit("--shard cannot be combined with spec.merge")
        try:
            index_text, count_text = args.shard.split("/")
            index, count = int(index_text), int(count_text)
        except ValueError:
            raise SystemExit("--shard must look like 0/3")
        if count <= 0 or not 0 <= index < count:
            raise SystemExit("--shard index out of range")
        conditions = [condition for position, condition in enumerate(conditions) if position % count == index]
        if not conditions:
            raise SystemExit(f"shard {args.shard} has no conditions")
        spec["conditions"] = conditions
        spec["lane"] = f"{spec['lane']}-shard{index + 1}of{count}"

    if args.jax_platform:
        spec.setdefault("paths", {})["jax_platform"] = args.jax_platform
    if args.eval_dtype:
        spec.setdefault("eval", {})["dtype"] = args.eval_dtype
    if args.parallel:
        spec.setdefault("convert", {})["parallel"] = args.parallel

    runner = Runner(spec, args.spec, dry_run=args.dry_run)
    if args.work:
        runner.work = Path(args.work).expanduser()
        runner.lane_dir = runner.work / "out" / runner.lane
    if not args.dry_run:
        runner.lane_dir.mkdir(parents=True, exist_ok=True)

    bf16 = runner.ensure_bf16()
    summaries = []
    for condition in conditions:
        summaries.append(runner.run_condition(condition, bf16))
    if not args.skip_merge:
        summaries.extend(runner.run_merge(bf16))

    problems = [] if args.dry_run else runner.validate(summaries)
    manifest = runner.write_manifest(summaries, problems)
    print(f"\nmanifest: {manifest}")
    if problems:
        print("VALIDATION PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    if args.hf_results:
        runner.upload(args.hf_results)
    print("LANE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
