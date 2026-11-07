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
    "max_tokens": 128,
    "output_block": "small",
    "output_block_limit": 2048,
    "damping": 1e-3,
    "raw_output": False,
}
DEFAULT_EVAL = {"max_sequences": 100, "max_tokens": 128, "min_tokens": 16}


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

    def run(self, step: str, command: list[str], log: Path) -> None:
        display = " ".join(str(part) for part in command)
        print(f"$ {display}", flush=True)
        self.commands.append([str(part) for part in command])
        if self.dry_run:
            return
        env = os.environ.copy()
        if self.jax_platform:
            env["JAX_PLATFORM_NAME"] = self.jax_platform
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
        model = self.spec.get("model", {})
        bf16 = self.resolve(model.get("bf16_dir", "bf16-lalamo"))
        assert bf16 is not None
        weights = bf16 / "model.safetensors"
        if weights.exists() or self.dry_run:
            return bf16
        source = self.resolve(model.get("bf16_source"))
        if source and (source / "model.safetensors").exists():
            print(f"copy bf16 from {source} -> {bf16}", flush=True)
            shutil.copytree(source, bf16, dirs_exist_ok=True)
            return bf16
        command = [
            sys.executable,
            str(self.scripts["convert"]),
            "--spec-name",
            model.get("spec_name", "Qwen3.5-0.8B"),
            "--output-dir",
            str(self.work / "import"),
            "--dry-run",
        ]
        if model.get("hf_repo"):
            command += ["--hf-repo", model["hf_repo"]]
        if model.get("hf_dir"):
            command += ["--hf-dir", str(self.resolve(model["hf_dir"]))]
        command += ["--bf16-dir", str(bf16)]
        self.run("bf16 import", command, self.lane_dir / "lane.log")
        return bf16

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
        corpus = self.resolve(condition["calibration_corpus"])
        assert corpus is not None

        existing = self.existing_result(name)
        if existing is not None:
            print(f"skip {name}: result.json already complete", flush=True)
            return existing

        skip_convert = bool(condition.get("skip_convert")) or (
            settings.get("raw_output") and bool(self.spec.get("merge")) and not condition.get("force_convert")
        )
        calib_npz = out / "calibration.npz"
        if not calib_npz.exists():
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
            "--preconditioners",
            str(calib_npz),
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
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    conditions = spec.get("conditions", [])
    if not conditions and not spec.get("merge"):
        raise SystemExit("spec has no conditions")

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
