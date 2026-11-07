#!/usr/bin/env python3
"""Fail fast when an experiment mixes incompatible eval/calibration settings."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def expected_eval(config: dict, corpus: str) -> dict:
    eval_cfg = config["eval"]
    corpus_path = Path(corpus)
    return {
        "corpus": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "max_sequences": eval_cfg["max_sequences"],
        "max_tokens": eval_cfg["max_tokens"],
        "min_tokens": eval_cfg["min_tokens"],
    }


def config_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    root = args.root or args.config.parent
    eval_cfg = config["eval"]
    calib_cfg = config["calibration"]
    corpora = {Path(path).name: path for path in eval_cfg["corpora"].values()}
    expected_by_name = {name: expected_eval(config, path) for name, path in corpora.items()}

    errors: list[str] = []
    warnings: list[str] = []
    ppl_rows = []
    baseline_by_corpus: dict[str, list[float]] = defaultdict(list)

    for path in sorted(root.rglob("ppl_*.json")):
        data = json.loads(path.read_text())
        corpus_name = Path(data.get("corpus", "")).name
        row = {"path": str(path), "corpus": corpus_name}
        ppl_rows.append(row)

        if corpus_name not in corpora:
            errors.append(f"{path}: unknown corpus {corpus_name!r}")
            continue

        expected = expected_by_name[corpus_name]
        if data.get("chunks") != expected["max_sequences"]:
            errors.append(
                f"{path}: chunks={data.get('chunks')} != config max_sequences={expected['max_sequences']}"
            )
        if data.get("tokens") is not None and data.get("chunks"):
            approx = data["chunks"] * expected["max_tokens"] - data["chunks"]
            if abs(int(data["tokens"]) - approx) > expected["max_sequences"]:
                warnings.append(
                    f"{path}: tokens={data.get('tokens')} differs from approx {approx}"
                )

        if data.get("config"):
            if data["config"] != expected:
                errors.append(f"{path}: embedded eval config differs from model config")
            if data.get("config_hash") != config_hash(expected):
                errors.append(f"{path}: config_hash mismatch")
        else:
            warnings.append(f"{path}: legacy result without config/config_hash")

        if "bf16_ppl" in data:
            baseline_by_corpus[corpus_name].append(float(data["bf16_ppl"]))

    for corpus_name, values in baseline_by_corpus.items():
        if max(values) - min(values) > 0.01:
            errors.append(
                f"BF16 baseline drift for {corpus_name}: {sorted(set(round(value, 4) for value in values))}"
            )

    calibration_rows = []
    calibration_corpora: set[Path] = set(
        Path(path) for path in calib_cfg.get("corpora", {}).values()
    )
    qad_corpus = config.get("qad", {}).get("corpus")
    if qad_corpus:
        calibration_corpora.add(Path(qad_corpus))
    for path in sorted(root.rglob("calibration.json")):
        data = json.loads(path.read_text())
        row = {"path": str(path), "corpus": Path(data.get("corpus", "")).name}
        calibration_rows.append(row)
        if data.get("corpus"):
            calibration_corpora.add(Path(data["corpus"]))
        checks = {
            "samples": calib_cfg["samples"],
            "max_tokens": calib_cfg["max_tokens"],
            "output_block": calib_cfg["output_block"],
        }
        for key, expected in checks.items():
            if data.get(key) != expected:
                errors.append(f"{path}: {key}={data.get(key)!r} != config {expected!r}")
        damping = float(data.get("damping", -1))
        if abs(damping - float(calib_cfg["damping"])) > 1e-12:
            errors.append(f"{path}: damping={damping} != config {calib_cfg['damping']}")

    eval_corpora = {Path(path) for path in corpora.values()}

    def long_lines(path: Path) -> set[str]:
        return {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if len(line.strip()) >= 20
        }

    for eval_corpus in sorted(eval_corpora):
        if not eval_corpus.exists():
            continue
        lines = [
            line.strip()
            for line in eval_corpus.read_text(encoding="utf-8").splitlines()
            if len(line.strip()) >= 20
        ]
        duplicates = len(lines) - len(set(lines))
        if duplicates:
            errors.append(f"{eval_corpus}: {duplicates} duplicate long lines; dedup required")

    overlap_checks = 0
    for calibration_corpus in sorted(calibration_corpora):
        if not calibration_corpus.exists():
            errors.append(f"missing calibration corpus: {calibration_corpus}")
            continue
        for eval_corpus in sorted(eval_corpora):
            if not eval_corpus.exists():
                errors.append(f"missing eval corpus: {eval_corpus}")
                continue
            overlap_checks += 1
            if calibration_corpus.resolve() == eval_corpus.resolve():
                errors.append(f"calibration and eval use the same file: {eval_corpus}")
                continue
            overlap = long_lines(calibration_corpus) & long_lines(eval_corpus)
            if overlap:
                sample = next(iter(overlap))[:80]
                errors.append(
                    f"corpus overlap: {calibration_corpus} vs {eval_corpus} "
                    f"({len(overlap)} lines, e.g. {sample!r})"
                )

    report = {
        "config": str(args.config),
        "config_name": config.get("name"),
        "ppl_rows": len(ppl_rows),
        "calibration_rows": len(calibration_rows),
        "corpus_overlap_checks": overlap_checks,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
