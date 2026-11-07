#!/usr/bin/env python3
"""Build a QAD-ready student dataset from a clean calibration.

QAD needs three things that the generic INT4 conversion does not hand you in one
place:

1. a student whose ``config.json`` is the raw lalamo form (``response_format``
   present); conversion rewrites the config for uzu
2. a ``provenance.json`` with ``overlap_checked: true``; QAD refuses a student
   without it
3. a tokenizer identical to the teacher

This script runs calibration -> convert -> lalamo copy, copies the provenance
next to the student, verifies all three conditions, and writes a manifest that
can be uploaded as a Kaggle dataset.

Usage:
    python scripts/make_qad_dataset.py \
        --bf16 /kaggle/working/bf16-lalamo \
        --out /kaggle/working/qad-student \
        --corpus data/calibration_mixed_en_ja_dedup.txt \
        --eval-corpus data/eval_en_wikitext2_dedup.txt \
        --eval-corpus data/eval_ja_wikipedia_dedup.txt \
        --kaggle-id <user>/uzu-qad-08b-clean
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(str(part) for part in command), flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in command], check=True)


def validate_student(bf16: Path, student: Path) -> list[str]:
    """Return the reasons QAD would refuse this student."""
    problems: list[str] = []
    config = json.loads((student / "config.json").read_text(encoding="utf-8"))
    if "response_format" not in config.get("token_codec_config", {}):
        problems.append("student config is uzu-normalized (no response_format)")

    provenance_path = student / "provenance.json"
    if not provenance_path.exists():
        problems.append("student provenance.json missing")
    else:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not provenance.get("overlap_checked"):
            problems.append("student provenance says overlap_checked=false")

    bf16_tokenizer = bf16 / "tokenizer.json"
    student_tokenizer = student / "tokenizer.json"
    if not student_tokenizer.exists():
        problems.append("student tokenizer.json missing")
    elif bf16_tokenizer.exists() and sha256(bf16_tokenizer) != sha256(student_tokenizer):
        problems.append("student tokenizer.json differs from bf16")

    for name in ("model.safetensors",):
        if not (student / name).exists():
            problems.append(f"student {name} missing")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bf16", type=Path, required=True, help="raw lalamo BF16 directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True, help="calibration corpus")
    parser.add_argument("--eval-corpus", type=Path, action="append", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output-block", default="small")
    parser.add_argument("--output-block-limit", type=int, default=2048)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--parallel", default="tensor")
    parser.add_argument("--calibration-json", type=Path, default=None, help="reuse an existing calibration.json")
    parser.add_argument("--calibration-npz", type=Path, default=None, help="reuse an existing calibration.npz")
    parser.add_argument("--kaggle-id", default=None, help="write dataset-metadata.json for this Kaggle id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = args.out
    calibration_json = args.calibration_json or (out / "calibration.json")
    calibration_npz = args.calibration_npz or (out / "calibration.npz")
    int4_dir = out / "int4"
    student = out / "student"

    if args.dry_run or not calibration_npz.exists():
        command = [
            sys.executable,
            str(HERE / "yaqa_calibration.py"),
            "--model-dir",
            str(args.bf16),
            "--corpus",
            str(args.corpus),
            "--output",
            str(calibration_npz),
            "--json-out",
            str(calibration_json),
            "--samples",
            str(args.samples),
            "--max-tokens",
            str(args.max_tokens),
            "--output-block",
            str(args.output_block),
            "--output-block-limit",
            str(args.output_block_limit),
            "--damping",
            str(args.damping),
            "--parallel",
            args.parallel,
        ]
        for corpus in args.eval_corpus:
            command += ["--eval-corpus", str(corpus)]
        run(command, args.dry_run)

    run(
        [
            sys.executable,
            str(HERE / "rht_int4_convert.py"),
            "--load-from",
            str(args.bf16),
            "--output-dir",
            str(int4_dir),
            "--preconditioners",
            str(calibration_npz),
            "--calibration-json",
            str(calibration_json),
            "--parallel",
            args.parallel,
        ],
        args.dry_run,
    )
    run(
        [
            sys.executable,
            str(HERE / "make_lalamo_eval_copy.py"),
            "--source",
            str(int4_dir),
            "--dest",
            str(student),
        ],
        args.dry_run,
    )

    if args.dry_run:
        print("[dry-run] would copy provenance.json and validate")
        return

    provenance_source = int4_dir / "provenance.json"
    if not provenance_source.exists():
        raise SystemExit(f"conversion did not write provenance: {provenance_source}")
    shutil.copy2(provenance_source, student / "provenance.json")

    problems = validate_student(args.bf16, student)
    if problems:
        print("STUDENT INVALID")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    provenance = json.loads((student / "provenance.json").read_text(encoding="utf-8"))
    manifest = {
        "kind": "qad-student",
        "teacher": str(args.bf16),
        "student": str(student),
        "calibration": {
            "corpus": str(args.corpus),
            "eval_corpora": [str(corpus) for corpus in args.eval_corpus],
            "samples": args.samples,
            "max_tokens": args.max_tokens,
            "output_block": args.output_block,
            "damping": args.damping,
            "corpus_sha256": [sha256(corpus) for corpus in [args.corpus, *args.eval_corpus]],
        },
        "provenance_sha256": sha256(student / "provenance.json"),
        "provenance_kind": provenance.get("kind"),
        "overlap_checked": bool(provenance.get("overlap_checked")),
        "tokenizer_sha256": sha256(student / "tokenizer.json"),
    }
    (out / "qad_dataset.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.kaggle_id:
        metadata = {
            "title": args.kaggle_id.split("/")[-1].replace("-", " "),
            "id": args.kaggle_id,
            "licenses": [{"name": "Apache-2.0"}],
            "subtitle": "QAD student with clean calibration provenance",
            "description": (
                "Student produced by clean YAQA calibration (calibration and eval corpora are disjoint) "
                "for uzu QAD. Contains student/, calibration.json, qad_dataset.json.\n\n"
                f"teacher: {args.bf16}\ncorpus: {args.corpus}\n"
                "See qad_dataset.json for hashes."
            ),
            "resources": [],
        }
        (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"student ready: {student}")
    print(f"manifest: {out / 'qad_dataset.json'}")


if __name__ == "__main__":
    main()
