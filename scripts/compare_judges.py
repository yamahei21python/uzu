#!/usr/bin/env python3
"""Compare two sets of judge scoring JSONLs.

Computes observed agreement (Po), expected agreement by chance (Pe),
Cohen's Kappa (kappa), and generates confusion matrices for overall
severity/verdict and per-check evaluations.

Usage:
    python scripts/compare_judges.py path/to/judge1.jsonl path/to/judge2.jsonl
    python scripts/compare_judges.py --judge1 j1.jsonl --judge2 j2.jsonl --threshold 0.70 --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at {path}:{idx}: {err}") from err
    return rows


def extract_record_fields(row: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    """Extract (id, severity_or_verdict, checks_dict) from a record."""
    record_id = str(row.get("id", ""))

    # If row is a calibration record with 'expected'
    if "expected" in row and isinstance(row["expected"], dict):
        sub = row["expected"]
        severity = str(sub.get("severity") or sub.get("verdict") or "none").lower()
        checks = sub.get("checks") or {}
    else:
        # Standard judge record
        severity = str(row.get("severity") or row.get("verdict") or row.get("overall_verdict") or "").lower()
        checks = row.get("checks") or row.get("check_results") or {}

    normalized_checks: dict[str, str] = {}
    if isinstance(checks, dict):
        for k, v in checks.items():
            if isinstance(v, dict):
                # e.g., {"status": "pass"}
                v_status = v.get("status") or v.get("verdict") or "na"
                normalized_checks[k] = str(v_status).lower()
            elif isinstance(v, bool):
                normalized_checks[k] = "pass" if v else "fail"
            else:
                normalized_checks[k] = str(v).lower()
    elif isinstance(checks, list):
        for item in checks:
            if isinstance(item, str):
                normalized_checks[item] = "pass"

    # Also extract root boolean checks if present (e.g., from eval_generation.py rows)
    standard_checks = {
        "refusal", "direct_answer", "over_refusal", "persona_hold",
        "format_ok", "length_ok", "numeric_ok", "ja_grammar"
    }
    for k in standard_checks:
        if k not in normalized_checks and k in row and isinstance(row[k], bool):
            normalized_checks[k] = "pass" if row[k] else "fail"

    # Derive overall verdict if severity is empty
    if not severity:
        if any(v == "fail" for v in normalized_checks.values()):
            severity = "fail"
        elif any(v == "pass" for v in normalized_checks.values()):
            severity = "pass"
        else:
            severity = "pass"

    return record_id, severity, normalized_checks


def compute_cohen_kappa(
    labels1: list[str],
    labels2: list[str],
    categories: list[str]
) -> tuple[float, float, float, dict[str, dict[str, int]]]:
    """Compute Po, Pe, Cohen's Kappa, and Confusion Matrix.

    Returns: (kappa, po, pe, confusion_matrix)
    """
    total = len(labels1)
    if total == 0:
        return 0.0, 0.0, 0.0, {}

    # Initialize confusion matrix: matrix[l1][l2]
    matrix: dict[str, dict[str, int]] = {c1: {c2: 0 for c2 in categories} for c1 in categories}

    agree_count = 0
    c1_counts = Counter(labels1)
    c2_counts = Counter(labels2)

    for l1, l2 in zip(labels1, labels2):
        if l1 in matrix and l2 in matrix[l1]:
            matrix[l1][l2] += 1
        if l1 == l2:
            agree_count += 1

    po = agree_count / total

    # Expected agreement Pe
    pe = sum((c1_counts.get(c, 0) / total) * (c2_counts.get(c, 0) / total) for c in categories)

    if abs(1.0 - pe) < 1e-9:
        # Perfect chance agreement or single class
        kappa = 1.0 if abs(po - 1.0) < 1e-9 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return kappa, po, pe, matrix


def format_confusion_matrix(
    matrix: dict[str, dict[str, int]],
    categories: list[str],
    label_j1: str = "Judge 1",
    label_j2: str = "Judge 2"
) -> str:
    """Format confusion matrix as an aligned text table."""
    col_width = max(max(len(c) for c in categories), 7)
    header_col = f"{label_j1} \\ {label_j2}"
    first_col_width = max(len(header_col), max(len(c) for c in categories) + 2)

    lines = []
    # Header
    header = f"{header_col:<{first_col_width}} | " + " | ".join(f"{c:>{col_width}}" for c in categories) + f" | {'Total':>{col_width}}"
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    col_totals = {c: 0 for c in categories}
    grand_total = 0

    for r in categories:
        row_vals = [matrix[r][c] for c in categories]
        row_sum = sum(row_vals)
        for c in categories:
            col_totals[c] += matrix[r][c]
        grand_total += row_sum
        row_str = f"{r:<{first_col_width}} | " + " | ".join(f"{v:>{col_width}}" for v in row_vals) + f" | {row_sum:>{col_width}}"
        lines.append(row_str)

    lines.append(sep)
    bottom_str = f"{'Total':<{first_col_width}} | " + " | ".join(f"{col_totals[c]:>{col_width}}" for c in categories) + f" | {grand_total:>{col_width}}"
    lines.append(bottom_str)

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement and confusion matrix between two judges."
    )
    parser.add_argument("pos_judge1", nargs="?", type=Path, help="Path to first judge JSONL")
    parser.add_argument("pos_judge2", nargs="?", type=Path, help="Path to second judge JSONL")
    parser.add_argument("--judge1", "-j1", type=Path, help="Path to first judge JSONL")
    parser.add_argument("--judge2", "-j2", type=Path, help="Path to second judge JSONL")
    parser.add_argument("--threshold", "-t", type=float, default=0.70, help="Cohen's kappa pass threshold (default: 0.70)")
    parser.add_argument("--field", "-f", choices=["primary", "checks", "both"], default="both", help="Evaluation focus")
    parser.add_argument("--json", "-o", type=Path, help="Save machine-readable summary report to JSON file")

    args = parser.parse_args()

    j1_path = args.judge1 or args.pos_judge1
    j2_path = args.judge2 or args.pos_judge2

    if not j1_path or not j2_path:
        parser.error("Two judge JSONL files must be specified.")

    j1_rows = read_jsonl(j1_path)
    j2_rows = read_jsonl(j2_path)

    j1_dict = {extract_record_fields(r)[0]: extract_record_fields(r) for r in j1_rows if extract_record_fields(r)[0]}
    j2_dict = {extract_record_fields(r)[0]: extract_record_fields(r) for r in j2_rows if extract_record_fields(r)[0]}

    common_ids = sorted(set(j1_dict.keys()) & set(j2_dict.keys()))

    if not common_ids:
        print(f"Error: No matching IDs found between {j1_path} and {j2_path}.", file=sys.stderr)
        return 1

    print("=" * 72)
    print(" uzu Judge Agreement & Calibration Evaluator")
    print("=" * 72)
    print(f"Judge 1: {j1_path} ({len(j1_rows)} records)")
    print(f"Judge 2: {j2_path} ({len(j2_rows)} records)")
    print(f"Paired comparisons: {len(common_ids)} items")
    print(f"Target Threshold: Cohen's Kappa >= {args.threshold:.2f}")
    print("-" * 72)

    # 1. Primary Verdict / Severity comparison
    sev1 = [j1_dict[i][1] for i in common_ids]
    sev2 = [j2_dict[i][1] for i in common_ids]

    all_severities = sorted(set(sev1) | set(sev2))
    kappa_sev, po_sev, pe_sev, matrix_sev = compute_cohen_kappa(sev1, sev2, all_severities)

    status_sev = "PASS" if kappa_sev >= args.threshold else "FAIL"

    print("\n[Primary Metric / Severity Agreement]")
    print(f"  Observed Agreement (Po): {po_sev:.4f} ({po_sev*100:.1f}%)")
    print(f"  Chance Agreement (Pe):   {pe_sev:.4f} ({pe_sev*100:.1f}%)")
    print(f"  Cohen's Kappa (κ):       {kappa_sev:.4f}  ==>  [{status_sev}]")
    print("\nConfusion Matrix (Primary Metric):")
    print(format_confusion_matrix(matrix_sev, all_severities, "Judge 1", "Judge 2"))

    # 2. Per-Check comparison
    all_check_names = sorted({
        k for i in common_ids
        for k in (set(j1_dict[i][2].keys()) | set(j2_dict[i][2].keys()))
    })

    checks_summary = {}
    if args.field in ("checks", "both") and all_check_names:
        print("\n" + "-" * 72)
        print("[Per-Check Breakdown (pass / fail / na)]")
        check_categories = ["pass", "fail", "na"]

        for check_name in all_check_names:
            c_vals1 = [j1_dict[i][2].get(check_name, "na") for i in common_ids]
            c_vals2 = [j2_dict[i][2].get(check_name, "na") for i in common_ids]

            # Only compute if at least one judge evaluated it as pass or fail
            active_pairs = [(v1, v2) for v1, v2 in zip(c_vals1, c_vals2) if v1 != "na" or v2 != "na"]
            active_count = len(active_pairs)

            if active_count == 0:
                continue

            k_c, po_c, pe_c, mat_c = compute_cohen_kappa(c_vals1, c_vals2, check_categories)
            checks_summary[check_name] = {
                "active_items": active_count,
                "observed_agreement": round(po_c, 4),
                "chance_agreement": round(pe_c, 4),
                "cohen_kappa": round(k_c, 4),
                "pass_threshold": bool(k_c >= args.threshold)
            }
            c_status = "PASS" if k_c >= args.threshold else "FAIL"
            print(f"  - {check_name:<15} (N={active_count:2d}): Po={po_c:.3f}, κ={k_c:.3f} [{c_status}]")

    print("\n" + "=" * 72)
    overall_pass = kappa_sev >= args.threshold
    if overall_pass:
        print(f"RESULT: OVERALL AGREEMENT PASSED (κ = {kappa_sev:.4f} >= {args.threshold:.2f})")
    else:
        print(f"RESULT: OVERALL AGREEMENT FAILED (κ = {kappa_sev:.4f} < {args.threshold:.2f})")
    print("=" * 72)

    report_data = {
        "judge1": str(j1_path),
        "judge2": str(j2_path),
        "paired_count": len(common_ids),
        "threshold": args.threshold,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "primary_metric": {
            "categories": all_severities,
            "observed_agreement": round(po_sev, 4),
            "chance_agreement": round(pe_sev, 4),
            "cohen_kappa": round(kappa_sev, 4),
            "status": status_sev,
            "confusion_matrix": matrix_sev
        },
        "checks": checks_summary
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Detailed JSON report written to {args.json}")

    return 0 if overall_pass else 2


if __name__ == "__main__":
    sys.exit(main())
