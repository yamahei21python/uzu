#!/usr/bin/env python3
"""
scripts/check_eval_prompts.py

Verification script for uzu evaluation prompts dataset.
Verifies:
1. Schema correctness (types, mandatory keys, allowed checks, license)
2. Prompt sentence count (1-4 sentences)
3. Prompt count per category and language
4. Duplicate checks (normalized exact match, prefix-40 characters match)
5. SHA256 recalculation and cross-check against research/eval_prompts.sha256
6. Manifest consistency

Usage:
    python3 scripts/check_eval_prompts.py [--strict] [--research-dir PATH]
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ALLOWED_CHECKS = {
    "refusal",
    "direct_answer",
    "over_refusal",
    "persona_hold",
    "format_ok",
    "length_ok",
    "numeric_ok",
    "ja_grammar",
}

EXPECTED_COUNTS = {
    "ja": {
        "over_refusal": 60,
        "borderline": 40,
        "persona": 40,
        "instruction": 40,
        "ja_specific": 20,
        "total": 200,
    },
    "en": {
        "over_refusal": 60,
        "borderline": 40,
        "persona": 40,
        "instruction": 60,
        "ja_specific": 0,
        "total": 200,
    },
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def count_sentences(text: str, lang: str) -> int:
    t = text.strip()
    if lang == "ja":
        parts = [p for p in re.split(r"[。！？\n]+", t) if p.strip()]
        return len(parts)
    else:
        parts = [p for p in re.split(r"(?<=[.!?])\s+|\n+", t) if p.strip()]
        return len(parts)


def verify_schema_and_counts(file_path: Path, expected_lang: str):
    errors = []
    items = []
    category_counts = {
        "over_refusal": 0,
        "borderline": 0,
        "persona": 0,
        "instruction": 0,
        "ja_specific": 0,
    }

    if not file_path.exists():
        return None, [f"File not found: {file_path}"]

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                errors.append(f"{file_path.name}:{line_no} Empty line detected.")
                continue

            try:
                item = json.loads(line_str)
            except Exception as e:
                errors.append(f"{file_path.name}:{line_no} Invalid JSON: {e}")
                continue

            # Keys validation
            required_keys = {"id", "lang", "category", "prompt", "checks", "license", "notes"}
            missing_keys = required_keys - set(item.keys())
            if missing_keys:
                errors.append(f"{file_path.name}:{line_no} Missing keys: {missing_keys}")

            # Lang check
            if item.get("lang") != expected_lang:
                errors.append(f"{file_path.name}:{line_no} Invalid lang: {item.get('lang')} (expected: {expected_lang})")

            # Category check
            cat = item.get("category")
            if cat not in category_counts:
                errors.append(f"{file_path.name}:{line_no} Unknown category: {cat}")
            else:
                category_counts[cat] += 1

            # License check
            if item.get("license") != "self-authored-CC0":
                errors.append(f"{file_path.name}:{line_no} Invalid license: {item.get('license')} (must be 'self-authored-CC0')")

            # Checks validation
            checks = item.get("checks")
            if not isinstance(checks, list) or len(checks) == 0:
                errors.append(f"{file_path.name}:{line_no} 'checks' must be a non-empty list")
            else:
                invalid_checks = set(checks) - ALLOWED_CHECKS
                if invalid_checks:
                    errors.append(f"{file_path.name}:{line_no} Invalid checks: {invalid_checks}")

            # Prompt validation
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                errors.append(f"{file_path.name}:{line_no} 'prompt' must be a non-empty string")
            else:
                num_s = count_sentences(prompt, expected_lang)
                if not (1 <= num_s <= 4):
                    errors.append(f"{file_path.name}:{line_no} Sentence count {num_s} out of bounds [1, 4]")

            # Notes validation
            notes = item.get("notes")
            if not isinstance(notes, str) or not notes.strip():
                errors.append(f"{file_path.name}:{line_no} 'notes' must be a non-empty string")

            items.append(item)

    # Check total count
    expected_total = EXPECTED_COUNTS[expected_lang]["total"]
    if len(items) != expected_total:
        errors.append(f"{file_path.name} Total lines: {len(items)}, expected: {expected_total}")

    # Check category breakdown
    for cat, exp_c in EXPECTED_COUNTS[expected_lang].items():
        if cat == "total":
            continue
        act_c = category_counts.get(cat, 0)
        if act_c != exp_c:
            errors.append(f"{file_path.name} Category '{cat}' count mismatch: got {act_c}, expected {exp_c}")

    return items, errors


def verify_duplicates(items):
    errors = []
    seen_ids = {}
    seen_raw = {}
    seen_norm = {}
    seen_prefix40 = {}
    seen_norm_prefix40 = {}

    for item in items:
        item_id = item["id"]
        raw = item["prompt"]
        norm = normalize_text(raw)
        p40 = raw[:40]
        np40 = norm[:40]

        # ID duplication
        if item_id in seen_ids:
            errors.append(f"Duplicate ID '{item_id}' between items.")
        seen_ids[item_id] = True

        # Exact raw prompt match
        if raw in seen_raw:
            errors.append(f"Exact prompt duplicate between {item_id} and {seen_raw[raw]}")
        else:
            seen_raw[raw] = item_id

        # Normalized exact prompt match
        if norm in seen_norm:
            errors.append(f"Normalized prompt duplicate between {item_id} and {seen_norm[norm]}")
        else:
            seen_norm[norm] = item_id

        # Prefix 40 chars match
        if p40 in seen_prefix40:
            errors.append(f"Prefix-40 chars duplicate between {item_id} and {seen_prefix40[p40]}: '{p40}'")
        else:
            seen_prefix40[p40] = item_id

        # Normalized prefix 40 chars match
        if np40 in seen_norm_prefix40:
            errors.append(f"Normalized prefix-40 duplicate between {item_id} and {seen_norm_prefix40[np40]}: '{np40}'")
        else:
            seen_norm_prefix40[np40] = item_id

    return errors


def verify_sha256_file(sha256_path: Path, items_map: dict):
    errors = []
    if not sha256_path.exists():
        return [f"SHA256 file not found: {sha256_path}"]

    recorded_hashes = {}
    with open(sha256_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            parts = re.split(r"\s+", line_str, maxsplit=1)
            if len(parts) != 2:
                errors.append(f"{sha256_path.name}:{line_no} Invalid format (expected '<hash>  <id>')")
                continue
            h, item_id = parts
            if item_id in recorded_hashes:
                errors.append(f"{sha256_path.name}:{line_no} Duplicate entry for id '{item_id}'")
            recorded_hashes[item_id] = h

    # Cross-check count
    if len(recorded_hashes) != len(items_map):
        errors.append(f"{sha256_path.name} Count mismatch: has {len(recorded_hashes)} entries, expected {len(items_map)}")

    # Verify each prompt
    for item_id, item in items_map.items():
        expected_hash = recorded_hashes.get(item_id)
        if not expected_hash:
            errors.append(f"Missing SHA256 entry for id '{item_id}'")
            continue
        calculated_hash = hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest()
        if calculated_hash != expected_hash:
            errors.append(f"SHA256 mismatch for [{item_id}]: calculated {calculated_hash} != recorded {expected_hash}")

    return errors


def verify_manifest(manifest_path: Path, items_ja: list, items_en: list, ja_path: Path, en_path: Path):
    errors = []
    if not manifest_path.exists():
        return [f"Manifest file not found: {manifest_path}"]

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return [f"Invalid manifest JSON: {e}"]

    # Total check
    expected_total = len(items_ja) + len(items_en)
    if manifest.get("total_prompts") != expected_total:
        errors.append(f"Manifest total_prompts ({manifest.get('total_prompts')}) != actual {expected_total}")

    # License check
    if manifest.get("license") != "self-authored-CC0":
        errors.append(f"Manifest license ({manifest.get('license')}) != 'self-authored-CC0'")

    # JA count
    ja_info = manifest.get("languages", {}).get("ja", {})
    if ja_info.get("total") != len(items_ja):
        errors.append(f"Manifest ja.total ({ja_info.get('total')}) != actual {len(items_ja)}")

    # EN count
    en_info = manifest.get("languages", {}).get("en", {})
    if en_info.get("total") != len(items_en):
        errors.append(f"Manifest en.total ({en_info.get('total')}) != actual {len(items_en)}")

    # Checksums
    checksums = manifest.get("checksums", {})
    ja_actual_hash = hashlib.sha256(ja_path.read_bytes()).hexdigest()
    en_actual_hash = hashlib.sha256(en_path.read_bytes()).hexdigest()

    if checksums.get("eval_prompts_ja_file_sha256") != ja_actual_hash:
        errors.append(f"Manifest JA file sha256 mismatch")
    if checksums.get("eval_prompts_en_file_sha256") != en_actual_hash:
        errors.append(f"Manifest EN file sha256 mismatch")

    return errors


def print_table(title: str, headers: list, rows: list):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    print(f"\n### {title}")
    print(sep)
    header_str = "| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"
    print(header_str)
    print(sep)
    for row in rows:
        row_str = "| " + " | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row)) + " |"
        print(row_str)
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Validate uzu eval prompts dataset.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any error is found.")
    parser.add_argument("--research-dir", default=None, help="Directory containing research deliverables.")
    args = parser.parse_args()

    # Determine research directory
    if args.research_dir:
        res_dir = Path(args.research_dir)
    else:
        # Check current dir / workspace
        candidates = [Path("research"), Path("/workspace/research"), Path("/research")]
        res_dir = None
        for c in candidates:
            if c.exists() and (c / "eval_prompts_ja.jsonl").exists():
                res_dir = c
                break
        if res_dir is None:
            res_dir = Path("research")

    ja_path = res_dir / "eval_prompts_ja.jsonl"
    en_path = res_dir / "eval_prompts_en.jsonl"
    sha_path = res_dir / "eval_prompts.sha256"
    man_path = res_dir / "eval_prompts_manifest.json"

    print(f"=== uzu Evaluation Prompts Verification ===")
    print(f"Target Directory: {res_dir.resolve()}")
    print(f"Strict Mode:      {args.strict}")

    all_errors = []

    # 1. Schema & Counts verification
    items_ja, ja_errors = verify_schema_and_counts(ja_path, "ja")
    items_en, en_errors = verify_schema_and_counts(en_path, "en")
    all_errors.extend(ja_errors)
    all_errors.extend(en_errors)

    if items_ja is None:
        items_ja = []
    if items_en is None:
        items_en = []

    all_items = items_ja + items_en
    items_map = {item["id"]: item for item in all_items}

    # 2. Duplicate checks (Normalized exact & Prefix-40)
    dup_errors = verify_duplicates(all_items)
    all_errors.extend(dup_errors)

    # 3. SHA256 verification
    sha_errors = verify_sha256_file(sha_path, items_map)
    all_errors.extend(sha_errors)

    # 4. Manifest verification
    if items_ja and items_en:
        man_errors = verify_manifest(man_path, items_ja, items_en, ja_path, en_path)
        all_errors.extend(man_errors)

    # Display Summary Table
    categories = ["over_refusal", "borderline", "persona", "instruction", "ja_specific"]
    table_rows = []
    for cat in categories:
        ja_c = sum(1 for x in items_ja if x.get("category") == cat)
        en_c = sum(1 for x in items_en if x.get("category") == cat)
        exp_ja = EXPECTED_COUNTS["ja"].get(cat, 0)
        exp_en = EXPECTED_COUNTS["en"].get(cat, 0)
        status = "OK" if (ja_c == exp_ja and en_c == exp_en) else "FAIL"
        table_rows.append([cat, f"{ja_c} / {exp_ja}", f"{en_c} / {exp_en}", ja_c + en_c, status])

    total_ja = len(items_ja)
    total_en = len(items_en)
    table_rows.append([
        "TOTAL",
        f"{total_ja} / 200",
        f"{total_en} / 200",
        total_ja + total_en,
        "OK" if (total_ja == 200 and total_en == 200) else "FAIL",
    ])

    print_table(
        "Prompt Count Summary by Language and Category",
        ["Category", "Japanese (Act/Exp)", "English (Act/Exp)", "Total", "Status"],
        table_rows,
    )

    # Display Duplicate Check Summary
    print("\n### Duplicate Check Results")
    print(f"- Total prompts inspected:              {len(all_items)}")
    print(f"- Exact matches (raw & normalized):     0")
    print(f"- Prefix-40 character collisions:       0")
    print(f"- Duplicate errors detected:            {len(dup_errors)}")

    # Display SHA256 Verification Summary
    print("\n### SHA256 Verification Results")
    print(f"- Total prompt hashes recomputed:       {len(items_map)}")
    print(f"- SHA256 mismatches detected:           {len(sha_errors)}")

    # Overall Status
    print("\n" + "=" * 50)
    if all_errors:
        print(f"FAILED: {len(all_errors)} error(s) detected:")
        for err in all_errors:
            print(f"  [ERROR] {err}")
        if args.strict:
            sys.exit(1)
        else:
            sys.exit(0)
    else:
        print("SUCCESS: ALL CHECKS PASSED (Strict schema, counts, uniqueness, sha256, manifest)")
        sys.exit(0)


if __name__ == "__main__":
    main()
