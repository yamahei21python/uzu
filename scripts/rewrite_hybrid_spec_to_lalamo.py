#!/usr/bin/env python3
"""Convert uzu-format HybridSpec metadata back to lalamo's nested form.

uzu stores:
  {"type":"HybridSpec", ..., "incoherence_block_size":32,
   "incoherence_processing_mode":"input_output"}
lalamo expects:
  {"type":"HybridSpec", ..., "incoherence_spec":
   {"type":"HadamardSpec","block_size":32,"processing_mode":"input_output"}}

Used to evaluate distributed Mirai checkpoints with lalamo (PPL / logits).
Tensor bytes are copied unchanged.

Usage:
  python scripts/rewrite_hybrid_spec_to_lalamo.py /path/to/model.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path


def parse_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("not a safetensors file: short header length")
        header_length = struct.unpack("<Q", raw_length)[0]
        header_bytes = handle.read(header_length)
    return json.loads(header_bytes), 8 + header_length


def nest_hybrid(spec: dict) -> dict:
    if spec.get("type") != "HybridSpec":
        return spec
    if "incoherence_spec" in spec and "incoherence_block_size" not in spec:
        return spec

    block_size = spec.pop("incoherence_block_size", None)
    processing_mode = spec.pop("incoherence_processing_mode", "input_output")
    if block_size is None:
        spec["incoherence_spec"] = None
    else:
        spec["incoherence_spec"] = {
            "type": "HadamardSpec",
            "block_size": block_size,
            "processing_mode": processing_mode,
        }
    return spec


def rewrite(path: Path, *, dry_run: bool = False) -> int:
    header, data_start = parse_header(path)
    metadata = header.get("__metadata__")
    if not metadata:
        print("no __metadata__")
        return 0

    changed = 0
    for key, value in list(metadata.items()):
        if not key.endswith(".spec") and key != "spec":
            continue
        try:
            spec = json.loads(value)
        except json.JSONDecodeError:
            continue
        if spec.get("type") != "HybridSpec" or "incoherence_block_size" not in spec:
            continue
        metadata[key] = json.dumps(nest_hybrid(spec), separators=(",", ":"))
        changed += 1

    if changed == 0:
        print("no uzu HybridSpec metadata found")
        return 0
    if dry_run:
        print(f"dry-run: would rewrite {changed} entries")
        return changed

    new_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    new_header += b" " * ((8 - len(new_header) % 8) % 8)

    temporary = path.with_name(path.name + ".rewrite.tmp")
    with path.open("rb") as source, temporary.open("wb") as destination:
        destination.write(struct.pack("<Q", len(new_header)))
        destination.write(new_header)
        source.seek(data_start)
        shutil.copyfileobj(source, destination, length=1024 * 1024 * 16)
    os.replace(temporary, path)
    print(f"rewrote {changed} metadata entries in {path}")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rewrite(args.path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
