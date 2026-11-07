from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_manifest(root: Path, name: str, results: list[dict], problems: list[str] | None = None) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    payload = {
        "lane": name.rsplit("-shard", 1)[0],
        "results": results,
        "problems": problems or [],
    }
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def result(name: str, en_ppl: float = 39.0, ja_ppl: float = 49.0, bf16: float = 35.38) -> dict:
    return {
        "name": name,
        "status": "complete",
        "ppl": {
            "en": {"chunks": 100, "bf16_ppl": bf16, "candidate_ppl": en_ppl, "bpc_candidate": 1.28},
            "ja": {"chunks": 100, "bf16_ppl": 43.42, "candidate_ppl": ja_ppl, "bpc_candidate": 3.58},
        },
    }


def run(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "combine_lane_manifests.py"),
            "--root",
            str(root),
            "--pattern",
            "sweep-shard*",
            *extra,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_combines_shards_in_order(tmp_path: Path) -> None:
    write_manifest(tmp_path, "sweep-shard1of2", [result("a"), result("b")])
    write_manifest(tmp_path, "sweep-shard2of2", [result("c")])
    out = tmp_path / "combined.json"
    completed = run(tmp_path, "--json", str(out))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    combined = json.loads(out.read_text(encoding="utf-8"))
    assert combined["count"] == 3
    assert [item["name"] for item in combined["results"]] == ["a", "b", "c"]
    assert combined["problems"] == []


def test_flags_duplicate_conditions(tmp_path: Path) -> None:
    write_manifest(tmp_path, "sweep-shard1of2", [result("a")])
    write_manifest(tmp_path, "sweep-shard2of2", [result("a", en_ppl=40.0)])
    out = tmp_path / "combined.json"
    completed = run(tmp_path, "--json", str(out), "--strict")
    assert completed.returncode == 1
    assert "duplicate condition a" in completed.stdout


def test_flags_baseline_drift(tmp_path: Path) -> None:
    write_manifest(tmp_path, "sweep-shard1of2", [result("a", bf16=35.38)])
    write_manifest(tmp_path, "sweep-shard2of2", [result("b", bf16=39.10)])
    completed = run(tmp_path, "--strict")
    assert completed.returncode == 1
    assert "bf16_ppl drift" in completed.stdout


def test_prints_table_with_bpc(tmp_path: Path) -> None:
    write_manifest(tmp_path, "sweep-shard1of1", [result("a")])
    completed = run(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "| condition |" in completed.stdout
    assert "1.2800" in completed.stdout
