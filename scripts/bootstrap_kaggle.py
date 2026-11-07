#!/usr/bin/env python3
"""Prepare a Kaggle/Colab session for a uzu lane.

Robustness notes:
- ``hf download --include`` behaviour differs between huggingface_hub versions
  (some accept a single value only). The kit is ~5 MB, so this script downloads
  the whole dataset and avoids ``--include`` entirely.
- The kit is rebuilt from a fixed manifest (``scripts/build_kit.py``), so a
  missing script cannot silently slip in.

Usage:
    python scripts/bootstrap_kaggle.py                 # install deps + fetch kit
    python scripts/bootstrap_kaggle.py --no-install    # kit only
    python scripts/bootstrap_kaggle.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

KIT_REPO = "yamahei21/uzu-quant-kit"
LALAMO = "git+https://github.com/trymirai/lalamo@e401e17761f2fcd337b0cf20218d67d656e6ecba"
PACKAGES = [LALAMO, "jax[cuda12]", "huggingface_hub", "tokenizers"]


def run(command: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--no-install", action="store_true", help="skip pip install")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    work = Path(args.work)
    kit = work / "kit"

    if not args.no_install:
        run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], args.dry_run)

    run(
        [
            "hf",
            "download",
            KIT_REPO,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(kit),
        ],
        args.dry_run,
    )

    print(
        "\nnext:\n"
        f"  python {kit}/scripts/run_lane.py --spec {kit}/lanes/clean-suite-08b.json "
        f"--hf-results <user>/uzu-results --work {work}",
        flush=True,
    )


if __name__ == "__main__":
    main()
