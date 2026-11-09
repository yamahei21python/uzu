#!/usr/bin/env python3
"""Read vLLM's spec-decode counters and report the draft acceptance.

Usage:
    python scripts/vllm_acceptance.py --url http://127.0.0.1:8000/metrics
    python scripts/vllm_acceptance.py --url ... --watch 10 --duration 300

Only the standard library is used; it runs anywhere the endpoint is reachable.

Why this exists: "acceptance" is easy to misread. vLLM exposes three counters:

    spec_decode_num_drafts_total           verification steps
    spec_decode_num_draft_tokens_total     tokens the draft proposed
    spec_decode_num_accepted_tokens_total  draft tokens the target accepted

The ratio of the last two is the per-token acceptance rate. On 0.32.0 without
the DFlash fix it sits near 0 even though the drafts keep running, and
`num_draft_tokens_total` stops growing proportionally to `num_drafts_total`
(the drafts shrink to one token). Both symptoms are reported here.
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.request

COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)

_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[0-9.eE+-]+)\s*$")


def parse_metrics(text: str) -> dict[str, float]:
    """Sum every sample of the counters we care about (labels omitted)."""
    totals = {name: 0.0 for name in COUNTERS}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name in totals:
            totals[name] += float(match.group("value"))
    return totals


def fetch(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def summarize(totals: dict[str, float], previous: dict[str, float] | None = None) -> str:
    drafts = totals["vllm:spec_decode_num_drafts_total"]
    proposed = totals["vllm:spec_decode_num_draft_tokens_total"]
    accepted = totals["vllm:spec_decode_num_accepted_tokens_total"]

    if previous is None:
        window = totals
        label = "total"
    else:
        window = {name: totals[name] - previous[name] for name in COUNTERS}
        label = "window"

    drafts = window["vllm:spec_decode_num_drafts_total"]
    proposed = window["vllm:spec_decode_num_draft_tokens_total"]
    accepted = window["vllm:spec_decode_num_accepted_tokens_total"]
    rate = accepted / proposed if proposed > 0 else 0.0
    per_draft = proposed / drafts if drafts > 0 else 0.0

    return (
        f"[{label}] drafts={drafts:.0f} proposed={proposed:.0f} accepted={accepted:.0f} "
        f"acceptance={rate:.3f} tokens/draft={per_draft:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--watch", type=float, default=0.0, help="seconds between samples; 0 prints once")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (watch mode)")
    args = parser.parse_args()

    if args.watch <= 0 and args.duration <= 0:
        totals = parse_metrics(fetch(args.url))
        print(summarize(totals))
        return

    start = time.time()
    previous: dict[str, float] | None = None
    while True:
        totals = parse_metrics(fetch(args.url))
        print(summarize(totals, previous), flush=True)
        previous = totals
        if args.duration and time.time() - start + args.watch > args.duration:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
