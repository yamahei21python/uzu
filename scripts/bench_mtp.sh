#!/usr/bin/env bash
# Compare uzu generation with and without the DFlash speculator.
#
# Usage:
#   bash scripts/bench_mtp.sh /path/to/model "prompt"
#
# The script temporarily renames model/speculator, runs the CLI, and restores it.
set -u

MODEL="${1:?model dir}"
PROMPT="${2:-富士山について3文で説明して}"
CLI="${CLI:-$(cd "$(dirname "$0")/.." && pwd)/target/release/cli}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
SPECULATOR="$MODEL/speculator"
BACKUP="$MODEL/speculator.mtp-bench-bak"

restore() {
  if [[ -d "$BACKUP" ]]; then
    rm -rf "$SPECULATOR"
    mv "$BACKUP" "$SPECULATOR"
  fi
}
trap restore EXIT

run_case() {
  local label="$1"
  local raw
  raw="$(mktemp)"
  ("$CLI" --model "$MODEL" -m "$PROMPT" --reasoning-effort disabled >"$raw" 2>&1) &
  local pid=$!
  for ((i = 0; i < TIMEOUT_SECONDS; i++)); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null
  fi
  wait "$pid" 2>/dev/null

  local tokens speed
  tokens="$(sed -n 's/.*tokens per forward pass: \(.*\) .*/\1/p' "$raw" | tail -1)"
  speed="$(sed -n 's/.*generation speed: \([0-9.]*\) t\/s.*/\1/p' "$raw" | tail -1)"
  [[ -z "$tokens" ]] && tokens="-"
  [[ -z "$speed" ]] && speed="-"
  echo "| $label | $tokens | $speed |"
  rm -f "$raw"
}

echo "| mode | tokens per forward pass | generation t/s |"
echo "|---|---:|---:|"

if [[ -d "$SPECULATOR" ]]; then
  mv "$SPECULATOR" "$BACKUP"
  run_case "without speculator"
  mv "$BACKUP" "$SPECULATOR"
  run_case "with speculator"
else
  echo "| without speculator | - | - |"
  echo
  echo "no $SPECULATOR; attach one with scripts/attach_speculator.py"
fi
