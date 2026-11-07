#!/bin/zsh
set -uo pipefail
cd /Users/kohei/Myproject/uzu
PY=.deps/venv-lalamo/bin/python
DATA=experiments/qwen35-0.8b/data
EXP=experiments/qwen35-0.8b-huihui
BF16=models/huihui-0.8b-abliterated-bf16-lalamo-eval
LOG=logs/2026-11-07/local/huihui08b_ratio.log
mkdir -p "$EXP"
: > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

if [ ! -f "$BF16/config.json" ]; then
  log "ERROR: missing $BF16. Run prep_huihui08b.sh first."
  exit 1
fi

run_condition() {
  local name="$1" corpus="$2"
  local out="$EXP/yaqa-hin-hout-${name}-int4-g32"
  local tmp="/tmp/huihui08b_yaqa_${name}_lalamo"
  local tmpeval="${tmp}_eval"
  mkdir -p "$out"

  log "=== $name calibration ==="
  JAX_PLATFORM_NAME=cpu "$PY" scripts/yaqa_calibration.py \
    --model-dir "$BF16" \
    --corpus "$corpus" \
    --eval-corpus "$DATA/eval_en_wikitext2_dedup.txt" \
    --eval-corpus "$DATA/eval_ja_wikipedia_dedup.txt" \
    --output "$out/calibration.npz" \
    --json-out "$out/calibration.json" \
    --samples 32 --max-tokens 128 \
    --output-block small --output-block-limit 2048 \
    --damping 1e-3 >> "$LOG" 2>&1 || { log "ERROR calibration $name"; return 1; }

  log "=== $name convert ==="
  JAX_PLATFORM_NAME=cpu "$PY" scripts/rht_int4_convert.py \
    --load-from "$BF16" \
    --lalamo-path .deps/lalamo-e401 \
    --output-dir "$tmp" \
    --preconditioners "$out/calibration.npz" >> "$LOG" 2>&1 || { log "ERROR convert $name"; return 1; }

  rm -f "$out/calibration.npz"
  rm -rf "$tmpeval"
  "$PY" scripts/make_lalamo_eval_copy.py --source "$tmp" --dest "$tmpeval" >> "$LOG" 2>&1 || { log "ERROR eval copy $name"; return 1; }

  log "=== $name eval EN ==="
  JAX_PLATFORM_NAME=cpu "$PY" scripts/eval_quantized.py \
    --bf16-dir "$BF16" \
    --candidate-dir "$tmpeval" \
    --corpus "$DATA/eval_en_wikitext2_dedup.txt" \
    --max-sequences 100 --max-tokens 128 \
    --json-out "$out/ppl_en.json" >> "$LOG" 2>&1 || { log "ERROR eval EN $name"; return 1; }

  log "=== $name eval JA ==="
  JAX_PLATFORM_NAME=cpu "$PY" scripts/eval_quantized.py \
    --bf16-dir "$BF16" \
    --candidate-dir "$tmpeval" \
    --corpus "$DATA/eval_ja_wikipedia_dedup.txt" \
    --max-sequences 100 --max-tokens 128 \
    --json-out "$out/ppl_ja.json" >> "$LOG" 2>&1 || { log "ERROR eval JA $name"; return 1; }

  rm -rf "$tmp" "$tmpeval"
  log "=== $name DONE ==="
}

run_condition enonly "$DATA/calibration_en_wikitext2_dedup.txt"
run_condition mixed50 "$DATA/calibration_mixed_en_ja_dedup.txt"
run_condition mix65 "$DATA/calibration_mixed_ja65_en35_dedup.txt"
run_condition mix35 "$DATA/calibration_mixed_ja35_en65_dedup.txt"
run_condition jaonly "$DATA/calibration_ja_wikipedia_dedup.txt"
log "HUIHUI SUITE DONE"
