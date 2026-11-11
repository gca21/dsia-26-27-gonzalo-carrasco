#!/usr/bin/env bash
# Runs the pipeline and checks its metrics.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

INPUT="${INPUT:-data/satellites/satellites.csv}"
MAX_ERROR_RATE="${MAX_ERROR_RATE:-0.1}"
MIN_F1="${MIN_F1:-0.5}"
MIN_ROWS="${MIN_ROWS:-1000}"

DAY="$(date +%F)"
OUT="output/${DAY}" # Same folder per day, re-running overwrites it
LOG="output/cron_logs/pipeline_${DAY}.log"
mkdir -p "$OUT" "$(dirname "$LOG")"

trap 'echo "FAILED ${DAY} (exit $?)" >> "$LOG"' ERR

{
    python -m satellites.cli --input "$INPUT" --output-dir "$OUT" \
        --max-error-rate "$MAX_ERROR_RATE" --min-f1 "$MIN_F1"
    python scripts/check_metrics.py --metrics "$OUT/metrics.json" \
        --max-error-rate "$MAX_ERROR_RATE" --min-f1 "$MIN_F1" --min-rows "$MIN_ROWS"
} >> "$LOG" 2>&1

echo "OK ${DAY}" | tee -a "$LOG"