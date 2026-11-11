"""Reads the metrics.json of a pipeline run. Exit 0 = OK, 1 = KO, 2 = file not found"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Checks the metrics of a run")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--max-error-rate", type=float, default=0.1)
    parser.add_argument("--min-f1", type=float, default=0.5)
    parser.add_argument("--min-rows", type=int, default=1000)
    args = parser.parse_args()

    if not args.metrics.exists():
        print(f"KO: {args.metrics} not found", file=sys.stderr)
        return 2
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))

    problems = []
    if metrics.get("status") != "ok":
        problems.append(f"status is {metrics.get('status')!r}")
    if metrics.get("error_rate", 1.0) > args.max_error_rate:
        problems.append(f"error_rate {metrics.get('error_rate')} > {args.max_error_rate}")
    if metrics.get("rows_valid", 0) < args.min_rows:
        problems.append(f"rows_valid {metrics.get('rows_valid')} < {args.min_rows}")
    f1_macro = metrics.get("model", {}).get("f1_macro", 0.0)
    if f1_macro < args.min_f1:
        problems.append(f"f1_macro {f1_macro:.3f} < {args.min_f1}")

    if problems:
        print("KO: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"OK: {metrics['rows_valid']} valid rows, macro F1 {f1_macro:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())