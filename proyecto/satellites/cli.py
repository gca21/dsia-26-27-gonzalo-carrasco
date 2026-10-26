from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from satellites.loader import CsvSatellitesRepository
from satellites.validator import SatellitesValidator
from satellites.metrics import SatellitesMetrics

# python -m satellites.cli --input data/satellites/satellites.csv --output data/satellites/clean_satellites.csv
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and analyze the satellites data"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the CSV input.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the clean CSV output.",
    )

    args = parser.parse_args()

    input = args.input
    output_csv = args.output
    # Same directory
    output_json = output_csv.with_name("quality_data.json")

    # Create output directory if it doesn't exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    satellites = CsvSatellitesRepository(input).load()
    valid, errors = SatellitesValidator(satellites).validate()

    print(f"Valid: {len(valid)} | Invalid: {len(errors)}\n")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    metrics = SatellitesMetrics(valid)
    launch_growth = metrics.launch_growth()
    print("|----- Launch growth over the years -----|\n", launch_growth, "\n")

    country_concentration = metrics.country_concentration(top_n=5)
    print("|----- Country concentration -----|\n", country_concentration, "\n")

    orbit_class = metrics.orbit_class_distribution()
    print("|----- Orbit class distribution -----|\n", orbit_class, "\n")

    valid.to_csv(output_csv, index=False)

    quality = {
        "Total rows": int(len(satellites)),
        "Valid rows": int(len(valid)),
        "Invalid rows": int(len(errors)),
    }

    output_json.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("CSV output:", output_csv)
    print("JSON output:", output_json)
    print(quality)


if __name__ == "__main__":
    main()