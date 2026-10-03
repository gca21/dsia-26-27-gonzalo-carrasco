from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ventas_app.loader import CsvSalesRepository
from ventas_app.metrics import SalesMetrics
from ventas_app.validator import SalesValidator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpia y analiza los datos de ventas."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Ruta al CSV de entrada.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Ruta al CSV de salida con las ventas válidas.",
    )

    args = parser.parse_args()

    input = args.input
    output_csv = args.output
    # Same directory
    output_json = output_csv.with_name("calidad_datos.json")

    # Create output directory if it doesn't exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    sales = CsvSalesRepository(input).load()
    valid, errors = SalesValidator(sales).validate()

    print(f"válidas: {len(valid)} | inválidas: {len(errors)}")

    metrics = SalesMetrics(valid)
    amount_by_region = metrics.amount_by_region()
    print("|----- Importes por región -----|\n", amount_by_region)

    max_amounts = metrics.top_3_amounts()
    print("|----- Top 3 productos por importe -----|\n", max_amounts)

    recurrent_clients = metrics.recurrent_clients()
    print(
        "|----- Clientes con más de una compra -----|\n",
        recurrent_clients,
    )

    valid.to_csv(output_csv, index=False)

    quality = {
        "filas_totales": int(len(sales)),
        "filas_validas": int(len(valid)),
        "filas_invalidas": int(len(errors)),
        "importe_total": float(valid["importe"].sum()),
    }

    output_json.write_text(
        json.dumps(quality, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("escrito:", output_csv)
    print("escrito:", output_json)
    print(quality)


if __name__ == "__main__":
    main()