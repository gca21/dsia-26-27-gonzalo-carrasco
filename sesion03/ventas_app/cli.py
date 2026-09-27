from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .loader import load
from .validator import validar_ventas
from .metrics import (
    importe_por_region,
    top_3_importe,
    clientes_mas_1_compras,
)


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

    entrada = args.input
    salida_csv = args.output
    # Same directory
    salida_json = salida_csv.with_name("calidad_datos.json")

    # Create output directory if it doesn't exist
    salida_csv.parent.mkdir(parents=True, exist_ok=True)

    ventas: pd.DataFrame = load(entrada)

    validos, errores = validar_ventas(ventas)

    print(f"válidas: {len(validos)} | inválidas: {len(errores)}")

    importe_region = importe_por_region(validos)
    print("|----- Importes por región -----|\n", importe_region)

    max_importes = top_3_importe(validos)
    print("|----- Top 3 productos por importe -----|\n", max_importes)

    clientes_recurrentes = clientes_mas_1_compras(validos)
    print(
        "|----- Clientes con más de una compra -----|\n",
        clientes_recurrentes,
    )

    validos.to_csv(salida_csv, index=False)

    calidad = {
        "filas_totales": int(len(ventas)),
        "filas_validas": int(len(validos)),
        "filas_invalidas": int(len(errores)),
        "importe_total": float(validos["importe"].sum()),
    }

    salida_json.write_text(
        json.dumps(calidad, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("escrito:", salida_csv)
    print("escrito:", salida_json)
    print(calidad)


if __name__ == "__main__":
    main()