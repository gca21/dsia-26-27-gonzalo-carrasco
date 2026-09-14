from __future__ import annotations

from loader import cargar_ventas

from pathlib import Path

import numpy as np
import pandas as pd


def validar_ventas(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["unidades"] = pd.to_numeric(work["unidades"], errors="coerce")
    work["precio_unitario"] = pd.to_numeric(work["precio_unitario"], errors="coerce")

    ok = (
        work["unidades"].notna()
        & (work["unidades"] > 0)
        & work["precio_unitario"].notna()
        & (work["precio_unitario"] > 0)
    )

    validos = work.loc[ok].copy()
    errores = work.loc[~ok].copy()
    validos["importe"] = validos["unidades"] * validos["precio_unitario"]
    return validos, errores

# Load dataset
DATA_DIR = Path("data")
ventas: pd.DataFrame = cargar_ventas(DATA_DIR / "ventas.csv")

# Get valid rows
validos, errores = validar_ventas(ventas)
print(f"válidas: {len(validos)} | inválidas: {len(errores)}")