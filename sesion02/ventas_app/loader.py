from __future__ import annotations

from datetime import date
from pathlib import Path
import json

import numpy as np
import pandas as pd



DATA_DIR = Path("data")

def cargar_ventas(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero: {path}")
    return pd.read_csv(path)


# Execute scrip directly for diagnosing
if __name__ == "__main__":
    ventas: pd.DataFrame = cargar_ventas(DATA_DIR / "ventas.csv")
    print("shape:", ventas.shape)
    print()
    print("dtypes:")
    print(ventas.dtypes)
    print()
    print("nulos por columna:")
    print(ventas.isna().sum())
    print("\n|--------------View of the dataframe--------------|")
    print(ventas.head())
    print("\n|--------------Description of the dataframe--------------|")
    print(ventas.describe(include="all"))

    # Row 4 is invalid due to NaN value in unidades
    # Through the method describe we know a row contains a negative precio_unitario value
    print("\nIndex of row with negative value in precio_unitario:", ventas.index[ventas['precio_unitario'] < -1].tolist())
    print("Row with the negative value:\n", ventas.iloc[8])
    # Row 8 is invalid due to negative value in precio_unitario
