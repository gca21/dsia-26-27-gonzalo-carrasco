from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from loader import cargar_ventas
from validator import validar_ventas
from metrics import importe_por_region, top_3_importe, clientes_mas_1_compras

DATA_DIR = Path("data")
ventas: pd.DataFrame = cargar_ventas(DATA_DIR / "ventas.csv")
validos, errores = validar_ventas(ventas)
print(f"válidas: {len(validos)} | inválidas: {len(errores)}")

importe_region = importe_por_region(validos)
print("|----- Importes por región -----|\n", importe_region)

max_importes = top_3_importe(validos)
print("|----- Top 3 productos por importe -----|\n",max_importes)

clientes_recurrentes = clientes_mas_1_compras(validos)
print("|----- Clientes con más de una compra -----|\n",clientes_recurrentes)