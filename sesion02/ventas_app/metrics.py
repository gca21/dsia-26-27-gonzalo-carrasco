from __future__ import annotations


import numpy as np
import pandas as pd

def importe_por_region(validos: pd.DataFrame) -> pd.DataFrame:
    # Importe total por región (desc)
    importe_por_region = (
        validos.groupby("region", as_index=False)["importe"]
        .sum()
        .sort_values("importe", ascending=False)
    )
    return importe_por_region

def top_3_importe(validos: pd.DataFrame) -> pd.DataFrame:
    # Top 3 productos por importe
    top_productos = (
        validos.groupby("producto", as_index=False)["importe"]
        .sum()
        .sort_values("importe", ascending=False)
        .head(3)
    )
    return top_productos

def clientes_mas_1_compras(validos: pd.DataFrame) -> pd.DataFrame:
    compras_por_cliente = validos["cliente_id"].value_counts()
    clientes_recurrentes = compras_por_cliente[compras_por_cliente > 1]
    return clientes_recurrentes