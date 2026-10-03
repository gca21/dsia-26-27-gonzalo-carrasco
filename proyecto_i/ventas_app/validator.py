from __future__ import annotations

import numpy as np
import pandas as pd


class DataValidationError(Exception):
    """Error validating data."""

class SalesValidator:
    """Class to validate the sales from a dataframe"""

    def __init__(self, sales: pd.Dataframe) -> None:
        self.sales = sales

    def validate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        work = self.sales.copy()
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
