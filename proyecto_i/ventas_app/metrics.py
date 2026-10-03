from __future__ import annotations


import numpy as np
import pandas as pd

class SalesMetrics:
    """Class to obtain metrics from the sales dataframe"""

    def __init__(self, sales: pd.DataFrame) -> None:
        self.sales = sales

    def amount_by_region(self) -> pd.DataFrame:
        # Amount by region (desc)
        amount_by_region = (
            self.sales.groupby("region", as_index=False)["importe"]
            .sum()
            .sort_values("importe", ascending=False)
        )
        return amount_by_region

    def top_3_amounts(self) -> pd.DataFrame:
        # Top 3 products by amount
        top_products = (
            self.sales.groupby("producto", as_index=False)["importe"]
            .sum()
            .sort_values("importe", ascending=False)
            .head(3)
        )
        return top_products

    def recurrent_clients(self) -> pd.DataFrame:
        purchases_by_clients = self.sales["cliente_id"].value_counts()
        recurrent_clients = purchases_by_clients[purchases_by_clients > 1]
        return recurrent_clients