from __future__ import annotations


import numpy as np
import pandas as pd

class SatellitesMetrics:
    """Class to obtain metrics from the satellites dataframe"""

    def __init__(self, satellites: pd.DataFrame) -> pd.DataFrame:
        self.satellites = satellites

    def launch_growth(self) -> pd.DataFrame:
        # Satellites per year, cumulative total and percentage change
        yearly = (
            self.satellites.groupby("Launch Year")
            .size()
            .rename("Satellites")
            .to_frame()
        )
        yearly["Cumulative"] = yearly["Satellites"].cumsum()
        yearly["Growth_pct"] = (yearly["Satellites"].pct_change() * 100).round(1)
        return yearly.reset_index()

    def operator_concentration(self, top_n: int = 5) -> pd.DataFrame:
        # Percentage of total satellites by country
        counts = self.satellites["Country of Operator/Owner"].value_counts()
        share = (counts / counts.sum() * 100).round(1)
        result = share.head(top_n).to_frame("Share percentage")
        result.loc["Rest of the world"] = 100 - result["Share percentage"].sum()
        return result.reset_index()