from __future__ import annotations


import numpy as np
import pandas as pd

class SatellitesMetrics:
    """Class to obtain metrics from the satellites dataframe"""

    def __init__(self, satellites: pd.DataFrame) -> None:
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