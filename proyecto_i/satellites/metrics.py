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

    def mass_class_distribution(self) -> pd.DataFrame:
        # Satellites grouped by launch mass class, with share, power and lifetime

        # Satellite categories
        bins = [0, 10, 100, 500, 1000, float("inf")]
        labels = [
            "Nano (<10 kg)",
            "Micro (10-100 kg)",
            "Mini (100-500 kg)",
            "Medium (500-1000 kg)",
            "Large (>1000 kg)",
        ]
        mass_class = pd.cut(self.satellites["Launch Mass (kg.)"], bins=bins, labels=labels)

        # Get satellite metrics by category
        result = (
            self.satellites.groupby(mass_class, observed=False)
            .agg(
                satellites=("Name", "count"),
                median_power_w=("Power (watts)", "median"),
                median_lifetime_yrs=("Expected Lifetime (yrs.)", "median"),
            )
            .rename_axis("mass_class")
            .reset_index()
        )
        result["share_pct"] = (result["satellites"] / result["satellites"].sum() * 100).round(1)
        result = result.rename(columns={
            "mass_class": "Mass class",
            "satellites": "Satellites",
            "share_pct": "Share percentage",
            "median_power_w": "Median power (W)",
            "median_lifetime_yrs": "Median lifetime (yrs.)",
        })
        return result