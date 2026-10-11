from __future__ import annotations

import pandas as pd


class SatellitesMetrics:
    """Class to obtain metrics from the satellites dataframe"""

    def __init__(self, satellites: pd.DataFrame) -> None:
        self.satellites = satellites

    def launch_growth(self) -> pd.DataFrame:
        # Objects per year, cumulative total and percentage change
        yearly = (
            self.satellites.groupby("LAUNCH_YEAR")
            .size()
            .rename("Objects")
            .to_frame()
        )
        yearly["Cumulative"] = yearly["Objects"].cumsum()
        yearly["Growth_pct"] = (yearly["Objects"].pct_change() * 100).round(1)
        yearly.index.name = "Launch year"
        return yearly.reset_index()

    def country_concentration(self, top_n: int = 5) -> pd.DataFrame:
        # Percentage of total objects by country
        counts = self.satellites["COUNTRY"].value_counts()
        share = (counts / counts.sum() * 100).round(1)
        result = share.head(top_n).to_frame("Share percentage")
        result.loc["Rest of the world"] = round(100 - result["Share percentage"].sum(), 1)
        result.index.name = "Country"
        return result.reset_index()

    def orbit_class_distribution(self) -> pd.DataFrame:
        # Objects grouped by orbit class, with share, median period and payload share
        result = (
            self.satellites.groupby("ORBIT_CLASS", observed=True)
            .agg(
                objects=("NORAD_CAT_ID", "count"),
                median_period_min=("PERIOD", "median"),
                median_inclination_deg=("INCLINATION", "median"),
                payload_pct=("OBJECT_TYPE", lambda s: (s == "PAYLOAD").mean() * 100),
            )
            .reset_index()
        )
        result["share_pct"] = (result["objects"] / result["objects"].sum() * 100).round(1)
        result["payload_pct"] = result["payload_pct"].round(1)
        result = result.rename(columns={
            "ORBIT_CLASS": "Orbit class",
            "objects": "Objects",
            "share_pct": "Share percentage",
            "median_period_min": "Median period (min)",
            "median_inclination_deg": "Median inclination (deg)",
            "payload_pct": "Payload percentage",
        })
        return result