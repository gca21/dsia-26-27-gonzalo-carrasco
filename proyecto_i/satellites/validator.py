from __future__ import annotations

import numpy as np
import pandas as pd


class DataValidationError(Exception):
    """Error validating data."""

class SatellitesValidator:
    """Class to validate satellites from a dataframe"""

    NUMERIC_COLUMNS = [
        "Perigee (km)",
        "Apogee (km)",
        "Inclination (degrees)",
        "Launch Mass (kg.)",
        "Power (watts)",
        "Expected Lifetime (yrs.)",
    ]

    VALID_ORBIT_CLASSES = [
        "LEO", "MEO", "GEO", "ELLIPTICAL"
    ]

    FIRST_LAUNCH = pd.Timestamp("1957-10-04")

    def __init__(self, satellites: pd.Dataframe) -> None:
        self.satellites = satellites

    def validate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        work = self.satellites.copy()

        # Parsing: unparseable values become NaT
        for col in self.NUMERIC_COLUMNS:
            work[col] = pd.to_numeric(
                work[col].astype("string").str.replace(",", "").str.strip(),
                errors="coerce",
            )
        work["Date of Launch"] = pd.to_datetime(work["Date of Launch"], format="mixed", errors="coerce")

        # Text cleaning
        text_cols = ["Name", "Country of Operator/Owner", "Users", "Purpose", "Class of Orbit"]
        for col in text_cols:
            work[col] = work[col].astype("string").str.strip()
        work["Class of Orbit"] = work["Class of Orbit"].str.upper()

        work = work.drop_duplicates()

        ok = (
            work["Name"].notna()
            & (work["Name"] != "")
            & work["Class of Orbit"].isin(self.VALID_ORBIT_CLASSES)
            & work["Date of Launch"].notna()
            & (work["Date of Launch"] <= pd.Timestamp.today())
            & (work["Date of Launch"] >= self.FIRST_LAUNCH)
            & work["Perigee (km)"].notna()
            & (work["Perigee (km)"] >= 0)
            & work["Apogee (km)"].notna()
            & (work["Apogee (km)"] >= work["Perigee (km)"])
            & (work["Inclination (degrees)"].isna() | work["Inclination (degrees)"].between(0, 180))
            & (work["Launch Mass (kg.)"].isna() | (work["Launch Mass (kg.)"] > 0))
            & (work["Power (watts)"].isna() | (work["Power (watts)"] > 0))
            & (work["Expected Lifetime (yrs.)"].isna() | (work["Expected Lifetime (yrs.)"] > 0))
        ).fillna(False)

        valid = work.loc[ok].copy()
        errors = work.loc[~ok].copy()
        valid["Launch Year"] = valid["Date of Launch"].dt.year
        return valid, errors