from __future__ import annotations

import numpy as np
import pandas as pd


class DataValidationError(Exception):
    """Error validating data."""


class SatellitesValidator:
    """Class to validate satellites from a dataframe"""

    NUMERIC_COLUMNS = [
        "NORAD_CAT_ID",
        "PERIOD",
        "INCLINATION",
        "APOGEE",
        "PERIGEE",
        "RCSVALUE",
    ]

    VALID_OBJECT_TYPES = [
        "PAYLOAD", "ROCKET BODY", "DEBRIS", "UNKNOWN", "TBA"
    ]

    FIRST_LAUNCH = pd.Timestamp("1957-10-04")

    # Orbit class limits (altitudes in km)
    ORBIT_CLASSES = ["LEO", "MEO", "GEO", "Elliptical"]
    LEO_MAX_ALTITUDE = 2000
    GEO_MIN_PERIGEE = 35586
    GEO_MAX_APOGEE = 35986

    def __init__(self, satellites: pd.DataFrame) -> None:
        self.satellites = satellites

    def validate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        work = self.satellites.copy()

        # Parsing: unparseable values become NaN / NaT
        for col in self.NUMERIC_COLUMNS:
            work[col] = pd.to_numeric(
                work[col].astype("string").str.replace(",", "").str.strip(),
                errors="coerce",
            )
        work["LAUNCH"] = pd.to_datetime(work["LAUNCH"], format="mixed", errors="coerce")
        work["DECAY"] = pd.to_datetime(work["DECAY"], format="mixed", errors="coerce")

        # Text cleaning
        text_cols = ["SATNAME", "OBJECT_TYPE", "COUNTRY", "SITE"]
        for col in text_cols:
            work[col] = work[col].astype("string").str.strip()
        work["OBJECT_TYPE"] = work["OBJECT_TYPE"].str.upper()

        is_duplicate = work.duplicated(subset=["NORAD_CAT_ID"], keep="first")

        ok = (
            work["NORAD_CAT_ID"].notna()
            & (work["NORAD_CAT_ID"] > 0)
            & work["SATNAME"].notna()
            & (work["SATNAME"] != "")
            & work["OBJECT_TYPE"].isin(self.VALID_OBJECT_TYPES)
            & work["LAUNCH"].notna()
            & (work["LAUNCH"] <= pd.Timestamp.today())
            & (work["LAUNCH"] >= self.FIRST_LAUNCH)
            & (work["DECAY"].isna() | ((work["DECAY"] >= work["LAUNCH"]) & (work["DECAY"] <= pd.Timestamp.today())))
            & work["PERIGEE"].notna()
            & (work["PERIGEE"] >= 0)
            & work["APOGEE"].notna()
            & (work["APOGEE"] >= work["PERIGEE"])
            & (work["PERIOD"].isna() | (work["PERIOD"] > 0))
            & (work["INCLINATION"].isna() | work["INCLINATION"].between(0, 180))
            & (work["RCSVALUE"].isna() | (work["RCSVALUE"] >= 0))
            & ~is_duplicate
        ).fillna(False)

        valid = work.loc[ok].copy()
        errors = work.loc[~ok].copy()
        valid["LAUNCH_YEAR"] = valid["LAUNCH"].dt.year
        valid["ORBIT_CLASS"] = self.__orbit_class(valid)
        return valid, errors

    def __orbit_class(self, satellites: pd.DataFrame) -> pd.Categorical:
        perigee = satellites["PERIGEE"]
        apogee = satellites["APOGEE"]

        conditions = [
            apogee < self.LEO_MAX_ALTITUDE,
            (perigee >= self.GEO_MIN_PERIGEE) & (apogee <= self.GEO_MAX_APOGEE),
            perigee < self.LEO_MAX_ALTITUDE,
        ]
        return pd.Categorical(
            np.select(conditions, ["LEO", "GEO", "Elliptical"], default="MEO"),
            categories=self.ORBIT_CLASSES,
        )