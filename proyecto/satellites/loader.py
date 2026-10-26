from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoadError(Exception):
    """Error loading data from the source"""


class CsvSatellitesRepository:
    """Class to import relevant satellite data from csv (Space-Track SATCAT)"""

    FEATURES = [
        "NORAD_CAT_ID",
        "SATNAME",
        "OBJECT_TYPE",
        "COUNTRY",
        "SITE",
        "LAUNCH",
        "DECAY",
        "PERIOD",
        "INCLINATION",
        "APOGEE",
        "PERIGEE",
        "RCSVALUE",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise DataLoadError(f"File not found: {self.path}")
        df = pd.read_csv(self.path, encoding="utf-8")
        return self.__filter_columns(df)

    def __filter_columns(self, satellites: pd.DataFrame) -> pd.DataFrame:
        # Remove irrelevant columns from the dataframe
        satellites = satellites.rename(columns=str.strip)
        missing = [c for c in self.FEATURES if c not in satellites.columns]
        if missing:
            raise DataLoadError(f"Missing columns in {self.path.name}: {missing}")
        return satellites[self.FEATURES]


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "data/satellites/satellites.csv"
    df = CsvSatellitesRepository(path).load()

    print(df.shape)
    print(df.duplicated().sum())