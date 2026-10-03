from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


class DataLoadError(Exception):
    """Error loading data from the source"""


class CsvSatellitesRepository:
    """Class to import relevant satellite data from csv"""

    FEATURES = [
        "Name",
        "Country of Operator/Owner",
        "Users",
        "Purpose",
        "Class of Orbit",
        "Perigee (km)",
        "Apogee (km)",
        "Inclination (degrees)",
        "Launch Mass (kg.)",
        "Power (watts)",
        "Expected Lifetime (yrs.)",
        "Date of Launch",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
    
    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise DataLoadError(f"File not found: {self.path}")
        
        df = pd.read_csv(self.path, encoding="latin-1")
        return self.__filter_columns(df)

    def __filter_columns(self, satellites: pd.DataFrame) -> pd.DataFrame:
        satellites = satellites.rename(columns=str.strip)
        satellites = satellites.rename(columns={"Current Official Name of Satellite": "Name"})
        return satellites[self.FEATURES]
        


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "data/satellites/satellites.csv"
    df = CsvSatellitesRepository(path).load()

    print(df.duplicated().sum())