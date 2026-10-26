from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


class DataLoadError(Exception):
    """Error loading data from the source"""


class CsvSalesRepository:
    """Class to import data from csv"""

    def __init__(self, path: Path) -> None:
        self.path = path
    
    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise DataLoadError(f"File not found: {self.path}")
        return pd.read_csv(self.path)


