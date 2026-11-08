import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataTransformationError(Exception):
    """Error transforming data into features."""


class SatellitesFeatureBuilder:
    """Builds the model features and the preprocessing from validated satellites"""

    TARGET = "OBJECT_TYPE"
    NUMERIC = [
        "PERIOD", "INCLINATION", "APOGEE", "PERIGEE",
        "LAUNCH_YEAR", "LAUNCH_MONTH", "DECAYED",
    ]
    CATEGORICAL = ["COUNTRY", "SITE", "ORBIT_CLASS", "RCS_SIZE"]

    # Expected columns from the input data
    RAW_REQUIRED = [
        TARGET, "LAUNCH", "DECAY", "PERIOD", "INCLINATION", "APOGEE",
        "PERIGEE", "LAUNCH_YEAR", "COUNTRY", "SITE", "ORBIT_CLASS", "RCS_SIZE",
    ]

    # Leakage or ID columns
    EXCLUDED = ["NORAD_CAT_ID", "SATNAME"]

    MIN_FREQ = 20  # Group categories with few values

    @staticmethod
    def add_derived(df: pd.DataFrame) -> pd.DataFrame:
        """Transform dates into numeric columns"""

        out = df.copy()
        launch = pd.to_datetime(out["LAUNCH"], errors="raise")
        out["LAUNCH_MONTH"] = launch.dt.month
        out["DECAYED"] = pd.to_datetime(out["DECAY"], errors="coerce").notna().astype(int)  # 1=Decayed, 0=In orbit
        return out

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns X from rows that may not have a target (inference)"""

        required = [c for c in self.RAW_REQUIRED if c != self.TARGET]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataTransformationError(f"Missing columns for features: {missing}")

        columns = self.NUMERIC + self.CATEGORICAL
        leaked = [c for c in self.EXCLUDED if c in columns]
        if leaked:
            raise DataTransformationError(f"Excluded columns present in features: {leaked}")

        X = self.add_derived(df)[columns].copy()
        # Pandas nulls to np.nan
        X[self.CATEGORICAL] = (
            X[self.CATEGORICAL].astype(object).where(X[self.CATEGORICAL].notna(), np.nan)
        )
        return X

    def select_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Returns (X, y), removes rows without target"""

        missing = [c for c in self.RAW_REQUIRED if c not in df.columns]
        if missing:
            raise DataTransformationError(f"Missing columns for features: {missing}")

        n_before = len(df)
        df = df.dropna(subset=[self.TARGET])
        if len(df) < n_before:
            logger.warning("Descartadas %d filas sin %s", n_before - len(df), self.TARGET)

        X = self.build_features(df)
        y = df[self.TARGET]

        # Check that each feature has more than 1 value
        constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if constant:
            logger.warning("Useless features (constants): %s", constant)

        logger.info("Features: %d rows, %d columns", len(X), X.shape[1])
        logger.info("Class distribution: %s", y.value_counts().to_dict())
        return X, y

    def build_preprocessor(self) -> ColumnTransformer:
        """Prepares a ColumnTransformer with the needed conversions"""

        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(
                handle_unknown="infrequent_if_exist", min_frequency=self.MIN_FREQ
            )),
        ])
        return ColumnTransformer([
            ("num", numeric, self.NUMERIC),
            ("cat", categorical, self.CATEGORICAL),
        ])