from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from satellites.train import SatellitesTrainer


@pytest.fixture
def trainer():
    trainer = SatellitesTrainer()
    trainer.N_TREES = 20  # No need for training a lot of trees for the tests
    trainer.MIN_ROWS = 100 
    return trainer


@pytest.fixture
def satellites_synthetic():
    # Create synthetic catalog: 300 objects from 60 launch events + 10 UNKNOWN
    rng = np.random.default_rng(0)
    n = 310
    launch_dates = pd.date_range("2000-01-01", periods=60, freq="30D").strftime("%Y-%m-%d")
    launch = rng.choice(launch_dates, n)
    object_type = np.append(
        rng.choice(["PAYLOAD", "ROCKET BODY", "DEBRIS"], 300, p=[0.3, 0.2, 0.5]),
        ["UNKNOWN"] * 10,
    )
    return pd.DataFrame(
        {
            "NORAD_CAT_ID": np.arange(1, n + 1),
            "SATNAME": [f"OBJECT {i}" for i in range(n)],
            "OBJECT_TYPE": object_type,
            "COUNTRY": rng.choice(["US", "RU", "CN"], n),
            "SITE": rng.choice(["AFETR", "TTMTR", "JSC", "PLMSC"], n),
            "LAUNCH": launch,
            "DECAY": rng.choice([None, "2024-01-01"], n),
            "PERIOD": rng.uniform(90, 120, n),
            "INCLINATION": rng.uniform(0, 100, n),
            "APOGEE": rng.uniform(500, 900, n),
            "PERIGEE": rng.uniform(300, 500, n),
            "LAUNCH_YEAR": [int(d[:4]) for d in launch],
            "ORBIT_CLASS": "LEO",
            "RCS_SIZE": rng.choice([None, "SMALL", "MEDIUM", "LARGE"], n),
        }
    )