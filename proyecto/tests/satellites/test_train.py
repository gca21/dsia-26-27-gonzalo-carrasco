from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from satellites.features import SatellitesFeatureBuilder
from satellites.train import SatellitesTrainer, TrainingError


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


def test_train_saves_model_and_metadata(trainer, satellites_synthetic, tmp_path):
    # Act
    metrics = trainer.train(satellites_synthetic, tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    # Assert
    assert (tmp_path / "model.joblib").exists()
    assert set(metadata["classes"]) == {"PAYLOAD", "ROCKET BODY", "DEBRIS"}
    assert metadata["metrics"]["f1_macro"] == metrics["f1_macro"]
    assert 0 <= metrics["f1_macro"] <= 1


def test_saved_model_loads_and_predicts(trainer, satellites_synthetic, tmp_path):
    # Arrange
    trainer.train(satellites_synthetic, tmp_path)
    X, _ = SatellitesFeatureBuilder().select_features(satellites_synthetic)

    # Act
    model = joblib.load(tmp_path / "model.joblib")
    predictions = model.predict(X.head(5))

    # Assert
    assert len(predictions) == 5
    assert set(predictions) <= set(trainer.TARGET_CLASSES)


def test_split_keeps_launch_groups_apart(trainer, satellites_synthetic):
    # Arrange
    X, y, groups = trainer.prepare(satellites_synthetic)

    # Act
    train_idx, test_idx = trainer.split(X, y, groups)

    # Assert
    assert set(y) == set(trainer.TARGET_CLASSES)  # UNKNOWN is discarded
    assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[test_idx])


def test_train_raises_if_there_are_too_few_rows(trainer, satellites_synthetic, tmp_path):
    # Act and assert
    with pytest.raises(TrainingError, match="Not enough rows"):
        trainer.train(satellites_synthetic.head(10), tmp_path)