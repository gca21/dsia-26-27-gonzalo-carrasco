from __future__ import annotations

import json

import joblib
import pytest

from satellites.features import SatellitesFeatureBuilder
from satellites.train import SatellitesTrainer, TrainingError


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