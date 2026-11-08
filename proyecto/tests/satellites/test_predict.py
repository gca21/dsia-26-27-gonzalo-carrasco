from __future__ import annotations

import json

import pytest

from satellites.predict import PredictionError, SatellitesPredictor


@pytest.fixture
def model_dir(trainer, satellites_synthetic, tmp_path):
    trainer.train(satellites_synthetic, tmp_path)
    return tmp_path


@pytest.fixture
def new_objects(satellites_synthetic):
    # Objects whose type is unknown
    return satellites_synthetic.drop(columns=["OBJECT_TYPE"]).head(5)


def test_predict_batch_adds_type_and_probability(model_dir, new_objects):
    # Arrange
    predictor = SatellitesPredictor.load(model_dir)

    # Act
    result = predictor.predict_batch(new_objects)

    # Assert
    assert len(result) == 5
    assert set(result["PREDICTED_TYPE"]) <= {"PAYLOAD", "ROCKET BODY", "DEBRIS"}
    assert result["PROBABILITY"].between(0, 1).all()
    assert "PREDICTED_TYPE" not in new_objects.columns  # The input is not modified


def test_predict_one_returns_type_and_probabilities(model_dir, new_objects):
    # Arrange
    predictor = SatellitesPredictor.load(model_dir)
    record = new_objects.iloc[0].to_dict()

    # Act
    result = predictor.predict_one(record)

    # Assert
    assert result["object_type"] in {"PAYLOAD", "ROCKET BODY", "DEBRIS"}
    assert result["probability"] == max(result["probabilities"].values())
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_load_raises_if_model_does_not_exist(tmp_path):
    # Act and assert
    with pytest.raises(PredictionError, match="Model not found"):
        SatellitesPredictor.load(tmp_path)


def test_load_warns_if_sklearn_version_differs(model_dir, caplog):
    # Arrange
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sklearn_version"] = "0.0.0"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    # Act
    with caplog.at_level("WARNING"):
        SatellitesPredictor.load(model_dir)

    # Assert
    assert "scikit-learn" in caplog.text