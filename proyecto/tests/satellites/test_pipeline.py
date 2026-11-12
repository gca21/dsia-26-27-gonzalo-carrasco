from __future__ import annotations

import json

import pandas as pd
import pytest

from satellites.ai.client import MockClient
from satellites.downloader import DataSourceError
from satellites.pipeline import GateError, PipelineConfig, SatellitesPipeline

pytestmark = pytest.mark.integration


def make_pipeline(raw_csv, output_dir, trainer, downloader=None, **options):
    config = PipelineConfig(input_path=raw_csv, output_dir=output_dir, **options)
    return SatellitesPipeline(config, MockClient(), trainer, downloader)


class FakeDownloader:
    """Writes a prepared csv in the input path, like the real download would"""

    def __init__(self, source_csv):
        self.source_csv = source_csv
        self.calls = 0

    def download(self, path):
        self.calls += 1
        path.write_text(self.source_csv.read_text(encoding="utf-8"), encoding="utf-8")
        return 310


class FailingDownloader:
    def download(self, path):
        raise DataSourceError("Space-Track is down")


def read_metrics(output_dir):
    return json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))


def test_pipeline_writes_artifacts_and_metrics(raw_csv, trainer, tmp_path):
    # Arrange
    output_dir = tmp_path / "output"
    pipeline = make_pipeline(raw_csv, output_dir, trainer, min_f1=0.0)

    # Act
    metrics = pipeline.run()

    # Assert
    expected = {
        "clean_satellites.csv", "invalid_satellites.csv", "model.joblib",
        "metadata.json", "predictions.csv", "metrics.json",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    assert metrics["status"] == "ok"
    assert metrics == read_metrics(output_dir)
    assert metrics["rows_in"] == 310


def test_pipeline_classifies_unknown_objects_and_explains_the_top(raw_csv, trainer, tmp_path):
    # Arrange
    output_dir = tmp_path / "output"
    pipeline = make_pipeline(raw_csv, output_dir, trainer, min_f1=0.0, explain_top=2)

    # Act
    metrics = pipeline.run()
    predictions = pd.read_csv(output_dir / "predictions.csv")

    # Assert
    assert len(predictions) == 10  # The 10 UNKNOWN objects of the fixture
    assert set(predictions["PREDICTED_TYPE"]) <= set(trainer.TARGET_CLASSES)
    assert predictions["EXPLANATION"].notna().sum() == 2
    assert predictions["EXPLANATION"].dropna().str.startswith("[mock]").all()
    assert metrics["predictions"]["explained"] == 2


def test_pipeline_data_gate_stops_before_training(raw_csv, trainer, tmp_path):
    # Arrange
    broken = pd.read_csv(raw_csv)
    broken.loc[:150, "NORAD_CAT_ID"] = 0  # Half of the rows are invalid
    broken.to_csv(raw_csv, index=False)
    output_dir = tmp_path / "output"
    pipeline = make_pipeline(raw_csv, output_dir, trainer)

    # Act and assert
    with pytest.raises(GateError, match="Data gate"):
        pipeline.run()
    assert read_metrics(output_dir)["status"] == "data_gate_failed"
    assert not (output_dir / "model.joblib").exists()


def test_pipeline_model_gate_stops_before_predicting(raw_csv, trainer, tmp_path):
    # Arrange
    output_dir = tmp_path / "output"
    pipeline = make_pipeline(raw_csv, output_dir, trainer, min_f1=1.01)

    # Act and assert
    with pytest.raises(GateError, match="Model gate"):
        pipeline.run()
    assert read_metrics(output_dir)["status"] == "model_gate_failed"
    assert not (output_dir / "predictions.csv").exists()


def test_pipeline_is_idempotent(raw_csv, trainer, tmp_path):
    # Arrange
    output_dir = tmp_path / "output"
    pipeline = make_pipeline(raw_csv, output_dir, trainer, min_f1=0.0)

    # Act
    pipeline.run()
    files_first_run = sorted(path.name for path in output_dir.iterdir())
    pipeline.run()
    files_second_run = sorted(path.name for path in output_dir.iterdir())

    # Assert
    assert files_first_run == files_second_run


def test_pipeline_downloads_the_data_before_loading_it(raw_csv, trainer, tmp_path):
    # Arrange: the input file does not exist until the download creates it
    input_path = tmp_path / "data" / "satellites.csv"
    input_path.parent.mkdir()
    downloader = FakeDownloader(raw_csv)
    pipeline = make_pipeline(input_path, tmp_path / "output", trainer, downloader, min_f1=0.0)

    # Act
    metrics = pipeline.run()

    # Assert
    assert downloader.calls == 1
    assert metrics["download"]["downloaded"] is True
    assert metrics["download"]["rows"] == 310
    assert metrics["rows_in"] == 310


def test_pipeline_stops_if_the_download_fails(raw_csv, trainer, tmp_path):
    # Arrange
    output_dir = tmp_path / "output"
    pipeline = make_pipeline(raw_csv, output_dir, trainer, FailingDownloader())

    # Act and assert
    with pytest.raises(DataSourceError, match="Space-Track is down"):
        pipeline.run()
    assert not (output_dir / "model.joblib").exists()