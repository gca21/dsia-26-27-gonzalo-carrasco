from __future__ import annotations

import logging

import pandas as pd
import pytest

from satellites.cli import main
from satellites.train import SatellitesTrainer


@pytest.fixture(autouse=True)
def fast_trainer_and_clean_logging(monkeypatch):
    # The CLI builds its own trainer, so the class defaults are patched
    monkeypatch.setattr(SatellitesTrainer, "N_TREES", 20)
    monkeypatch.setattr(SatellitesTrainer, "MIN_ROWS", 100)
    
    # main() configures the root logger: restore it after each test
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers[:]:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


def test_cli_returns_0_and_writes_the_artifacts(raw_csv, tmp_path):
    # Arrange
    output_dir = tmp_path / "output"

    # Act
    code = main(["--input", str(raw_csv), "--output-dir", str(output_dir), "--min-f1", "0"])

    # Assert
    assert code == 0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "run.log").exists()
    assert (output_dir / "predictions.csv").exists()


def test_cli_returns_1_if_the_data_gate_fails(raw_csv, tmp_path):
    # Arrange
    broken = pd.read_csv(raw_csv)
    broken.loc[:150, "NORAD_CAT_ID"] = 0
    broken.to_csv(raw_csv, index=False)

    # Act
    code = main(["--input", str(raw_csv), "--output-dir", str(tmp_path / "output")])

    # Assert
    assert code == 1


def test_cli_returns_2_if_the_input_does_not_exist(tmp_path):
    # Act
    code = main(["--input", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path / "out")])

    # Assert
    assert code == 2


def test_cli_returns_2_if_the_ai_provider_is_misconfigured(raw_csv, tmp_path, monkeypatch):
    # Arrange
    monkeypatch.setenv("AI_PROVIDER", "hf")  # Without HF_TOKEN

    # Act
    code = main(["--input", str(raw_csv), "--output-dir", str(tmp_path / "output")])

    # Assert
    assert code == 2


@pytest.mark.parametrize("args", [["--output-dir", "output"], ["--input", "satellites.csv"]])
def test_cli_requires_input_and_output_dir(args):
    # Act and assert
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2