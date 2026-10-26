from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from satellites.cli import main


def test_cli_processes_satellites_and_writes_files(tmp_path, monkeypatch):
    # Arrange
    input_csv = tmp_path / "satellites.csv"
    output_csv = tmp_path / "output" / "clean_satellites.csv"
    input_csv.write_text(
        "NORAD_CAT_ID,SATNAME,OBJECT_TYPE,COUNTRY,SITE,LAUNCH,DECAY,"
        "PERIOD,INCLINATION,APOGEE,PERIGEE,RCSVALUE\n"
        "1,Satellite A,PAYLOAD,US,AFETR,2020-01-01,,90,51.6,500,400,0.1\n"
        "0,Invalid satellite,DEBRIS,RU,TTMTR,2021-01-01,,100,60,500,400,0.2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["satellites", "--input", str(input_csv), "--output", str(output_csv)],
    )

    # Act
    main()
    satellites_validos = pd.read_csv(output_csv)
    calidad = json.loads((output_csv.parent / "quality_data.json").read_text())

    # Assert
    assert satellites_validos["NORAD_CAT_ID"].tolist() == [1]
    assert satellites_validos["ORBIT_CLASS"].tolist() == ["LEO"]
    assert calidad == {"Total rows": 2, "Valid rows": 1, "Invalid rows": 1}


@pytest.mark.parametrize(
    "args",
    [
        ["--output", "output.csv"],
        ["--input", "satellites.csv"],
    ],
)
def test_cli_requires_input_and_output(args, monkeypatch):
    # Arrange
    monkeypatch.setattr(sys, "argv", ["satellites", *args])

    # Act and assert
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
