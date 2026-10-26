from __future__ import annotations

import pandas as pd
import pytest

from satellites.loader import CsvSatellitesRepository, DataLoadError


def test_load_raises_if_file_does_not_exist(tmp_path):
    # Arrange
    repo = CsvSatellitesRepository(tmp_path / "missing.csv")

    # Act and assert
    with pytest.raises(DataLoadError, match="File not found"):
        repo.load()


def test_load_returns_only_satellite_features(tmp_path):
    # Arrange
    path = tmp_path / "satellites.csv"
    path.write_text(
        "NORAD_CAT_ID,SATNAME,OBJECT_TYPE,COUNTRY,SITE,LAUNCH,DECAY,"
        "PERIOD,INCLINATION,APOGEE,PERIGEE,RCSVALUE,EXTRA\n"
        "1,Test satellite,PAYLOAD,US,AFETR,2020-01-01,,90,50,500,400,0.1,ignored\n",
        encoding="utf-8",
    )

    # Act
    df = CsvSatellitesRepository(path).load()

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 12)
    assert "EXTRA" not in df.columns
    assert df.loc[0, "SATNAME"] == "Test satellite"


def test_load_raises_if_feature_is_missing(tmp_path):
    # Arrange
    path = tmp_path / "satellites.csv"
    path.write_text("NORAD_CAT_ID,SATNAME\n1,Test satellite\n", encoding="utf-8")
    repo = CsvSatellitesRepository(path)

    # Act and assert
    with pytest.raises(DataLoadError, match="Missing columns"):
        repo.load()
