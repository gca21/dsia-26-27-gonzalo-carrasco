from __future__ import annotations

import pandas as pd
import pytest

from ventas_app.loader import CsvSalesRepository, DataLoadError


def test_load_raises_if_file_does_not_exist(tmp_path):
    # Arrange
    repo = CsvSalesRepository(tmp_path / "missing.csv")

    # Act and assert
    with pytest.raises(DataLoadError, match="File not found"):
        repo.load()


def test_load_returns_dataframe_with_all_rows_and_columns(tmp_path):
    # Arrange
    path = tmp_path / "sales.csv"
    path.write_text("region,unidades\nN,1\nS,2\nE,3\n", encoding="utf-8")

    # Act
    df = CsvSalesRepository(path).load()

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["region", "unidades"]


def test_load_preserves_values(tmp_path):
    # Arrange
    path = tmp_path / "sales.csv"
    path.write_text("region,unidades\nN,1\nS,2\n", encoding="utf-8")

    # Act
    df = CsvSalesRepository(path).load()

    # Assert
    assert df["region"].tolist() == ["N", "S"]
    assert df["unidades"].tolist() == [1, 2]