from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from satellites.features import DataTransformationError, SatellitesFeatureBuilder


@pytest.fixture
def builder():
    return SatellitesFeatureBuilder()


@pytest.fixture
def satellites_mini():
    return pd.DataFrame(
        {
            "NORAD_CAT_ID": [1, 2, 3, 4],
            "SATNAME": ["Satellite A", "Debris B DEB", "Satellite C", "Rocket D R/B"],
            "OBJECT_TYPE": ["PAYLOAD", "DEBRIS", "PAYLOAD", "ROCKET BODY"],
            "COUNTRY": ["US", "RU", "US", "CN"],
            "SITE": ["AFETR", "TTMTR", "AFETR", "JSC"],
            "LAUNCH": ["2020-01-15", "2009-02-10", "2021-06-01", "2019-11-30"],
            "DECAY": [None, "2015-03-01", None, None],
            "PERIOD": [90.0, 100.0, 1436.0, 110.0],
            "INCLINATION": [51.6, 74.0, 0.1, 98.0],
            "APOGEE": [500, 800, 35800, 900],
            "PERIGEE": [400, 700, 35700, 800],
            "LAUNCH_YEAR": [2020, 2009, 2021, 2019],
            "ORBIT_CLASS": ["LEO", "LEO", "GEO", "LEO"],
            "RCS_SIZE": ["LARGE", None, "MEDIUM", "SMALL"],
        }
    )


def test_add_derived_computes_year_month_and_decay_flag(builder, satellites_mini):
    # Act
    result = builder.add_derived(satellites_mini)

    # Assert
    assert result["LAUNCH_YEAR"].tolist() == [2020, 2009, 2021, 2019]
    assert result["LAUNCH_MONTH"].tolist() == [1, 2, 6, 11]
    assert result["DECAYED"].tolist() == [0, 1, 0, 0]


def test_select_features_returns_only_model_columns(builder, satellites_mini):
    # Act
    X, y = builder.select_features(satellites_mini)

    # Assert
    assert X.columns.tolist() == builder.NUMERIC + builder.CATEGORICAL
    assert "SATNAME" not in X.columns
    assert "NORAD_CAT_ID" not in X.columns
    assert y.tolist() == ["PAYLOAD", "DEBRIS", "PAYLOAD", "ROCKET BODY"]


def test_select_features_drops_rows_without_target(builder, satellites_mini):
    # Arrange
    satellites_mini.loc[0, "OBJECT_TYPE"] = None

    # Act
    X, y = builder.select_features(satellites_mini)

    # Assert
    assert len(X) == len(y) == 3


def test_select_features_raises_if_column_is_missing(builder, satellites_mini):
    # Arrange
    incomplete = satellites_mini.drop(columns=["RCS_SIZE"])

    # Act and assert
    with pytest.raises(DataTransformationError, match="Missing columns"):
        builder.select_features(incomplete)


def test_preprocessor_leaves_no_nulls(builder, satellites_mini):
    # Arrange
    satellites_mini.loc[0, "PERIOD"] = np.nan
    satellites_mini.loc[1, "SITE"] = None
    X, _ = builder.select_features(satellites_mini)

    # Act
    result = builder.build_preprocessor().fit_transform(X)
    result = result.toarray() if hasattr(result, "toarray") else result

    # Assert
    assert result.shape[0] == len(X)
    assert not np.isnan(result).any()

def test_build_features_does_not_need_the_target(builder, satellites_mini):
    # Arrange
    new_objects = satellites_mini.drop(columns=["OBJECT_TYPE"])

    # Act
    X = builder.build_features(new_objects)

    # Assert
    assert X.columns.tolist() == builder.NUMERIC + builder.CATEGORICAL
    assert len(X) == len(new_objects)