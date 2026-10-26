from __future__ import annotations

import pandas as pd
import pytest

from satellites.metrics import SatellitesMetrics


@pytest.fixture
def metrics():
    # Arrange for all tests
    satellites_mini = pd.DataFrame(
        {
            "LAUNCH_YEAR": [2020, 2020, 2021, 2022, 2022],
            "COUNTRY": ["US", "US", "RU", "CN", "UK"],
            "ORBIT_CLASS": pd.Categorical(
                ["LEO", "LEO", "GEO", "GEO", "LEO"],
                categories=["LEO", "MEO", "GEO", "Elliptical"],
            ),
            "NORAD_CAT_ID": [1, 2, 3, 4, 5],
            "PERIOD": [90, 95, 1436, 1436, 100],
            "INCLINATION": [50, 98, 0.1, 0.2, 60],
            "OBJECT_TYPE": ["PAYLOAD", "DEBRIS", "PAYLOAD", "PAYLOAD", "PAYLOAD"],
        }
    )
    return SatellitesMetrics(satellites_mini)


def test_launch_growth(metrics):
    # Act
    resultado = metrics.launch_growth()

    # Assert
    assert resultado["Launch year"].tolist() == [2020, 2021, 2022]
    assert resultado["Objects"].tolist() == [2, 1, 2]
    assert resultado["Cumulative"].tolist() == [2, 3, 5]
    assert pd.isna(resultado.loc[0, "Growth_pct"])
    assert resultado.loc[1:, "Growth_pct"].tolist() == [-50.0, 100.0]


def test_country_concentration_returns_top_countries_and_rest(metrics):
    # Act
    resultado = metrics.country_concentration(top_n=2)

    # Assert
    assert resultado.to_dict("records") == [
        {"Country": "US", "Share percentage": 40.0},
        {"Country": "RU", "Share percentage": 20.0},
        {"Country": "Rest of the world", "Share percentage": 40.0},
    ]


def test_orbit_class_distribution(metrics):
    # Act
    resultado = metrics.orbit_class_distribution()

    # Assert
    assert resultado["Orbit class"].tolist() == ["LEO", "GEO"]
    assert resultado["Objects"].tolist() == [3, 2]
    assert resultado["Median period (min)"].tolist() == pytest.approx([95.0, 1436.0])
    assert resultado["Median inclination (deg)"].tolist() == pytest.approx([60.0, 0.15])
    assert resultado["Payload percentage"].tolist() == [66.7, 100.0]
    assert resultado["Share percentage"].tolist() == [60.0, 40.0]
