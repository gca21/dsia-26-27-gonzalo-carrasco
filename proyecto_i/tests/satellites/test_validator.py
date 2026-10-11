from __future__ import annotations

import pandas as pd
import pytest

from satellites.validator import SatellitesValidator


@pytest.fixture
def satellites_mini():
    return pd.DataFrame(
        {
            "NORAD_CAT_ID": [1, 2, 1],
            "SATNAME": ["Satellite A", "Satellite B", "Duplicate"],
            "OBJECT_TYPE": [" payload ", "ROCKET BODY", "PAYLOAD"],
            "COUNTRY": ["US", "RU", "US"],
            "SITE": ["AFETR", "TTMTR", "AFETR"],
            "LAUNCH": ["2020-01-01", "2021-01-01", "2020-01-01"],
            "DECAY": ["", "", ""],
            "PERIOD": [90, 1436, 90],
            "INCLINATION": [51.6, 0.1, 51.6],
            "APOGEE": [500, 35800, 500],
            "PERIGEE": [400, 35700, 400],
            "RCSVALUE": [0.1, 1.0, 0.1],
        }
    )


def test_validate_separa_validos_y_duplicados(satellites_mini):
    # Arrange
    validator = SatellitesValidator(satellites_mini)

    # Act
    validos, errores = validator.validate()

    # Assert
    assert len(validos) == 2
    assert len(errores) == 1
    assert validos["SATNAME"].tolist() == ["Satellite A", "Satellite B"]


def test_validate_limpia_texto_y_calcula_anio_y_clase_orbital(satellites_mini):
    # Arrange
    validator = SatellitesValidator(satellites_mini)

    # Act
    validos, _ = validator.validate()

    # Assert
    assert validos["OBJECT_TYPE"].tolist() == ["PAYLOAD", "ROCKET BODY"]
    assert validos["LAUNCH_YEAR"].tolist() == [2020, 2021]
    assert validos["ORBIT_CLASS"].astype(str).tolist() == ["LEO", "GEO"]


@pytest.mark.parametrize("norad_id", [0, -1])
def test_norad_id_no_positivo_no_es_valido(satellites_mini, norad_id):
    # Arrange
    satellite = satellites_mini.iloc[[0]].copy()
    satellite.loc[:, "NORAD_CAT_ID"] = norad_id
    validator = SatellitesValidator(satellite)

    # Act
    validos, errores = validator.validate()

    # Assert
    assert len(validos) == 0
    assert len(errores) == 1
