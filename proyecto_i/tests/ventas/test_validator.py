import pytest
import pandas as pd

from ventas_app.validator import SalesValidator

@pytest.fixture
def ventas_mini():
    return pd.DataFrame({
        "fecha": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "region": ["Norte", "Sur", "Norte"],
        "producto": ["A", "B", "A"],
        "unidades": [2, None, 5],
        "precio_unitario": [10.0, 20.0, -1.0],
        "cliente_id": ["C1", "C2", "C1"],
    })


def test_validar_ventas_cuenta_validos_e_invalidos(ventas_mini):
    # Arrange
    validator = SalesValidator(ventas_mini)

    # Act
    validos, errores = validator.validate()

    #Assert
    assert len(validos) == 1
    assert len(errores) == 2


@pytest.mark.parametrize("precio", [0, -1])
def test_precio_no_positivo_no_es_valido(precio):
    # Arrange
    df = pd.DataFrame({"unidades": [1], "precio_unitario": [precio]})
    validator = SalesValidator(df)

    # Act
    validos, errores = validator.validate()
    assert len(validos) == 0
    assert len(errores) == 1