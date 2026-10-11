import pandas as pd
import pytest

from ventas_app.metrics import SalesMetrics


@pytest.fixture
def metrics():
    # Arrange for all tests
    ventas_mini = pd.DataFrame(
        {
            "region": ["Norte", "Sur", "Norte", "Este", "Sur", "Oeste", "Norte"],
            "producto": ["A", "B", "B", "C", "A", "D", "A"],
            "importe": [25, 10, 30, 14, 5, 50, 5],
            "cliente_id": ["C1", "C2", "C1", "C3", "C2", "C4", "C1"],
        }
    )
    return SalesMetrics(ventas_mini)


def test_importe_por_region_ordenado_de_mayor_a_menor(metrics):
    # Act
    resultado = metrics.amount_by_region()

    # Assert
    assert resultado.to_dict("records") == [
        {"region": "Norte", "importe": 60},
        {"region": "Oeste", "importe": 50},
        {"region": "Sur", "importe": 15},
        {"region": "Este", "importe": 14},
    ]


def test_top_3_productos_por_importe(metrics):
    # Act
    resultado = metrics.top_3_amounts()

    # Assert
    assert resultado.to_dict("records") == [
        {"producto": "D", "importe": 50},
        {"producto": "B", "importe": 40},
        {"producto": "A", "importe": 35},
    ]


def test_clientes_recurrentes_y_numero_de_compras(metrics):
    # Act
    resultado = metrics.recurrent_clients()

    # Assert
    assert resultado.to_dict() == {"C1": 3, "C2": 2}