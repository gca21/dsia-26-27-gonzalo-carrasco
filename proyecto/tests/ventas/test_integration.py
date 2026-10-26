from pathlib import Path

import pytest

from ventas_app.loader import CsvSalesRepository
from ventas_app.validator import SalesValidator


@pytest.mark.integration
def test_pipeline_con_csv_real_de_ventas():
    # Arrange
    project_dir = Path(__file__).resolve().parents[2]
    csv_path = project_dir / "data" / "ventas.csv"

    # Act
    sales = CsvSalesRepository(csv_path).load()
    valid, errors = SalesValidator(sales).validate()

    # Assert
    assert len(sales) == 150
    assert len(valid) == 140
    assert len(errors) == 10
    assert valid["importe"].sum() == pytest.approx(96598.17)
