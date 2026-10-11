import json
import sys

import pandas as pd
import pytest

from ventas_app.cli import main


def test_cli_procesa_ventas_y_escribe_archivos(tmp_path, monkeypatch):
    # Arrange
    input_csv = tmp_path / "ventas.csv"
    output_csv = tmp_path / "salida" / "ventas_validas.csv"
    input_csv.write_text(
        "region,producto,unidades,precio_unitario,cliente_id\n"
        "Norte,A,2,10,C1\n"
        "Sur,B,1,5,C2\n"
        "Norte,C,0,20,C1\n",
        encoding="utf-8",
    )
    # Mock terminal arguments
    monkeypatch.setattr(
        sys,
        "argv",
        ["ventas", "--input", str(input_csv), "--output", str(output_csv)],
    )

    # Act
    main()
    ventas_validas = pd.read_csv(output_csv)
    calidad = json.loads((output_csv.parent / "calidad_datos.json").read_text())

    # Assert
    assert ventas_validas["importe"].tolist() == [20, 5]
    assert calidad == {
        "filas_totales": 3,
        "filas_validas": 2,
        "filas_invalidas": 1,
        "importe_total": 25.0,
    }
@pytest.mark.parametrize(
    "args",
    [
        ["--output", "salida.csv"],
        ["--input", "ventas.csv"],
    ],
)
def test_cli_requiere_input_y_output(args, monkeypatch):
    # Arrange
    monkeypatch.setattr(sys, "argv", ["ventas", *args])

    # Act and assert
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2