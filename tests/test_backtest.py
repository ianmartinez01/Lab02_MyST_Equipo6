import numpy as np
import pytest

from src.backtest import COMISION, ejecutar_backtest
from src.data import cargar_datos
from src.signals import generar_senales


@pytest.fixture(scope="module")
def resultado():
    datos = generar_senales(cargar_datos().iloc[:4000])
    return datos, ejecutar_backtest(datos)


def test_contabilidad_valor_final(resultado):
    """El valor final es el efectivo más el valor de las posiciones abiertas."""
    datos, r = resultado
    port = r["portafolio"]
    assert len(r["operaciones"]) > 0
    assert r["valor"].iloc[-1] == pytest.approx(port.efectivo + port.acciones * datos["Close"].iloc[-1])


def test_contabilidad_costos(resultado):
    """La suma de costos es la comisión por el monto de cada apertura y cada cierre."""
    _, r = resultado
    ops = r["operaciones"]
    montos = ops["acciones"] * (ops["precio_entrada"] + ops["precio_salida"])
    assert r["portafolio"].costos == pytest.approx(COMISION * montos.sum())
    assert ops["costos"].sum() == pytest.approx(r["portafolio"].costos)
    assert np.all(ops["costos"] > 0)
