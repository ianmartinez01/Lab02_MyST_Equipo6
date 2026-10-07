import numpy as np
import pandas as pd
import pytest

from src.metrics import calmar, max_drawdown, rendimiento_anualizado, tabla_retornos, win_rate


def test_max_drawdown_y_calmar():
    valor = pd.Series([100.0, 120.0, 90.0, 110.0])
    assert max_drawdown(valor) == pytest.approx(90 / 120 - 1)
    assert calmar(valor, velas_por_anio=3) == pytest.approx(0.10 / 0.25)
    assert rendimiento_anualizado(valor, velas_por_anio=3) == pytest.approx(0.10)


def test_tabla_retornos_mensual():
    fechas = pd.to_datetime(["2026-01-05", "2026-01-30", "2026-02-10", "2026-02-27"]).tz_localize("America/New_York")
    valor = pd.Series([100.0, 110.0, 99.0, 121.0], index=fechas)
    tabla = tabla_retornos(valor, "ME")
    np.testing.assert_allclose(tabla.to_numpy(), [0.10, 0.10])


def test_win_rate():
    assert win_rate(pd.DataFrame({"pnl": [5.0, -2.0, 1.0, -1.0]})) == 0.5


def test_payoff_ratio():
    from src.metrics import payoff_ratio
    assert payoff_ratio(pd.DataFrame({"pnl": [6.0, 2.0, -1.0, -3.0]})) == pytest.approx(4.0 / 2.0)
