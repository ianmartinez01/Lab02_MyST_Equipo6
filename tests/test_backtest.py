"""Backtest: contabilidad, golden file calculado a mano, truncamiento del pipeline completo,
stop primero en la misma vela, sizing por riesgo exacto y una sola posición a la vez."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import SIN_COSTOS, Costos, Posicion, backtest, tamano_por_riesgo
from src.data import cargar_datos
from src.signals import generar_senales


@pytest.fixture(scope="module")
def datos():
    return cargar_datos().iloc[:6000]


@pytest.fixture(scope="module")
def resultado(datos):
    s = generar_senales(datos)
    return s, backtest(s, s["objetivo"], liquidar_al_final=True)


def test_contabilidad_valor_final(resultado):
    """El equity final es el efectivo más el valor de la posición abierta."""
    s, r = resultado
    assert len(r["operaciones"]) > 0
    assert r["equity"].iloc[-1] == pytest.approx(r["efectivo"].iloc[-1] + r["posicion"].iloc[-1] * s["Close"].iloc[-1])


def test_contabilidad_costos(resultado):
    """La suma de comisiones es la tasa por el monto de cada apertura y cada cierre."""
    _, r = resultado
    ops = r["operaciones"]
    monto = (ops["acciones"] * (ops["precio_entrada"] + ops["precio_salida"])).sum()
    assert r["comisiones"] == pytest.approx(Costos().comision * monto)
    assert ops["comision"].sum() == pytest.approx(r["comisiones"])


def test_golden_file_equity_calculado_a_mano():
    """Cuentas en papel (capital 100,000; comisión 0.1%; sin spread ni impacto; SL 2 ATR, TP 3 ATR; riesgo 1%):

    Objetivo +1 desde el cierre de 09:45 → compra en la apertura de 09:50 a 100.
    Stop = 100 − 2·1 = 98; objetivo = 100 + 3·1 = 103; acciones = 0.01 · 100,000 / 2 = 500.
    Comisión de entrada = 0.001 · 500 · 100 = 50 → efectivo = 100,000 − 50,000 − 50 = 49,950.
    09:50: equity = 49,950 + 500 · 100.2 = 100,050.   09:55: equity = 49,950 + 500 · 101.0 = 100,450.
    10:00: el máximo 103.5 toca el objetivo → venta a 103; comisión = 0.001 · 500 · 103 = 51.5.
           efectivo = 49,950 + 51,500 − 51.5 = 101,398.5 = equity final; P&L = 1,500 − 101.5 = 1,398.5.
    """
    indice = pd.date_range("2026-03-02 09:30", periods=8, freq="5min", tz="America/New_York")
    datos = pd.DataFrame({
        "Open":  [100, 100, 100, 100, 100.0, 100.2, 101.0, 103.2],
        "High":  [100, 100, 100, 100, 100.5, 101.5, 103.5, 103.4],
        "Low":   [100, 100, 100, 100, 99.5, 100.0, 100.8, 103.0],
        "Close": [100, 100, 100, 100, 100.2, 101.0, 103.2, 103.2],
        "atr_14": 1.0,
    }, index=indice)
    objetivo = pd.Series([0, 0, 0, 1, 1, 1, 1, 1], index=indice)
    r = backtest(datos, objetivo, theta={"sl_atr": 2.0, "tp_atr": 3.0, "riesgo": 0.01, "max_velas": 48},
                 costos=Costos(0.001, 0.0, 0.0), capital=100_000)
    esperado = [100_000, 100_000, 100_000, 100_000, 100_050, 100_450, 101_398.5, 101_398.5]
    assert r["equity"].tolist() == pytest.approx(esperado, abs=1e-9)
    assert r["operaciones"].iloc[0]["motivo"] == "take_profit"
    assert r["operaciones"].iloc[0]["pnl"] == pytest.approx(1_398.5)


def test_stop_y_objetivo_en_la_misma_vela_cierra_por_stop():
    pos = Posicion(lado=1, acciones=100, precio_entrada=100.0, stop_loss=98.0, take_profit=103.0)
    assert pos.salida_intrabar(alto=104.0, bajo=97.0, apertura=100.5) == (98.0, "stop_loss")


def test_sizing_regresa_exactamente_el_riesgo_presupuestado():
    acciones = tamano_por_riesgo(1_000_000.0, 200.0, 198.0, 0.0025)
    assert acciones * 2.0 == pytest.approx(2_500.0, rel=1e-12)


def test_nunca_hay_dos_posiciones_abiertas():
    indice = pd.date_range("2026-03-02 09:30", periods=78, freq="5min", tz="America/New_York")
    indice = indice.append([indice + pd.Timedelta(days=d) for d in range(1, 6)])
    rng = np.random.default_rng(42)
    cierre = 100 + rng.normal(0, 0.3, len(indice)).cumsum()
    datos = pd.DataFrame({"Open": cierre, "High": cierre + 0.4, "Low": cierre - 0.4, "Close": cierre,
                          "atr_14": 0.5}, index=indice)
    r = backtest(datos, pd.Series(rng.choice([-1, 1], len(indice)), index=indice), costos=SIN_COSTOS)
    ops = r["operaciones"]
    assert len(ops) > 10
    assert (ops["entrada"].iloc[1:].to_numpy() >= ops["salida"].iloc[:-1].to_numpy()).all()


@pytest.mark.parametrize("t", [1500, 3700, 5999])
def test_truncamiento_de_todo_el_pipeline(datos, t):
    """Indicadores → señales → posición objetivo → backtest sobre df.iloc[:t+1]: el valor en t no cambia."""
    completo = generar_senales(datos)
    r_completo = backtest(completo, completo["objetivo"])
    parcial = generar_senales(datos.iloc[: t + 1])
    r_parcial = backtest(parcial, parcial["objetivo"])
    assert parcial["objetivo"].iloc[-1] == completo["objetivo"].iloc[t]
    assert r_parcial["equity"].iloc[-1] == pytest.approx(r_completo["equity"].iloc[t], rel=1e-12)


def test_la_posicion_pasa_la_noche_y_el_corto_paga_prestamo():
    """Un corto abierto el lunes sigue abierto el martes (no hay cierre de 15:55) y paga préstamo."""
    lunes = pd.date_range("2026-03-02 09:30", periods=78, freq="5min", tz="America/New_York")
    indice = lunes.append(lunes + pd.Timedelta(days=1))
    datos = pd.DataFrame({"Open": 100.0, "High": 100.1, "Low": 99.9, "Close": 100.0, "atr_14": 0.5}, index=indice)
    objetivo = pd.Series(-1, index=indice)
    objetivo.iloc[0] = 0
    r = backtest(datos, objetivo, theta={"sl_atr": 6.0, "tp_atr": 18.0, "riesgo": 0.005, "max_velas": 780},
                 costos=Costos(0.0, 0.0, 0.0, 0.01))
    assert len(r["operaciones"]) == 1
    assert r["posicion_abierta"] is not None and r["posicion_abierta"].lado == -1
    assert r["prestamo"] > 0


def test_despues_del_stop_no_reentra_hasta_que_cambie_el_objetivo():
    indice = pd.date_range("2026-03-02 09:30", periods=12, freq="5min", tz="America/New_York")
    precio = [100, 100, 100, 99, 95, 95, 95, 95, 95, 95, 95, 95.0]
    datos = pd.DataFrame({"Open": precio, "High": precio, "Low": precio, "Close": precio, "atr_14": 1.0}, index=indice)
    objetivo = pd.Series([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], index=indice)
    r = backtest(datos, objetivo, theta={"sl_atr": 2.0, "tp_atr": 6.0, "riesgo": 0.01, "max_velas": 780},
                 costos=SIN_COSTOS)
    ops = r["operaciones"]
    assert ops["motivo"].iloc[0] == "stop_loss"
    assert len(ops) == 2 and ops["entrada"].iloc[1] == indice[9]


def test_compra_nocturna_entra_a_las_1555_y_sale_en_la_apertura_siguiente():
    lunes = pd.date_range("2026-03-02 09:30", periods=78, freq="5min", tz="America/New_York")
    indice = lunes.append(lunes + pd.Timedelta(days=1))
    precio = np.r_[np.full(78, 100.0), np.full(78, 101.0)]
    datos = pd.DataFrame({"Open": precio, "High": precio, "Low": precio, "Close": precio, "atr_14": 0.5}, index=indice)
    nocturna = pd.Series(0, index=indice)
    nocturna[indice.strftime("%H:%M") == "15:50"] = 1
    r = backtest(datos, pd.Series(0, index=indice), theta={"sl_atr": 6.0, "tp_atr": 18.0, "riesgo": 0.005, "max_velas": 780},
                 costos=SIN_COSTOS, nocturna=nocturna, liquidar_al_final=True)
    op = r["operaciones"].iloc[0]
    assert op["tipo"] == "noche" and op["entrada"] == indice[77] and op["salida"] == indice[78]
    assert op["motivo"] == "fin_nocturna" and op["pnl"] > 0
