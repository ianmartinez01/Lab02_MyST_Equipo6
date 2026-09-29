"""Métricas de desempeño sobre la curva de valor del portafolio.

Convenciones declaradas:
- Retornos simples por vela de 5 min; tasa libre de riesgo igual a 0.
- Anualización con 78 velas por sesión regular × 252 sesiones = 19,656 velas por año.
- Rendimiento anualizado compuesto: (V_final / V_inicial)^(velas_por_año / n) − 1.
- Calmar = rendimiento anualizado / |máximo drawdown|.
- Win Rate = operaciones con P&L neto de comisiones > 0 entre el total de operaciones cerradas.
"""

import numpy as np
import pandas as pd

VELAS_POR_ANIO = 78 * 252


def retornos(valor: pd.Series) -> pd.Series:
    """Retornos simples por vela de la curva de valor."""
    return valor.pct_change().dropna()


def rendimiento_anualizado(valor: pd.Series, velas_por_anio: int = VELAS_POR_ANIO) -> float:
    n = len(valor) - 1
    if n <= 0:
        return np.nan
    return (valor.iloc[-1] / valor.iloc[0]) ** (velas_por_anio / n) - 1


def sharpe(valor: pd.Series, velas_por_anio: int = VELAS_POR_ANIO) -> float:
    """Sharpe = E[r] / σ(r) · √(velas por año)."""
    r = retornos(valor)
    sd = r.std()
    return np.nan if sd == 0 or np.isnan(sd) else r.mean() / sd * np.sqrt(velas_por_anio)


def sortino(valor: pd.Series, velas_por_anio: int = VELAS_POR_ANIO) -> float:
    """Sortino = E[r] / σ₋ · √(velas por año), con σ₋ = √(E[min(r, 0)²])."""
    r = retornos(valor)
    sd_neg = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return np.nan if sd_neg == 0 or np.isnan(sd_neg) else r.mean() / sd_neg * np.sqrt(velas_por_anio)


def curva_drawdown(valor: pd.Series) -> pd.Series:
    """Drawdown en cada vela: V_t / máx(V_0..V_t) − 1 (valores ≤ 0)."""
    return valor / valor.cummax() - 1


def max_drawdown(valor: pd.Series) -> float:
    """Máximo drawdown como número negativo (−0.10 = caída de 10% desde el máximo)."""
    return float(curva_drawdown(valor).min())


def calmar(valor: pd.Series, velas_por_anio: int = VELAS_POR_ANIO) -> float:
    """Calmar = rendimiento anualizado / |máximo drawdown|."""
    mdd = abs(max_drawdown(valor))
    return np.nan if mdd == 0 else rendimiento_anualizado(valor, velas_por_anio) / mdd


def win_rate(operaciones: pd.DataFrame) -> float:
    """Proporción de operaciones cerradas con P&L neto positivo."""
    if operaciones.empty or "pnl" not in operaciones:
        return np.nan
    pnl = operaciones["pnl"].dropna()
    return float((pnl > 0).mean()) if len(pnl) else np.nan


def resumen_metricas(valor: pd.Series, operaciones: pd.DataFrame) -> dict:
    """Las cinco métricas obligatorias más el retorno total y el número de operaciones."""
    return {
        "retorno_total": valor.iloc[-1] / valor.iloc[0] - 1,
        "rendimiento_anualizado": rendimiento_anualizado(valor),
        "sharpe": sharpe(valor),
        "sortino": sortino(valor),
        "calmar": calmar(valor),
        "max_drawdown": max_drawdown(valor),
        "win_rate": win_rate(operaciones),
        "operaciones": int(len(operaciones)),
    }


def tabla_retornos(valor: pd.Series, frecuencia: str) -> pd.Series:
    """Retorno por periodo calendario: 'ME' mensual, 'QE' trimestral, 'YE' anual.

    El primer periodo se mide contra el valor inicial de la curva.
    """
    cierres = valor.resample(frecuencia).last().dropna()
    previo = pd.concat([pd.Series([valor.iloc[0]]), cierres.iloc[:-1]]).to_numpy()
    return pd.Series(cierres.to_numpy() / previo - 1, index=cierres.index, name="retorno")


def tablas_retornos(valor: pd.Series) -> dict:
    """Tablas de retornos mensuales, trimestrales y anuales."""
    return {nombre: tabla_retornos(valor, f) for nombre, f in
            (("mensual", "ME"), ("trimestral", "QE"), ("anual", "YE"))}
