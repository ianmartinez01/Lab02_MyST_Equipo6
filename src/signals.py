"""Indicadores técnicos y regla de confirmación 2 de 3 con gatillo EMA.

Convención de señales: +1 compra (abre largo), -1 venta (abre corto), 0 sin señal.
Todo se calcula con información hasta el cierre de la vela t; la ejecución ocurre en t+1
y es responsabilidad del motor de backtesting.
"""

import numpy as np
import pandas as pd
import ta

PARAMETROS_BASE = {
    "bb_n": 20, "bb_k": 2.0,
    "rsi_n": 14, "rsi_bajo": 30, "rsi_alto": 70,
    "sto_n": 14, "sto_d": 3, "sto_bajo": 20, "sto_alto": 80,
    "macd_rapida": 12, "macd_lenta": 26, "macd_senal": 9, "macd_umbral": 0.8,
    "ema_n": 9, "atr_n": 14,
    "memoria": 3,
}

MODOS = ("dos_de_tres", "estricta", "solo_bollinger", "solo_momento", "solo_macd")


# Indicadores con la librería ta, con las mismas funciones y nombres de columna de la Act04.

def ema(df: pd.DataFrame, window: int = 9):
    return ta.trend.EMAIndicator(df["Close"], window=window).ema_indicator()


def bollinger(df: pd.DataFrame, window: int = 20, std_dev: float = 2.0):
    bb = ta.volatility.BollingerBands(df["Close"], window=window, window_dev=std_dev)
    return pd.DataFrame({
        "bb_high": bb.bollinger_hband(),
        "bb_low": bb.bollinger_lband(),
        "bb_mid": bb.bollinger_mavg(),
        "bb_pct": bb.bollinger_pband(),
    }, index=df.index)


def atr(df: pd.DataFrame, window: int = 14):
    # La firma de ta es (high, low, close); el orden importa.
    return ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=window
    ).average_true_range()


def rsi(df: pd.DataFrame, window: int = 14):
    return ta.momentum.RSIIndicator(df["Close"], window=window).rsi()


def macd(df: pd.DataFrame, window_fast: int = 12, window_slow: int = 26, window_sign: int = 9):
    m = ta.trend.MACD(df["Close"], window_slow=window_slow,
                      window_fast=window_fast, window_sign=window_sign)
    return pd.DataFrame({
        "macd": m.macd(),
        "macd_signal": m.macd_signal(),
        "macd_hist": m.macd_diff(),
    }, index=df.index)


def stochastic(df: pd.DataFrame, window: int = 14, smooth: int = 3):
    # %K sin suavizar y %D = media de 3 de %K: equivale al Estocástico 14,1,3 de TradingView.
    st = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=window, smooth_window=smooth
    )
    return pd.DataFrame({
        "stoch_k": st.stoch(),
        "stoch_d": st.stoch_signal(),
    }, index=df.index)


def calcular_indicadores(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Agrega al OHLC los indicadores de la estrategia y el MACD normalizado por ATR."""
    out = df[["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df else [])].copy()
    out["ema_9"] = ema(df, p["ema_n"])
    out = out.join(bollinger(df, p["bb_n"], p["bb_k"]))
    out["atr_14"] = atr(df, p["atr_n"])
    out["rsi_14"] = rsi(df, p["rsi_n"])
    out = out.join(macd(df, p["macd_rapida"], p["macd_lenta"], p["macd_senal"]))
    out = out.join(stochastic(df, p["sto_n"], p["sto_d"]))
    out["macd_atr"] = out["macd"] / out["atr_14"].replace(0, np.nan)
    return out


def _cruce_arriba(a: pd.Series, b) -> pd.Series:
    return (a.shift(1) <= (b.shift(1) if isinstance(b, pd.Series) else b)) & (a > b)


def _cruce_abajo(a: pd.Series, b) -> pd.Series:
    return (a.shift(1) >= (b.shift(1) if isinstance(b, pd.Series) else b)) & (a < b)


def _direccion(compra: pd.Series, venta: pd.Series) -> pd.Series:
    return compra.astype(int) - venta.astype(int)


def detectar_eventos(ind: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Evento de cada indicador en la vela t (+1 sobreventa, -1 sobrecompra) y gatillo EMA."""
    return pd.DataFrame({
        "bollinger": _direccion(ind["Close"] < ind["bb_low"], ind["Close"] > ind["bb_high"]),
        "rsi": _direccion(_cruce_abajo(ind["rsi_14"], p["rsi_bajo"]), _cruce_arriba(ind["rsi_14"], p["rsi_alto"])),
        "estocastico": _direccion(
            _cruce_arriba(ind["stoch_k"], ind["stoch_d"]) & (ind["stoch_k"] < p["sto_bajo"]),
            _cruce_abajo(ind["stoch_k"], ind["stoch_d"]) & (ind["stoch_k"] > p["sto_alto"]),
        ),
        "macd": _direccion(ind["macd_atr"] < -p["macd_umbral"], ind["macd_atr"] > p["macd_umbral"]),
        "gatillo_ema": _direccion(_cruce_arriba(ind["Close"], ind["ema_9"]), _cruce_abajo(ind["Close"], ind["ema_9"])),
    }, index=ind.index)


def mantener_activo(evento: pd.Series, memoria: int) -> pd.Series:
    """Un evento sigue vigente `memoria` velas (incluida la suya). Si hay compra y venta vigentes, 0."""
    compra = (evento == 1).astype(int).rolling(memoria, min_periods=1).max()
    venta = (evento == -1).astype(int).rolling(memoria, min_periods=1).max()
    return (compra - venta).astype(int)


def calcular_votos(eventos: pd.DataFrame, memoria: int) -> pd.DataFrame:
    """Tres votos de preparación: volatilidad (Bollinger), momento (RSI o Estocástico) y MACD."""
    rsi = mantener_activo(eventos["rsi"], memoria)
    sto = mantener_activo(eventos["estocastico"], memoria)
    momento = np.sign(rsi + sto).where((rsi * sto) >= 0, 0).astype(int)
    return pd.DataFrame({
        "volatilidad": mantener_activo(eventos["bollinger"], memoria),
        "momento": momento,
        "tendencia": mantener_activo(eventos["macd"], memoria),
    }, index=eventos.index)


def regla_confirmacion(votos: pd.DataFrame, minimo: int = 2) -> pd.Series:
    """+1 si al menos `minimo` votos son +1, -1 si al menos `minimo` son -1, 0 en otro caso.

    señal_t = +1 si Σ 1[v_i,t = +1] ≥ m ;  -1 si Σ 1[v_i,t = -1] ≥ m ;  0 si no.
    """
    compra = (votos == 1).sum(axis=1) >= minimo
    venta = (votos == -1).sum(axis=1) >= minimo
    return _direccion(compra & ~venta, venta & ~compra)


def generar_senales(df: pd.DataFrame, p: dict | None = None, modo: str = "dos_de_tres") -> pd.DataFrame:
    """Señal de entrada al cierre de t: preparación según `modo` y gatillo EMA en la misma dirección.

    Modos: dos_de_tres (oficial), estricta (los tres votos), o un solo voto
    (solo_bollinger, solo_momento, solo_macd) para la comparación de la pregunta 1.
    """
    if modo not in MODOS:
        raise ValueError(f"modo desconocido: {modo}")
    p = {**PARAMETROS_BASE, **(p or {})}
    ind = calcular_indicadores(df, p)
    eventos = detectar_eventos(ind, p)
    votos = calcular_votos(eventos, p["memoria"])
    if modo == "dos_de_tres":
        preparacion = regla_confirmacion(votos, 2)
    elif modo == "estricta":
        preparacion = regla_confirmacion(votos, 3)
    else:
        preparacion = votos[{"solo_bollinger": "volatilidad", "solo_momento": "momento", "solo_macd": "tendencia"}[modo]]
    senal = preparacion.where(preparacion == eventos["gatillo_ema"], 0)
    senal[ind.isna().any(axis=1)] = 0
    return ind.join(votos).assign(preparacion=preparacion, gatillo_ema=eventos["gatillo_ema"], senal=senal.astype(int))
