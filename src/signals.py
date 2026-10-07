"""Indicadores técnicos, regla de confirmación 2 de 3 y posición objetivo.

Tres votos de tres familias distintas, calculados con la librería ta:
- Volatilidad: ruptura del canal de Keltner (el cierre cruza la banda superior o la inferior).
- Momento: RSI arriba o abajo de 50.
- Tendencia: signo del TRIX.
La señal es +1 (compra) o −1 (venta) cuando al menos 2 de los 3 votos coinciden. La posición objetivo
opera la señal solo a favor de la tendencia diaria: la señal a favor abre, la señal en contra cierra, y la
posición pasa la noche.

Módulo de compra nocturna: en la vela de las 15:50 se compra (en la apertura de las 15:55, para vender en la
apertura del día siguiente) si al menos 2 de 3 votos dicen que el día cerró débil: distancia al VWAP de la
sesión ≤ −umbral (volumen), cierre en el tercio inferior del rango del día (momento, como un Estocástico de la
sesión) y RSI(14) < 40 (momento). Solo hay lado de compra.

Todo se calcula con información hasta el cierre de la vela t; la ejecución ocurre en t+1 y es
responsabilidad del motor de backtesting.
"""

import numpy as np
import pandas as pd
import ta

PARAMETROS_BASE = {
    "kc_n": 48, "kc_atr": 28,
    "rsi_n": 24,
    "trix_n": 25,
    "memoria": 14,
    "dias_tendencia": 21,
    "atr_n": 14,
    "umbral_vwap": 0.005,
}

MODOS = ("dos_de_tres", "estricta", "solo_keltner", "solo_rsi", "solo_trix")
MODOS_NOCHE = ("dos_de_tres", "simple", "ninguna")
HORA_NOCHE = "15:50"
RSI_NOCHE = 40
RANGO_NOCHE = 1 / 3


# Indicadores con la librería ta, con el mismo estilo de la Act04.

def keltner(df: pd.DataFrame, window: int = 20, window_atr: int = 10):
    kc = ta.volatility.KeltnerChannel(df["High"], df["Low"], df["Close"], window=window,
                                      window_atr=window_atr, original_version=False)
    return pd.DataFrame({
        "kc_high": kc.keltner_channel_hband(),
        "kc_mid": kc.keltner_channel_mband(),
        "kc_low": kc.keltner_channel_lband(),
    }, index=df.index)


def atr(df: pd.DataFrame, window: int = 14):
    # La firma de ta es (high, low, close); el orden importa.
    return ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=window
    ).average_true_range()


def rsi(df: pd.DataFrame, window: int = 14):
    return ta.momentum.RSIIndicator(df["Close"], window=window).rsi()


def trix(df: pd.DataFrame, window: int = 15):
    return ta.trend.TRIXIndicator(df["Close"], window=window).trix()


def vwap_sesion(df: pd.DataFrame) -> pd.Series:
    """VWAP que se reinicia cada sesión: Σ precio típico · volumen / Σ volumen desde las 09:30 hasta t."""
    dia = df.index.date
    tipico = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tipico * df["Volume"]).groupby(dia).cumsum() / df["Volume"].groupby(dia).cumsum().replace(0, np.nan)


def posicion_en_rango(df: pd.DataFrame) -> pd.Series:
    """(Close − mínimo de la sesión) / (máximo − mínimo de la sesión) hasta t: 0 = en el mínimo, 1 = en el máximo."""
    dia = df.index.date
    alto, bajo = df["High"].groupby(dia).cummax(), df["Low"].groupby(dia).cummin()
    return (df["Close"] - bajo) / (alto - bajo).replace(0, np.nan)


def calcular_indicadores(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Agrega al OHLC el canal de Keltner, el RSI, el TRIX y el ATR."""
    out = df[["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df else [])].copy()
    out = out.join(keltner(df, p["kc_n"], p["kc_atr"]))
    out["rsi"] = rsi(df, p["rsi_n"])
    out["trix"] = trix(df, p["trix_n"])
    out["atr_14"] = atr(df, p["atr_n"])
    out["dist_vwap"] = df["Close"] / vwap_sesion(df) - 1
    out["pos_rango"] = posicion_en_rango(df)
    out["rsi_14"] = rsi(df, 14)
    return out


def _cruce_arriba(a: pd.Series, b) -> pd.Series:
    return (a.shift(1) <= (b.shift(1) if isinstance(b, pd.Series) else b)) & (a > b)


def _cruce_abajo(a: pd.Series, b) -> pd.Series:
    return (a.shift(1) >= (b.shift(1) if isinstance(b, pd.Series) else b)) & (a < b)


def _direccion(compra: pd.Series, venta: pd.Series) -> pd.Series:
    return compra.astype(int) - venta.astype(int)


def mantener_activo(evento: pd.Series, memoria: int) -> pd.Series:
    """Un evento sigue vigente `memoria` velas (incluida la suya). Si hay compra y venta vigentes, 0."""
    compra = (evento == 1).astype(int).rolling(memoria, min_periods=1).max()
    venta = (evento == -1).astype(int).rolling(memoria, min_periods=1).max()
    return (compra - venta).astype(int)


def calcular_votos(ind: pd.DataFrame, memoria: int) -> pd.DataFrame:
    """Tres votos: volatilidad (ruptura de Keltner vigente `memoria` velas), momento (RSI vs 50) y tendencia (TRIX)."""
    ruptura = _direccion(_cruce_arriba(ind["Close"], ind["kc_high"]), _cruce_abajo(ind["Close"], ind["kc_low"]))
    return pd.DataFrame({
        "volatilidad": mantener_activo(ruptura, memoria),
        "momento": np.sign(ind["rsi"] - 50).fillna(0).astype(int),
        "tendencia": np.sign(ind["trix"]).fillna(0).astype(int),
    }, index=ind.index)


def regla_confirmacion(votos: pd.DataFrame, minimo: int = 2) -> pd.Series:
    """+1 si al menos `minimo` votos son +1, -1 si al menos `minimo` son -1, 0 en otro caso.

    señal_t = +1 si Σ 1[v_i,t = +1] ≥ m ;  -1 si Σ 1[v_i,t = -1] ≥ m ;  0 si no.
    """
    compra = (votos == 1).sum(axis=1) >= minimo
    venta = (votos == -1).sum(axis=1) >= minimo
    return _direccion(compra & ~venta, venta & ~compra)


def tendencia_diaria(df: pd.DataFrame, dias: int = 20) -> pd.Series:
    """+1 si el cierre del día anterior está arriba de la media de sus últimos `dias` cierres diarios,
    −1 si está abajo, 0 sin historia suficiente. Usa solo sesiones ya cerradas (causal)."""
    fecha = pd.Series(df.index.date, index=df.index)
    cierre = df["Close"].groupby(fecha.to_numpy()).last()
    signo = np.sign(cierre - cierre.rolling(dias).mean()).shift(1)
    return fecha.map(signo).fillna(0).astype(int).rename("tendencia_diaria")


def posicion_objetivo(senal: pd.Series, tendencia: pd.Series) -> pd.Series:
    """Posición deseada al cierre de t:

    objetivo_t = tendencia_t     si señal_t = tendencia_t ≠ 0     (la señal a favor abre o mantiene)
    objetivo_t = 0               si señal_t = −tendencia_t o tendencia_t = 0 (la señal en contra cierra)
    objetivo_t = 0               si objetivo_{t−1} = −tendencia_t (la tendencia se volteó)
    objetivo_t = objetivo_{t−1}  en otro caso (la posición se mantiene, también de un día a otro)
    """
    s, d = senal.to_numpy(int), tendencia.to_numpy(int)
    out = np.zeros(len(s), dtype=int)
    previo = 0
    for t in range(len(s)):
        if d[t] == 0 or s[t] == -d[t] or previo == -d[t]:
            previo = 0
        elif s[t] == d[t]:
            previo = d[t]
        out[t] = previo
    return pd.Series(out, index=senal.index, name="objetivo")


def senal_nocturna(ind: pd.DataFrame, umbral_vwap: float, modo: str = "dos_de_tres") -> pd.DataFrame:
    """Votos de la compra nocturna y señal (+1 o 0) solo en la vela de las 15:50.

    noche_t = +1  si  hora_t = 15:50  y  1[dist_VWAP_t ≤ −umbral] + 1[posición en el rango_t < 1/3] + 1[RSI(14)_t < 40] ≥ 2
    En modo "simple" basta la condición del VWAP; en modo "ninguna" no hay compra nocturna.
    """
    if modo not in MODOS_NOCHE:
        raise ValueError(f"modo nocturno desconocido: {modo}")
    votos = pd.DataFrame({
        "noche_volumen": (ind["dist_vwap"] <= -umbral_vwap).astype(int),
        "noche_rango": (ind["pos_rango"] < RANGO_NOCHE).astype(int),
        "noche_rsi": (ind["rsi_14"] < RSI_NOCHE).astype(int),
    }, index=ind.index)
    en_hora = pd.Series(ind.index.strftime("%H:%M") == HORA_NOCHE, index=ind.index)
    if modo == "dos_de_tres":
        cumple = votos.sum(axis=1) >= 2
    elif modo == "simple":
        cumple = votos["noche_volumen"] == 1
    else:
        cumple = pd.Series(False, index=ind.index)
    return votos.assign(nocturna=(en_hora & cumple & ind[["dist_vwap", "pos_rango", "rsi_14"]].notna().all(axis=1)).astype(int))


def generar_senales(df: pd.DataFrame, p: dict | None = None, modo: str = "dos_de_tres",
                    modo_noche: str = "dos_de_tres") -> pd.DataFrame:
    """Señal al cierre de t según `modo`, posición objetivo con el filtro de tendencia diaria y señal nocturna.

    Modos: dos_de_tres (oficial), estricta (los tres votos), o un solo voto
    (solo_keltner, solo_rsi, solo_trix) para la comparación de reglas. `modo_noche` en MODOS_NOCHE.
    """
    if modo not in MODOS:
        raise ValueError(f"modo desconocido: {modo}")
    p = {**PARAMETROS_BASE, **(p or {})}
    ind = calcular_indicadores(df, p)
    votos = calcular_votos(ind, p["memoria"])
    if modo == "dos_de_tres":
        senal = regla_confirmacion(votos, 2)
    elif modo == "estricta":
        senal = regla_confirmacion(votos, 3)
    else:
        senal = votos[{"solo_keltner": "volatilidad", "solo_rsi": "momento", "solo_trix": "tendencia"}[modo]]
    senal = senal.where(ind[["kc_high", "kc_low", "rsi", "trix", "atr_14"]].notna().all(axis=1), 0).astype(int)
    tendencia = tendencia_diaria(df, p["dias_tendencia"])
    return ind.join(votos).join(senal_nocturna(ind, p["umbral_vwap"], modo_noche)).assign(
        senal=senal, tendencia_diaria=tendencia, objetivo=posicion_objetivo(senal, tendencia))


def contar_aperturas(objetivo: pd.Series) -> pd.Series:
    """Número de veces que la posición objetivo pasa a largo (compras) o a corto (ventas)."""
    nuevo = objetivo.ne(objetivo.shift())
    return pd.Series({"compras": int((nuevo & (objetivo == 1)).sum()), "ventas": int((nuevo & (objetivo == -1)).sum())})
