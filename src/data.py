"""Carga, validación y auditoría de los datos de NVDA a 5 minutos.

Fuente: velas públicas de Binance Stocks (NVDA, ajustadas). La descarga se hace
una sola vez con `python -m src.data` y se congela en data/nvda_5m.csv;
el resto del proyecto solo lee el CSV.
"""

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL_VELAS = "https://www.binance.com/bapi/equity/v1/public/equity/kline/chart"
INTERVALOS = {"1m": "1T", "5m": "5T", "1h": "1H", "4h": "4H", "1d": "1D"}
RUTA_CRUDOS = Path(__file__).resolve().parents[1] / "data" / "nvda_5m.csv"
ZONA = "America/New_York"
INICIO_RTH, FIN_RTH = "09:30", "15:55"   # última vela de 5 min abre a las 15:55
FIN_TRAIN = "2026-05-29"
FIN_TEST = "2026-07-31"


def descargar(ticker: str = "NVDA", start: str = "2026-01-01", end: str = "2026-09-26",
              interval: str = "5m", pausa: float = 0.3) -> pd.DataFrame:
    """Descarga velas OHLCV de un ticker entre start y end.

    Yahoo Finance solo guarda 60 días de velas de 5 minutos, así que se usa el histórico
    público de Binance Stocks (precios ajustados). Pagina hacia atrás de 1000 en 1000 velas.
    """
    inicio = pd.Timestamp(start, tz=ZONA).value // 10**6
    fin = pd.Timestamp(end, tz=ZONA).value // 10**6
    base = f"{URL_VELAS}?symbol={ticker}&adjustmentMode=ADJUSTED&timeframe={INTERVALOS[interval]}&limit=1000"
    filas, corte = [], fin - 1
    while corte >= inicio:
        with urllib.request.urlopen(f"{base}&endTime={corte}", timeout=30) as r:
            velas = json.load(r)["data"]["bars"]
        if not velas:
            break
        filas.extend(velas)
        corte = velas[0]["t"] - 1
        time.sleep(pausa)
    crudo = pd.DataFrame(filas).drop_duplicates("t").sort_values("t")
    crudo = crudo[(crudo["t"] >= inicio) & (crudo["t"] < fin)]
    df = crudo[["o", "h", "l", "c", "v"]].astype(float)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df.index = pd.to_datetime(crudo["t"], unit="ms", utc=True).dt.tz_convert(ZONA)
    df.index.name = "Datetime"
    return df


def congelar_datos(ruta: Path = RUTA_CRUDOS, **kwargs) -> Path:
    """Descarga con `descargar` y guarda el CSV que usa todo el proyecto."""
    descargar(**kwargs).to_csv(ruta)
    return ruta


def cargar_datos(ruta: Path = RUTA_CRUDOS, solo_rth: bool = True) -> pd.DataFrame:
    """Lee el CSV congelado como en clase; opcionalmente filtra a la sesión regular."""
    df = pd.read_csv(ruta, index_col="Datetime")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ZONA)
    if solo_rth:
        df = df.between_time(INICIO_RTH, FIN_RTH)
    return df


def auditar_datos(df: pd.DataFrame, velas_por_dia: int = 78) -> dict:
    """Revisa duplicados, nulos, coherencia OHLC y días incompletos. Devuelve un resumen."""
    ohlc_incoherente = (
        (df["High"] < df[["Open", "Close"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close"]].min(axis=1))
        | (df["High"] < df["Low"])
    )
    conteo_dia = df.groupby(df.index.date).size()
    return {
        "velas": len(df),
        "inicio": str(df.index.min()),
        "fin": str(df.index.max()),
        "dias": int(conteo_dia.size),
        "duplicados": int(df.index.duplicated().sum()),
        "nulos": int(df[["Open", "High", "Low", "Close"]].isna().sum().sum()),
        "ohlc_incoherente": int(ohlc_incoherente.sum()),
        "precios_no_positivos": int((df[["Open", "High", "Low", "Close"]] <= 0).sum().sum()),
        "dias_incompletos": {str(d): int(n) for d, n in conteo_dia[conteo_dia < velas_por_dia].items()},
    }


def separar_train_test(df: pd.DataFrame, fin_train: str = FIN_TRAIN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa por fecha: train hasta `fin_train` inclusive, test desde el día siguiente."""
    corte = pd.Timestamp(fin_train, tz=ZONA) + pd.Timedelta(days=1)
    return df[df.index < corte], df[df.index >= corte]


def separar_train_test_validacion(df: pd.DataFrame, fin_train: str = FIN_TRAIN,
                                  fin_test: str = FIN_TEST) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tres periodos cronológicos sin mezclar: train (≈ 55%), test (≈ 23%) y validation (≈ 22%)."""
    train, resto = separar_train_test(df, fin_train)
    test, validacion = separar_train_test(resto, fin_test)
    return train, test, validacion


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Descarga y congela las velas de NVDA a 5 min en data/.")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-09-26", help="fecha final exclusiva (hora de Nueva York)")
    parser.add_argument("--interval", default="5m", choices=sorted(INTERVALOS))
    args = parser.parse_args()
    ruta = congelar_datos(ticker=args.ticker, start=args.start, end=args.end, interval=args.interval)
    print(f"Datos guardados en {ruta}")
    for clave, valor in auditar_datos(cargar_datos(ruta)).items():
        print(f"  {clave}: {valor}")

