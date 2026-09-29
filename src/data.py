"""Carga, validación y auditoría de los datos de NVDA a 5 minutos.

Fuente: velas públicas de Binance Stocks (NVDA, ajustadas). La descarga se hace
una sola vez y se congela en data/; el resto del proyecto solo lee el CSV.
"""

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

URL_VELAS = (
    "https://www.binance.com/bapi/equity/v1/public/equity/kline/chart"
    "?symbol=NVDA&adjustmentMode=ADJUSTED&timeframe=5T&limit=1000"
)
RUTA_CRUDOS = Path(__file__).resolve().parents[1] / "data" / "nvda_5m_crudo.csv"
ZONA = "America/New_York"
INICIO_RTH, FIN_RTH = "09:30", "15:55"   # última vela de 5 min abre a las 15:55
FIN_TRAIN = "2026-06-30"


def descargar_velas(desde: str = "2026-01-01", pausa: float = 0.3) -> pd.DataFrame:
    """Descarga todas las velas de 5 min hacia atrás hasta `desde`, paginando con endTime."""
    limite = pd.Timestamp(desde, tz="UTC").value // 10**6
    filas, fin = [], None
    while True:
        url = URL_VELAS + (f"&endTime={fin}" if fin else "")
        with urllib.request.urlopen(url, timeout=30) as r:
            velas = json.load(r)["data"]["bars"]
        if not velas:
            break
        filas.extend(velas)
        fin = velas[0]["t"] - 1
        if velas[0]["t"] <= limite:
            break
        time.sleep(pausa)
    df = pd.DataFrame(filas).drop_duplicates("t").sort_values("t")
    df = df[df["t"] >= limite]
    return pd.DataFrame({
        "timestamp_ms": df["t"].astype("int64"),
        "Open": df["o"].astype(float),
        "High": df["h"].astype(float),
        "Low": df["l"].astype(float),
        "Close": df["c"].astype(float),
        "Volume": df["v"].astype(float),
    }).reset_index(drop=True)


def congelar_datos(ruta: Path = RUTA_CRUDOS, hasta: str | None = None) -> Path:
    """Descarga y guarda el CSV crudo. Solo se usa para regenerar data/; main.py no descarga."""
    df = descargar_velas()
    if hasta:
        df = df[df["timestamp_ms"] < pd.Timestamp(hasta, tz=ZONA).value // 10**6]
    df.to_csv(ruta, index=False)
    return ruta


def cargar_datos(ruta: Path = RUTA_CRUDOS, solo_rth: bool = True) -> pd.DataFrame:
    """Lee el CSV congelado con índice en hora de Nueva York; opcionalmente filtra a sesión regular."""
    df = pd.read_csv(ruta)
    df.index = pd.to_datetime(df.pop("timestamp_ms"), unit="ms", utc=True).dt.tz_convert(ZONA)
    df.index.name = "fecha"
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Descarga y congela las velas de NVDA a 5 min en data/.")
    parser.add_argument("--hasta", default="2026-09-26", help="fecha de corte exclusiva (hora de Nueva York)")
    args = parser.parse_args()
    ruta = congelar_datos(hasta=args.hasta)
    print(f"Datos guardados en {ruta}")
    for clave, valor in auditar_datos(cargar_datos(ruta)).items():
        print(f"  {clave}: {valor}")
