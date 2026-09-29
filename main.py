"""Ejecuta el proyecto completo: python main.py

Flujo: datos → señales → backtest → métricas, en train y en test.
Las etapas de régimen, optimización walk-forward y figuras se agregan al integrar sus módulos.
"""

import numpy as np
import pandas as pd

from src.backtest import COMISION, ejecutar_backtest
from src.data import RUTA_CRUDOS, auditar_datos, cargar_datos, congelar_datos, separar_train_test
from src.metrics import resumen_metricas, tablas_retornos
from src.signals import PARAMETROS_BASE, generar_senales

SEMILLA = 42


def titulo(texto: str):
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def main():
    np.random.seed(SEMILLA)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    titulo("PASO 1 · Datos: NVDA velas de 5 min, sesión regular 09:30–16:00 ET")
    if not RUTA_CRUDOS.exists():
        print("No existe data/nvda_5m.csv: descargando…")
        congelar_datos()
    datos = cargar_datos()
    for clave, valor in auditar_datos(datos).items():
        print(f"  {clave}: {valor}")
    train, test = separar_train_test(datos)
    print(f"  train: {train.index[0]:%Y-%m-%d} a {train.index[-1]:%Y-%m-%d} ({len(train)} velas)")
    print(f"  test : {test.index[0]:%Y-%m-%d} a {test.index[-1]:%Y-%m-%d} ({len(test)} velas)")

    titulo("PASO 2 · Señales: 3 votos (Bollinger, RSI/Estocástico, MACD) + gatillo EMA 9")
    print("  Compra si ≥ 2 de 3 votos marcan sobreventa y el precio cruza arriba de la EMA 9.")
    print("  Venta (corto) si ≥ 2 de 3 marcan sobrecompra y el precio cruza abajo de la EMA 9.")
    print(f"  Parámetros: {PARAMETROS_BASE}")
    senales = {nombre: generar_senales(tramo) for nombre, tramo in (("train", train), ("test", test))}
    for nombre, s in senales.items():
        print(f"  {nombre}: {int((s['senal'] == 1).sum())} señales de compra, "
              f"{int((s['senal'] == -1).sum())} de venta en {len(s)} velas")
    ejemplo = senales["train"][senales["train"]["senal"] != 0].head(3)
    print("\n  Primeras señales en train (lo que vio el modelo al cierre de la vela):")
    print(ejemplo[["Close", "bb_low", "bb_high", "rsi_14", "stoch_k", "macd_atr", "ema_9",
                   "volatilidad", "momento", "tendencia", "senal"]].round(2).to_string())

    titulo(f"PASO 3 · Backtest: $1,000,000, comisión {COMISION:.3%} por lado, SL 1.5 ATR, TP 2 ATR")
    print("  La orden entra en la apertura de la vela siguiente; todo se cierra al final del día.")
    filas, resultados = {}, {}
    for nombre, s in senales.items():
        r = ejecutar_backtest(s)
        resultados[nombre] = r
        ops = r["operaciones"]
        print(f"\n  {nombre}: {len(ops)} operaciones · valor final ${r['valor'].iloc[-1]:,.0f} · "
              f"comisiones pagadas ${r['portafolio'].costos:,.0f}")
        print("  Motivos de salida:", ops["motivo"].value_counts().to_dict())
        vista = ops.head(5).assign(direccion=ops["direccion"].map({1: "compra", -1: "venta"}))
        print(vista[["direccion", "entrada", "precio_entrada", "salida", "precio_salida", "motivo", "pnl"]]
              .round({"precio_entrada": 2, "precio_salida": 2, "pnl": 2}).to_string(index=False))

    titulo("PASO 4 · Métricas")
    for nombre, r in resultados.items():
        filas[nombre] = resumen_metricas(r["valor"], r["operaciones"])
        filas[f"buy&hold {nombre}"] = resumen_metricas(senales[nombre]["Close"] / senales[nombre]["Close"].iloc[0] * 1e6,
                                                       pd.DataFrame())
        mensual = tablas_retornos(r["valor"])["mensual"]
        mensual.index = mensual.index.strftime("%Y-%m")
        print(f"\n  Retornos mensuales {nombre} (%): {(mensual * 100).round(2).to_dict()}")
    print()
    print(pd.DataFrame(filas).round(4).to_string())


if __name__ == "__main__":
    main()
