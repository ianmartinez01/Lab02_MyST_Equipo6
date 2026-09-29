"""Ejecuta el laboratorio completo: python main.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import ejecutar_backtest
from src.data import RUTA_CRUDOS, auditar_datos, cargar_datos, congelar_datos, separar_train_test
from src.metrics import resumen_metricas, tablas_retornos
from src.plots import plot_indicadores, plot_operaciones, plot_valor_drawdown
from src.signals import MODOS, generar_senales

SEED = 42
FIGURES_DIR = Path("docs/figures")
DIA_EJEMPLO = "2026-03-16"


def buy_and_hold(datos, capital=1_000_000):
    return datos["Close"] / datos["Close"].iloc[0] * capital


def main():
    np.random.seed(SEED)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 140)

    print("1. Cargando datos de NVDA (5 min, sesión regular)...")
    if not RUTA_CRUDOS.exists():
        print("   No existe data/nvda_5m.csv: descargando...")
        congelar_datos()
    datos = cargar_datos()
    auditoria = auditar_datos(datos)
    print(f"   {auditoria['velas']} velas en {auditoria['dias']} días, de {auditoria['inicio']} a {auditoria['fin']}")
    print(f"   Duplicados: {auditoria['duplicados']}, nulos: {auditoria['nulos']}, "
          f"OHLC incoherente: {auditoria['ohlc_incoherente']}")
    train, test = separar_train_test(datos)
    print(f"   Train: {len(train)} velas · Test: {len(test)} velas")

    print("\n2. Generando señales (2 de 3 votos + gatillo EMA 9)...")
    senales = {"train": generar_senales(train), "test": generar_senales(test)}
    conteo = pd.DataFrame({
        modo: generar_senales(train, modo=modo)["senal"].value_counts().reindex([1, -1], fill_value=0)
        for modo in MODOS
    }).rename(index={1: "compras", -1: "ventas"})
    print("   Señales en train por regla:")
    print("   " + conteo.to_string().replace("\n", "\n   "))

    print("\n3. Ejecutando backtest ($1,000,000, comisión 0.125%, SL 1.5 ATR, TP 2 ATR)...")
    resultados = {nombre: ejecutar_backtest(s) for nombre, s in senales.items()}
    for nombre, r in resultados.items():
        ops = r["operaciones"]
        print(f"   {nombre.capitalize()}: {len(ops)} operaciones, valor final ${r['valor'].iloc[-1]:,.2f}, "
              f"comisiones ${r['portafolio'].costos:,.2f}")

    print("\n4. Calculando métricas...")
    metricas = {}
    for nombre, r in resultados.items():
        metricas[nombre] = resumen_metricas(r["valor"], r["operaciones"])
        metricas[f"buy&hold {nombre}"] = resumen_metricas(buy_and_hold(senales[nombre]), pd.DataFrame())
    print("   " + pd.DataFrame(metricas).round(4).to_string().replace("\n", "\n   "))
    for nombre, r in resultados.items():
        mensual = tablas_retornos(r["valor"])["mensual"] * 100
        mensual.index = mensual.index.strftime("%Y-%m")
        print(f"   Retornos mensuales {nombre} (%): {mensual.round(2).to_dict()}")

    print("\n5. Generando figuras...")
    figures = {
        "01_indicadores.png": plot_indicadores(
            senales["train"].loc[DIA_EJEMPLO], titulo=f"NVDA {DIA_EJEMPLO}: indicadores de la estrategia"),
        "02_operaciones.png": plot_operaciones(
            senales["train"].loc[DIA_EJEMPLO], resultados["train"]["operaciones"],
            titulo=f"NVDA {DIA_EJEMPLO}: señales y operaciones del modelo"),
        "03_valor_drawdown.png": plot_valor_drawdown(
            {nombre: (r["valor"], buy_and_hold(senales[nombre])) for nombre, r in resultados.items()},
            titulo="Valor del portafolio con parámetros base vs buy & hold"),
    }
    for filename, fig in figures.items():
        fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"   Figuras guardadas en: {FIGURES_DIR}")
    print("\nProyecto ejecutado correctamente.")


if __name__ == "__main__":
    main()
