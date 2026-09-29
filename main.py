"""Ejecuta el proyecto completo: python main.py

Flujo: datos → señales → backtest → métricas, en train y en test.
Las etapas de régimen, optimización walk-forward y figuras se agregan al integrar sus módulos.
"""

import numpy as np
import pandas as pd

from src.backtest import ejecutar_backtest
from src.data import RUTA_CRUDOS, auditar_datos, cargar_datos, congelar_datos, separar_train_test
from src.metrics import resumen_metricas, tablas_retornos
from src.signals import PARAMETROS_BASE, generar_senales

SEMILLA = 42


def main():
    np.random.seed(SEMILLA)
    pd.set_option("display.width", 120)

    if not RUTA_CRUDOS.exists():
        print("No existe data/nvda_5m.csv: descargando…")
        congelar_datos()
    datos = cargar_datos()
    print("Auditoría de datos (sesión regular):")
    for clave, valor in auditar_datos(datos).items():
        print(f"  {clave}: {valor}")

    train, test = separar_train_test(datos)
    print(f"\nParámetros base: {PARAMETROS_BASE}")
    filas = {}
    for nombre, tramo in (("train", train), ("test", test)):
        resultado = ejecutar_backtest(generar_senales(tramo))
        filas[nombre] = resumen_metricas(resultado["valor"], resultado["operaciones"])
        mensual = tablas_retornos(resultado["valor"])["mensual"]
        print(f"\nRetornos mensuales {nombre} (%):")
        print((mensual * 100).round(2).to_string())

    print("\nMétricas por conjunto:")
    print(pd.DataFrame(filas).round(4).to_string())


if __name__ == "__main__":
    main()
