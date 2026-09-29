"""Ejecuta el laboratorio completo: python main.py"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import ejecutar_backtest
from src.data import RUTA_CRUDOS, auditar_datos, cargar_datos, congelar_datos, separar_train_test
from src.metrics import metricas_por_regimen, prueba_diferencia_regimenes, resumen_metricas, tablas_retornos
from src.optimize import (CALENTAMIENTO, MIN_OPERACIONES, N_PRUEBAS, backtest_por_regimen, barrido_costos,
                          modelo_final, sensibilidad, walk_forward)
from src.plots import (plot_costos, plot_distribuciones_regimen, plot_indicadores, plot_operaciones,
                       plot_regimenes_precio, plot_sensibilidad, plot_tabla_retornos, plot_valor_drawdown,
                       plot_valor_regimenes, plot_walk_forward)
from src.regimes import ajustar_modelo, calcular_variables_regimen, etiquetar, validar_regimenes
from src.signals import MODOS, generar_senales

SEED = 42
FIGURES_DIR = Path("docs/figures")
RESULTS_DIR = Path("resultados")
DIA_EJEMPLO = "2026-03-16"


def buy_and_hold(datos, capital=1_000_000):
    return datos["Close"] / datos["Close"].iloc[0] * capital


def imprimir(tabla):
    print("   " + tabla.to_string().replace("\n", "\n   "))


def main():
    np.random.seed(SEED)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)
    resultados = {}

    print("1. Cargando datos de NVDA (5 min, sesión regular)...")
    if not RUTA_CRUDOS.exists():
        print("   No existe data/nvda_5m.csv: descargando...")
        congelar_datos()
    datos = cargar_datos()
    auditoria = auditar_datos(datos)
    train, test = separar_train_test(datos)
    print(f"   {auditoria['velas']} velas en {auditoria['dias']} días; duplicados {auditoria['duplicados']}, "
          f"nulos {auditoria['nulos']}, OHLC incoherente {auditoria['ohlc_incoherente']}")
    print(f"   Train: {train.index[0]:%Y-%m-%d} a {train.index[-1]:%Y-%m-%d} ({len(train)} velas) · "
          f"Test: {test.index[0]:%Y-%m-%d} a {test.index[-1]:%Y-%m-%d} ({len(test)} velas)")

    print("\n2. Detectando regímenes con K-means (ventana de 1 semana, actualización cada 4 h)...")
    variables = calcular_variables_regimen(datos)
    modelo = ajustar_modelo(variables.loc[train.index])
    etiquetas = etiquetar(modelo, variables)
    validacion = {n: validar_regimenes(modelo, variables.loc[d.index], etiquetas.loc[d.index])
                  for n, d in (("train", train), ("test", test))}
    imprimir(pd.DataFrame(validacion))
    resultados["regimenes"] = validacion

    print("\n3. Generando señales con los parámetros base (2 de 3 votos + gatillo EMA 9)...")
    senales_base = generar_senales(train)
    conteo = pd.DataFrame({m: generar_senales(train, modo=m)["senal"].value_counts().reindex([1, -1], fill_value=0)
                           for m in MODOS}).rename(index={1: "compras", -1: "ventas"})
    imprimir(conteo)

    print(f"\n4. Walk-forward sobre train (1 mes / 1 semana, {N_PRUEBAS} pruebas por régimen, "
          f"mínimo {MIN_OPERACIONES} operaciones)...")
    wf = walk_forward(train)
    resumen_wf = wf["resumen"]
    print(f"   {len(resumen_wf)} ventanas, {wf['configuraciones']} configuraciones evaluadas en {wf['segundos']:.0f} s")
    imprimir(resumen_wf.round(3))
    metricas_wf = resumen_metricas(wf["valor"], wf["operaciones"])
    dentro = resumen_wf["calmar_dentro"].median()
    fuera = resumen_wf["calmar_fuera"].median()
    print(f"   Calmar mediano dentro de muestra {dentro:.2f} · fuera de muestra {fuera:.2f}")
    resultados["walk_forward"] = {"ventanas": len(resumen_wf), "configuraciones": wf["configuraciones"],
                                  "segundos": wf["segundos"], "calmar_mediano_dentro": dentro,
                                  "calmar_mediano_fuera": fuera,
                                  "retorno_medio_dentro": resumen_wf["retorno_dentro"].mean(),
                                  "retorno_medio_fuera": resumen_wf["retorno_fuera"].mean(),
                                  "semanas_positivas": int((resumen_wf["retorno_fuera"] > 0).sum())}
    resumen_wf.to_csv(RESULTS_DIR / "walk_forward.csv", index=False)

    print("\n5. Modelo final: K-means con todo train y Optuna por régimen en su último mes...")
    final = modelo_final(train)
    print(f"   Calmar de entrenamiento por régimen: "
          f"{ {k: round(v, 2) for k, v in final['calmar_por_regimen'].items()} }")
    for regimen, p in final["parametros"].items():
        texto = "apagado (sin Calmar positivo)" if p is None else {k: round(v, 2) for k, v in p.items()}
        print(f"   {regimen}: {texto}")
    etiquetas_final = etiquetar(final["modelo"], variables)
    tramo_test = datos.iloc[len(train) - CALENTAMIENTO:]
    resultado_test = backtest_por_regimen(tramo_test, etiquetas_final, final["parametros"], test.index[0])
    metricas = {
        "train (walk-forward)": metricas_wf,
        "buy&hold train": resumen_metricas(buy_and_hold(train.loc[wf["valor"].index[0]:]), pd.DataFrame()),
        "test": resumen_metricas(resultado_test["valor"], resultado_test["operaciones"]),
        "buy&hold test": resumen_metricas(buy_and_hold(test), pd.DataFrame()),
    }
    tabla_metricas = pd.DataFrame(metricas)
    imprimir(tabla_metricas.round(4))
    tabla_metricas.to_csv(RESULTS_DIR / "metricas.csv")
    tablas = {"train": tablas_retornos(wf["valor"]), "test": tablas_retornos(resultado_test["valor"])}
    por_regimen = metricas_por_regimen(resultado_test["valor"], resultado_test["operaciones"], etiquetas_final)
    print("   Métricas por régimen en test:")
    imprimir(por_regimen.astype(float).round(4))
    diferencia = prueba_diferencia_regimenes(pd.concat([wf["operaciones"], resultado_test["operaciones"]]))
    print(f"   Kruskal-Wallis del retorno por operación entre regímenes (train WF + test): "
          f"H = {diferencia['estadistico']:.2f}, p = {diferencia['p_valor']:.3f}")
    por_regimen.to_csv(RESULTS_DIR / "metricas_por_regimen.csv")
    resultados["metricas"] = tabla_metricas.to_dict()
    resultados["kruskal"] = diferencia
    resultados["parametros_finales"] = final["parametros"]

    print("\n6. Robustez en test: reglas de confirmación, sensibilidad ±20% y costos...")
    reglas = {}
    for modo in MODOS:
        r = backtest_por_regimen(tramo_test, etiquetas_final, final["parametros"], test.index[0], modo=modo)
        m = resumen_metricas(r["valor"], r["operaciones"])
        reglas[modo] = {"operaciones": m["operaciones"], "calmar": m["calmar"], "retorno_total": m["retorno_total"],
                        "win_rate": m["win_rate"]}
    tabla_reglas = pd.DataFrame(reglas).T
    imprimir(tabla_reglas.astype(float).round(4))
    tabla_sens = sensibilidad(tramo_test, etiquetas_final, final["parametros"], test.index[0])
    imprimir(tabla_sens.round(3))
    costos = barrido_costos(tramo_test, etiquetas_final, final["parametros"], test.index[0])
    costos_train = barrido_costos(datos.iloc[: len(train)], etiquetas_final, final["parametros"],
                                  train.index[CALENTAMIENTO])
    positivos = costos[costos["retorno_neto"] > 0]
    equilibrio = positivos["comision"].max() if len(positivos) else None
    print(f"   Retorno neto en test sin comisión: {costos['retorno_neto'].iloc[0]:.2%}; "
          f"comisión máxima con retorno positivo: {equilibrio if equilibrio is not None else 'ninguna'}")
    tabla_reglas.to_csv(RESULTS_DIR / "reglas.csv")
    tabla_sens.to_csv(RESULTS_DIR / "sensibilidad.csv")
    costos.to_csv(RESULTS_DIR / "costos_test.csv", index=False)
    resultados["reglas"] = reglas
    resultados["equilibrio_costos"] = equilibrio

    print("\n7. Generando figuras...")
    figures = {
        "01_indicadores.png": plot_indicadores(senales_base.loc[DIA_EJEMPLO],
                                               titulo=f"NVDA {DIA_EJEMPLO}: indicadores de la estrategia"),
        "02_operaciones.png": plot_operaciones(
            senales_base.loc[DIA_EJEMPLO], ejecutar_backtest(senales_base)["operaciones"],
            titulo=f"NVDA {DIA_EJEMPLO}: señales y operaciones con parámetros base"),
        "03_valor_drawdown.png": plot_valor_drawdown(
            {"train (walk-forward)": (wf["valor"], buy_and_hold(train.loc[wf["valor"].index[0]:])),
             "test": (resultado_test["valor"], buy_and_hold(test))},
            titulo="Valor del portafolio y drawdown: estrategia contra buy & hold"),
        "04_tabla_retornos.png": plot_tabla_retornos(tablas),
        "05_sensibilidad.png": plot_sensibilidad(tabla_sens),
        "06_costos.png": plot_costos({"test": costos, "train": costos_train}),
        "07_regimenes_precio.png": plot_regimenes_precio(datos["Close"], etiquetas_final, test.index[0],
                                                         titulo="Regímenes detectados por K-means sobre el precio de NVDA"),
        "08_distribuciones_regimen.png": plot_distribuciones_regimen(variables.loc[train.index], etiquetas_final),
        "09_valor_regimenes.png": plot_valor_regimenes(resultado_test["valor"], etiquetas_final, buy_and_hold(test),
                                                       titulo="Valor del portafolio en test con regímenes superpuestos"),
        "10_walk_forward.png": plot_walk_forward(resumen_wf),
    }
    for filename, fig in figures.items():
        fig.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    (RESULTS_DIR / "resumen.json").write_text(json.dumps(resultados, indent=2, default=str), encoding="utf-8")

    print(f"   Figuras guardadas en: {FIGURES_DIR}")
    print(f"   Tablas guardadas en: {RESULTS_DIR}")
    print("\nProyecto ejecutado correctamente.")


if __name__ == "__main__":
    main()
