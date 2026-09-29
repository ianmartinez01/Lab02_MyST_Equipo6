"""Ejecuta el laboratorio completo: python main.py

Split cronológico: train (ene–may) → test (jun–jul) → validation (ago–sep).
El walk-forward y la elección de θ* usan solo train + test. θ* se congela en docs/theta_congelado.json
(`python main.py --congelar`) y se versiona antes de evaluar validation una sola vez.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import ejecutar_backtest
from src.data import RUTA_CRUDOS, auditar_datos, cargar_datos, congelar_datos, separar_train_test_validacion
from src.metrics import metricas_por_regimen, prueba_diferencia_regimenes, resumen_metricas, tablas_retornos
from src.optimize import (CALENTAMIENTO, EMBARGO, MIN_OPERACIONES, N_PRUEBAS, RANGOS, backtest_por_regimen,
                          barrido_costos, cargar_theta, encadenar, guardar_theta, importancia, modelo_final,
                          sensibilidad, superficie, walk_forward)
from src.plots import (plot_anchored_rolling, plot_costos, plot_distribuciones_regimen, plot_historia,
                       plot_importancia, plot_indicadores, plot_operaciones, plot_regimenes_precio,
                       plot_sensibilidad, plot_slices, plot_superficie, plot_tabla_retornos, plot_valor_drawdown,
                       plot_valor_regimenes, plot_walk_forward)
from src.regimes import calcular_variables_regimen, etiquetar, validar_regimenes
from src.signals import MODOS, generar_senales

SEED = 42
FIGURES_DIR = Path("docs/figures")
RESULTS_DIR = Path("resultados")
THETA = Path("docs/theta_congelado.json")
DIA_EJEMPLO = "2026-03-16"


def buy_and_hold(datos, capital=1_000_000):
    return datos["Close"] / datos["Close"].iloc[0] * capital


def imprimir(tabla):
    print("   " + tabla.to_string().replace("\n", "\n   "))


def iguales(a, b):
    """Compara dos θ por régimen con tolerancia numérica."""
    if (a is None) != (b is None):
        return False
    return a is None or all(np.isclose(a[k], b[k]) for k in RANGOS)


def main(solo_congelar=False):
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
    train, test, validacion = separar_train_test_validacion(datos)
    desarrollo = pd.concat([train, test])
    print(f"   {auditoria['velas']} velas en {auditoria['dias']} días; duplicados {auditoria['duplicados']}, "
          f"nulos {auditoria['nulos']}, OHLC incoherente {auditoria['ohlc_incoherente']}")
    for nombre, d in (("Train", train), ("Test", test), ("Validation", validacion)):
        print(f"   {nombre}: {d.index[0]:%Y-%m-%d} a {d.index[-1]:%Y-%m-%d} "
              f"({len(d)} velas, {len(d) / len(datos):.0%})")

    print("\n2. Modelo final sobre train + test: K-means y, por régimen, random search y TPE en julio...")
    final = modelo_final(desarrollo)
    diagnostico = {}
    for regimen in final["tpe"]:
        a, t = final["aleatorio"][regimen], final["tpe"][regimen]
        if t["estudio"] is None:
            print(f"   {regimen}: sin velas en el último mes")
            continue
        diagnostico[regimen] = {"n_random": a["pruebas"], "mejor_random": a["calmar"], "n_tpe": t["pruebas"],
                                "mejor_tpe": t["calmar"], "meseta_tpe": t["calmar_meseta"],
                                "theta_es_argmax": t["es_argmax"], "activo": t["parametros"] is not None}
        print(f"   {regimen}: random search N = {a['pruebas']} (mejor Calmar {a['calmar']:.2f}) · "
              f"TPE N = {t['pruebas']} (mejor Calmar {t['calmar']:.2f}, meseta {t['calmar_meseta']:.2f}, "
              f"θ* {'= argmax' if t['es_argmax'] else 'del centro de la meseta'}, "
              f"{'activo' if t['parametros'] is not None else 'apagado'})")
    resultados["diagnostico"] = diagnostico

    if solo_congelar or not THETA.exists():
        guardar_theta(THETA, final, desarrollo.index[-1])
        print(f"\n   θ* congelado en {THETA}. Haz commit de este archivo y vuelve a correr python main.py.")
        return
    congelado = cargar_theta(THETA)
    theta = congelado["parametros"]
    coincide = all(iguales(theta[r], final["parametros"][r]) for r in theta)
    print(f"\n   θ* leído de {THETA} (congelado antes de tocar validation); "
          f"{'coincide' if coincide else 'NO coincide'} con el recalculado.")
    resultados["theta_congelado"] = congelado

    print("\n3. Validando regímenes (modelo ajustado con train + test)...")
    variables = calcular_variables_regimen(datos)
    etiquetas = etiquetar(final["modelo"], variables)
    validacion_reg = {n: validar_regimenes(final["modelo"], variables.loc[d.index], etiquetas.loc[d.index])
                      for n, d in (("train", train), ("test", test), ("validation", validacion))}
    imprimir(pd.DataFrame(validacion_reg))
    resultados["regimenes"] = validacion_reg

    print("\n4. Señales con parámetros base (2 de 3 votos + gatillo EMA 9) en train...")
    senales_base = generar_senales(train)
    conteo = pd.DataFrame({m: generar_senales(train, modo=m)["senal"].value_counts().reindex([1, -1], fill_value=0)
                           for m in MODOS}).rename(index={1: "compras", -1: "ventas"})
    imprimir(conteo)

    print(f"\n5. Walk-forward sobre train + test (prueba de 1 semana, {N_PRUEBAS} pruebas TPE por régimen, "
          f"mínimo {MIN_OPERACIONES} operaciones, embargo de {EMBARGO} velas)...")
    wf = walk_forward(desarrollo, "rolling")
    anclado = walk_forward(desarrollo, "anchored")
    for w in (wf, anclado):
        print(f"   {w['modo'].capitalize()}: {len(w['resumen'])} ventanas, {w['configuraciones']} configuraciones "
              f"en {w['segundos']:.0f} s · rendimiento anualizado dentro {w['anualizado_dentro']:.2%}, "
              f"fuera {w['anualizado_fuera']:.2%} · walk-forward efficiency {w['eficiencia']:.2f}")
    resumen_wf = wf["resumen"]
    imprimir(resumen_wf.round(3))
    resumen_wf.to_csv(RESULTS_DIR / "walk_forward.csv", index=False)
    corte = test.index[0]
    valor_train, ops_train = encadenar([v for v in wf["ventanas"] if v["inicio_test"] < corte])
    valor_test, ops_test = encadenar([v for v in wf["ventanas"] if v["inicio_test"] >= corte])
    resultados["walk_forward"] = {
        "ventanas": len(resumen_wf), "configuraciones": wf["configuraciones"], "segundos": wf["segundos"],
        "calmar_mediano_dentro": resumen_wf["calmar_dentro"].median(),
        "calmar_mediano_fuera": resumen_wf["calmar_fuera"].median(),
        "retorno_medio_dentro": resumen_wf["retorno_dentro"].mean(),
        "retorno_medio_fuera": resumen_wf["retorno_fuera"].mean(),
        "semanas_positivas": int((resumen_wf["retorno_fuera"] > 0).sum()),
        **{f"{k}_{w['modo']}": w[k] for w in (wf, anclado)
           for k in ("eficiencia", "anualizado_dentro", "anualizado_fuera", "configuraciones", "segundos")},
        "retorno_fuera_rolling": float(wf["valor"].iloc[-1] / 1e6 - 1),
        "retorno_fuera_anchored": float(anclado["valor"].iloc[-1] / 1e6 - 1),
    }

    print("\n6. Evaluando validation UNA sola vez con θ* congelado...")
    tramo_val = datos.iloc[len(desarrollo) - CALENTAMIENTO:]
    resultado_val = backtest_por_regimen(tramo_val, etiquetas, theta, validacion.index[0])
    metricas = {
        "train (WF)": resumen_metricas(valor_train, ops_train),
        "B&H train": resumen_metricas(buy_and_hold(train.loc[valor_train.index[0]:]), pd.DataFrame()),
        "test (WF)": resumen_metricas(valor_test, ops_test),
        "B&H test": resumen_metricas(buy_and_hold(test), pd.DataFrame()),
        "validation": resumen_metricas(resultado_val["valor"], resultado_val["operaciones"]),
        "B&H validation": resumen_metricas(buy_and_hold(validacion), pd.DataFrame()),
    }
    tabla_metricas = pd.DataFrame(metricas)
    imprimir(tabla_metricas.round(4))
    tabla_metricas.to_csv(RESULTS_DIR / "metricas.csv")
    print(f"   Deslizamiento (spread + impacto) en validation: ${resultado_val['portafolio'].deslizamiento:,.0f}; "
          f"comisiones: ${resultado_val['portafolio'].costos:,.0f}")
    tablas = {"train": tablas_retornos(valor_train), "test": tablas_retornos(valor_test),
              "validation": tablas_retornos(resultado_val["valor"])}
    por_regimen = metricas_por_regimen(resultado_val["valor"], resultado_val["operaciones"], etiquetas)
    print("   Métricas por régimen en validation:")
    imprimir(por_regimen.astype(float).round(4))
    diferencia = prueba_diferencia_regimenes(pd.concat([wf["operaciones"], resultado_val["operaciones"]]))
    print(f"   Kruskal-Wallis del retorno por operación entre regímenes (walk-forward + validation): "
          f"H = {diferencia['estadistico']:.2f}, p = {diferencia['p_valor']:.3f}")
    por_regimen.to_csv(RESULTS_DIR / "metricas_por_regimen.csv")
    resultados["metricas"] = tabla_metricas.to_dict()
    resultados["kruskal"] = diferencia
    resultados["deslizamiento_validation"] = resultado_val["portafolio"].deslizamiento
    resultados["comisiones_validation"] = resultado_val["portafolio"].costos

    print("\n7. Robustez de θ* congelado en validation: reglas, sensibilidad ±20% y costos (sin reoptimizar)...")
    reglas = {}
    for modo in MODOS:
        r = backtest_por_regimen(tramo_val, etiquetas, theta, validacion.index[0], modo=modo)
        m = resumen_metricas(r["valor"], r["operaciones"])
        reglas[modo] = {"operaciones": m["operaciones"], "calmar": m["calmar"], "retorno_total": m["retorno_total"],
                        "win_rate": m["win_rate"]}
    tabla_reglas = pd.DataFrame(reglas).T
    imprimir(tabla_reglas.astype(float).round(4))
    tabla_sens = sensibilidad(tramo_val, etiquetas, theta, validacion.index[0])
    imprimir(tabla_sens.round(3))
    costos = barrido_costos(tramo_val, etiquetas, theta, validacion.index[0])
    costos_desarrollo = barrido_costos(datos.iloc[: len(desarrollo)], etiquetas, theta, desarrollo.index[CALENTAMIENTO])
    positivos = costos[costos["retorno_neto"] > 0]
    equilibrio = positivos["comision"].max() if len(positivos) else None
    print(f"   Retorno neto en validation sin comisión: {costos['retorno_neto'].iloc[0]:.2%}; "
          f"comisión máxima con retorno positivo: {equilibrio if equilibrio is not None else 'ninguna'}")
    tabla_reglas.to_csv(RESULTS_DIR / "reglas.csv")
    tabla_sens.to_csv(RESULTS_DIR / "sensibilidad.csv")
    costos.to_csv(RESULTS_DIR / "costos_validation.csv", index=False)
    resultados["reglas"] = reglas
    resultados["equilibrio_costos"] = equilibrio

    print("\n8. Generando figuras...")
    activos = {r: final["tpe"][r]["estudio"] for r in diagnostico}
    importancias = {r: importancia(e) for r, e in activos.items()}
    figures = {
        "01_indicadores.png": plot_indicadores(senales_base.loc[DIA_EJEMPLO],
                                               titulo=f"NVDA {DIA_EJEMPLO}: indicadores de la estrategia"),
        "02_operaciones.png": plot_operaciones(
            senales_base.loc[DIA_EJEMPLO], ejecutar_backtest(senales_base)["operaciones"],
            titulo=f"NVDA {DIA_EJEMPLO}: señales y operaciones con parámetros base"),
        "03_valor_drawdown.png": plot_valor_drawdown(
            {"train (WF)": (valor_train, buy_and_hold(train.loc[valor_train.index[0]:])),
             "test (WF)": (valor_test, buy_and_hold(test)),
             "validation": (resultado_val["valor"], buy_and_hold(validacion))},
            titulo="Valor del portafolio y drawdown: estrategia contra buy & hold"),
        "04_tabla_retornos.png": plot_tabla_retornos(tablas),
        "05_sensibilidad.png": plot_sensibilidad(tabla_sens, "Sensibilidad del Calmar ante ±20% en θ* (validation)"),
        "06_costos.png": plot_costos({"validation": costos, "train + test": costos_desarrollo}),
        "07_regimenes_precio.png": plot_regimenes_precio(datos["Close"], etiquetas, validacion.index[0],
                                                         titulo="Regímenes detectados por K-means sobre el precio de NVDA"),
        "08_distribuciones_regimen.png": plot_distribuciones_regimen(variables.loc[desarrollo.index], etiquetas),
        "09_valor_regimenes.png": plot_valor_regimenes(resultado_val["valor"], etiquetas, buy_and_hold(validacion),
                                                       titulo="Valor del portafolio en validation con regímenes superpuestos"),
        "10_walk_forward.png": plot_walk_forward(resumen_wf),
        "11_historia_optimizacion.png": plot_historia(
            {r: (final["aleatorio"][r]["estudio"], e) for r, e in activos.items()}),
        "12_importancia_parametros.png": plot_importancia(importancias),
        "15_anchored_rolling.png": plot_anchored_rolling(
            {"Rolling (1 mes)": (wf["valor"], wf["eficiencia"]), "Anchored": (anclado["valor"], anclado["eficiencia"])},
            buy_and_hold(desarrollo.loc[wf["valor"].index[0]:])),
    }
    for r, e in activos.items():
        elegido = {k: v for k, v in (final["parametros"][r] or e.best_params).items() if k in RANGOS}
        figures[f"13_slices_{r}.png"] = plot_slices(e, elegido, r)
        ejes = tuple(importancias[r].index[:2])
        figures[f"14_superficie_{r}.png"] = plot_superficie(superficie(final, r, ejes), r)
        resultados["diagnostico"][r]["ejes_superficie"] = list(ejes)
        resultados["diagnostico"][r]["importancia"] = importancias[r].round(4).to_dict()
    for filename, fig in figures.items():
        fig.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
    (RESULTS_DIR / "resumen.json").write_text(json.dumps(resultados, indent=2, default=str), encoding="utf-8")

    print(f"   Figuras guardadas en: {FIGURES_DIR}")
    print(f"   Tablas guardadas en: {RESULTS_DIR}")
    print("\nProyecto ejecutado correctamente.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laboratorio 02: estrategia de trading con análisis técnico")
    parser.add_argument("--congelar", action="store_true",
                        help="solo calcula θ* con train + test y lo guarda en docs/theta_congelado.json")
    main(parser.parse_args().congelar)
