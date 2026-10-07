"""Ejecuta el laboratorio completo: python main.py

Split cronológico: train (ene–may) → test (jun–jul) → validation (ago–sep).
θ* se optimiza por régimen con todo train y se evalúa fuera de muestra en test y en validation; el walk-forward
de 1 mes sobre train + test mide la degradación. θ* se congela en docs/theta_congelado.json
(`python main.py --congelar`) y se versiona antes de evaluar validation.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import THETA_OPERACION, backtest
from src.data import RUTA_CRUDOS, auditar_datos, cargar_datos, congelar_datos, separar_train_test_validacion
from src.metrics import (descomposicion_dia_noche, metricas_por_regimen, prueba_diferencia_regimenes, resumen_metricas, tablas_retornos,
                         turnover, win_rate, winrate_equilibrio)
from src.optimize import (CALENTAMIENTO, EMBARGO, MIN_OPERACIONES, N_PRUEBAS, RANGOS, RANGOS_VENTANAS, backtest_por_regimen,
                          barrido_costos, cargar_theta, encadenar, guardar_theta, importancia, modelo_final,
                          sensibilidad, superficie, walk_forward)
from src.plots import (plot_sensibilidad_costos, plot_linea_tiempo, plot_distribuciones_regimen, plot_historia,
                       plot_importancia, plot_indicadores, plot_operaciones, plot_regimenes_precio,
                       plot_sensibilidad, plot_slices, plot_superficie, plot_tabla_retornos, plot_valor_drawdown,
                       plot_valor_regimenes, plot_walk_forward)
from src.regimes import HMM, KMedias, Reglas, calcular_variables_regimen, comparar, etiquetar, validar_regimenes
from src.signals import MODOS, contar_aperturas, generar_senales

SEED = 42
FIGURES_DIR = Path("docs/figures")
RESULTS_DIR = Path("resultados")
THETA = Path("docs/theta_congelado.json")
EJEMPLO = ("2026-03-02", "2026-03-13")  # dos semanas de train para las figuras de indicadores y operaciones


def buy_and_hold(datos, capital=1_000_000):
    return datos["Close"] / datos["Close"].iloc[0] * capital


def imprimir(tabla):
    print("   " + tabla.to_string().replace("\n", "\n   "))


def iguales(a, b):
    """Compara dos θ por régimen con tolerancia numérica."""
    if (a is None) != (b is None):
        return False
    return a is None or all(np.isclose(a[k], b[k]) for k in {**RANGOS_VENTANAS, **RANGOS})


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
    resultados["dia_noche_nvda"] = {n: descomposicion_dia_noche(d) for n, d in
                                    (("train", train), ("test", test), ("validation", validacion))}
    print(f"   {auditoria['velas']} velas en {auditoria['dias']} días; duplicados {auditoria['duplicados']}, "
          f"nulos {auditoria['nulos']}, OHLC incoherente {auditoria['ohlc_incoherente']}")
    for nombre, d in (("Train", train), ("Test", test), ("Validation", validacion)):
        print(f"   {nombre}: {d.index[0]:%Y-%m-%d} a {d.index[-1]:%Y-%m-%d} "
              f"({len(d)} velas, {len(d) / len(datos):.0%})")
    print("   Suma de log-retornos de NVDA: " + " · ".join(
        f"{n} día {v['dia']:.2%} / noche {v['noche']:.2%}" for n, v in resultados["dia_noche_nvda"].items()))

    print("\n2. Modelo final sobre train: K-means y, por régimen, random search y TPE de enero a mayo...")
    final = modelo_final(train)
    e1 = final["etapa1"]
    print(f"   Etapa 1, ventanas de los indicadores (un θ para todo train): TPE N = {e1['pruebas']}, "
          f"mejor Calmar {e1['calmar']:.2f}, meseta {e1['calmar_meseta']:.2f}, {e1['segundos']:.0f} s → {final['ventanas']}")
    resultados["etapa1"] = {k: e1[k] for k in ("ventanas", "calmar", "calmar_meseta", "es_argmax", "pruebas", "segundos")}
    diagnostico = {}
    for regimen in final["tpe"]:
        a, t = final["aleatorio"][regimen], final["tpe"][regimen]
        if t["estudio"] is None:
            print(f"   {regimen}: sin velas en train")
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
        guardar_theta(THETA, final, train.index[-1])
        print(f"\n   θ* congelado en {THETA}. Haz commit de este archivo y vuelve a correr python main.py.")
        return
    congelado = cargar_theta(THETA)
    theta = congelado["parametros"]
    coincide = all(iguales(theta[r], final["parametros"][r]) for r in theta)
    print(f"\n   θ* leído de {THETA} (congelado antes de tocar validation); "
          f"{'coincide' if coincide else 'NO coincide'} con el recalculado.")
    resultados["theta_congelado"] = congelado

    print("\n3. Validando regímenes (modelo ajustado con train)...")
    variables = calcular_variables_regimen(datos)
    etiquetas = etiquetar(final["modelo"], variables)
    validacion_reg = {n: validar_regimenes(final["modelo"], variables.loc[d.index], etiquetas.loc[d.index])
                      for n, d in (("train", train), ("test", test), ("validation", validacion))}
    imprimir(pd.DataFrame(validacion_reg))
    resultados["regimenes"] = validacion_reg
    print("   Comparación de clasificadores (ajustados con train): reglas, K-means y HMM filtrado")
    f_train = variables.loc[train.index]
    clasificadores = {"reglas": Reglas().ajustar(f_train), "kmeans": KMedias().ajustar(f_train), "hmm": HMM().ajustar(f_train)}
    etiq_clas = {n: m.etiquetar(variables) for n, m in clasificadores.items()}
    etiq_clas["hmm_viterbi"] = clasificadores["hmm"].etiquetar_viterbi(variables.loc[desarrollo.index])
    tabla_clas = pd.DataFrame({(n, c): comparar(variables.loc[d.index], e.reindex(d.index).fillna("sin_datos"))
                               for n, e in etiq_clas.items() for c, d in (("train", train), ("test", test))}).T
    imprimir(tabla_clas.round(3))
    tabla_clas.to_csv(RESULTS_DIR / "clasificadores.csv")
    coincidencia = float((etiq_clas["hmm"].loc[train.index] == etiq_clas["hmm_viterbi"].loc[train.index]).mean())
    print(f"   Coincidencia HMM filtrado vs Viterbi en train: {coincidencia:.1%} · elegido: K-means")
    resultados["clasificadores"] = {"/".join(k): v for k, v in tabla_clas.to_dict(orient="index").items()}
    resultados["coincidencia_filtrada_viterbi"] = coincidencia

    print("\n4. Señales con parámetros base (Keltner + RSI + TRIX, 2 de 3, a favor de la tendencia diaria) en train...")
    senales_base = generar_senales(train)

    conteo = pd.DataFrame({m: contar_aperturas(generar_senales(train, modo=m)["objetivo"]) for m in MODOS})
    imprimir(conteo)

    print(f"\n5. Walk-forward sobre train + test (prueba de 1 semana, {N_PRUEBAS} pruebas TPE por régimen, "
          f"mínimo {MIN_OPERACIONES} operaciones, embargo de {EMBARGO} velas)...")
    wf = walk_forward(desarrollo, ventanas=congelado.get("ventanas", final["ventanas"]))
    print(f"   {len(wf['resumen'])} ventanas, {wf['configuraciones']} configuraciones en {wf['segundos']:.0f} s · "
          f"rendimiento anualizado dentro {wf['anualizado_dentro']:.2%}, fuera {wf['anualizado_fuera']:.2%} · "
          f"walk-forward efficiency {wf['eficiencia']:.2f}")
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
        **{k: wf[k] for k in ("eficiencia", "anualizado_dentro", "anualizado_fuera")},
        "retorno_fuera": float(wf["valor"].iloc[-1] / 1e6 - 1),
    }

    print("\n6. θ* congelado en train (dentro de muestra), test y validation (fuera de muestra)...")
    tramo_val = datos.iloc[len(desarrollo) - CALENTAMIENTO:]

    def con_theta(desde, hasta):
        pos = datos.index.get_indexer([desde])[0]
        return backtest_por_regimen(datos.iloc[max(0, pos - CALENTAMIENTO):].loc[:hasta], etiquetas, theta, desde)

    resultado_train = con_theta(final["inicio"], train.index[-1])
    resultado_test = con_theta(test.index[0], test.index[-1])
    resultado_val = con_theta(validacion.index[0], validacion.index[-1])
    curvas = {"train": resultado_train, "test": resultado_test, "validation": resultado_val}
    metricas = {
        "train": resumen_metricas(resultado_train["equity"], resultado_train["operaciones"]),
        "B&H train": resumen_metricas(buy_and_hold(train.loc[final["inicio"]:]), pd.DataFrame()),
        "test": resumen_metricas(resultado_test["equity"], resultado_test["operaciones"]),
        "B&H test": resumen_metricas(buy_and_hold(test), pd.DataFrame()),
        "validation": resumen_metricas(resultado_val["equity"], resultado_val["operaciones"]),
        "B&H validation": resumen_metricas(buy_and_hold(validacion), pd.DataFrame()),
        "train (WF)": resumen_metricas(valor_train, ops_train),
        "test (WF)": resumen_metricas(valor_test, ops_test),
    }
    for nombre, (eq, ops) in {**{n: (r["equity"], r["operaciones"]) for n, r in curvas.items()},
                              "train (WF)": (valor_train, ops_train), "test (WF)": (valor_test, ops_test)}.items():
        metricas[nombre].update({**turnover(ops, eq), "win_rate_empirico": win_rate(ops)})
    tabla_metricas = pd.DataFrame(metricas)
    imprimir(tabla_metricas.round(4))
    tabla_metricas.to_csv(RESULTS_DIR / "metricas.csv")
    print(f"   Deslizamiento (spread + impacto) en validation: ${resultado_val['deslizamiento']:,.0f}; "
          f"comisiones: ${resultado_val['comisiones']:,.0f}")
    atr_rel = float((senales_base["atr_14"] / senales_base["Close"]).median())
    desliz = resultado_val["deslizamiento"] / max(1.0, (resultado_val["operaciones"]["acciones"] * (
        resultado_val["operaciones"]["precio_entrada"] + resultado_val["operaciones"]["precio_salida"])).sum())
    costo_rt = 2 * (0.00125 + desliz)
    equilibrios = {"SPEC": winrate_equilibrio(THETA_OPERACION["sl_atr"], THETA_OPERACION["tp_atr"], atr_rel, costo_rt),
                   **{r: winrate_equilibrio(p["sl_atr"], p["tp_atr"], atr_rel, costo_rt) for r, p in theta.items() if p}}
    print(f"   Win rate de equilibrio p* (ATR/precio {atr_rel:.4%}, costo ida y vuelta {costo_rt:.4%}): "
          f"{ {k: round(v, 3) for k, v in equilibrios.items()} }")
    resultados["win_rate_equilibrio"] = equilibrios
    resultados["costo_ida_vuelta"] = costo_rt
    tablas = {n: tablas_retornos(r["equity"]) for n, r in curvas.items()}
    por_regimen = metricas_por_regimen(resultado_val["equity"], resultado_val["operaciones"], etiquetas)
    print("   Métricas por régimen en validation:")
    imprimir(por_regimen.astype(float).round(4))
    diferencia = prueba_diferencia_regimenes(pd.concat([r["operaciones"] for r in curvas.values()]))
    print(f"   Kruskal-Wallis del retorno por operación entre regímenes (train + test + validation): "
          f"H = {diferencia['estadistico']:.2f}, p = {diferencia['p_valor']:.3f}")
    por_regimen.to_csv(RESULTS_DIR / "metricas_por_regimen.csv")
    print("   Aporte de la parte de día y de la compra nocturna:")
    filas_tipo = []
    for nombre, ops in [(n, r["operaciones"]) for n, r in curvas.items()] + [("train (WF)", ops_train), ("test (WF)", ops_test)]:
        for tipo in ("dia", "noche"):
            o = ops[ops["tipo"] == tipo] if len(ops) else ops
            filas_tipo.append({"conjunto": nombre, "parte": tipo, "operaciones": len(o),
                               "pnl_usd": float(o["pnl"].sum()) if len(o) else 0.0,
                               "pnl_pct_capital": float(o["pnl"].sum()) / 1e6 if len(o) else 0.0,
                               "win_rate": win_rate(o) if len(o) else np.nan})
    por_tipo = pd.DataFrame(filas_tipo).set_index(["conjunto", "parte"])
    imprimir(por_tipo.round(4))
    por_tipo.to_csv(RESULTS_DIR / "dia_noche.csv")
    resultados["dia_noche"] = {"/".join(k): v for k, v in por_tipo.to_dict(orient="index").items()}
    resultados["metricas"] = tabla_metricas.to_dict()
    resultados["kruskal"] = diferencia
    resultados["deslizamiento_validation"] = resultado_val["deslizamiento"]
    resultados["comisiones_validation"] = resultado_val["comisiones"]

    print("\n7. Robustez de θ* congelado en validation: reglas, sensibilidad ±20% y costos (sin reoptimizar)...")
    reglas = {}
    variantes = [(m, "dos_de_tres") for m in MODOS] + [("dos_de_tres", "simple"), ("dos_de_tres", "ninguna")]
    for modo, modo_noche in variantes:
        r = backtest_por_regimen(tramo_val, etiquetas, theta, validacion.index[0], modo=modo, modo_noche=modo_noche)
        modo = modo if modo_noche == "dos_de_tres" else f"noche_{modo_noche}"
        m = resumen_metricas(r["equity"], r["operaciones"])
        reglas[modo] = {"operaciones": m["operaciones"], "calmar": m["calmar"], "retorno_total": m["retorno_total"],
                        "win_rate": m["win_rate"]}
    tabla_reglas = pd.DataFrame(reglas).T
    imprimir(tabla_reglas.astype(float).round(4))
    tabla_sens = sensibilidad(tramo_val, etiquetas, theta, validacion.index[0])
    imprimir(tabla_sens.round(3))
    costos = barrido_costos(tramo_val, etiquetas, theta, validacion.index[0])
    costos_desarrollo = barrido_costos(datos.iloc[: len(desarrollo)], etiquetas, theta, desarrollo.index[CALENTAMIENTO])
    positivos = costos[costos["retorno_neto"] > 0]
    equilibrio = int(positivos["ida_vuelta_bps"].max()) if len(positivos) else None
    print(f"   Retorno neto en validation con costo cero: {costos['retorno_neto'].iloc[0]:.2%}; "
          f"costo máximo de ida y vuelta con retorno positivo: {f'{equilibrio} pb' if equilibrio is not None else 'ninguno'}")
    tabla_reglas.to_csv(RESULTS_DIR / "reglas.csv")
    tabla_sens.to_csv(RESULTS_DIR / "sensibilidad.csv")
    costos.to_csv(RESULTS_DIR / "costos_validation.csv", index=False)
    resultados["reglas"] = reglas
    resultados["equilibrio_costos"] = equilibrio

    print("\n8. Generando figuras...")
    activos = {r: final["tpe"][r]["estudio"] for r in diagnostico}
    importancias = {r: importancia(e) for r, e in activos.items()}
    figures = {
        "01_indicadores.png": plot_indicadores(senales_base.loc[EJEMPLO[0]:EJEMPLO[1]],
                                               titulo=f"NVDA {EJEMPLO[0]} a {EJEMPLO[1]}: indicadores de la estrategia"),
        "02_operaciones.png": plot_operaciones(
            senales_base.loc[EJEMPLO[0]:EJEMPLO[1]],
            backtest(senales_base, senales_base["objetivo"])["operaciones"],
            titulo=f"NVDA {EJEMPLO[0]} a {EJEMPLO[1]}: posición objetivo y operaciones con parámetros base"),
        "03_valor_drawdown.png": plot_valor_drawdown(
            {"train": (resultado_train["equity"], buy_and_hold(train.loc[final["inicio"]:])),
             "test": (resultado_test["equity"], buy_and_hold(test)),
             "validation": (resultado_val["equity"], buy_and_hold(validacion))},
            titulo="Valor del portafolio y drawdown: estrategia contra buy & hold"),
        "04_tabla_retornos.png": plot_tabla_retornos(tablas),
        "05_sensibilidad.png": plot_sensibilidad(tabla_sens, "Sensibilidad del Calmar ante ±20% en θ* (validation)"),
        "06_costos.png": plot_sensibilidad_costos({"validation": costos, "train + test": costos_desarrollo},
                                                  marca_bps=round(costo_rt * 1e4, 1)),
        "07_regimenes_precio.png": plot_regimenes_precio(datos["Close"], etiquetas, validacion.index[0],
                                                         titulo="Regímenes detectados por K-means sobre el precio de NVDA"),
        "08_distribuciones_regimen.png": plot_distribuciones_regimen(variables.loc[desarrollo.index], etiquetas),
        "09_valor_regimenes.png": plot_valor_regimenes(resultado_val["equity"], etiquetas, buy_and_hold(validacion),
                                                       titulo="Valor del portafolio en validation con regímenes superpuestos"),
        "10_walk_forward.png": plot_walk_forward(resumen_wf),
        "11_historia_optimizacion.png": plot_historia(
            {r: (final["aleatorio"][r]["estudio"], e) for r, e in activos.items()}),
        "12_importancia_parametros.png": plot_importancia(importancias),
        "15_hmm_filtrada_viterbi.png": plot_linea_tiempo(desarrollo["Close"], etiq_clas["hmm"].loc[desarrollo.index],
                                                         etiq_clas["hmm_viterbi"]),
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
