"""Optimización por régimen con Optuna, walk-forward y análisis de robustez.

- Objetivo: θ* = argmax_θ Calmar(backtest(train, θ)), con el backtest restringido a las velas del
  régimen dentro de la ventana de entrenamiento y con comisión, spread e impacto.
- Restricción: una configuración con menos de `MIN_OPERACIONES` operaciones vale −10 en lugar de su
  Calmar, para que el optimizador no premie tres operaciones ganadoras.
- θ* se toma del centro de la mejor meseta (promedio de los k vecinos más cercanos en el espacio
  normalizado de parámetros), no del argmax literal.
- Si el mejor Calmar de un régimen en la ventana no es positivo, ese régimen queda apagado.
- Walk-forward: entrenamiento de 1 mes (rolling) o desde el inicio (anchored), prueba de 1 semana,
  paso semanal. Purga: señales y variables de régimen solo miran hacia atrás y toda posición se cierra
  el mismo día, así que ninguna observación de entrenamiento usa datos de la prueba. Embargo: se
  descarta la última sesión antes de cada semana de prueba.
"""

import json
import time

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from src.backtest import COMISION, ejecutar_backtest
from src.metrics import calmar, rendimiento_anualizado, resumen_metricas
from src.regimes import REGIMENES, ajustar_modelo, calcular_variables_regimen, etiquetar
from src.signals import PARAMETROS_BASE, generar_senales

N_PRUEBAS = 150
N_DIAGNOSTICO = 200
MIN_OPERACIONES = 10
CALENTAMIENTO = 200
EMBARGO = 78
VECINOS_MESETA = 10
PENALIZACION = -10.0
SEMILLA = 42
PARAMETROS_OPERACION = ("sl_atr", "tp_atr", "fraccion", "enfriamiento")
BASE_OPERACION = {"sl_atr": 1.5, "tp_atr": 2.0, "fraccion": 1.0, "enfriamiento": 0}

# Espacio de búsqueda: (mínimo, máximo, tipo, justificación)
RANGOS = {
    "bb_n": (14, 30, int, "alrededor de las 20 velas de la gráfica: de 70 min a 2.5 h de historia"),
    "bb_k": (1.5, 3.0, float, "de bandas estrechas (más señales) a solo movimientos extremos"),
    "rsi_bajo": (20, 40, int, "umbral de sobreventa alrededor de 30; sobrecompra = 100 − bajo"),
    "sto_bajo": (10, 30, int, "zona del Estocástico alrededor de 20; alta = 100 − baja"),
    "macd_umbral": (0.3, 1.5, float, "MACD en múltiplos de ATR: de desviación leve a extrema"),
    "ema_n": (5, 15, int, "gatillo alrededor de la EMA 9: de reacción rápida a más filtrada"),
    "memoria": (1, 12, int, "de 5 min a 1 h para que preparación y gatillo coincidan"),
    "sl_atr": (1.0, 4.0, float, "stop entre 1 y 4 rangos típicos de vela"),
    "tp_atr": (2.0, 8.0, float, "objetivo que supere el costo de 0.25% ida y vuelta (ATR ≈ 0.22%)"),
    "fraccion": (0.25, 1.0, float, "de un cuarto del capital al total, sin apalancamiento"),
    "enfriamiento": (0, 24, int, "de reentrar de inmediato a esperar 2 h tras un stop-loss"),
}

optuna.logging.set_verbosity(optuna.logging.WARNING)


def sugerir_parametros(trial) -> dict:
    """θ del trial dentro de RANGOS; los umbrales altos son simétricos a los bajos."""
    p = {}
    for nombre, (bajo, alto, tipo, _) in RANGOS.items():
        p[nombre] = trial.suggest_int(nombre, bajo, alto) if tipo is int else trial.suggest_float(nombre, bajo, alto)
    p["rsi_alto"], p["sto_alto"] = 100 - p["rsi_bajo"], 100 - p["sto_bajo"]
    return p


def completar(params: dict) -> dict:
    """Agrega los umbrales simétricos a un diccionario con solo los parámetros de RANGOS."""
    return {**params, "rsi_alto": 100 - params["rsi_bajo"], "sto_alto": 100 - params["sto_bajo"]}


def _separar(p: dict) -> tuple[dict, dict]:
    senal = {k: v for k, v in p.items() if k not in PARAMETROS_OPERACION}
    operacion = {k: p.get(k, BASE_OPERACION[k]) for k in PARAMETROS_OPERACION}
    return senal, operacion


def senales_por_regimen(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict,
                        modo: str = "dos_de_tres") -> pd.DataFrame:
    """Combina las señales de cada régimen con sus propios parámetros.

    `parametros` = {regimen: dict o None}; None significa régimen apagado (sin entradas).
    Agrega por vela las columnas regimen, sl_atr, tp_atr, fraccion y enfriamiento.
    """
    base = generar_senales(datos, None, modo)
    base["senal"] = 0
    base["regimen"] = etiquetas.reindex(datos.index).to_numpy()
    for k, v in BASE_OPERACION.items():
        base[k] = v
    for regimen, p in parametros.items():
        mascara = base["regimen"] == regimen
        if p is None or not mascara.any():
            continue
        p_senal, p_operacion = _separar(p)
        propia = generar_senales(datos, p_senal, modo)
        base.loc[mascara, "senal"] = propia.loc[mascara, "senal"]
        for k, v in p_operacion.items():
            base.loc[mascara, k] = v
    return base


def backtest_por_regimen(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, desde=None,
                         comision: float = COMISION, modo: str = "dos_de_tres") -> dict:
    """Backtest con parámetros por régimen; `desde` recorta el calentamiento de indicadores."""
    senales = senales_por_regimen(datos, etiquetas, parametros, modo)
    if desde is not None:
        senales = senales.loc[desde:]
    return ejecutar_backtest(senales, comision=comision)


def seleccionar_meseta(estudio, vecinos: int = VECINOS_MESETA) -> dict:
    """Centro de la mejor meseta: el trial cuyo vecindario (k vecinos más cercanos, parámetros
    normalizados a [0, 1]) tiene el Calmar promedio más alto."""
    trials = [t for t in estudio.trials if t.value is not None and t.value > PENALIZACION]
    if len(trials) <= vecinos:
        mejor = estudio.best_trial
        return {"params": mejor.params, "valor": mejor.value, "meseta": mejor.value, "es_argmax": True}
    X = np.array([[(t.params[k] - RANGOS[k][0]) / (RANGOS[k][1] - RANGOS[k][0]) for k in RANGOS] for t in trials])
    valores = np.array([t.value for t in trials])
    distancias = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    cercanos = np.argsort(distancias, axis=1)[:, :vecinos]
    promedio = valores[cercanos].mean(axis=1)
    i = int(np.argmax(promedio))
    return {"params": trials[i].params, "valor": float(valores[i]), "meseta": float(promedio[i]),
            "es_argmax": bool(trials[i].number == estudio.best_trial.number)}


def optimizar_regimen(datos: pd.DataFrame, etiquetas: pd.Series, regimen: str, desde,
                      n_pruebas: int = N_PRUEBAS, semilla: int = SEMILLA, sampler: str = "tpe",
                      devolver_estudio: bool = False) -> dict:
    """Optuna maximizando el Calmar del backtest que solo opera en `regimen`."""
    if not (etiquetas.loc[desde:].reindex(datos.loc[desde:].index) == regimen).any():
        return {"parametros": None, "calmar": np.nan, "factibles": 0, "pruebas": 0, "estudio": None}

    def objetivo(trial):
        p = sugerir_parametros(trial)
        r = backtest_por_regimen(datos, etiquetas, {regimen: p}, desde)
        n = len(r["operaciones"])
        trial.set_user_attr("operaciones", n)
        if n < MIN_OPERACIONES:
            return PENALIZACION
        valor = calmar(r["valor"])
        return valor if np.isfinite(valor) else PENALIZACION

    muestreador = (optuna.samplers.RandomSampler(seed=semilla) if sampler == "random"
                   else optuna.samplers.TPESampler(seed=semilla))
    estudio = optuna.create_study(direction="maximize", sampler=muestreador)
    if sampler == "tpe":
        estudio.enqueue_trial({**{k: PARAMETROS_BASE[k] for k in
                                  ("bb_n", "bb_k", "rsi_bajo", "sto_bajo", "macd_umbral", "ema_n", "memoria")},
                               **BASE_OPERACION})
    estudio.optimize(objetivo, n_trials=n_pruebas)
    factibles = sum(1 for t in estudio.trials if t.value is not None and t.value > PENALIZACION)
    meseta = seleccionar_meseta(estudio)
    activo = estudio.best_value > 0 and meseta["valor"] > 0
    return {"parametros": completar(meseta["params"]) if activo else None, "calmar": estudio.best_value,
            "calmar_meseta": meseta["meseta"], "es_argmax": meseta["es_argmax"],
            "factibles": factibles, "pruebas": len(estudio.trials),
            "estudio": estudio if devolver_estudio else None}


def optimizar_ventana(datos: pd.DataFrame, variables: pd.DataFrame, inicio_train, inicio_test, fin_test,
                      n_pruebas: int = N_PRUEBAS, semilla: int = SEMILLA) -> dict:
    """Una ventana del walk-forward: K-means con la historia disponible (sin la sesión de embargo),
    Optuna por régimen, desempeño dentro de muestra y en la semana de prueba."""
    t0 = time.perf_counter()
    corte = datos.index.get_indexer([inicio_test])[0] - EMBARGO
    historia = variables.iloc[:corte]
    modelo = ajustar_modelo(historia)
    etiquetas = etiquetar(modelo, variables)
    pos = datos.index.get_indexer([inicio_train])[0]
    tramo_train = datos.iloc[max(0, pos - CALENTAMIENTO):corte]
    resultados = {r: optimizar_regimen(tramo_train, etiquetas, r, inicio_train, n_pruebas, semilla + i)
                  for i, r in enumerate(REGIMENES)}
    parametros = {r: res["parametros"] for r, res in resultados.items()}
    dentro = backtest_por_regimen(tramo_train, etiquetas, parametros, inicio_train)
    pos = datos.index.get_indexer([inicio_test])[0]
    tramo_test = datos.iloc[pos - CALENTAMIENTO:].loc[:fin_test]
    fuera = backtest_por_regimen(tramo_test, etiquetas, parametros, inicio_test)
    return {
        "inicio_train": inicio_train, "inicio_test": inicio_test, "fin_test": fin_test,
        "parametros": parametros,
        "calmar_por_regimen": {r: res["calmar"] for r, res in resultados.items()},
        "factibles": sum(res["factibles"] for res in resultados.values()),
        "pruebas": sum(res["pruebas"] for res in resultados.values()),
        "dentro": resumen_metricas(dentro["valor"], dentro["operaciones"]),
        "fuera": resumen_metricas(fuera["valor"], fuera["operaciones"]),
        "valor_fuera": fuera["valor"],
        "operaciones_fuera": fuera["operaciones"],
        "segundos": time.perf_counter() - t0,
    }


def ventanas_walk_forward(datos: pd.DataFrame, variables: pd.DataFrame, modo: str = "rolling") -> list:
    """(inicio_train, inicio_test, fin_test) por semana de prueba.

    rolling: entrenamiento del último mes; anchored: desde la primera vela con variables completas.
    """
    primera_valida = variables.dropna().index[0]
    semanas = datos.groupby(datos.index.tz_localize(None).to_period("W-FRI")).apply(lambda s: (s.index[0], s.index[-1]))
    ventanas = []
    for inicio_test, fin_test in semanas:
        inicio_mes = inicio_test - pd.DateOffset(months=1)
        if inicio_mes < primera_valida:
            continue
        inicio_train = primera_valida if modo == "anchored" else datos.index[datos.index >= inicio_mes][0]
        ventanas.append((inicio_train, inicio_test, fin_test))
    return ventanas


def walk_forward(datos: pd.DataFrame, modo: str = "rolling", n_pruebas: int = N_PRUEBAS, n_jobs: int = -1) -> dict:
    """Corre todas las ventanas en paralelo, encadena las semanas fuera de muestra y calcula
    la walk-forward efficiency = rendimiento anualizado fuera / promedio dentro de muestra."""
    variables = calcular_variables_regimen(datos)
    ventanas = ventanas_walk_forward(datos, variables, modo)
    t0 = time.perf_counter()
    resultados = Parallel(n_jobs=n_jobs)(
        delayed(optimizar_ventana)(datos, variables, a, b, c, n_pruebas, SEMILLA + 10 * i)
        for i, (a, b, c) in enumerate(ventanas))
    segundos = time.perf_counter() - t0
    rendimientos = pd.concat([r["valor_fuera"].pct_change().fillna(r["valor_fuera"].iloc[0] / 1e6 - 1)
                              for r in resultados])
    valor = (1 + rendimientos).cumprod() * 1e6
    operaciones = pd.concat([r["operaciones_fuera"] for r in resultados], ignore_index=True)
    resumen = pd.DataFrame([{
        "inicio_test": r["inicio_test"].date(), "calmar_dentro": r["dentro"]["calmar"],
        "calmar_fuera": r["fuera"]["calmar"], "retorno_dentro": r["dentro"]["retorno_total"],
        "retorno_fuera": r["fuera"]["retorno_total"],
        "anualizado_dentro": r["dentro"]["rendimiento_anualizado"],
        "operaciones_fuera": r["fuera"]["operaciones"],
        **{f"activo_{k}": v is not None for k, v in r["parametros"].items()},
    } for r in resultados])
    dentro = resumen["anualizado_dentro"].mean()
    fuera = rendimiento_anualizado(valor)
    return {"modo": modo, "ventanas": resultados, "resumen": resumen, "valor": valor, "operaciones": operaciones,
            "segundos": segundos, "configuraciones": sum(r["pruebas"] for r in resultados),
            "anualizado_dentro": dentro, "anualizado_fuera": fuera, "eficiencia": fuera / dentro}


def modelo_final(train: pd.DataFrame, n_pruebas: int = N_DIAGNOSTICO) -> dict:
    """K-means con todo train; por régimen, random search y TPE en su último mes. θ* sale de la
    meseta del TPE; el random search queda como referencia y para fijar la superficie 3D."""
    variables = calcular_variables_regimen(train)
    modelo = ajustar_modelo(variables)
    inicio = train.index[train.index >= train.index[-1] - pd.DateOffset(months=1)][0]
    etiquetas = etiquetar(modelo, variables)
    tramo = train.iloc[train.index.get_indexer([inicio])[0] - CALENTAMIENTO:]
    tpe, aleatorio = {}, {}
    for i, r in enumerate(REGIMENES):
        aleatorio[r] = optimizar_regimen(tramo, etiquetas, r, inicio, n_pruebas, SEMILLA + 200 + i, "random", True)
        tpe[r] = optimizar_regimen(tramo, etiquetas, r, inicio, n_pruebas, SEMILLA + 100 + i, "tpe", True)
    return {"modelo": modelo, "etiquetas": etiquetas, "tramo": tramo, "inicio": inicio,
            "parametros": {r: res["parametros"] for r, res in tpe.items()},
            "tpe": tpe, "aleatorio": aleatorio,
            "calmar_por_regimen": {r: res["calmar"] for r, res in tpe.items()},
            "pruebas": sum(res["pruebas"] for res in list(tpe.values()) + list(aleatorio.values()))}


def encadenar(ventanas: list) -> tuple[pd.Series, pd.DataFrame]:
    """Encadena la curva de valor fuera de muestra de un subconjunto de ventanas desde $1,000,000."""
    rendimientos = pd.concat([r["valor_fuera"].pct_change().fillna(r["valor_fuera"].iloc[0] / 1e6 - 1)
                              for r in ventanas])
    operaciones = pd.concat([r["operaciones_fuera"] for r in ventanas], ignore_index=True)
    return (1 + rendimientos).cumprod() * 1e6, operaciones


def guardar_theta(ruta, final: dict, datos_hasta) -> dict:
    """Congela θ* (parámetros por régimen) y la huella del modelo de régimen en un JSON."""
    modelo = final["modelo"]
    congelado = {
        "entrenado_hasta": str(datos_hasta),
        "mes_de_optimizacion_desde": str(final["inicio"]),
        "parametros": final["parametros"],
        "centros_kmeans": np.round(modelo.kmeans.cluster_centers_, 6).tolist(),
        "nombres_grupos": {str(k): v for k, v in modelo.nombres.items()},
    }
    ruta.write_text(json.dumps(congelado, indent=2, ensure_ascii=False), encoding="utf-8")
    return congelado


def cargar_theta(ruta) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def importancia(estudio) -> pd.Series:
    """Importancia de cada parámetro (fANOVA de Optuna) sobre los trials factibles."""
    return pd.Series(optuna.importance.get_param_importances(estudio)).sort_values(ascending=False)


def superficie(final: dict, regimen: str, ejes: tuple, puntos: int = 12) -> dict:
    """Calmar en una malla de los dos parámetros `ejes`, con el resto de θ fijo en el mejor
    punto del random search del régimen."""
    base = final["aleatorio"][regimen]["estudio"].best_params
    xs = np.linspace(*RANGOS[ejes[0]][:2], puntos)
    ys = np.linspace(*RANGOS[ejes[1]][:2], puntos)
    Z = np.full((puntos, puntos), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            p = dict(base)
            for eje, v in ((ejes[0], x), (ejes[1], y)):
                p[eje] = int(round(v)) if RANGOS[eje][2] is int else float(v)
            r = backtest_por_regimen(final["tramo"], final["etiquetas"], {regimen: completar(p)}, final["inicio"])
            Z[i, j] = calmar(r["valor"]) if len(r["operaciones"]) >= MIN_OPERACIONES else np.nan
    return {"x": xs, "y": ys, "z": Z, "ejes": ejes, "base": base}


def sensibilidad(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, desde,
                 variacion: float = 0.20) -> pd.DataFrame:
    """Varía cada parámetro ±20% en todos los regímenes activos y mide el Calmar."""
    base = backtest_por_regimen(datos, etiquetas, parametros, desde)
    filas = []
    for clave in RANGOS:
        fila = {"parametro": clave, "base": calmar(base["valor"])}
        for etiqueta, factor in (("-20%", 1 - variacion), ("+20%", 1 + variacion)):
            variado = {}
            for r, p in parametros.items():
                if p is None:
                    variado[r] = None
                    continue
                q = dict(p)
                nuevo = q[clave] * factor
                q[clave] = int(round(nuevo)) if RANGOS[clave][2] is int else nuevo
                q["fraccion"] = min(q["fraccion"], 1.0)
                variado[r] = completar(q)
            fila[etiqueta] = calmar(backtest_por_regimen(datos, etiquetas, variado, desde)["valor"])
        filas.append(fila)
    return pd.DataFrame(filas).set_index("parametro")[["-20%", "base", "+20%"]]


def barrido_costos(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, desde,
                   comisiones=np.arange(0, 0.00501, 0.00025)) -> pd.DataFrame:
    """Retorno neto y Calmar contra el nivel de comisión por lado (spread e impacto fijos)."""
    filas = []
    for c in comisiones:
        r = backtest_por_regimen(datos, etiquetas, parametros, desde, comision=c)
        filas.append({"comision": c, "retorno_neto": r["valor"].iloc[-1] / r["valor"].iloc[0] - 1,
                      "calmar": calmar(r["valor"]), "operaciones": len(r["operaciones"])})
    return pd.DataFrame(filas)
