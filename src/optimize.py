"""Optimización por régimen con Optuna, walk-forward y análisis de robustez.

- Función objetivo: Calmar Ratio del backtest restringido a las velas del régimen, dentro de la
  ventana de entrenamiento. Una configuración con menos de `MIN_OPERACIONES` operaciones se
  descarta (Calmar = −10) para que el optimizador no premie tres operaciones ganadoras.
- Si el mejor Calmar de un régimen en la ventana no es positivo, ese régimen queda apagado
  (no se opera en la semana de prueba).
- Walk-forward: entrenamiento de 1 mes, prueba de 1 semana, paso semanal. K-means se reajusta
  en cada ventana con toda la historia disponible hasta el fin del entrenamiento.
"""

import time

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from src.backtest import COMISION, ejecutar_backtest
from src.metrics import calmar, resumen_metricas
from src.regimes import REGIMENES, ajustar_modelo, calcular_variables_regimen, etiquetar
from src.signals import PARAMETROS_BASE, generar_senales

N_PRUEBAS = 150
MIN_OPERACIONES = 5
CALENTAMIENTO = 200
PENALIZACION = -10.0
SEMILLA = 42
PARAMETROS_OPERACION = ("sl_atr", "tp_atr", "fraccion", "enfriamiento")
BASE_OPERACION = {"sl_atr": 1.5, "tp_atr": 2.0, "fraccion": 1.0, "enfriamiento": 0}

optuna.logging.set_verbosity(optuna.logging.WARNING)


def sugerir_parametros(trial) -> dict:
    """Espacio de búsqueda alrededor de la configuración de la gráfica del equipo."""
    rsi_bajo = trial.suggest_int("rsi_bajo", 20, 40)
    sto_bajo = trial.suggest_int("sto_bajo", 10, 30)
    return {
        "bb_n": trial.suggest_int("bb_n", 14, 30),
        "bb_k": trial.suggest_float("bb_k", 1.5, 3.0),
        "rsi_bajo": rsi_bajo, "rsi_alto": 100 - rsi_bajo,
        "sto_bajo": sto_bajo, "sto_alto": 100 - sto_bajo,
        "macd_umbral": trial.suggest_float("macd_umbral", 0.3, 1.5),
        "ema_n": trial.suggest_int("ema_n", 5, 15),
        "memoria": trial.suggest_int("memoria", 1, 12),
        "sl_atr": trial.suggest_float("sl_atr", 1.0, 4.0),
        "tp_atr": trial.suggest_float("tp_atr", 2.0, 8.0),
        "fraccion": trial.suggest_float("fraccion", 0.25, 1.0),
        "enfriamiento": trial.suggest_int("enfriamiento", 0, 24),
    }


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


def optimizar_regimen(datos: pd.DataFrame, etiquetas: pd.Series, regimen: str, desde,
                      n_pruebas: int = N_PRUEBAS, semilla: int = SEMILLA) -> dict:
    """Optuna (TPE) maximizando el Calmar del backtest que solo opera en `regimen`."""
    if not (etiquetas.loc[desde:] == regimen).any():
        return {"parametros": None, "calmar": np.nan, "factibles": 0, "pruebas": 0}

    def objetivo(trial):
        p = sugerir_parametros(trial)
        r = backtest_por_regimen(datos, etiquetas, {regimen: p}, desde)
        n = len(r["operaciones"])
        trial.set_user_attr("operaciones", n)
        if n < MIN_OPERACIONES:
            return PENALIZACION
        valor = calmar(r["valor"])
        return valor if np.isfinite(valor) else PENALIZACION

    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semilla))
    estudio.enqueue_trial({**{k: PARAMETROS_BASE[k] for k in
                              ("bb_n", "bb_k", "rsi_bajo", "sto_bajo", "macd_umbral", "ema_n", "memoria")},
                           **BASE_OPERACION, "fraccion": 1.0})
    estudio.optimize(objetivo, n_trials=n_pruebas)
    factibles = [t for t in estudio.trials if t.value is not None and t.value > PENALIZACION]
    mejor = estudio.best_trial
    activo = mejor.value > 0
    parametros = sugerir_parametros(optuna.trial.FixedTrial(mejor.params)) if activo else None
    return {"parametros": parametros, "calmar": mejor.value, "factibles": len(factibles),
            "pruebas": len(estudio.trials)}


def optimizar_ventana(datos: pd.DataFrame, variables: pd.DataFrame, inicio_train, inicio_test, fin_test,
                      n_pruebas: int = N_PRUEBAS, semilla: int = SEMILLA) -> dict:
    """Una ventana del walk-forward: K-means con la historia disponible, Optuna por régimen,
    desempeño dentro de muestra (entrenamiento) y fuera de muestra (semana de prueba)."""
    t0 = time.perf_counter()
    historia = variables.loc[:inicio_test].iloc[:-1]
    modelo = ajustar_modelo(historia)
    etiquetas = etiquetar(modelo, variables)
    pos = datos.index.get_indexer([inicio_train])[0]
    tramo_train = datos.iloc[max(0, pos - CALENTAMIENTO):].loc[:inicio_test].iloc[:-1]
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


def ventanas_walk_forward(datos: pd.DataFrame, variables: pd.DataFrame) -> list:
    """(inicio_train, inicio_test, fin_test): 1 mes de entrenamiento, 1 semana de prueba, paso semanal."""
    primera_valida = variables.dropna().index[0]
    semanas = datos.groupby(datos.index.tz_localize(None).to_period("W-FRI")).apply(lambda s: (s.index[0], s.index[-1]))
    ventanas = []
    for inicio_test, fin_test in semanas:
        inicio_train = inicio_test - pd.DateOffset(months=1)
        if inicio_train < primera_valida:
            continue
        inicio_train = datos.index[datos.index >= inicio_train][0]
        ventanas.append((inicio_train, inicio_test, fin_test))
    return ventanas


def walk_forward(datos: pd.DataFrame, n_pruebas: int = N_PRUEBAS, n_jobs: int = -1) -> dict:
    """Corre todas las ventanas en paralelo y encadena las semanas fuera de muestra."""
    variables = calcular_variables_regimen(datos)
    ventanas = ventanas_walk_forward(datos, variables)
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
        "retorno_fuera": r["fuera"]["retorno_total"], "operaciones_fuera": r["fuera"]["operaciones"],
        **{f"activo_{k}": v is not None for k, v in r["parametros"].items()},
    } for r in resultados])
    return {"ventanas": resultados, "resumen": resumen, "valor": valor, "operaciones": operaciones,
            "segundos": segundos, "configuraciones": sum(r["pruebas"] for r in resultados)}


def modelo_final(train: pd.DataFrame, n_pruebas: int = N_PRUEBAS) -> dict:
    """Parámetros finales: K-means con todo el train y Optuna por régimen en su último mes."""
    variables = calcular_variables_regimen(train)
    modelo = ajustar_modelo(variables)
    inicio = train.index[train.index >= train.index[-1] - pd.DateOffset(months=1)][0]
    etiquetas = etiquetar(modelo, variables)
    tramo = train.iloc[train.index.get_indexer([inicio])[0] - CALENTAMIENTO:]
    resultados = {r: optimizar_regimen(tramo, etiquetas, r, inicio, n_pruebas, SEMILLA + 100 + i)
                  for i, r in enumerate(REGIMENES)}
    return {"modelo": modelo, "parametros": {r: res["parametros"] for r, res in resultados.items()},
            "calmar_por_regimen": {r: res["calmar"] for r, res in resultados.items()},
            "pruebas": sum(res["pruebas"] for res in resultados.values())}


def sensibilidad(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, desde,
                 variacion: float = 0.20) -> pd.DataFrame:
    """Varía cada parámetro ±20% en todos los regímenes activos y mide el Calmar."""
    base = backtest_por_regimen(datos, etiquetas, parametros, desde)
    filas = []
    claves = next(p for p in parametros.values() if p is not None).keys()
    for clave in claves:
        if clave in ("rsi_alto", "sto_alto"):
            continue
        fila = {"parametro": clave, "base": calmar(base["valor"])}
        for etiqueta, factor in (("-20%", 1 - variacion), ("+20%", 1 + variacion)):
            variado = {}
            for r, p in parametros.items():
                if p is None:
                    variado[r] = None
                    continue
                q = dict(p)
                nuevo = q[clave] * factor
                q[clave] = int(round(nuevo)) if isinstance(q[clave], (int, np.integer)) else nuevo
                q["fraccion"] = min(q["fraccion"], 1.0)
                q["rsi_alto"], q["sto_alto"] = 100 - q["rsi_bajo"], 100 - q["sto_bajo"]
                variado[r] = q
            fila[etiqueta] = calmar(backtest_por_regimen(datos, etiquetas, variado, desde)["valor"])
        filas.append(fila)
    return pd.DataFrame(filas).set_index("parametro")[["-20%", "base", "+20%"]]


def barrido_costos(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, desde,
                   comisiones=np.arange(0, 0.00501, 0.00025)) -> pd.DataFrame:
    """Retorno neto y Calmar contra el nivel de comisión por lado."""
    filas = []
    for c in comisiones:
        r = backtest_por_regimen(datos, etiquetas, parametros, desde, comision=c)
        filas.append({"comision": c, "retorno_neto": r["valor"].iloc[-1] / r["valor"].iloc[0] - 1,
                      "calmar": calmar(r["valor"]), "operaciones": len(r["operaciones"])})
    return pd.DataFrame(filas)
