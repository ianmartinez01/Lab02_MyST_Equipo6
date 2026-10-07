"""Optimización por régimen con Optuna, walk-forward y análisis de robustez.

- Objetivo: θ* = argmax_θ Calmar(backtest(train, θ)), con el backtest restringido a las velas del
  régimen dentro de la ventana de entrenamiento y con comisión, spread e impacto. θ* final se optimiza con
  todo train y se evalúa fuera de muestra en test y en validation; el walk-forward de 1 mes mide la degradación.
- Restricción: una configuración con menos de `MIN_OPERACIONES` operaciones abiertas en el régimen dentro de
  la ventana vale −10 en lugar de su Calmar. La estrategia opera unas 10 veces al mes en total, así que el
  mínimo por régimen y por ventana de 1 mes es 4 (con 10 por régimen ninguna ventana sería factible).
- θ* se toma del centro de la mejor meseta (promedio de los k vecinos más cercanos en el espacio
  normalizado de parámetros), no del argmax literal.
- Si el mejor Calmar de un régimen en la ventana no es positivo, ese régimen queda apagado.
- Walk-forward: entrenamiento del último mes, prueba de 1 semana, paso semanal. Purga: señales y variables de régimen solo miran hacia atrás, el backtest de entrenamiento
  termina antes del embargo y la posición que siga abierta se liquida al cierre de su última vela, así que
  ninguna observación de entrenamiento usa datos de la prueba. Embargo: se descarta la última sesión antes
  de cada semana de prueba. Cada semana de prueba arranca sin posición y liquida la suya al final.
"""

import json
import time

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from src.backtest import THETA_OPERACION, Costos, backtest
from src.metrics import calmar, rendimiento_anualizado, resumen_metricas, sharpe
from src.regimes import REGIMENES, ajustar_modelo, calcular_variables_regimen, etiquetar
from src.signals import PARAMETROS_BASE, generar_senales, posicion_objetivo

N_PRUEBAS = 150
N_DIAGNOSTICO = 200
MIN_OPERACIONES = 4
CALENTAMIENTO = 32 * 78  # 32 sesiones: cubre la media de hasta 30 días de la tendencia diaria
EMBARGO = 78
VECINOS_MESETA = 10
PENALIZACION = -10.0
SEMILLA = 42
COMISION = Costos().comision
PARAMETROS_OPERACION = ("sl_atr", "tp_atr", "riesgo", "max_velas", "tamano_noche")
BASE_OPERACION = dict(THETA_OPERACION)

# Espacio de búsqueda: (mínimo, máximo, tipo, justificación)
# Etapa 1: ventanas de los indicadores, un solo θ para todo train (no por régimen), con N_PRUEBAS_VENTANAS
# pruebas y al menos MIN_OPERACIONES_VENTANAS operaciones. Optimizarlas por régimen en ventanas de 1 mes, con 3 a 4
# operaciones por régimen, sobreajustaba (walk-forward: +120% anualizado dentro de muestra contra −18% fuera).
RANGOS_VENTANAS = {
    "kc_n": (20, 60, int, "media del canal de Keltner: de 1.7 a 5 horas de historia"),
    "kc_atr": (5, 30, int, "ventana del ATR que da el ancho del canal"),
    "rsi_n": (7, 42, int, "RSI de 35 min a 3.5 horas"),
    "trix_n": (10, 60, int, "TRIX de 50 min a 5 horas (triple suavizado)"),
    "memoria": (1, 24, int, "la ruptura de Keltner sigue vigente de 5 min a 2 horas"),
    "dias_tendencia": (10, 30, int, "media de cierres diarios del filtro de tendencia: de 2 a 6 semanas"),
}
N_PRUEBAS_VENTANAS = 200
MIN_OPERACIONES_VENTANAS = 30

# Etapa 2: por régimen, con las ventanas de la etapa 1 fijas, se optimiza lo operativo.
RANGOS = {
    "sl_atr": (2.0, 8.0, float, "stop amplio, porque la posición pasa la noche y hay huecos de apertura"),
    "tp_atr": (4.0, 24.0, float, "objetivo que supere varias veces el costo de 0.28% ida y vuelta (ATR ≈ 0.28%)"),
    "riesgo": (0.0025, 0.01, float, "tamaño de posición: tocar el stop cuesta de 0.25% a 1% del equity"),
    "umbral_vwap": (0.003, 0.008, float, "compra nocturna: cierre de 0.3% a 0.8% bajo el VWAP de la sesión"),
    "tamano_noche": (0.25, 1.0, float, "compra nocturna: fracción del equity (sin apalancamiento)"),
}

optuna.logging.set_verbosity(optuna.logging.WARNING)


def sugerir_parametros(trial, rangos: dict = None) -> dict:
    """θ del trial dentro de `rangos` (RANGOS por defecto)."""
    p = {}
    for nombre, (bajo, alto, tipo, _) in (rangos or RANGOS).items():
        p[nombre] = trial.suggest_int(nombre, bajo, alto) if tipo is int else trial.suggest_float(nombre, bajo, alto)
    return p


def completar(params: dict) -> dict:
    """θ completo a partir de los parámetros de RANGOS (no hay umbrales derivados)."""
    return dict(params)


def _separar(p: dict) -> tuple[dict, dict]:
    senal = {k: v for k, v in p.items() if k not in PARAMETROS_OPERACION}
    operacion = {k: p.get(k, BASE_OPERACION[k]) for k in PARAMETROS_OPERACION}
    return senal, operacion


def senales_por_regimen(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict,
                        modo: str = "dos_de_tres", modo_noche: str = "dos_de_tres") -> pd.DataFrame:
    """Combina las señales de cada régimen con sus propios parámetros.

    `parametros` = {regimen: dict o None}; None significa régimen apagado (fuera del mercado).
    Se combinan la señal y la tendencia diaria de cada régimen y con ellas se calcula una sola posición
    objetivo, así que al cambiar de régimen la posición abierta sigue mientras el objetivo con el θ del
    régimen nuevo no cambie. Agrega por vela las columnas regimen, sl_atr, tp_atr, riesgo y max_velas.
    """
    base = generar_senales(datos, None, modo, modo_noche)
    base["senal"] = 0
    base["tendencia_diaria"] = 0
    base["nocturna"] = 0
    base["regimen"] = etiquetas.reindex(datos.index).to_numpy()
    for k, v in BASE_OPERACION.items():
        base[k] = v
    for regimen, p in parametros.items():
        mascara = base["regimen"] == regimen
        if p is None or not mascara.any():
            continue
        p_senal, p_operacion = _separar(p)
        propia = generar_senales(datos, p_senal, modo, modo_noche)
        base.loc[mascara, "senal"] = propia.loc[mascara, "senal"]
        base.loc[mascara, "tendencia_diaria"] = propia.loc[mascara, "tendencia_diaria"]
        base.loc[mascara, "nocturna"] = propia.loc[mascara, "nocturna"]
        for k, v in p_operacion.items():
            base.loc[mascara, k] = v
    base["objetivo"] = posicion_objetivo(base["senal"], base["tendencia_diaria"])
    return base


def backtest_por_regimen(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, desde=None,
                         costos: Costos = Costos(), modo: str = "dos_de_tres", modo_noche: str = "dos_de_tres") -> dict:
    """Backtest con parámetros por régimen, parte de día y compra nocturna; `desde` recorta el calentamiento."""
    senales = senales_por_regimen(datos, etiquetas, parametros, modo, modo_noche)
    if desde is not None:
        senales = senales.loc[desde:]
    return backtest(senales, senales["objetivo"], costos=costos, por_vela=senales[list(PARAMETROS_OPERACION)],
                    regimen=senales["regimen"], nocturna=senales["nocturna"], liquidar_al_final=True)


def seleccionar_meseta(estudio, vecinos: int = VECINOS_MESETA, rangos: dict = None) -> dict:
    """Centro de la mejor meseta: el trial cuyo vecindario (k vecinos más cercanos, parámetros
    normalizados a [0, 1]) tiene el Calmar promedio más alto."""
    rangos = rangos or RANGOS
    trials = [t for t in estudio.trials if t.value is not None and t.value > PENALIZACION]
    if len(trials) <= vecinos:
        mejor = estudio.best_trial
        return {"params": mejor.params, "valor": mejor.value, "meseta": mejor.value, "es_argmax": True}
    X = np.array([[(t.params[k] - rangos[k][0]) / (rangos[k][1] - rangos[k][0]) for k in rangos] for t in trials])
    valores = np.array([t.value for t in trials])
    distancias = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    cercanos = np.argsort(distancias, axis=1)[:, :vecinos]
    promedio = valores[cercanos].mean(axis=1)
    i = int(np.argmax(promedio))
    return {"params": trials[i].params, "valor": float(valores[i]), "meseta": float(promedio[i]),
            "es_argmax": bool(trials[i].number == estudio.best_trial.number)}


def optimizar_regimen(datos: pd.DataFrame, etiquetas: pd.Series, regimen: str, desde,
                      n_pruebas: int = N_PRUEBAS, semilla: int = SEMILLA, sampler: str = "tpe",
                      devolver_estudio: bool = False, ventanas: dict | None = None) -> dict:
    """Optuna maximizando el Calmar del backtest que solo opera en `regimen`, con las `ventanas` fijas."""
    ventanas = ventanas or {}
    if not (etiquetas.loc[desde:].reindex(datos.loc[desde:].index) == regimen).any():
        return {"parametros": None, "calmar": np.nan, "factibles": 0, "pruebas": 0, "estudio": None}

    def objetivo(trial):
        p = {**ventanas, **sugerir_parametros(trial)}
        r = backtest_por_regimen(datos, etiquetas, {regimen: p}, desde)
        ops = r["operaciones"]
        n = int((ops["regimen"] == regimen).sum()) if len(ops) else 0
        trial.set_user_attr("operaciones", n)
        if n < MIN_OPERACIONES:
            return PENALIZACION
        valor = calmar(r["equity"])
        return valor if np.isfinite(valor) else PENALIZACION

    muestreador = (optuna.samplers.RandomSampler(seed=semilla) if sampler == "random"
                   else optuna.samplers.TPESampler(seed=semilla))
    estudio = optuna.create_study(direction="maximize", sampler=muestreador)
    if sampler == "tpe":
        estudio.enqueue_trial({k: (PARAMETROS_BASE[k] if k in PARAMETROS_BASE else BASE_OPERACION[k]) for k in RANGOS})
    estudio.optimize(objetivo, n_trials=n_pruebas)
    factibles = sum(1 for t in estudio.trials if t.value is not None and t.value > PENALIZACION)
    meseta = seleccionar_meseta(estudio)
    activo = estudio.best_value > 0 and meseta["valor"] > 0
    return {"parametros": {**ventanas, **completar(meseta["params"])} if activo else None, "calmar": estudio.best_value,
            "calmar_meseta": meseta["meseta"], "es_argmax": meseta["es_argmax"],
            "factibles": factibles, "pruebas": len(estudio.trials),
            "estudio": estudio if devolver_estudio else None}


def optimizar_ventana(datos: pd.DataFrame, variables: pd.DataFrame, inicio_train, inicio_test, fin_test,
                      n_pruebas: int = N_PRUEBAS, semilla: int = SEMILLA, ventanas: dict | None = None) -> dict:
    """Una ventana del walk-forward: K-means con la historia disponible (sin la sesión de embargo),
    Optuna por régimen con las ventanas de la etapa 1 fijas, desempeño dentro de muestra y en la semana de prueba."""
    t0 = time.perf_counter()
    corte = datos.index.get_indexer([inicio_test])[0] - EMBARGO
    historia = variables.iloc[:corte]
    modelo = ajustar_modelo(historia)
    etiquetas = etiquetar(modelo, variables)
    pos = datos.index.get_indexer([inicio_train])[0]
    tramo_train = datos.iloc[max(0, pos - CALENTAMIENTO):corte]
    resultados = {r: optimizar_regimen(tramo_train, etiquetas, r, inicio_train, n_pruebas, semilla + i,
                                       ventanas=ventanas)
                  for i, r in enumerate(REGIMENES)}
    parametros = {r: res["parametros"] for r, res in resultados.items()}
    dentro = backtest_por_regimen(tramo_train, etiquetas, parametros, inicio_train)
    pos = datos.index.get_indexer([inicio_test])[0]
    tramo_test = datos.iloc[max(0, pos - CALENTAMIENTO):].loc[:fin_test]
    fuera = backtest_por_regimen(tramo_test, etiquetas, parametros, inicio_test)
    return {
        "inicio_train": inicio_train, "inicio_test": inicio_test, "fin_test": fin_test,
        "parametros": parametros,
        "calmar_por_regimen": {r: res["calmar"] for r, res in resultados.items()},
        "factibles": sum(res["factibles"] for res in resultados.values()),
        "pruebas": sum(res["pruebas"] for res in resultados.values()),
        "dentro": resumen_metricas(dentro["equity"], dentro["operaciones"]),
        "fuera": resumen_metricas(fuera["equity"], fuera["operaciones"]),
        "valor_fuera": fuera["equity"],
        "operaciones_fuera": fuera["operaciones"],
        "segundos": time.perf_counter() - t0,
    }


def ventanas_walk_forward(datos: pd.DataFrame, variables: pd.DataFrame) -> list:
    """(inicio_train, inicio_test, fin_test) por semana de prueba, con entrenamiento del último mes."""
    primera_valida = variables.dropna().index[0]
    semanas = datos.groupby(datos.index.tz_localize(None).to_period("W-FRI")).apply(lambda s: (s.index[0], s.index[-1]))
    ventanas = []
    for inicio_test, fin_test in semanas:
        inicio_mes = inicio_test - pd.DateOffset(months=1)
        if inicio_mes < primera_valida:
            continue
        inicio_train = datos.index[datos.index >= inicio_mes][0]
        ventanas.append((inicio_train, inicio_test, fin_test))
    return ventanas


def walk_forward(datos: pd.DataFrame, n_pruebas: int = N_PRUEBAS, n_jobs: int = -1, ventanas: dict | None = None) -> dict:
    """Corre todas las ventanas en paralelo, encadena las semanas fuera de muestra y calcula
    la walk-forward efficiency = rendimiento anualizado fuera / promedio dentro de muestra."""
    variables = calcular_variables_regimen(datos)
    cortes = ventanas_walk_forward(datos, variables)
    t0 = time.perf_counter()
    resultados = Parallel(n_jobs=n_jobs)(
        delayed(optimizar_ventana)(datos, variables, a, b, c, n_pruebas, SEMILLA + 10 * i, ventanas)
        for i, (a, b, c) in enumerate(cortes))
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
    return {"ventanas": resultados, "resumen": resumen, "valor": valor, "operaciones": operaciones,
            "segundos": segundos, "configuraciones": sum(r["pruebas"] for r in resultados),
            "anualizado_dentro": dentro, "anualizado_fuera": fuera, "eficiencia": fuera / dentro}


def optimizar_ventanas(train: pd.DataFrame, n_pruebas: int = N_PRUEBAS_VENTANAS, semilla: int = SEMILLA) -> dict:
    """Etapa 1: Optuna (TPE) sobre las ventanas de los indicadores de la parte de día, con un solo θ para todo
    train y sin régimen. Las ventanas solo afectan la parte de día, así que la compra nocturna queda apagada; el
    stop, el objetivo y el riesgo se buscan junto con las ventanas porque interactúan con ellas, pero de esta
    etapa solo se conservan las ventanas. Objetivo: Calmar con al menos MIN_OPERACIONES_VENTANAS operaciones; las
    ventanas salen del centro de la mejor meseta, como θ* por régimen."""
    conjunto = {**RANGOS_VENTANAS, **{k: RANGOS[k] for k in ("sl_atr", "tp_atr", "riesgo")}}
    inicio = train.index[min(CALENTAMIENTO, len(train) - 1)]

    def objetivo(trial):
        p = sugerir_parametros(trial, conjunto)
        s = generar_senales(train, p, modo_noche="ninguna")
        operacion = {k: p[k] for k in ("sl_atr", "tp_atr", "riesgo")}
        r = backtest(s.loc[inicio:], s.loc[inicio:, "objetivo"], operacion, liquidar_al_final=True)
        n = len(r["operaciones"])
        trial.set_user_attr("operaciones", n)
        valor = calmar(r["equity"])
        return valor if n >= MIN_OPERACIONES_VENTANAS and np.isfinite(valor) else PENALIZACION

    t0 = time.perf_counter()
    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=semilla))
    estudio.enqueue_trial({k: (PARAMETROS_BASE[k] if k in PARAMETROS_BASE else BASE_OPERACION[k]) for k in conjunto})
    estudio.optimize(objetivo, n_trials=n_pruebas)
    meseta = seleccionar_meseta(estudio, rangos=conjunto)
    return {"ventanas": {k: meseta["params"][k] for k in RANGOS_VENTANAS}, "calmar": estudio.best_value, "calmar_meseta": meseta["meseta"],
            "es_argmax": meseta["es_argmax"], "pruebas": len(estudio.trials), "estudio": estudio,
            "segundos": time.perf_counter() - t0}


def modelo_final(train: pd.DataFrame, n_pruebas: int = N_DIAGNOSTICO) -> dict:
    """K-means con todo train; por régimen, random search y TPE sobre todo train (enero a mayo). θ* sale de la
    meseta del TPE; el random search queda como referencia y para fijar la superficie 3D. Con 5 meses cada
    régimen junta de 15 a 25 operaciones; con el último mes solo juntaba 3 o 4 y θ* seguía al ruido.
    Antes, la etapa 1 fija las ventanas de los indicadores con todo train."""
    etapa1 = optimizar_ventanas(train)
    ventanas = etapa1["ventanas"]
    variables = calcular_variables_regimen(train)
    modelo = ajustar_modelo(variables)
    inicio = train.index[min(CALENTAMIENTO, len(train) - 1)]
    etiquetas = etiquetar(modelo, variables)
    tramo = train
    trabajos = [(r, s, SEMILLA + d + i) for i, r in enumerate(REGIMENES) for s, d in (("random", 200), ("tpe", 100))]
    salida = Parallel(n_jobs=len(trabajos))(
        delayed(optimizar_regimen)(tramo, etiquetas, r, inicio, n_pruebas, semilla, s, True, ventanas)
        for r, s, semilla in trabajos)
    aleatorio = {r: res for (r, s, _), res in zip(trabajos, salida) if s == "random"}
    tpe = {r: res for (r, s, _), res in zip(trabajos, salida) if s == "tpe"}
    return {"modelo": modelo, "etiquetas": etiquetas, "tramo": tramo, "inicio": inicio,
            "ventanas": ventanas, "etapa1": etapa1,
            "parametros": {r: res["parametros"] for r, res in tpe.items()},
            "tpe": tpe, "aleatorio": aleatorio,
            "calmar_por_regimen": {r: res["calmar"] for r, res in tpe.items()},
            "pruebas": etapa1["pruebas"] + sum(res["pruebas"] for res in list(tpe.values()) + list(aleatorio.values()))}


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
        "ventanas": final["ventanas"],
        "parametros": final["parametros"],
        "centros_kmeans": np.round(modelo.modelo.cluster_centers_, 6).tolist(),
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
    base = {**final["ventanas"], **final["aleatorio"][regimen]["estudio"].best_params}
    xs = np.linspace(*RANGOS[ejes[0]][:2], puntos)
    ys = np.linspace(*RANGOS[ejes[1]][:2], puntos)
    Z = np.full((puntos, puntos), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            p = dict(base)
            for eje, v in ((ejes[0], x), (ejes[1], y)):
                p[eje] = int(round(v)) if RANGOS[eje][2] is int else float(v)
            r = backtest_por_regimen(final["tramo"], final["etiquetas"], {regimen: completar(p)}, final["inicio"])
            Z[i, j] = calmar(r["equity"]) if len(r["operaciones"]) >= MIN_OPERACIONES else np.nan
    return {"x": xs, "y": ys, "z": Z, "ejes": ejes, "base": base}


def sensibilidad(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, desde,
                 variacion: float = 0.20) -> pd.DataFrame:
    """Varía cada parámetro ±20% en todos los regímenes activos y mide el Calmar."""
    base = backtest_por_regimen(datos, etiquetas, parametros, desde)
    filas = []
    todos = {**RANGOS_VENTANAS, **RANGOS}
    for clave in todos:
        fila = {"parametro": clave, "base": calmar(base["equity"])}
        for etiqueta, factor in (("-20%", 1 - variacion), ("+20%", 1 + variacion)):
            variado = {}
            for r, p in parametros.items():
                if p is None:
                    variado[r] = None
                    continue
                q = dict(p)
                nuevo = q[clave] * factor
                q[clave] = max(1, int(round(nuevo))) if todos[clave][2] is int else nuevo
                variado[r] = completar(q)
            fila[etiqueta] = calmar(backtest_por_regimen(datos, etiquetas, variado, desde)["equity"])
        filas.append(fila)
    return pd.DataFrame(filas).set_index("parametro")[["-20%", "base", "+20%"]]


def barrido_costos(datos: pd.DataFrame, etiquetas: pd.Series, parametros: dict, desde,
                   ida_vuelta_bps=np.arange(0, 51, 5)) -> pd.DataFrame:
    """Sharpe y equity final contra el costo total de ida y vuelta (comisión repartida en dos lados,
    sin spread ni impacto, para aislar el efecto del costo)."""
    filas = []
    for bps in ida_vuelta_bps:
        r = backtest_por_regimen(datos, etiquetas, parametros, desde, costos=Costos(bps / 2 / 1e4, 0.0, 0.0))
        filas.append({"ida_vuelta_bps": int(bps), "sharpe": sharpe(r["equity"]),
                      "retorno_neto": r["equity"].iloc[-1] / r["equity"].iloc[0] - 1,
                      "equity_final": r["equity"].iloc[-1], "calmar": calmar(r["equity"]),
                      "operaciones": len(r["operaciones"])})
    return pd.DataFrame(filas)
