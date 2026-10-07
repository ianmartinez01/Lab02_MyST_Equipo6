"""Figuras del Lab 02 con el mismo estilo que las de la Act04.

Todas las funciones regresan la figura de matplotlib para poder mostrarla o guardarla.
Las gráficas intradía usan el número de vela en el eje x para no dibujar los huecos
nocturnos, y etiquetan con la hora de Nueva York.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.metrics import curva_drawdown

plt.style.use("seaborn-v0_8-whitegrid")

COLOR_PRECIO = "#222222"
COLOR_COMPRA = "#2ca02c"
COLOR_VENTA = "#d62728"
COLOR_EMA = "#ff7f0e"
COLOR_BANDA = "#7f7f7f"


def _eje_horas(ax, indice, cada=6):
    posiciones = np.arange(len(indice))[::cada]
    ax.set_xticks(posiciones)
    ax.set_xticklabels([indice[i].strftime("%d-%b %H:%M") for i in posiciones], rotation=45, ha="right")


def plot_indicadores(ind, titulo="Precio e indicadores de la estrategia"):
    """Panel de 3 filas: precio con el canal de Keltner, RSI contra 50 y TRIX contra cero."""
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    x = np.arange(len(ind))
    cada = max(6, len(ind) // 24)
    ax[0].plot(x, ind["Close"], color=COLOR_PRECIO, linewidth=1, label="Close")
    ax[0].plot(x, ind["kc_mid"], color=COLOR_EMA, linewidth=1.2, label="Media del canal")
    ax[0].fill_between(x, ind["kc_low"], ind["kc_high"], color=COLOR_BANDA, alpha=0.15, label="Canal de Keltner")
    ax[0].set_title(titulo)
    ax[0].set_ylabel("Precio (USD)")
    ax[0].legend(loc="upper left")
    ax[1].plot(x, ind["rsi"], color="#9467bd", label="RSI")
    ax[1].axhline(50, color="black", linewidth=0.8)
    ax[1].fill_between(x, 50, ind["rsi"], where=ind["rsi"] > 50, color=COLOR_COMPRA, alpha=0.15)
    ax[1].fill_between(x, 50, ind["rsi"], where=ind["rsi"] < 50, color=COLOR_VENTA, alpha=0.15)
    ax[1].set_ylabel("RSI")
    ax[1].legend(loc="upper left")
    ax[2].bar(x, ind["trix"], color=np.where(ind["trix"] > 0, COLOR_COMPRA, COLOR_VENTA), alpha=0.6, label="TRIX")
    ax[2].set_ylabel("TRIX")
    ax[2].set_xlabel("Vela (hora de Nueva York)")
    ax[2].legend(loc="upper left")
    _eje_horas(ax[2], ind.index, cada)
    fig.tight_layout()
    return fig


def plot_operaciones(sen, operaciones, titulo="Señales y operaciones de la estrategia"):
    """Precio con el canal de Keltner; fondo = posición objetivo, ▲/▼ = entrada, ✕ = salida."""
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(sen))
    pos = {t: i for i, t in enumerate(sen.index)}
    ax.plot(x, sen["Close"], color=COLOR_PRECIO, linewidth=1, label="Close")
    ax.fill_between(x, sen["kc_low"], sen["kc_high"], color=COLOR_BANDA, alpha=0.15, label="Canal de Keltner")
    bajo, alto = sen["Low"].min(), sen["High"].max()
    ax.fill_between(x, bajo, alto, where=sen["objetivo"] > 0, color=COLOR_COMPRA, alpha=0.07, step="mid",
                    label="Objetivo largo")
    ax.fill_between(x, bajo, alto, where=sen["objetivo"] < 0, color=COLOR_VENTA, alpha=0.07, step="mid",
                    label="Objetivo corto")
    ops = operaciones[operaciones["entrada"].isin(sen.index) & operaciones["salida"].isin(sen.index)]
    for d, color, marca, nombre in ((1, COLOR_COMPRA, "^", "Compra"), (-1, COLOR_VENTA, "v", "Venta en corto")):
        o = ops[ops["lado"] == d]
        ax.scatter([pos[t] for t in o["entrada"]], o["precio_entrada"], marker=marca, s=150, color=color,
                   zorder=5, label=f"{nombre} ({len(o)})")
    ax.scatter([pos[t] for t in ops["salida"]], ops["precio_salida"], marker="x", s=80, color="black",
               zorder=5, label="Salida")
    for op in ops.itertuples():
        ax.annotate(f"{op.motivo}\n${op.pnl:,.0f}", (pos[op.salida], op.precio_salida),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_title(titulo)
    ax.set_ylabel("Precio (USD)")
    ax.set_xlabel("Vela (hora de Nueva York)")
    ax.legend(loc="upper left", fontsize=9)
    _eje_horas(ax, sen.index, max(6, len(sen) // 24))
    fig.tight_layout()
    return fig


def plot_valor_drawdown(curvas, titulo="Valor del portafolio y drawdown"):
    """`curvas` = {nombre: (valor_estrategia, valor_benchmark)}; arriba el valor, abajo el drawdown."""
    fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colores = ["#1f77b4", COLOR_EMA, COLOR_COMPRA, "#9467bd"]
    for (nombre, (valor, benchmark)), color in zip(curvas.items(), colores):
        ax[0].plot(valor.index, valor, color=color, label=f"Estrategia {nombre}")
        ax[0].plot(benchmark.index, benchmark, color=color, linestyle=":", label=f"Buy & hold {nombre}")
        ax[1].fill_between(valor.index, curva_drawdown(valor) * 100, color=color, alpha=0.4, label=f"Drawdown {nombre}")
    ax[0].set_title(titulo)
    ax[0].set_ylabel("Valor (USD)")
    ax[0].legend(loc="upper left")
    ax[1].set_ylabel("Drawdown (%)")
    ax[1].set_xlabel("Fecha")
    ax[1].legend(loc="lower left")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    fig.tight_layout()
    return fig


COLORES_REGIMEN = {"tendencia": "#2ca02c", "reversion": "#1f77b4", "crisis": "#d62728"}
NOMBRE_REGIMEN = {"tendencia": "tendencia", "reversion": "reversión", "crisis": "crisis"}


def _sombrear_regimenes(ax, etiquetas, x=None):
    """Sombrea el fondo según el régimen; x = posiciones (si None, fechas)."""
    x = etiquetas.index if x is None else x
    racha = (etiquetas != etiquetas.shift()).cumsum()
    vistos = set()
    for _, tramo in etiquetas.groupby(racha):
        nombre = tramo.iloc[0]
        if nombre not in COLORES_REGIMEN:
            continue
        i0, i1 = etiquetas.index.get_loc(tramo.index[0]), etiquetas.index.get_loc(tramo.index[-1])
        ax.axvspan(x[i0], x[i1], color=COLORES_REGIMEN[nombre], alpha=0.15, linewidth=0,
                   label=None if nombre in vistos else f"Régimen {NOMBRE_REGIMEN[nombre]}")
        vistos.add(nombre)


def plot_tabla_retornos(tablas, titulo="Retornos mensuales, trimestrales y anuales (%)"):
    """`tablas` = {conjunto: {'mensual','trimestral','anual': Series}} dibujado como tabla."""
    filas = []
    for conjunto, t in tablas.items():
        for periodo, serie in t.items():
            formato = {"mensual": "%Y-%m", "trimestral": "%Y-T", "anual": "%Y"}[periodo]
            for fecha, valor in serie.items():
                etiqueta = f"{fecha.year}-T{fecha.quarter}" if periodo == "trimestral" else fecha.strftime(formato)
                filas.append([conjunto, periodo, etiqueta, f"{valor * 100:.2f}"])
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(filas) + 1.2))
    ax.axis("off")
    tabla = ax.table(cellText=filas, colLabels=["Conjunto", "Periodo", "Fecha", "Retorno (%)"],
                     loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1, 1.3)
    for (fila, col), celda in tabla.get_celld().items():
        if fila > 0 and col == 3:
            celda.set_text_props(color=COLOR_VENTA if float(filas[fila - 1][3]) < 0 else COLOR_COMPRA)
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def plot_sensibilidad(tabla, titulo="Sensibilidad del Calmar ante variaciones de ±20% (test)"):
    """Calmar con cada parámetro al −20%, base y +20%."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(tabla))
    ancho = 0.27
    for i, (col, color) in enumerate(zip(["-20%", "base", "+20%"], ["#9467bd", COLOR_BANDA, COLOR_EMA])):
        ax.bar(x + (i - 1) * ancho, tabla[col], width=ancho, color=color, label=col)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tabla.index, rotation=30, ha="right")
    ax.set_title(titulo)
    ax.set_xlabel("Parámetro")
    ax.set_ylabel("Calmar Ratio")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sensibilidad_costos(barridos: dict, marca_bps=None, titulo="Sensibilidad a costos de ida y vuelta"):
    """`barridos` = {nombre: DataFrame de barrido_costos}; Sharpe y equity final contra costo."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for (nombre, b), color in zip(barridos.items(), ["#1f77b4", "#ff7f0e"]):
        ax[0].plot(b["ida_vuelta_bps"], b["sharpe"], marker="o", color=color, label=nombre)
        ax[1].plot(b["ida_vuelta_bps"], b["equity_final"], marker="o", color=color, label=nombre)
    for eje, etiqueta in zip(ax, ("Sharpe anualizado", "Equity final (USD)")):
        if marca_bps is not None:
            eje.axvline(marca_bps, color="#d62728", linestyle="--", label=f"Costo del SPEC ({marca_bps:.1f} pb)")
        eje.set_xlabel("Costo de ida y vuelta (pb)")
        eje.set_ylabel(etiqueta)
        eje.legend()
    ax[1].axhline(1_000_000, color="black", linewidth=0.8)
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def plot_regimenes_precio(precio, etiquetas, corte=None, titulo="Línea de tiempo de regímenes sobre el precio"):
    """Precio de cierre con el fondo coloreado por régimen; `corte` marca el inicio de test."""
    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(precio))
    _sombrear_regimenes(ax, etiquetas.reindex(precio.index), x)
    ax.plot(x, precio, color=COLOR_PRECIO, linewidth=0.8, label="Close")
    if corte is not None:
        ax.axvline(precio.index.get_indexer([corte])[0], color="black", linestyle="--", label="Inicio de test")
    meses = pd.Series(np.arange(len(precio)), index=precio.index).groupby(precio.index.tz_localize(None).to_period("M")).first()
    ax.set_xticks(meses.to_numpy())
    ax.set_xticklabels([str(m) for m in meses.index])
    ax.set_title(titulo)
    ax.set_xlabel("Fecha (velas de sesión regular)")
    ax.set_ylabel("Precio (USD)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_distribuciones_regimen(variables, etiquetas, titulo="Distribución de las variables por régimen"):
    """Histogramas de volatilidad, eficiencia y retorno semanal para cada régimen."""
    columnas = {"volatilidad": "Volatilidad diaria", "eficiencia": "Eficiencia de Kaufman",
                "retorno_semana": "Retorno de la semana (log)"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    datos = variables.join(etiquetas.rename("regimen"))
    for eje, (col, nombre) in zip(ax, columnas.items()):
        for regimen, color in COLORES_REGIMEN.items():
            valores = datos.loc[datos["regimen"] == regimen, col].dropna()
            eje.hist(valores, bins=40, alpha=0.5, color=color, density=True, label=NOMBRE_REGIMEN[regimen])
        eje.set_title(nombre)
        eje.set_xlabel(nombre)
        eje.set_ylabel("Densidad")
        eje.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def plot_valor_regimenes(valor, etiquetas, benchmark=None, titulo="Valor del portafolio con regímenes superpuestos"):
    """Curva de valor con el fondo coloreado por régimen."""
    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(valor))
    _sombrear_regimenes(ax, etiquetas.reindex(valor.index), x)
    ax.plot(x, valor, color=COLOR_PRECIO, linewidth=1.2, label="Estrategia")
    if benchmark is not None:
        ax.plot(x, benchmark.reindex(valor.index), color=COLOR_EMA, linestyle=":", label="Buy & hold")
    semanas = pd.Series(x, index=valor.index).groupby(valor.index.tz_localize(None).to_period("W")).first()
    paso = max(1, len(semanas) // 12)
    ax.set_xticks(semanas.to_numpy()[::paso])
    ax.set_xticklabels([p.start_time.strftime("%d-%b") for p in semanas.index[::paso]])
    ax.set_title(titulo)
    ax.set_xlabel("Fecha (velas de sesión regular)")
    ax.set_ylabel("Valor (USD)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_walk_forward(resumen, titulo="Calmar dentro y fuera de muestra por ventana del walk-forward"):
    """Calmar en la ventana de entrenamiento contra la semana de prueba siguiente."""
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(resumen))
    ax.bar(x - 0.2, resumen["calmar_dentro"], width=0.4, color="#1f77b4", label="Dentro de muestra (1 mes)")
    ax.bar(x + 0.2, resumen["calmar_fuera"].fillna(0), width=0.4, color=COLOR_VENTA, label="Fuera de muestra (1 semana)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in resumen["inicio_test"]], rotation=45, ha="right")
    ax.set_title(titulo)
    ax.set_xlabel("Semana de prueba")
    ax.set_ylabel("Calmar Ratio")
    ax.legend()
    fig.tight_layout()
    return fig


def _valores(estudio, penalizacion=-10.0):
    trials = [t for t in estudio.trials if t.value is not None]
    x = np.array([t.number for t in trials])
    y = np.array([t.value for t in trials])
    return x, y, y > penalizacion


def plot_historia(finales, titulo="Historia de optimización: random search contra TPE"):
    """`finales` = {regimen: (estudio_random, estudio_tpe)}; valor por trial y mejor acumulado."""
    fig, ax = plt.subplots(1, len(finales), figsize=(7 * len(finales), 4.8), squeeze=False)
    for eje, (regimen, (aleatorio, tpe)) in zip(ax[0], finales.items()):
        for estudio, color, nombre in ((aleatorio, COLOR_BANDA, "Random search"), (tpe, "#1f77b4", "TPE")):
            x, y, ok = _valores(estudio)
            eje.scatter(x[ok], y[ok], s=10, color=color, alpha=0.5, label=f"{nombre} (N = {len(x)})")
            mejor = np.maximum.accumulate(np.where(ok, y, -np.inf))
            eje.plot(x, mejor, color=color, linewidth=2, label=f"Mejor acumulado {nombre}")
        eje.set_title(f"Régimen {NOMBRE_REGIMEN[regimen]}")
        eje.set_xlabel("Trial")
        eje.set_ylabel("Calmar (trials con operaciones suficientes)")
        eje.legend(fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def plot_importancia(importancias, titulo="Importancia de parámetros en el TPE (fANOVA)"):
    """`importancias` = {regimen: Series}."""
    fig, ax = plt.subplots(1, len(importancias), figsize=(6.5 * len(importancias), 4.8), squeeze=False)
    for eje, (regimen, imp) in zip(ax[0], importancias.items()):
        imp = imp.sort_values()
        eje.barh(imp.index, imp.to_numpy(), color=COLORES_REGIMEN[regimen], label="Importancia")
        eje.set_title(f"Régimen {NOMBRE_REGIMEN[regimen]}")
        eje.set_xlabel("Proporción de la varianza del Calmar explicada")
        eje.set_ylabel("Parámetro")
        eje.legend(loc="lower right")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def plot_slices(estudio, elegido, regimen, titulo=None):
    """Valor de cada parámetro contra el Calmar de cada trial del TPE; marca argmax y θ* de meseta."""
    trials = [t for t in estudio.trials if t.value is not None and t.value > -10]
    nombres = list(estudio.best_params)
    columnas = 4
    filas = int(np.ceil(len(nombres) / columnas))
    fig, ax = plt.subplots(filas, columnas, figsize=(16, 3.4 * filas), squeeze=False)
    for eje, nombre in zip(ax.flat, nombres):
        eje.scatter([t.params[nombre] for t in trials], [t.value for t in trials], s=12, alpha=0.5,
                    color="#1f77b4", label="Trial")
        eje.axvline(estudio.best_params[nombre], color=COLOR_VENTA, linestyle="--", label="Argmax")
        eje.axvline(elegido[nombre], color=COLOR_COMPRA, linewidth=2, label="θ* (meseta)")
        eje.set_xlabel(nombre)
        eje.set_ylabel("Calmar")
    for eje in list(ax.flat)[len(nombres):]:
        eje.axis("off")
    ax.flat[0].legend(fontsize=8)
    fig.suptitle(titulo or f"Slice plots del TPE, régimen {NOMBRE_REGIMEN[regimen]}")
    fig.tight_layout()
    return fig


def plot_superficie(sup, regimen, titulo=None):
    """Superficie 3D del Calmar en los dos parámetros más importantes, resto fijo en el mejor
    punto del random search."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(sup["x"], sup["y"])
    Z = np.ma.masked_invalid(sup["z"])
    superficie = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    fig.colorbar(superficie, ax=ax, shrink=0.6, label="Calmar")
    ax.set_xlabel(sup["ejes"][0])
    ax.set_ylabel(sup["ejes"][1])
    ax.set_zlabel("Calmar")
    ax.set_title(titulo or f"Calmar en función de {sup['ejes'][0]} y {sup['ejes'][1]} "
                           f"(régimen {NOMBRE_REGIMEN[regimen]}; resto de θ en el mejor random search)")
    return fig


def plot_linea_tiempo(precio: pd.Series, filtrada: pd.Series, viterbi: pd.Series,
                      titulo="Regímenes del HMM: etiqueta filtrada contra Viterbi"):
    """Dos paneles: precio coloreado por la etiqueta filtrada (causal) y por la ruta de Viterbi."""
    fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    x = np.arange(len(precio))
    for eje, etiquetas, nombre in ((ax[0], filtrada, "Filtrada (solo información hasta t)"),
                                   (ax[1], viterbi, "Viterbi (usa toda la muestra: mira al futuro)")):
        _sombrear_regimenes(eje, etiquetas.reindex(precio.index), x)
        eje.plot(x, precio, color="#222222", linewidth=0.8, label="Close")
        eje.set_title(nombre)
        eje.set_ylabel("Precio (USD)")
        eje.legend(loc="upper left", fontsize=9)
    meses = pd.Series(x, index=precio.index).groupby(precio.index.tz_localize(None).to_period("M")).first()
    ax[1].set_xticks(meses.to_numpy())
    ax[1].set_xticklabels([str(m) for m in meses.index])
    ax[1].set_xlabel("Fecha (velas de sesión regular)")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig
