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
    """Panel de 4 filas: precio con Bollinger y EMA 9, RSI, Estocástico y MACD/ATR."""
    fig, ax = plt.subplots(4, 1, figsize=(15, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    x = np.arange(len(ind))
    ax[0].plot(x, ind["Close"], color=COLOR_PRECIO, linewidth=1, label="Close")
    ax[0].plot(x, ind["ema_9"], color=COLOR_EMA, linewidth=1.3, label="EMA 9")
    ax[0].fill_between(x, ind["bb_low"], ind["bb_high"], color=COLOR_BANDA, alpha=0.15, label="Bollinger (20, 2σ)")
    ax[0].set_title(titulo)
    ax[0].set_ylabel("Precio (USD)")
    ax[0].legend(loc="upper left")
    ax[1].plot(x, ind["rsi_14"], color="#9467bd", label="RSI 14")
    ax[1].axhspan(70, 100, color=COLOR_VENTA, alpha=0.08)
    ax[1].axhspan(0, 30, color=COLOR_COMPRA, alpha=0.08)
    ax[1].set_ylabel("RSI")
    ax[1].legend(loc="upper left")
    ax[2].plot(x, ind["stoch_k"], color="#1f77b4", label="%K")
    ax[2].plot(x, ind["stoch_d"], color=COLOR_EMA, label="%D")
    ax[2].axhspan(80, 100, color=COLOR_VENTA, alpha=0.08)
    ax[2].axhspan(0, 20, color=COLOR_COMPRA, alpha=0.08)
    ax[2].set_ylabel("Estocástico")
    ax[2].legend(loc="upper left")
    ax[3].bar(x, ind["macd_atr"], color=np.where(ind["macd_atr"] > 0, COLOR_VENTA, COLOR_COMPRA), alpha=0.6,
              label="MACD / ATR")
    ax[3].set_ylabel("MACD / ATR")
    ax[3].set_xlabel("Vela (hora de Nueva York)")
    ax[3].legend(loc="upper left")
    _eje_horas(ax[3], ind.index)
    fig.tight_layout()
    return fig


def plot_operaciones(sen, operaciones, titulo="Señales y operaciones de la estrategia"):
    """Precio con Bollinger y EMA 9; círculos = señal al cierre, ▲/▼ = entrada, ✕ = salida."""
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(sen))
    pos = {t: i for i, t in enumerate(sen.index)}
    ax.plot(x, sen["Close"], color=COLOR_PRECIO, linewidth=1, label="Close")
    ax.plot(x, sen["ema_9"], color=COLOR_EMA, linewidth=1.2, label="EMA 9")
    ax.fill_between(x, sen["bb_low"], sen["bb_high"], color=COLOR_BANDA, alpha=0.15, label="Bollinger (20, 2σ)")
    senales = sen[sen["senal"] != 0]
    ax.scatter([pos[t] for t in senales.index], senales["Close"], s=60, facecolors="none", edgecolors="black",
               zorder=4, label="Señal al cierre de la vela")
    ops = operaciones[operaciones["entrada"].isin(sen.index) & operaciones["salida"].isin(sen.index)]
    for d, color, marca, nombre in ((1, COLOR_COMPRA, "^", "Compra"), (-1, COLOR_VENTA, "v", "Venta en corto")):
        o = ops[ops["direccion"] == d]
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
    _eje_horas(ax, sen.index)
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


def plot_costos(barridos, comision_base=0.00125, titulo="Retorno neto contra nivel de comisión por lado"):
    """`barridos` = {conjunto: DataFrame de barrido_costos}."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for (nombre, b), color in zip(barridos.items(), ["#1f77b4", COLOR_EMA]):
        ax.plot(b["comision"] * 100, b["retorno_neto"] * 100, marker="o", color=color, label=nombre)
    ax.axvline(comision_base * 100, color=COLOR_VENTA, linestyle="--", label="Comisión del laboratorio (0.125%)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(titulo)
    ax.set_xlabel("Comisión por lado (%)")
    ax.set_ylabel("Retorno neto (%)")
    ax.legend()
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
