"""Figuras del Lab 02 con el mismo estilo que las de la Act04.

Todas las funciones regresan la figura de matplotlib para poder mostrarla o guardarla.
Las gráficas intradía usan el número de vela en el eje x para no dibujar los huecos
nocturnos, y etiquetan con la hora de Nueva York.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
