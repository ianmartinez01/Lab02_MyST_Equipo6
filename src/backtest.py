"""Motor de backtesting orientado a eventos (no vectorizado), con la posición abierta y el sizing del SPEC.

`backtest()` es una función pura: recibe datos, señales y parámetros, no lee ni escribe nada fuera de sus
argumentos y regresa todo el estado en el tiempo (efectivo, posición, equity) más la lista de operaciones.
El estado vive en tres variables explícitas: `efectivo`, `posicion` y `equity`.

Convenciones (docs/SPEC.md):
- La posición objetivo (+1, 0, −1) se decide al cierre de t y se ejecuta en la apertura de t+1. Se abre cuando el
  objetivo deja de ser 0 y se cierra cuando cambia.
- La posición pasa la noche: la orden decidida al cierre de las 15:55 se ejecuta en la apertura del día siguiente,
  así que un hueco de apertura se llena a la apertura.
- Stop-loss primero si stop y objetivo caen en la misma vela; si la vela abre más allá, se llena a la apertura.
- Después de un stop, un take-profit o el holding máximo no se vuelve a entrar hasta que el objetivo cambie.
- Sizing por riesgo: tocar el stop cuesta `riesgo` del equity, sin apalancamiento. Los cortos pagan préstamo de
  acciones por cada vela abierta.
- Compra nocturna (serie `nocturna`): si la señal del cierre de las 15:50 es +1, se compra en la apertura de las
  15:55 y se vende en la apertura del día siguiente. Si la parte de día está corta a esa hora, el corto se cierra;
  si está larga, sigue y no hay compra nocturna. El tamaño es una fracción fija del equity (`tamano_noche`), no
  depende del stop, porque la posición sale siempre en la apertura. La lógica de día no cierra esta posición, y su
  stop no bloquea la reentrada de día.
- Nunca hay más de una posición abierta.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# θ de operación base: stop 6 ATR, objetivo 18.7 ATR, riesgo 0.5%, holding máximo de 10 sesiones (780 velas)
# y compra nocturna con el 100% del equity.
THETA_OPERACION = {"sl_atr": 6.0, "tp_atr": 18.7, "riesgo": 0.005, "max_velas": 780, "tamano_noche": 1.0}


@dataclass
class Posicion:
    """Una posición abierta: lado (+1 largo, −1 corto), acciones, precio de entrada, stop y objetivo."""
    lado: int
    acciones: float
    precio_entrada: float
    stop_loss: float
    take_profit: float
    entrada: pd.Timestamp = None
    velas: int = 0
    tipo: str = "dia"

    def __post_init__(self):
        if self.lado not in (1, -1):
            raise ValueError("lado debe ser +1 (largo) o −1 (corto)")
        if self.acciones <= 0:
            raise ValueError("acciones debe ser positivo")
        if self.lado * (self.precio_entrada - self.stop_loss) <= 0:
            raise ValueError("el stop debe quedar del lado de la pérdida")
        if self.lado * (self.take_profit - self.precio_entrada) <= 0:
            raise ValueError("el objetivo debe quedar del lado de la ganancia")

    def salida_intrabar(self, alto: float, bajo: float, apertura: float):
        """Precio y motivo de salida si la vela toca stop u objetivo; None si no toca ninguno.

        Convención conservadora: si stop y objetivo caen dentro de la misma vela, gana el stop.
        Si la vela abre más allá de un nivel, se llena a la apertura.
        """
        if self.lado == 1:
            if bajo <= self.stop_loss:
                return min(apertura, self.stop_loss), "stop_loss"
            if alto >= self.take_profit:
                return max(apertura, self.take_profit), "take_profit"
        else:
            if alto >= self.stop_loss:
                return max(apertura, self.stop_loss), "stop_loss"
            if bajo <= self.take_profit:
                return min(apertura, self.take_profit), "take_profit"
        return None


def niveles(lado: int, precio: float, atr: float, sl_atr: float, tp_atr: float) -> tuple[float, float]:
    """Stop y objetivo a sl_atr y tp_atr rangos típicos (ATR) del precio de entrada."""
    return precio - lado * sl_atr * atr, precio + lado * tp_atr * atr


def tamano_por_riesgo(equity: float, precio: float, stop: float, riesgo: float) -> float:
    """Acciones tales que tocar el stop cueste exactamente riesgo × equity (sin apalancamiento).

    acciones = riesgo · equity / |precio − stop|, recortado a equity / precio.
    """
    distancia = abs(precio - stop)
    if distancia <= 0:
        raise ValueError("el stop no puede estar en el precio de entrada")
    return min(riesgo * equity / distancia, equity / precio)


CAPITAL = 1_000_000.0


@dataclass(frozen=True)
class Costos:
    """Comisión por lado (fracción del monto), medio spread (USD por acción), impacto de raíz cuadrada y
    tasa anual de préstamo de acciones para cortos (cobrada por vela sobre el valor de la posición)."""
    comision: float = 0.00125
    medio_spread: float = 0.005
    coef_impacto: float = 1.0
    prestamo_anual: float = 0.005


SIN_COSTOS = Costos(0.0, 0.0, 0.0, 0.0)
VELAS_POR_ANIO = 252 * 78


def _deslizamiento(costos: Costos, precio: float, acciones: float, sigma: float, adv: float) -> float:
    """USD por acción: medio spread + precio · coef · σ_diaria · √(acciones / volumen diario promedio)."""
    impacto = 0.0
    if costos.coef_impacto > 0 and adv > 0 and np.isfinite(sigma):
        impacto = precio * costos.coef_impacto * sigma * np.sqrt(acciones / adv)
    return costos.medio_spread + impacto


def _volumen_diario_previo(datos: pd.DataFrame) -> np.ndarray:
    """Volumen diario promedio de las 20 sesiones anteriores (sin la sesión en curso)."""
    if "Volume" not in datos:
        return np.zeros(len(datos))
    dia = pd.Series(datos.index.date, index=datos.index)
    por_dia = datos["Volume"].groupby(dia.to_numpy()).sum()
    previo = por_dia.rolling(20, min_periods=1).mean().shift(1)
    return dia.map(previo).fillna(0.0).to_numpy()


def backtest(datos: pd.DataFrame, objetivo: pd.Series, theta: dict | None = None, costos: Costos = Costos(),
             capital: float = CAPITAL, por_vela: pd.DataFrame | None = None, regimen: pd.Series | None = None,
             nocturna: pd.Series | None = None, liquidar_al_final: bool = False) -> dict:
    """Simula la estrategia vela por vela.

    `datos` necesita Open, High, Low, Close y atr_14 (Volume opcional, para el impacto).
    `objetivo` ∈ {−1, 0, +1} es la posición deseada al cierre de cada vela; `nocturna` ∈ {0, 1} la compra nocturna.
    `por_vela` (opcional) trae sl_atr, tp_atr, riesgo, max_velas y tamano_noche por vela (θ distinto por régimen);
    `regimen` solo etiqueta las operaciones: al cambiar de régimen la posición se mantiene mientras el objetivo
    (ya calculado con el θ del régimen nuevo) no cambie.
    `liquidar_al_final` cierra la posición que siga abierta al cierre de la última vela, con sus costos.
    """
    p = {**THETA_OPERACION, **(theta or {})}
    o, h, l, c = (datos[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    atr = datos["atr_14"].to_numpy(float)
    s = objetivo.reindex(datos.index).fillna(0).to_numpy(int)
    noche = (nocturna.reindex(datos.index).fillna(0).to_numpy(int) if nocturna is not None
             else np.zeros(len(datos), dtype=int))
    sigma = atr / c * np.sqrt(78)
    adv = _volumen_diario_previo(datos)
    dia = np.asarray(datos.index.date)
    nuevo_dia = np.r_[True, dia[1:] != dia[:-1]]
    param = {k: (por_vela[k].to_numpy(float) if por_vela is not None and k in por_vela else np.full(len(datos), p[k]))
             for k in ("sl_atr", "tp_atr", "riesgo", "max_velas", "tamano_noche")}
    reg = regimen.reindex(datos.index).to_numpy() if regimen is not None else None

    efectivo, posicion = float(capital), None
    hist_efectivo, hist_posicion, hist_equity = np.empty(len(datos)), np.zeros(len(datos)), np.empty(len(datos))
    operaciones, comisiones, deslizamientos, prestamo = [], 0.0, 0.0, 0.0
    bloqueado = False

    def abrir(t, lado, tipo="dia"):
        nonlocal efectivo, posicion, comisiones, deslizamientos
        equity_ahora = efectivo
        stop, tp = niveles(lado, o[t], atr[t - 1], param["sl_atr"][t - 1], param["tp_atr"][t - 1])
        if tipo == "noche":
            acciones = param["tamano_noche"][t - 1] * equity_ahora / o[t]
        else:
            acciones = tamano_por_riesgo(equity_ahora, o[t], stop, param["riesgo"][t - 1])
        acciones = min(acciones, equity_ahora / (o[t] * (1 + costos.comision) + 1e-9))
        d = _deslizamiento(costos, o[t], acciones, sigma[t - 1], adv[t])
        precio = o[t] + lado * d
        comision = costos.comision * acciones * precio
        efectivo -= lado * acciones * precio + comision
        comisiones += comision
        deslizamientos += acciones * d
        posicion = Posicion(lado, acciones, precio, stop, tp, datos.index[t], tipo=tipo)
        operaciones.append({"tipo": tipo, "lado": lado, "entrada": datos.index[t], "precio_entrada": precio,
                            "acciones": acciones, "stop_loss": stop, "take_profit": tp, "comision": comision,
                            "deslizamiento": acciones * d, "prestamo": 0.0, "riesgo_usd": acciones * abs(o[t] - stop),
                            "regimen": reg[t - 1] if reg is not None else None})

    def cerrar(t, precio_ref, motivo):
        nonlocal efectivo, posicion, comisiones, deslizamientos
        d = _deslizamiento(costos, precio_ref, posicion.acciones, sigma[t - 1] if t > 0 else np.nan, adv[t])
        precio = precio_ref - posicion.lado * d
        comision = costos.comision * posicion.acciones * precio
        efectivo += posicion.lado * posicion.acciones * precio - comision
        comisiones += comision
        deslizamientos += posicion.acciones * d
        op = operaciones[-1]
        op.update({"salida": datos.index[t], "precio_salida": precio, "motivo": motivo, "velas": posicion.velas})
        op["comision"] += comision
        op["deslizamiento"] += posicion.acciones * d
        op["pnl"] = posicion.lado * posicion.acciones * (precio - op["precio_entrada"]) - op["comision"] - op["prestamo"]
        posicion = None

    for t in range(len(datos)):
        # 1) Órdenes decididas al cierre de t−1, en la apertura de t (también en la primera vela del día).
        if t > 0:
            if posicion is not None and posicion.tipo == "noche" and nuevo_dia[t]:
                cerrar(t, o[t], "fin_nocturna")
            if posicion is not None and posicion.tipo == "dia" and s[t - 1] != posicion.lado:
                cerrar(t, o[t], "objetivo")
            if posicion is not None and posicion.tipo == "dia" and posicion.lado == -1 and noche[t - 1] == 1:
                cerrar(t, o[t], "cede_a_nocturna")
            if t > 1 and s[t - 1] != s[t - 2]:
                bloqueado = False
            atr_ok = np.isfinite(atr[t - 1]) and atr[t - 1] > 0
            if posicion is None and noche[t - 1] == 1 and atr_ok:
                abrir(t, 1, "noche")
            elif posicion is None and s[t - 1] != 0 and not bloqueado and atr_ok:
                abrir(t, int(s[t - 1]))
        # 2) Stop y objetivo dentro de la vela, préstamo de los cortos y holding máximo.
        if posicion is not None:
            salida = posicion.salida_intrabar(h[t], l[t], o[t])
            if salida is not None:
                bloqueado = bloqueado or posicion.tipo == "dia"
                cerrar(t, *salida)
            else:
                posicion.velas += 1
                if posicion.lado == -1 and costos.prestamo_anual > 0:
                    cargo = costos.prestamo_anual / VELAS_POR_ANIO * posicion.acciones * c[t]
                    efectivo -= cargo
                    prestamo += cargo
                    operaciones[-1]["prestamo"] += cargo
                if posicion.velas >= param["max_velas"][t]:
                    bloqueado = bloqueado or posicion.tipo == "dia"
                    cerrar(t, c[t], "holding_maximo")
        if liquidar_al_final and posicion is not None and t == len(datos) - 1:
            cerrar(t, c[t], "fin_de_periodo")
        hist_efectivo[t] = efectivo
        hist_posicion[t] = 0.0 if posicion is None else posicion.lado * posicion.acciones
        hist_equity[t] = efectivo + hist_posicion[t] * c[t]

    return {
        "equity": pd.Series(hist_equity, index=datos.index, name="equity"),
        "efectivo": pd.Series(hist_efectivo, index=datos.index, name="efectivo"),
        "posicion": pd.Series(hist_posicion, index=datos.index, name="posicion"),
        "operaciones": pd.DataFrame(operaciones),
        "comisiones": comisiones,
        "deslizamiento": deslizamientos,
        "prestamo": prestamo,
        "posicion_abierta": posicion,
    }
