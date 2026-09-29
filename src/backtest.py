"""Motor de backtesting event-driven con costos, stop-loss y take-profit.

Convenciones declaradas:
- La señal se genera al cierre de la vela t y la orden se ejecuta en la apertura de t+1.
- Comisión proporcional al monto operado, en cada apertura y en cada cierre.
- Sin apalancamiento: el monto de cada posición es una fracción ≤ 1 del valor del portafolio.
- Si stop-loss y take-profit caen dentro del rango de la misma vela, se ejecuta primero el
  stop-loss (convención conservadora). Si la vela abre más allá del stop, se ejecuta en la apertura.
- Toda posición se cierra en la última vela de cada sesión (sin exposición nocturna).
- Deslizamiento en cada entrada y cada salida (incluidos stops): medio spread de $0.005 por acción
  (spread típico de NVDA de 1 centavo) más impacto de mercado con la ley de raíz cuadrada,
  impacto = σ_diaria × √(acciones / volumen diario promedio de 20 sesiones), con σ_diaria ≈ ATR/P·√78.
  Compra paga precio + deslizamiento y venta recibe precio − deslizamiento.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COMISION = 0.00125
CAPITAL_INICIAL = 1_000_000.0
MEDIO_SPREAD = 0.005
COEF_IMPACTO = 1.0


@dataclass
class Portafolio:
    """Estado explícito del portafolio: efectivo, acciones (negativas en corto) y costos acumulados."""
    efectivo: float
    acciones: float = 0.0
    costos: float = 0.0
    deslizamiento: float = 0.0
    operaciones: list = field(default_factory=list)

    def valor(self, precio: float) -> float:
        return self.efectivo + self.acciones * precio

    def abrir(self, direccion: int, precio: float, monto: float, comision: float, fecha, sl: float, tp: float,
              desliz: float = 0.0):
        """`precio` es el de referencia; se ejecuta a precio ± desliz (USD por acción)."""
        ejecucion = precio + direccion * desliz
        acciones = monto / ejecucion
        costo = comision * monto
        self.efectivo -= direccion * acciones * ejecucion + costo
        self.acciones = direccion * acciones
        self.costos += costo
        self.deslizamiento += acciones * desliz
        self.operaciones.append({"direccion": direccion, "entrada": fecha, "precio_entrada": ejecucion,
                                 "acciones": acciones, "sl": sl, "tp": tp, "costos": costo,
                                 "deslizamiento": acciones * desliz})

    def cerrar(self, precio: float, comision: float, fecha, motivo: str, desliz: float = 0.0):
        ejecucion = precio - np.sign(self.acciones) * desliz
        monto = abs(self.acciones) * ejecucion
        costo = comision * monto
        self.efectivo += self.acciones * ejecucion - costo
        self.costos += costo
        self.deslizamiento += abs(self.acciones) * desliz
        op = self.operaciones[-1]
        op.update({"salida": fecha, "precio_salida": ejecucion, "motivo": motivo})
        op["costos"] += costo
        op["deslizamiento"] += abs(self.acciones) * desliz
        op["pnl"] = op["direccion"] * op["acciones"] * (precio - op["precio_entrada"]) - op["costos"]
        self.acciones = 0.0


def ejecutar_backtest(datos: pd.DataFrame, sl_atr: float = 1.5, tp_atr: float = 2.0, fraccion: float = 1.0,
                      comision: float = COMISION, capital: float = CAPITAL_INICIAL,
                      velas_sin_entrada: int = 3, enfriamiento: int = 0, max_por_dia: int = 99,
                      medio_spread: float = MEDIO_SPREAD, coef_impacto: float = COEF_IMPACTO) -> dict:
    """Recorre las velas una por una. `datos` requiere Open, High, Low, Close, atr_14 y senal.

    Si trae la columna `regimen`, una posición abierta se cierra en la apertura siguiente al cambio.
    Si trae columnas `sl_atr`, `tp_atr`, `fraccion` o `enfriamiento`, se usan los valores de la vela
    de la señal en lugar de los argumentos (parámetros distintos por régimen).
    Devuelve la curva de valor, la tabla de operaciones y el estado final del portafolio.
    """
    if not 0 < fraccion <= 1:
        raise ValueError("fraccion debe estar en (0, 1]: no se permite apalancamiento")
    o, h, l, c = (datos[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    atr, senal = datos["atr_14"].to_numpy(), datos["senal"].to_numpy()
    regimen = datos["regimen"].to_numpy() if "regimen" in datos else None
    por_vela = {k: (datos[k].to_numpy() if k in datos else np.full(len(datos), v))
                for k, v in (("sl_atr", sl_atr), ("tp_atr", tp_atr), ("fraccion", fraccion),
                             ("enfriamiento", enfriamiento))}
    if np.any(por_vela["fraccion"] > 1) or np.any(por_vela["fraccion"] <= 0):
        raise ValueError("fraccion debe estar en (0, 1]: no se permite apalancamiento")
    fechas = datos.index
    dia = np.asarray(fechas.date)
    if "Volume" in datos and coef_impacto > 0:
        volumen_dia = datos["Volume"].groupby(dia).sum()
        adv = volumen_dia.rolling(20, min_periods=1).mean().shift(1).fillna(volumen_dia.iloc[0])
        adv = pd.Series(dia).map(adv).to_numpy()
        sigma = (datos["atr_14"] / datos["Close"]).to_numpy() * np.sqrt(78)
    else:
        adv, sigma = None, None

    def desliz(t, precio, acciones):
        """Medio spread más impacto de raíz cuadrada con información hasta t-1."""
        if adv is None:
            return medio_spread
        s = sigma[t - 1] if t > 0 and np.isfinite(sigma[t - 1]) else 0.0
        return medio_spread + precio * coef_impacto * s * np.sqrt(acciones / adv[t])

    ultima = np.r_[dia[1:] != dia[:-1], True]
    pos_en_dia = pd.Series(1, index=fechas).groupby(dia).cumsum().to_numpy() - 1
    velas_dia = pd.Series(1, index=fechas).groupby(dia).transform("size").to_numpy()

    port = Portafolio(efectivo=capital)
    valor = np.empty(len(datos))
    bloqueo = {1: -1, -1: -1}
    abiertas_hoy = 0

    for t in range(len(datos)):
        d = int(np.sign(port.acciones))
        if t == 0 or dia[t] != dia[t - 1]:
            abiertas_hoy = 0
        # 1) Órdenes decididas al cierre de t-1, ejecutadas en la apertura de t.
        if t > 0 and not ultima[t - 1]:
            s = senal[t - 1]
            cambio_regimen = regimen is not None and regimen[t - 1] != regimen[t - 2] if t > 1 else False
            if d != 0 and (s == -d or cambio_regimen):
                port.cerrar(o[t], comision, fechas[t], "cambio de régimen" if cambio_regimen else "señal contraria",
                            desliz(t, o[t], abs(port.acciones)))
                d = 0
            permitido = velas_sin_entrada <= pos_en_dia[t] < velas_dia[t] - velas_sin_entrada
            permitido = permitido and abiertas_hoy < max_por_dia and t > bloqueo.get(s, -1)
            if d == 0 and s != 0 and permitido and np.isfinite(atr[t - 1]):
                monto = por_vela["fraccion"][t - 1] * port.valor(o[t])
                port.abrir(s, o[t], monto / (1 + comision), comision, fechas[t],
                           sl=o[t] - s * por_vela["sl_atr"][t - 1] * atr[t - 1],
                           tp=o[t] + s * por_vela["tp_atr"][t - 1] * atr[t - 1],
                           desliz=desliz(t, o[t], monto / o[t]))
                port.operaciones[-1]["enfriamiento"] = int(por_vela["enfriamiento"][t - 1])
                if regimen is not None:
                    port.operaciones[-1]["regimen"] = regimen[t - 1]
                abiertas_hoy += 1
                d = s
        # 2) Stop-loss y take-profit dentro de la vela t (stop-loss primero).
        if d != 0:
            op = port.operaciones[-1]
            toca_sl = l[t] <= op["sl"] if d == 1 else h[t] >= op["sl"]
            toca_tp = h[t] >= op["tp"] if d == 1 else l[t] <= op["tp"]
            if toca_sl:
                precio = min(o[t], op["sl"]) if d == 1 else max(o[t], op["sl"])
                port.cerrar(precio, comision, fechas[t], "stop-loss", desliz(t, precio, abs(port.acciones)))
                bloqueo[d] = t + op["enfriamiento"]
                d = 0
            elif toca_tp:
                precio = max(o[t], op["tp"]) if d == 1 else min(o[t], op["tp"])
                port.cerrar(precio, comision, fechas[t], "take-profit", desliz(t, precio, abs(port.acciones)))
                d = 0
        # 3) Cierre obligatorio al final de la sesión.
        if d != 0 and ultima[t]:
            port.cerrar(c[t], comision, fechas[t], "fin de sesión", desliz(t, c[t], abs(port.acciones)))
        valor[t] = port.valor(c[t])

    operaciones = pd.DataFrame(port.operaciones)
    return {"valor": pd.Series(valor, index=fechas, name="valor"),
            "operaciones": operaciones, "portafolio": port}
