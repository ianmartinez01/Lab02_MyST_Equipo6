"""Motor de backtesting event-driven con costos, stop-loss y take-profit.

Convenciones declaradas:
- La señal se genera al cierre de la vela t y la orden se ejecuta en la apertura de t+1.
- Comisión proporcional al monto operado, en cada apertura y en cada cierre.
- Sin apalancamiento: el monto de cada posición es una fracción ≤ 1 del valor del portafolio.
- Si stop-loss y take-profit caen dentro del rango de la misma vela, se ejecuta primero el
  stop-loss (convención conservadora). Si la vela abre más allá del stop, se ejecuta en la apertura.
- Toda posición se cierra en la última vela de cada sesión (sin exposición nocturna).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COMISION = 0.00125
CAPITAL_INICIAL = 1_000_000.0


@dataclass
class Portafolio:
    """Estado explícito del portafolio: efectivo, acciones (negativas en corto) y costos acumulados."""
    efectivo: float
    acciones: float = 0.0
    costos: float = 0.0
    operaciones: list = field(default_factory=list)

    def valor(self, precio: float) -> float:
        return self.efectivo + self.acciones * precio

    def abrir(self, direccion: int, precio: float, monto: float, comision: float, fecha, sl: float, tp: float):
        acciones = monto / precio
        costo = comision * monto
        self.efectivo -= direccion * acciones * precio + costo
        self.acciones = direccion * acciones
        self.costos += costo
        self.operaciones.append({"direccion": direccion, "entrada": fecha, "precio_entrada": precio,
                                 "acciones": acciones, "sl": sl, "tp": tp, "costos": costo})

    def cerrar(self, precio: float, comision: float, fecha, motivo: str):
        monto = abs(self.acciones) * precio
        costo = comision * monto
        self.efectivo += self.acciones * precio - costo
        self.costos += costo
        op = self.operaciones[-1]
        op.update({"salida": fecha, "precio_salida": precio, "motivo": motivo})
        op["costos"] += costo
        op["pnl"] = op["direccion"] * op["acciones"] * (precio - op["precio_entrada"]) - op["costos"]
        self.acciones = 0.0


def ejecutar_backtest(datos: pd.DataFrame, sl_atr: float = 1.5, tp_atr: float = 2.0, fraccion: float = 1.0,
                      comision: float = COMISION, capital: float = CAPITAL_INICIAL,
                      velas_sin_entrada: int = 3, enfriamiento: int = 0, max_por_dia: int = 99) -> dict:
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
                port.cerrar(o[t], comision, fechas[t], "cambio de régimen" if cambio_regimen else "señal contraria")
                d = 0
            permitido = velas_sin_entrada <= pos_en_dia[t] < velas_dia[t] - velas_sin_entrada
            permitido = permitido and abiertas_hoy < max_por_dia and t > bloqueo.get(s, -1)
            if d == 0 and s != 0 and permitido and np.isfinite(atr[t - 1]):
                monto = por_vela["fraccion"][t - 1] * port.valor(o[t])
                port.abrir(s, o[t], monto, comision, fechas[t],
                           sl=o[t] - s * por_vela["sl_atr"][t - 1] * atr[t - 1],
                           tp=o[t] + s * por_vela["tp_atr"][t - 1] * atr[t - 1])
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
                port.cerrar(precio, comision, fechas[t], "stop-loss")
                bloqueo[d] = t + op["enfriamiento"]
                d = 0
            elif toca_tp:
                precio = max(o[t], op["tp"]) if d == 1 else min(o[t], op["tp"])
                port.cerrar(precio, comision, fechas[t], "take-profit")
                d = 0
        # 3) Cierre obligatorio al final de la sesión.
        if d != 0 and ultima[t]:
            port.cerrar(c[t], comision, fechas[t], "fin de sesión")
        valor[t] = port.valor(c[t])

    operaciones = pd.DataFrame(port.operaciones)
    return {"valor": pd.Series(valor, index=fechas, name="valor"),
            "operaciones": operaciones, "portafolio": port}
