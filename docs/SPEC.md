# SPEC · Estrategia del Lab 02 (NVDA, 5 minutos)

Ian Carlo Escalante Martínez · 731828 · Equipo 8

## 1. Universo y frecuencia

| Concepto | Valor |
|---|---|
| Activo | NVDA (NVIDIA), precios ajustados |
| Barras | 5 minutos, solo sesión regular 09:30–16:00 ET (78 velas por sesión) |
| Fuente | Histórico público de Binance Stocks, congelado en `data/nvda_5m.csv` |
| Rango | 2 de enero a 25 de septiembre de 2026 (184 sesiones, 14,352 velas) |
| Train | 2 ene – 29 may 2026 (7,956 velas, 55%) |
| Test | 1 jun – 31 jul 2026 (3,354 velas, 23%) |
| Validation | 3 ago – 25 sep 2026 (3,042 velas, 21%) |

## 2. Idea de la estrategia

En train, NVDA ganó todo su rendimiento de un día a otro: la suma de log-retornos de la apertura al cierre fue −1.8% y la del cierre a la apertura siguiente +12.4% (paso 1 de `main.py`, `descomposicion_dia_noche()` en `src/metrics.py`). Además, el costo de ida y vuelta (0.28%) es igual al rango típico de una vela de 5 minutos (ATR ≈ 0.28% del precio), así que apostar a movimientos de pocas velas no deja margen.

De ahí salen dos decisiones de diseño:

1. Parte de día: seguir rupturas a favor de la tendencia de varias semanas, con objetivos lejanos, y mantener la posición de un día a otro en lugar de cerrar a las 15:55.
2. Compra nocturna: cuando la sesión cierra castigada, comprar al cierre y vender en la apertura siguiente.

## 3. Features

| Indicador | Ventanas base | Familia | Uso |
|---|---|---|---|
| Canal de Keltner | media 48, ATR 28 | Volatilidad | Voto de volatilidad (ruptura del canal) |
| RSI | 24 | Momento | Voto de momento (RSI contra 50) |
| TRIX | 25 | Tendencia | Voto de tendencia (signo) |
| Media de cierres diarios | 21 días | Tendencia | Filtro: solo se opera a favor de la tendencia |
| ATR | 14 | Volatilidad | Distancia de SL/TP y sizing |
| VWAP de la sesión | se reinicia cada día | Volumen | Compra nocturna: distancia del cierre al VWAP |
| Posición en el rango del día | (Close − mínimo) / (máximo − mínimo) de la sesión | Momento | Compra nocturna |
| RSI | 14 | Momento | Compra nocturna |

Keltner, RSI, TRIX y ATR se calculan con `ta` (`KeltnerChannel`, `RSIIndicator`, `TRIXIndicator`, `AverageTrueRange`). El VWAP de sesión (Σ precio típico · volumen / Σ volumen desde las 09:30) y la posición en el rango del día, que funciona como un Estocástico de la sesión, se calculan directamente. Las ventanas de esta tabla son el punto de partida; las finales están en la sección 5. Implementación: `calcular_indicadores()` en `src/signals.py`.

## 4. Regla de entrada

Votos al cierre de t (compra; la venta es el espejo):

- v_vol = +1 si en las últimas m = 14 velas el cierre cruzó hacia arriba la banda superior de Keltner; −1 si cruzó hacia abajo la inferior (0 si hubo las dos).
- v_mom = +1 si RSI_t > 50; −1 si RSI_t < 50.
- v_ten = +1 si TRIX_t > 0; −1 si TRIX_t < 0.

```
señal_t = +1  si  Σ_i 1[v_i,t = +1] ≥ 2
señal_t = −1  si  Σ_i 1[v_i,t = −1] ≥ 2
señal_t =  0  en otro caso
```

Filtro de tendencia: T_t = signo(cierre del día anterior − media de los últimos 21 cierres diarios), conocido desde la apertura.

```
objetivo_t = T_t            si señal_t = T_t ≠ 0                 (abre o mantiene a favor de la tendencia)
objetivo_t = 0              si señal_t = −T_t, o T_t = 0, o objetivo_{t−1} = −T_t
objetivo_t = objetivo_{t−1} en otro caso                          (la posición se mantiene, también de noche)
```

Con un solo voto a favor no se abre posición; con dos o más, sí. Implementación: `regla_confirmacion()` y `posicion_objetivo()` en `src/signals.py`.

## 5. Regla de salida

| Elemento | Valor |
|---|---|
| Stop-loss | Entrada − lado · 6.0 · ATR14 de la vela de la señal |
| Take-profit | Entrada + lado · 18.7 · ATR14 de la vela de la señal |
| Señal en contra | El objetivo pasa a 0: se cierra en la apertura siguiente, sin voltear |
| Volteo de la tendencia diaria | Se cierra en la apertura siguiente |
| Fin de sesión | La posición pasa la noche; la orden decidida a las 15:55 se ejecuta en la apertura del día siguiente |
| Holding máximo | 780 velas (10 sesiones) |
| Después de un stop, take-profit u holding máximo | No se vuelve a entrar hasta que el objetivo cambie |
| Cambio de régimen | La posición sigue mientras el objetivo con el θ del régimen nuevo no cambie; conserva su stop y su objetivo |

Los valores de esta sección son el punto de partida de la búsqueda (`PARAMETROS_BASE` y `THETA_OPERACION`). La optimización tiene dos etapas, ambas con todo train (enero a mayo):

1. Ventanas de los indicadores (Keltner, ATR del canal, RSI, TRIX, memoria y días de tendencia): un solo θ para todo train, con la parte de día, 200 pruebas TPE y al menos 30 operaciones (`optimizar_ventanas()`, `RANGOS_VENTANAS`).
2. Por régimen, con esas ventanas fijas: stop, objetivo, riesgo, umbral del VWAP y tamaño de la compra nocturna (`RANGOS`), 200 pruebas TPE y 200 de random search por régimen.

Las ventanas no se optimizan por régimen: con 3 a 4 operaciones por régimen al mes, en el walk-forward de 1 mes ajustaban ruido. El θ* final queda en `docs/theta_congelado.json`:

| Ventanas (etapa 1, todos los regímenes) | Keltner | ATR del canal | RSI | TRIX | Memoria | Días de tendencia |
|---|---:|---:|---:|---:|---:|---:|
| θ* | 50 | 26 | 15 | 28 | 15 | 20 |

| Régimen (etapa 2) | Stop (ATR) | Take-profit (ATR) | Riesgo por operación | Umbral VWAP | Tamaño nocturno |
|---|---:|---:|---:|---:|---:|
| Tendencia | 7.73 | 7.07 | 0.26% | 0.31% | 98% |
| Reversión | 3.29 | 8.89 | 0.77% | 0.47% | 27% |
| Crisis | 2.58 | 21.39 | 0.27% | 0.55% | 97% |

## 6. Compra nocturna

Complemento de la parte de día: solo actúa al cierre. Cuando NVDA cierra la sesión castigada, la apertura siguiente tiende a rebotar. Con θ* la compra nocturna ganó 5.98% del capital en 18 noches de train (66.7% ganadoras), 3.80% en 12 noches de test y 1.13% en 10 noches de validation (`resultados/dia_noche.csv`).

Votos al cierre de la vela de las 15:50:

- n_vol = 1 si Close / VWAP_sesión − 1 ≤ −u, con u = 0.5% (volumen).
- n_rng = 1 si el cierre quedó en el tercio inferior del rango del día: (Close − mínimo) / (máximo − mínimo) < 1/3 (momento).
- n_rsi = 1 si RSI(14) < 40 (momento).

```
noche_t = +1  si  hora_t = 15:50  y  n_vol + n_rng + n_rsi ≥ 2
noche_t =  0  en otro caso
```

| Elemento | Regla |
|---|---|
| Entrada | Compra en la apertura de la vela de las 15:55. Si la parte de día está corta, el corto se cierra en esa misma apertura; si está larga, la posición de día sigue (ya pasa la noche comprada) |
| Tamaño | Fracción fija del equity (`tamano_noche`, base 100%, optimizable de 25% a 100%), sin apalancamiento. No depende del stop porque la posición sale siempre en la apertura siguiente |
| Salida | Venta en la apertura de la primera vela del día siguiente (09:30) |
| Stop-loss | El del régimen (sl_atr · ATR14), revisado dentro de la vela de las 15:55; en la apertura siguiente la posición sale de todos modos, así que un hueco en contra se llena a la apertura |
| Lado | Solo compra: el espejo (vender cuando cierra muy por arriba del VWAP) no pagaba la comisión en train |
| Frecuencia | Máximo una operación nocturna por día |
| Optimización | u (0.3% a 0.8%) y el tamaño se optimizan por régimen; los umbrales de rango (1/3) y RSI (40) quedan fijos |
| Stop de día | Un stop de la compra nocturna no bloquea la reentrada de la parte de día |

La regla simple (solo n_vol) se reporta como comparación en la tabla de reglas, igual que las variantes de un solo voto de la parte de día. Implementación: `senal_nocturna()` en `src/signals.py` y el bloque nocturno de `backtest()` en `src/backtest.py`.

## 7. Sizing

Parte de día: riesgo fijo por operación, tocar el stop cuesta `riesgo` del equity (base 0.5%, $5,000 con $1,000,000; θ* entre 0.26% y 0.77% según el régimen).

```
acciones = riesgo · equity / |entrada − stop|,  recortado a equity / (precio · (1 + comisión))
```

Compra nocturna: acciones = tamaño nocturno · equity / precio (θ* entre 27% y 98%), con el mismo recorte.

Sin apalancamiento: ninguna posición supera el equity disponible y nunca hay dos posiciones abiertas. Implementación: `tamano_por_riesgo()` y `abrir()` en `src/backtest.py`.

## 8. Costos

| Costo | Valor | Fuente o justificación |
|---|---|---|
| Comisión | 0.125% por lado, también en la compra nocturna | Fijada por el enunciado del Lab 02 |
| Spread | $0.005 por acción por lado | Medio spread de 1 centavo, el tick de NVDA en sesión regular |
| Impacto | σ_diaria · √(acciones / volumen diario promedio de 20 sesiones) | Ley de raíz cuadrada del impacto; en train promedia 1.36 pb por lado (spread incluido) |
| Préstamo para cortos | 0.5% anual sobre el valor del corto, cobrado por vela | Tasa típica de una acción fácil de pedir prestada; ahora importa porque los cortos pasan la noche |

Costo total ida y vuelta: 2 · (0.125% + 0.0136%) = 0.277% del monto, más el préstamo de los cortos.

## 9. Convenciones

- Se actúa sobre la señal en la apertura de la vela siguiente (t+1); nada de la vela t+1 entra en la señal.
- Empate intrabar: si stop y take-profit caen en la misma vela, se ejecuta el stop (convención conservadora). Si la vela abre más allá de un nivel, se llena a la apertura: un hueco de apertura en contra puede costar más que el riesgo presupuestado.
- Una sola posición abierta a la vez.
- Cada periodo evaluado (ventana de entrenamiento, semana de prueba, validation) arranca sin posición y liquida la que quede abierta al cierre de su última vela, con costos.

## 10. Win rate de equilibrio

Con ATR mediano en train de 0.278% del precio y costo ida y vuelta C = 0.277%:

```
ganancia W = 18.7 · 0.278% − 0.277% = 4.92%
pérdida  L =  6.0 · 0.278% + 0.277% = 1.95%
p* = L / (W + L) = 28.3%
```

Con la configuración base la estrategia necesita ganar más del 28.3% de sus operaciones (24.3% sin costos); con θ* hace falta 59.0% en tendencia (objetivo cerca y stop lejos), 35.2% en reversión y 14.9% en crisis (objetivo de 21 ATR). Con objetivos de 2 ATR y stops de 1.5 ATR haría falta 71.3%: con objetivos lejanos el costo pesa mucho menos. Implementación: `winrate_equilibrio()` en `src/metrics.py`.
