# Laboratorio 02: Estrategia de trading con análisis técnico

- Ian Carlo Escalante Martínez

**Nivel de alcance: B** (estrategia multi-indicador, backtesting, optimización, walk-forward y detección dinámica de régimen de mercado).

## Descripción

Este proyecto desarrolla una estrategia sistemática sobre NVDA en velas de 5 minutos (sesión regular, enero a septiembre de 2026) con dos partes. La parte de día combina tres indicadores de tres familias, el canal de Keltner (volatilidad), el RSI (momento) y el TRIX (tendencia): abre cuando al menos 2 de 3 coinciden y solo a favor de la tendencia de los últimos 20 días, y mantiene la posición de un día a otro hasta que la señal se voltea o toca su stop-loss o take-profit. La compra nocturna compra a las 15:55 y vende en la apertura siguiente cuando la sesión cerró castigada, también con una regla 2 de 3 (distancia al VWAP, posición en el rango del día y RSI). Se evalúa en un motor event-driven con comisión de 0.125% por lado, spread, impacto de mercado y préstamo de acciones para cortos; los hiperparámetros se optimizan con Optuna maximizando el Calmar Ratio, con parámetros distintos para cada régimen detectado por K-means (tendencia, reversión y crisis), y se valida con un walk-forward de 1 mes de entrenamiento y 1 semana de prueba.

## Estructura del proyecto

```text
Lab02_MyST_Equipo8/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   └── nvda_5m.csv
├── src/
│   ├── data.py
│   ├── signals.py
│   ├── backtest.py
│   ├── metrics.py
│   ├── optimize.py
│   ├── regimes.py
│   └── plots.py
├── tests/
│   ├── conftest.py
│   ├── test_signals.py
│   ├── test_backtest.py
│   ├── test_metrics.py
│   └── test_regimes.py
├── notebooks/
│   └── analysis.ipynb
└── docs/
    ├── SPEC.md
    ├── theta_congelado.json
    ├── figures/
    ├── reporte.pdf
    └── presentacion.pdf
```

## Instalación

Requiere Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproducción de resultados

Para ejecutar el proyecto completo (datos, optimización, regímenes, walk-forward, validation, robustez y figuras) con un solo comando:

```bash
python main.py
```

El proyecto tarda alrededor de 10 minutos con 8 núcleos; la optimización y el walk-forward corren en paralelo. Las figuras se guardan en `docs/figures/` y las tablas en `resultados/`.

θ* se optimiza solo con train y se congela en `docs/theta_congelado.json` antes de evaluar validation. Para recalcularlo sin tocar validation:

```bash
python main.py --congelar
```

`python main.py` lee ese archivo, verifica que coincide con el recalculado y después evalúa test y validation.

Para ejecutar las pruebas automáticas:

```bash
python -m pytest -v
```

Los datos ya están congelados en `data/nvda_5m.csv`. Para volver a descargarlos:

```bash
python -m src.data --ticker NVDA --start 2026-01-01 --end 2026-09-26 --interval 5m
```

La semilla aleatoria utilizada es `42`, definida en `main.py` y en `src/optimize.py`. Cada estudio de Optuna usa `TPESampler` o `RandomSampler` con semilla `42 + desplazamiento` por etapa, régimen y ventana, y K-means usa `random_state=42`, por lo que los resultados son reproducibles.

## Datos

| Concepto | Valor |
|---|---|
| Activo | NVDA (NVIDIA Corporation), en lugar de BTCUSDT, con autorización del profesor |
| Fuente | Histórico público de velas de Binance Stocks, precios ajustados |
| Frecuencia | 5 minutos, sesión regular 09:30–16:00 ET (78 velas por día) |
| Train | 2 de enero a 29 de mayo de 2026 (7,956 velas, 55%) |
| Test | 1 de junio a 31 de julio de 2026 (3,354 velas, 23%) |
| Validation | 3 de agosto a 25 de septiembre de 2026 (3,042 velas, 21%) |
| Auditoría | 0 duplicados, 0 nulos, 0 velas con OHLC incoherente, 0 días incompletos |

Yahoo Finance solo conserva 60 días de velas de 5 minutos, por eso la descarga usa el histórico de Binance Stocks. Las primeras 32 sesiones de train sirven de calentamiento para la tendencia diaria; las métricas de train se miden desde el 17 de febrero.

## Estrategia

La idea sale de los datos: la suma de log-retornos de NVDA de la apertura al cierre fue −1.8% en train, −6.5% en test y −7.2% en validation, y la del cierre a la apertura siguiente +12.4%, −0.7% y +20.1%. Una estrategia que cierra cada día a las 15:55 se queda con la parte que pierde. Además, el costo de ida y vuelta (0.28%) es igual al rango típico de una vela de 5 minutos, así que la estrategia busca movimientos grandes y mantiene la posición de un día a otro.

**Parte de día.** Votos al cierre de la vela t (la venta es el espejo):

| Voto | Indicador | Familia | Compra (+1) |
|---|---|---|---|
| Volatilidad | Canal de Keltner (50, ATR 26) | Volatilidad | El cierre rompió la banda superior en las últimas 15 velas |
| Momento | RSI 15 | Momento | RSI_t > 50 |
| Tendencia | TRIX 28 | Tendencia | TRIX_t > 0 |

señal_t = +1 si Σ 1[v_i,t = +1] ≥ 2; −1 si Σ 1[v_i,t = −1] ≥ 2; 0 en otro caso.

Filtro de tendencia: T_t = signo(cierre de ayer − media de los últimos 20 cierres diarios). Posición objetivo: objetivo_t = T_t si señal_t = T_t; 0 si señal_t = −T_t, si T_t = 0 o si objetivo_{t−1} = −T_t; objetivo_{t−1} en otro caso. La señal a favor de la tendencia abre, la señal en contra cierra sin voltear, y si no hay señal la posición se mantiene, también de noche.

**Compra nocturna.** Al cierre de la vela de las 15:50: n_vol = 1[Close / VWAP_sesión − 1 ≤ −u], n_rng = 1[posición en el rango del día < 1/3], n_rsi = 1[RSI(14) < 40]. Si n_vol + n_rng + n_rsi ≥ 2, se compra en la apertura de las 15:55 y se vende en la apertura del día siguiente. Si la parte de día está corta a esa hora, el corto se cierra; si está larga, sigue.

La señal se calcula con información hasta el cierre de la vela t y la orden se ejecuta en la apertura de t+1. La especificación completa está en `docs/SPEC.md`.

## Parámetros fijos del problema

| Parámetro | Valor |
|---|---|
| Capital inicial | $1,000,000 |
| Comisión | 0.125% por lado, en cada apertura y cada cierre |
| Spread | Medio spread de $0.005 por acción en cada entrada y salida |
| Impacto de mercado | Ley de raíz cuadrada: σ_diaria × √(acciones / volumen diario promedio de 20 sesiones) |
| Préstamo para cortos | 0.5% anual sobre el valor del corto, por vela |
| Sizing | Parte de día: tocar el stop cuesta el `riesgo` del equity. Compra nocturna: fracción fija del equity. Sin apalancamiento |
| Salidas de día | Señal en contra, volteo de la tendencia, stop-loss, take-profit y holding máximo de 10 sesiones |
| Empate intrabar | Si stop y take-profit caen en la misma vela, se ejecuta primero el stop; si la vela abre más allá, se llena a la apertura |
| Cambio de régimen | La posición sigue mientras el objetivo con el θ del régimen nuevo no cambie y conserva su stop y su objetivo |
| Anualización | 78 velas × 252 sesiones = 19,656 velas por año, tasa libre de riesgo 0 |

Win rate de equilibrio p* = L / (W + L), con W = tp·ATR − C y L = sl·ATR + C (C = 0.276% de costo ida y vuelta, ATR = 0.278% del precio): 59.0% con θ* de tendencia, 35.2% con el de reversión y 14.9% con el de crisis.

## Detección de régimen

Se compararon tres clasificadores ajustados solo con train, sobre ln(volatilidad realizada) y √(eficiencia de Kaufman) en una ventana móvil de 1 semana, reclasificando cada 4 horas de sesión. El HMM etiqueta con probabilidades filtradas (algoritmo forward); la ruta de Viterbi coincide con la filtrada en el 98.2% de train, pero usa el futuro y solo se muestra para comparar.

| Clasificador | Silhouette train / test | Duración media (h) train / test | Transiciones por mes train / test |
|---|---|---|---|
| Reglas (percentiles de train) | 0.283 / 0.195 | 17.4 / 12.7 | 7.6 / 10.3 |
| K-means | 0.387 / 0.290 | 13.6 / 17.5 | 9.8 / 7.3 |
| HMM filtrado | 0.360 / 0.128 | 17.9 / 17.5 | 7.4 / 7.3 |

Se eligió K-means: mejor silhouette en train y en test, y es el único que conserva los tres regímenes fuera de muestra (el HMM detecta tendencia solo en 1.4% de test).

| Métrica (K-means) | Train | Test | Validation |
|---|---:|---:|---:|
| Silhouette | 0.387 | 0.290 | 0.267 |
| Duración media del régimen (horas de sesión) | 13.6 | 17.5 | 14.9 |
| Transiciones por mes | 9.8 | 7.3 | 8.6 |
| Tiempo en tendencia / reversión / crisis | 26% / 32% / 43% | 13% / 46% / 41% | 38% / 22% / 40% |

La persistencia supera el objetivo de 12 horas en los tres conjuntos; la separación (silhouette) queda por debajo del objetivo de 0.4 y se degrada fuera de muestra.

## Optimización

| Etapa | Qué se optimiza | Datos | Método | Restricción |
|---|---|---|---|---|
| 1. Ventanas | Keltner, ATR del canal, RSI, TRIX, memoria y días de tendencia, un solo θ, con la parte de día | Todo train | TPE, 200 pruebas | Al menos 30 operaciones |
| 2. Por régimen | Stop, take-profit, riesgo, umbral del VWAP y tamaño de la compra nocturna | Todo train, solo las velas del régimen | TPE y random search, 200 pruebas cada uno | Al menos 10 operaciones abiertas en el régimen |

El objetivo es el Calmar Ratio con comisión, spread e impacto; una configuración que no cumple la restricción vale −10. θ* se toma del centro de la mejor meseta (vecindario de 10 trials), no del argmax, y salió solo del TPE.

| Régimen | Mejor Calmar random search | Mejor Calmar TPE | Calmar de la meseta | Stop / take-profit (ATR) | Riesgo | Umbral VWAP | Tamaño nocturno |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tendencia | 3.51 | 10.91 | 10.02 | 7.73 / 7.07 | 0.26% | 0.31% | 98% |
| Reversión | 6.00 | 8.63 | 7.62 | 3.29 / 8.89 | 0.77% | 0.47% | 27% |
| Crisis | 14.74 | 18.07 | 17.33 | 2.58 / 21.39 | 0.27% | 0.55% | 97% |

Ventanas de la etapa 1 (mejor Calmar 10.39, meseta 8.85): Keltner 50, ATR 26, RSI 15, TRIX 28, memoria 15 y 20 días de tendencia.

En total se evaluaron 11,300 configuraciones en unos 8 minutos con 8 núcleos: 200 en la etapa 1 (25 s), 1,200 en la etapa 2 y 9,900 en el walk-forward (323 s).

## Walk-forward

Entrenamiento de 1 mes, prueba de 1 semana, paso semanal, sobre train + test, con embargo de 1 sesión. En cada ventana se optimiza por régimen con 150 pruebas TPE y al menos 4 operaciones por régimen, con las ventanas de la etapa 1 fijas; cada semana de prueba arranca sin posición y liquida la suya al final.

| Ventanas | Configuraciones | Anualizado dentro | Anualizado fuera | Walk-forward efficiency | Semanas positivas |
|---:|---:|---:|---:|---:|---:|
| 24 | 9,900 | 85.09% | −25.62% | −0.30 | 6 de 24 |

## Resultados

θ* optimizado con train y congelado; test y validation quedan fuera de muestra.

| Métrica | Train | B&H train | Test | B&H test | Validation | B&H validation |
|---|---:|---:|---:|---:|---:|---:|
| Retorno total | 14.25% | 13.04% | −1.99% | −7.88% | 1.10% | 12.84% |
| Volatilidad anualizada | 11.87% | 35.92% | 12.66% | 38.32% | 20.99% | 35.20% |
| Sharpe | 4.10 | 1.41 | −0.87 | −1.06 | 0.44 | 2.39 |
| Sortino | 8.75 | 2.01 | −1.60 | −1.44 | 0.79 | 3.84 |
| Calmar | 26.35 | 3.30 | −2.75 | −2.14 | 1.17 | 10.91 |
| Máximo drawdown | −2.34% | −16.81% | −4.05% | −17.88% | −6.23% | −10.84% |
| Win rate | 48.4% | | 34.0% | | 29.4% | |
| Payoff ratio | 2.18 | | 1.65 | | 2.59 | |
| Operaciones | 62 | | 47 | | 34 | |
| Turnover anual (× equity) | 240 | | 240 | | 227 | |

Aporte de cada parte (P&L como porcentaje del capital inicial):

| Conjunto | Día: operaciones | Día: P&L | Noche: operaciones | Noche: P&L | Noche: win rate |
|---|---:|---:|---:|---:|---:|
| Train | 44 | +8.27% | 18 | +5.98% | 66.7% |
| Test | 35 | −5.80% | 12 | +3.80% | 58.3% |
| Validation | 24 | −0.03% | 10 | +1.13% | 40.0% |

## Preguntas de análisis

### 1. ¿Qué aporta la regla de confirmación de 2 de 3 frente a usar un solo indicador?

En validation, la regla 2 de 3 abrió 34 operaciones con Calmar 1.17 y retorno de 1.10%. Solo RSI abrió 179 operaciones y perdió 24.13% (Calmar −3.44): el RSI contra 50 cambia de lado constantemente y los costos se lo comen. Solo Keltner abrió 29 (Calmar 1.41) y solo TRIX 31 (0.14). La regla estricta, 3 de 3, abrió 27 con Calmar 3.63. Sin la compra nocturna, el Calmar baja de 1.17 a 0.65. La confirmación sirve sobre todo como filtro de frecuencia: evita las operaciones de un indicador ruidoso.

### 2. ¿Cuánto se degrada el desempeño entre entrenamiento y prueba en el walk-forward?

El rendimiento anualizado pasa de 85.09% dentro de muestra a −25.62% fuera (walk-forward efficiency −0.30), el Calmar mediano cae de 66.27 a −19.70 y solo 6 de 24 semanas fueron positivas. Con un mes de historia cada régimen junta 3 a 4 operaciones y la compra nocturna 1 o 2 noches, así que la optimización por ventana sigue al ruido; no sobrevive la ventaja. Por eso θ* se optimiza con los cinco meses de train, donde cada régimen junta de 15 a 25 operaciones: con ese θ* la estrategia pierde 1.99% en test frente a −7.88% de buy & hold y gana 1.10% en validation.

### 3. ¿Qué tan sensible es la estrategia a una variación de ±20% en sus parámetros?

En validation el Calmar base es 1.17. Los parámetros de la compra nocturna (umbral del VWAP y tamaño) y el RSI forman una meseta (1.07 a 2.20). Los más sensibles son el stop (−20% lo lleva a −2.18), los días de tendencia (+20% a −0.33), el TRIX (−20% a −0.01) y la media de Keltner (−20% a 0.28). La parte de día está cerca de un borde; la compra nocturna no.

### 4. ¿A qué nivel de costo de transacción deja de ser rentable?

Con costo cero la estrategia gana 6.12% en validation; deja de ser rentable entre 30 y 35 pb de ida y vuelta (con 30 pb gana 0.67%, con 35 pb pierde 0.21%). El costo modelado es de 27.6 pb (2 × 0.125% de comisión + 1.3 pb de spread e impacto por lado), así que el margen de seguridad es de unos 3 a 7 pb de ida y vuelta, entre 1.5 y 3.5 pb por lado frente a la comisión de 0.125%. En validation pagó $45,357 de comisiones y $4,718 de spread e impacto.

### 5. ¿El desempeño difiere de forma significativa entre regímenes?

En validation, reversión ganó $35,817 (7 operaciones, win rate 57%), crisis perdió $8,517 (10, 20%) y tendencia perdió $16,329 (17, 24%). Kruskal-Wallis sobre el retorno por operación en train, test y validation: H = 0.32, p = 0.85, sin diferencia significativa. La capa de régimen no cambia la señal sino el riesgo: el tamaño nocturno baja a 27% en reversión y el stop pasa de 2.6 ATR en crisis a 7.7 ATR en tendencia. Sirve como control de exposición, no como fuente de rentabilidad.

### 7. Tres limitaciones para operar con capital real

1. La ventaja de la parte de día no sobrevive fuera de muestra: gana 8.27% en train y pierde en test y validation. Lo que gana en los tres conjuntos es la compra nocturna, con solo 10 a 18 noches por periodo.
2. La compra nocturna está expuesta a huecos de apertura: el stop no funciona con el mercado cerrado y una noticia nocturna se llena completa a la apertura.
3. Un solo activo, elegido hoy y ganador en la muestra (sesgo de supervivencia), con nueve meses de datos de un tercero.

Advertencia de interpretación: el backtest asume ejecución completa al precio modelado. Incluye comisión, spread e impacto de raíz cuadrada, que el modelo estima en ≈ 1.3 pb por lado ($4,718 en validation), pero no incluye fallas de ejecución ni llenados parciales. La compra nocturna opera hasta el 98% del equity en la vela de las 15:55; esa orden es alrededor del 0.1% del volumen de la vela (máximo 0.18%), así que el impacto real debería ser chico, pero un hueco de apertura o una vela de poco volumen se ejecutarían peor.

## Uso de inteligencia artificial

Se utilizó asistencia de IA (Claude) para: estructurar el proyecto según lo especificado, implementar el motor de backtesting con costos, la optimización en dos etapas con Optuna (random search y TPE), el walk-forward, la validación con θ* congelado y la detección de régimen, escribir las pruebas y las figuras, y organizar este README. La idea de la estrategia y la compra nocturna fueron propuestas por el autor. Todo el código y los resultados numéricos fueron revisados y ejecutados por el autor, que es responsable de explicar cualquier parte del proyecto entregado.
