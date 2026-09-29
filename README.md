# Laboratorio 02: Estrategia de trading con análisis técnico

- Ian Carlo Escalante Martínez
- Abdon Islas

**Nivel de alcance: B** (estrategia multi-indicador, backtesting, optimización, walk-forward y detección dinámica de régimen de mercado).

## Descripción

Este proyecto desarrolla una estrategia sistemática sobre NVDA en velas de 5 minutos (sesión regular, enero a septiembre de 2026) que combina cinco indicadores técnicos: Bandas de Bollinger, RSI, Estocástico, MACD y EMA 9. Tres votos de preparación (volatilidad, momento y tendencia) deben coincidir al menos 2 de 3 y el precio debe cruzar la EMA 9 para abrir una posición larga o corta. La estrategia se evalúa en un motor de backtesting event-driven con comisión de 0.125% por lado, stop-loss y take-profit, y sus hiperparámetros se optimizan con Optuna maximizando el Calmar Ratio en un walk-forward de 1 mes de entrenamiento y 1 semana de prueba, con parámetros distintos para cada régimen detectado por K-means (tendencia, reversión y crisis).

## Estructura del proyecto

```text
Lab02_MyST_Equipo6/
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
    ├── figures/
    ├── theta_congelado.json
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

Para ejecutar el proyecto completo (datos, modelo final, regímenes, walk-forward anchored y rolling, validation, robustez y figuras) con un solo comando:

```bash
python main.py
```

El walk-forward corre en paralelo en todos los núcleos disponibles y el proyecto completo tarda alrededor de 15 a 20 minutos. Las figuras se guardan en `docs/figures/` y las tablas en `resultados/`.

θ* se calcula solo con train + test y se congela en `docs/theta_congelado.json` antes de evaluar validation. Para recalcularlo sin tocar validation:

```bash
python main.py --congelar
```

`python main.py` lee ese archivo, verifica que coincide con el recalculado y evalúa validation una sola vez.

Para ejecutar las pruebas automáticas:

```bash
python -m pytest -v
```

Los datos ya están congelados en `data/nvda_5m.csv`. Para volver a descargarlos:

```bash
python -m src.data --ticker NVDA --start 2026-01-01 --end 2026-09-26 --interval 5m
```

La semilla aleatoria utilizada es `42`, definida en `main.py` y en `src/optimize.py`. Cada estudio de Optuna usa `TPESampler` o `RandomSampler` con semilla `42 + desplazamiento` por ventana y régimen, y K-means usa `random_state=42`, por lo que los resultados son reproducibles.

## Datos

| Concepto | Valor |
|---|---|
| Activo | NVDA (NVIDIA Corporation) |
| Fuente | Histórico público de velas de Binance Stocks, precios ajustados |
| Frecuencia | 5 minutos, sesión regular 09:30–16:00 ET (78 velas por día) |
| Train | 2 de enero a 29 de mayo de 2026 (7,956 velas, 55%) |
| Test | 1 de junio a 31 de julio de 2026 (3,354 velas, 23%) |
| Validation | 3 de agosto a 25 de septiembre de 2026 (3,042 velas, 21%) |
| Auditoría | 0 duplicados, 0 nulos, 0 velas con OHLC incoherente, 0 días incompletos |

Yahoo Finance solo conserva 60 días de velas de 5 minutos, por eso la descarga usa el histórico de Binance Stocks.

## Estrategia

| Voto | Indicador | Compra (+1) | Venta (−1) |
|---|---|---|---|
| Volatilidad | Bollinger (20, 2) | Close < banda inferior | Close > banda superior |
| Momento | RSI 14 o Estocástico (14, 1, 3) | RSI cruza abajo de 30, o %K cruza arriba de %D con %K < 20 | RSI cruza arriba de 70, o %K cruza abajo de %D con %K > 80 |
| Tendencia | MACD (12, 26, 9) / ATR 14 | < −0.8 | > 0.8 |

Cada evento sigue vigente N velas (memoria). Regla de confirmación:

señal_t = +1 si Σ 1[v_i,t = +1] ≥ 2 y Close_t cruza arriba de EMA9_t; −1 si Σ 1[v_i,t = −1] ≥ 2 y Close_t cruza abajo de EMA9_t; 0 en otro caso.

La señal se calcula con información hasta el cierre de la vela t y la orden se ejecuta en la apertura de t+1.

## Parámetros fijos del problema

| Parámetro | Valor |
|---|---|
| Capital inicial | $1,000,000 |
| Comisión | 0.125% por lado, en cada apertura y cada cierre |
| Apalancamiento | No permitido (fracción del capital ≤ 1) |
| Posiciones | Largas y cortas |
| Stop-loss y take-profit en la misma vela | Se ejecuta primero el stop-loss |
| Cierre | Toda posición se cierra en la última vela de la sesión |
| Entradas | No se abren posiciones en los primeros ni en los últimos 15 minutos |
| Spread | Medio spread de $0.005 por acción en cada entrada y salida |
| Impacto de mercado | Ley de raíz cuadrada: σ_diaria × √(acciones / volumen diario promedio de 20 sesiones) |
| Anualización | 78 velas × 252 sesiones = 19,656 velas por año, tasa libre de riesgo 0 |

## Detección de régimen

K-means (k = 3) sobre ln(volatilidad realizada) y √(eficiencia de Kaufman), calculadas en una ventana móvil de 1 semana (390 velas) y actualizadas cada 4 horas de sesión (48 velas). Crisis es el grupo de mayor volatilidad; de los otros dos, tendencia es el de mayor eficiencia y reversión el restante. Al cambiar el régimen, la posición abierta se cierra en la apertura siguiente. El modelo final se ajusta con train + test.

| Métrica | Train | Test | Validation | Objetivo |
|---|---:|---:|---:|---:|
| Silhouette | 0.376 | 0.470 | 0.410 | > 0.4 |
| Duración media del régimen (horas de sesión) | 13.3 | 17.5 | 16.9 | > 12 |
| Transiciones por semana | 2.4 | 1.7 | 1.8 | |
| Tiempo en tendencia / reversión / crisis | 32% / 30% / 38% | 13% / 46% / 41% | 46% / 19% / 35% | |

## Optimización

| Elemento | Valor |
|---|---|
| Objetivo | θ* = argmax Calmar(backtest(θ)) con comisión, spread e impacto |
| Espacio de búsqueda | 11 parámetros por régimen, con rango y justificación en `RANGOS` de `src/optimize.py` |
| Restricción | Menos de 10 operaciones por régimen en la ventana (≈ 30 por ventana) → Calmar = −10 |
| Selección de θ* | Centro de la mejor meseta: trial cuyo vecindario de 10 trials tiene el Calmar promedio más alto |
| Régimen apagado | Si su mejor Calmar no es positivo, no se opera en ese régimen |
| Modelo final (julio) | Random search N = 200 y TPE N = 200 por régimen |

| Régimen | Mejor Calmar random search | Mejor Calmar TPE | Calmar de la meseta | θ* | Estado |
|---|---:|---:|---:|---|---|
| Tendencia | −0.06 | −2.56 | −3.19 | meseta | apagado |
| Reversión | 13.90 | 59.38 | 56.71 | meseta | activo |
| Crisis | −4.63 | 11.68 | 8.72 | meseta | activo |

En los tres regímenes el argmax del TPE está aislado de sus vecinos, así que θ* se tomó del centro de la meseta. Las dos dimensiones más importantes (fANOVA) definen la superficie 3D de cada régimen; por ejemplo, fracción del capital y ventana de Bollinger en reversión.

θ* quedó congelado en `docs/theta_congelado.json` en el commit `9a11fab6634e00a0848a1defbce71a36763eead9`, antes de evaluar validation.

## Walk-forward

Entrenamiento de 1 mes (rolling) o desde el inicio (anchored), prueba de 1 semana, paso semanal, sobre train + test. Purga: señales y variables de régimen solo miran hacia atrás y toda posición se cierra el mismo día, por lo que ninguna observación de entrenamiento usa datos de la prueba. Embargo: se descarta la última sesión (78 velas) antes de cada semana de prueba.

| Variante | Ventanas | Configuraciones | Tiempo | Anualizado dentro | Anualizado fuera | Walk-forward efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Rolling (1 mes) | 25 | 10,350 | 315 s | 55.37% | −45.48% | −0.82 |
| Anchored | 25 | 11,250 | 404 s | 28.47% | −23.60% | −0.83 |

## Resultados

Train y test son las semanas fuera de muestra del walk-forward rolling en cada periodo; validation es θ* congelado evaluado una sola vez.

| Métrica | Train (WF) | B&H train | Test (WF) | B&H test | Validation | B&H validation |
|---|---:|---:|---:|---:|---:|---:|
| Retorno total | −15.31% | 12.73% | −11.55% | −7.88% | −7.23% | 12.84% |
| Rendimiento anualizado | −41.95% | 48.04% | −51.29% | −38.18% | −38.42% | 118.28% |
| Volatilidad anualizada | 6.98% | 35.99% | 9.86% | 38.32% | 6.51% | 35.20% |
| Sharpe | −7.75 | 1.27 | −7.24 | −1.06 | −7.41 | 2.39 |
| Sortino | −9.65 | 1.83 | −9.01 | −1.44 | −9.49 | 3.84 |
| Calmar | −2.64 | 2.86 | −3.87 | −2.14 | −5.32 | 10.91 |
| Máximo drawdown | −15.88% | −16.81% | −13.25% | −17.88% | −7.23% | −10.84% |
| Win Rate | 23.4% | | 19.4% | | 35.3% | |
| Payoff ratio | 0.59 | | 1.12 | | 0.48 | |
| Operaciones | 64 | | 31 | | 34 | |

Las tablas de retornos mensuales, trimestrales y anuales están en `docs/figures/04_tabla_retornos.png`.

## Preguntas de análisis

### 1. ¿Qué aporta la regla de confirmación de 2 de 3 frente a usar un solo indicador?

En validation, con θ* congelado, la regla 2 de 3 abrió 34 operaciones con Calmar −5.32. Solo Bollinger abrió 44 (Calmar −4.84), solo momento 48 (−4.86) y solo MACD 31 (−5.40). La confirmación filtra operaciones frente a Bollinger y momento, pero no mejora el Calmar: todas las variantes pierden. La versión estricta (los tres votos) abrió 24 con Calmar −3.88, la menos mala porque opera menos.

### 2. ¿Cuánto se degrada el desempeño entre entrenamiento y prueba en el walk-forward?

El rendimiento anualizado pasa de 55.37% dentro de muestra a −45.48% fuera en rolling (walk-forward efficiency −0.82) y de 28.47% a −23.60% en anchored (−0.83). El Calmar mediano cae de 20.53 a −23.40 y solo 2 de 25 semanas fueron positivas. Una eficiencia negativa significa que no sobrevive nada de la ventaja: el resultado dentro de muestra es ruido ajustado. Anchored pierde menos (−12.0% contra −25.1% acumulado) porque su ventana más larga cambia menos los parámetros entre semanas, pero la eficiencia es igual, así que no hay un edge estable que se degrade con el tiempo.

### 3. ¿Qué tan sensible es la estrategia a una variación de ±20% en sus parámetros?

En validation el Calmar se mueve entre −5.53 y −5.12 al variar cualquier parámetro ±20%. Es una meseta, no un pico aislado, pero es una meseta de pérdidas. En la optimización de julio, en cambio, los slice plots muestran el argmax separado de sus vecinos en los tres regímenes, por lo que θ* se tomó del centro de la meseta.

### 4. ¿A qué nivel de costo deja de ser rentable?

La estrategia no es rentable en ningún nivel de costo: con comisión cero pierde 2.51% en validation y con 0.125% pierde 7.23%. En validation las comisiones sumaron $47,851 y el spread con impacto $5,495. No hay margen de seguridad frente al 0.125%.

### 5. ¿El desempeño difiere de forma significativa entre regímenes?

En validation, crisis abrió 24 operaciones (win rate 41.7%, P&L −$49,630), reversión 10 (20.0%, −$20,289) y tendencia quedó apagada durante el 46% del tiempo. La prueba de Kruskal-Wallis sobre el retorno por operación (walk-forward y validation) da H = 0.36 y p = 0.835: la diferencia no es significativa. La capa de régimen aporta control de exposición: apaga la estrategia en tendencia y cierra posiciones en cada transición.

### 6. Tres limitaciones para operar con capital real

1. La ventaja no sobrevive fuera de muestra: walk-forward efficiency de −0.82 y validation negativo.
2. Nueve meses de velas de 5 minutos dejan pocos episodios por régimen y el optimizador encuentra picos aislados.
3. Los datos provienen de un tercero (Binance Stocks) y la estrategia concentra el capital en un solo activo.

Advertencia de interpretación: el backtest incluye comisión, spread e impacto de raíz cuadrada, pero asume ejecución completa sin fallas. Una orden de $1,000,000 equivale a unas 4,973 acciones, alrededor de 0.41% del volumen de una vela típica de 5 minutos. El modelo de impacto estima unos 2 puntos base por lado ($5,495 en las 34 operaciones de validation); un impacto temporal mayor en velas de poco volumen o stops disparados por huecos se ejecutarían peor de lo modelado.

## Uso de inteligencia artificial

Se utilizó asistencia de IA (Claude) para: estructurar el proyecto según lo especificado, implementar el motor de backtesting con costos, la optimización con Optuna (random search y TPE), el walk-forward, la validación con θ* congelado y la detección de régimen, escribir las pruebas y las figuras, y organizar este README. La estrategia de indicadores fue propuesta por el equipo. Todo el código y los resultados numéricos fueron revisados y ejecutados por los integrantes, quienes son responsables de explicar cualquier parte del proyecto entregado.
