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

Para ejecutar el proyecto completo (datos, regímenes, walk-forward, modelo final, robustez y figuras) con un solo comando:

```bash
python main.py
```

El walk-forward corre en paralelo en todos los núcleos disponibles y el proyecto completo tarda entre 7 y 12 minutos según la carga de la máquina. Las figuras se guardan en `docs/figures/` y las tablas en `resultados/`.

Para ejecutar las pruebas automáticas:

```bash
python -m pytest -v
```

Los datos ya están congelados en `data/nvda_5m.csv`. Para volver a descargarlos (misma interfaz que `yf.download`):

```bash
python -m src.data --ticker NVDA --start 2026-01-01 --end 2026-09-26 --interval 5m
```

La semilla aleatoria utilizada es `42`, definida en `main.py` y en `src/optimize.py`. Cada estudio de Optuna usa un `TPESampler` con semilla `42 + desplazamiento` por ventana y régimen, y K-means usa `random_state=42`, por lo que los resultados son reproducibles.

## Datos

| Concepto | Valor |
|---|---|
| Activo | NVDA (NVIDIA Corporation) |
| Fuente | Histórico público de velas de Binance Stocks, precios ajustados |
| Frecuencia | 5 minutos, sesión regular 09:30–16:00 ET (78 velas por día) |
| Train | 2 de enero a 30 de junio de 2026 (9,594 velas) |
| Test | 1 de julio a 25 de septiembre de 2026 (4,758 velas) |
| Auditoría | 0 duplicados, 0 nulos, 0 velas con OHLC incoherente, 0 días incompletos |

Yahoo Finance solo conserva 60 días de velas de 5 minutos, por eso la descarga usa el histórico de Binance Stocks con la misma interfaz que `yf.download`.

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
| Anualización | 78 velas × 252 sesiones = 19,656 velas por año, tasa libre de riesgo 0 |

## Detección de régimen

K-means (k = 3) sobre ln(volatilidad realizada) y √(eficiencia de Kaufman), calculadas en una ventana móvil de 1 semana (390 velas) y actualizadas cada 4 horas de sesión (48 velas). Crisis es el grupo de mayor volatilidad; de los otros dos, tendencia es el de mayor eficiencia y reversión el restante. Al cambiar el régimen, la posición abierta se cierra en la apertura siguiente.

| Métrica | Train | Test | Objetivo |
|---|---:|---:|---:|
| Silhouette | 0.400 | 0.374 | > 0.4 |
| Duración media del régimen (horas de sesión) | 13.4 | 15.3 | > 12 |
| Transiciones por semana | 2.4 | 2.0 | |
| Tiempo en tendencia / reversión / crisis | 22% / 33% / 45% | 34% / 34% / 31% | |

## Optimización y walk-forward

| Elemento | Valor |
|---|---|
| Método | Optuna, TPE, maximizando el Calmar Ratio |
| Pruebas | 150 por régimen por ventana (la primera es la configuración base) |
| Operaciones mínimas | 5 por régimen por ventana; si no se cumplen, Calmar = −10 |
| Régimen apagado | Si su mejor Calmar en la ventana no es positivo, no se opera en ese régimen |
| Ventanas | 21 (entrenamiento 1 mes, prueba 1 semana, paso semanal) |
| Configuraciones evaluadas | 8,700 |
| Tiempo de optimización | 370 s en paralelo (8 núcleos) |
| Modelo final | K-means con todo train y Optuna por régimen en junio; se evalúa sin cambios en test |

## Resultados

| Métrica | Train (walk-forward) | Buy & hold train | Test | Buy & hold test |
|---|---:|---:|---:|---:|
| Retorno total | −21.56% | 6.65% | −8.14% | 16.24% |
| Sharpe | −6.45 | 0.63 | −4.79 | 1.92 |
| Sortino | −8.21 | 0.91 | −6.10 | 2.90 |
| Calmar | −2.11 | 0.94 | −3.27 | 7.63 |
| Máximo drawdown | −22.05% | −19.25% | −9.06% | −11.30% |
| Win Rate | 21.8% | | 29.2% | |
| Operaciones | 87 | | 24 | |

Las tablas de retornos mensuales, trimestrales y anuales están en `docs/figures/04_tabla_retornos.png`.

## Preguntas de análisis

### 1. ¿Qué aporta la regla de confirmación de 2 de 3 frente a usar un solo indicador?

En test, con los parámetros finales, la regla 2 de 3 abrió 24 operaciones con Calmar −3.27. Usar solo el momento (RSI o Estocástico) abrió 49 con Calmar −3.04 y retorno −17.29%, y solo el MACD abrió 62 con Calmar −2.80. La confirmación reduce las operaciones a la mitad o menos y con ello el costo total en comisiones, pero no mejora el Calmar: todas las variantes pierden. La versión estricta (los tres votos) solo abrió 2 operaciones.

### 2. ¿Cuánto se degrada el desempeño entre entrenamiento y prueba en el walk-forward?

El Calmar mediano dentro de muestra fue 85.28 y fuera de muestra −18.82. El retorno medio de la ventana de entrenamiento fue +6.38% en un mes y el de la semana de prueba −1.14%; solo 4 de 21 semanas fueron positivas. No sobrevive ninguna proporción de la ventaja: el desempeño dentro de muestra es ajuste a la muestra histórica.

### 3. ¿Qué tan sensible es la estrategia a una variación de ±20% en sus parámetros?

El Calmar en test se mueve entre −2.92 y −3.46 al variar cualquier parámetro ±20%. Es una meseta, no un pico aislado, pero es una meseta de pérdidas: ninguna variación local vuelve rentable a la estrategia.

### 4. ¿A qué nivel de costo deja de ser rentable?

La estrategia no es rentable en ningún nivel de costo: aun con comisión cero pierde 2.98% en test. Con la comisión del laboratorio (0.125%) pierde 8.14%, así que las comisiones explican unos 5 puntos de la pérdida y la selección de entradas los otros 3. No hay margen de seguridad frente al 0.125%.

### 5. ¿El desempeño difiere de forma significativa entre regímenes?

En test, las operaciones abiertas en crisis ganaron 16.7% de las veces con P&L de −$61,204 y las de reversión 41.7% con −$20,200; tendencia quedó apagada. La prueba de Kruskal-Wallis sobre el retorno por operación (train y test) da H = 1.57 y p = 0.456, así que la diferencia no es significativa. La capa de régimen aporta control de exposición: apagó la estrategia en tendencia durante 34% del tiempo de test y cerró posiciones en cada transición, y muestra que la lógica contraria pierde más en crisis.

### 6. Tres limitaciones para operar con capital real

1. La ventaja no sobrevive fuera de muestra: el walk-forward muestra que la optimización ajusta ruido de cada mes.
2. Seis meses de entrenamiento con velas de 5 minutos cubren pocos episodios de cada régimen; la separación de K-means queda en el límite (silhouette 0.40 en train, 0.37 en test).
3. Los datos provienen de un tercero (Binance Stocks) y la estrategia opera 100% del capital en un solo activo, sin diversificación.

Advertencia de interpretación: el backtest asume ejecución completa al precio de apertura o al nivel del stop y del take-profit, sin impacto de mercado ni fallas de ejecución. Una orden de $1,000,000 equivale a unas 4,973 acciones, alrededor de 0.41% del volumen de una vela típica de 5 minutos (1.23 millones de acciones). Con un spread de 1 a 2 centavos sobre un precio cercano a $200 y un impacto de 1 a 2 puntos base por lado, las 24 operaciones de test costarían entre 0.05% y 0.10% adicional del capital; los stops disparados por huecos se ejecutan peor de lo modelado.

## Uso de inteligencia artificial

Se utilizó asistencia de IA (Claude) para: estructurar el proyecto según lo especificado, implementar el motor de backtesting, la optimización con Optuna, el walk-forward y la detección de régimen, escribir las pruebas y las figuras, y organizar este README. La estrategia de indicadores fue propuesta por el equipo. Todo el código y los resultados numéricos fueron revisados y ejecutados por los integrantes, quienes son responsables de explicar cualquier parte del proyecto entregado.
