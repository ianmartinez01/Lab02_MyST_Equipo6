"""Detección dinámica de régimen de mercado con K-means.

Variables, calculadas sobre una ventana móvil de 1 semana (5 sesiones × 78 velas = 390 velas):
- Volatilidad realizada: desviación estándar de los log-retornos de 5 min, escalada a un día.
- Eficiencia de Kaufman: |ln P_t − ln P_{t−390}| / Σ |Δ ln P| en la ventana. Vale 1 si el precio
  se movió en línea recta (tendencia) y cerca de 0 si fue y vino sin avanzar (reversión).
K-means usa ln(volatilidad) y √eficiencia estandarizadas, porque ambas variables son sesgadas.

Etiquetas: crisis = centroide de mayor volatilidad; de los otros dos, tendencia = mayor eficiencia
y reversión = el restante. La clasificación se actualiza cada 48 velas (4 horas de sesión) y se
mantiene fija entre actualizaciones. Todo usa solo información hasta la vela t.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

VENTANA = 390
CADA = 48
VELAS_POR_HORA = 12
REGIMENES = ("tendencia", "reversion", "crisis")
SIN_DATOS = "sin_datos"
COLUMNAS_MODELO = ["log_volatilidad", "raiz_eficiencia"]


@dataclass
class ModeloRegimen:
    escalador: StandardScaler
    kmeans: KMeans
    nombres: dict

    def predecir(self, variables: pd.DataFrame) -> np.ndarray:
        grupos = self.kmeans.predict(self.escalador.transform(variables[COLUMNAS_MODELO]))
        return np.array([self.nombres[g] for g in grupos])


def calcular_variables_regimen(df: pd.DataFrame, ventana: int = VENTANA) -> pd.DataFrame:
    """Variables de régimen en cada vela con la semana previa (incluye la vela t)."""
    log_precio = np.log(df["Close"])
    r = log_precio.diff()
    volatilidad = r.rolling(ventana).std() * np.sqrt(78)
    eficiencia = log_precio.diff(ventana).abs() / r.abs().rolling(ventana).sum()
    return pd.DataFrame({
        "volatilidad": volatilidad,
        "eficiencia": eficiencia,
        "retorno_semana": log_precio.diff(ventana),
        "log_volatilidad": np.log(volatilidad),
        "raiz_eficiencia": np.sqrt(eficiencia),
    }, index=df.index)


def puntos_actualizacion(variables: pd.DataFrame, cada: int = CADA) -> pd.DataFrame:
    """Velas donde se reclasifica: posiciones múltiplo de `cada` con variables completas."""
    posiciones = np.arange(len(variables))
    return variables[(posiciones % cada == 0) & variables[COLUMNAS_MODELO].notna().all(axis=1).to_numpy()]


def ajustar_modelo(variables: pd.DataFrame, cada: int = CADA, semilla: int = 42) -> ModeloRegimen:
    """Ajusta el escalador y K-means (k=3) con los puntos de actualización y nombra cada grupo."""
    muestra = puntos_actualizacion(variables, cada)[COLUMNAS_MODELO]
    escalador = StandardScaler().fit(muestra)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=semilla).fit(escalador.transform(muestra))
    centros = pd.DataFrame(escalador.inverse_transform(kmeans.cluster_centers_), columns=COLUMNAS_MODELO)
    crisis = int(centros["log_volatilidad"].idxmax())
    resto = centros.drop(index=crisis)
    tendencia = int(resto["raiz_eficiencia"].idxmax())
    reversion = int(resto.index.drop(tendencia)[0])
    return ModeloRegimen(escalador, kmeans, {tendencia: "tendencia", reversion: "reversion", crisis: "crisis"})


def etiquetar(modelo: ModeloRegimen, variables: pd.DataFrame, cada: int = CADA) -> pd.Series:
    """Régimen vigente en cada vela: el de la última actualización ≤ t (sin_datos antes de la primera)."""
    puntos = puntos_actualizacion(variables, cada)
    etiquetas = pd.Series(np.nan, index=variables.index, dtype=object)
    if len(puntos):
        etiquetas.loc[puntos.index] = modelo.predecir(puntos)
    return etiquetas.ffill().fillna(SIN_DATOS).rename("regimen")


def duraciones(etiquetas: pd.Series) -> pd.DataFrame:
    """Rachas consecutivas de cada régimen con su duración en horas de sesión."""
    validas = etiquetas[etiquetas != SIN_DATOS]
    racha = (validas != validas.shift()).cumsum()
    tabla = validas.groupby(racha).agg(["first", "size"])
    tabla.columns = ["regimen", "velas"]
    tabla["horas"] = tabla["velas"] / VELAS_POR_HORA
    return tabla.reset_index(drop=True)


def validar_regimenes(modelo: ModeloRegimen, variables: pd.DataFrame, etiquetas: pd.Series,
                      cada: int = CADA) -> dict:
    """Silhouette, persistencia, transiciones y proporción de tiempo en cada régimen."""
    puntos = puntos_actualizacion(variables, cada)
    X = modelo.escalador.transform(puntos[COLUMNAS_MODELO])
    grupos = modelo.kmeans.predict(X)
    rachas = duraciones(etiquetas)
    validas = etiquetas[etiquetas != SIN_DATOS]
    semanas = len(validas) / VENTANA
    return {
        "silhouette": float(silhouette_score(X, grupos)) if len(set(grupos)) > 1 else np.nan,
        "duracion_media_horas": float(rachas["horas"].mean()),
        "duracion_media_por_regimen": rachas.groupby("regimen")["horas"].mean().round(1).to_dict(),
        "transiciones": int(len(rachas) - 1),
        "transiciones_por_semana": float((len(rachas) - 1) / semanas) if semanas else np.nan,
        "proporcion": validas.value_counts(normalize=True).reindex(REGIMENES, fill_value=0).round(3).to_dict(),
    }
