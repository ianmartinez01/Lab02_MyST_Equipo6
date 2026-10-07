import pandas as pd
import pytest

from src.data import cargar_datos
from src.signals import generar_senales, regla_confirmacion


@pytest.fixture(scope="module")
def datos():
    return cargar_datos().iloc[:3000]


@pytest.mark.parametrize("t", [500, 1500, 2999])
def test_causalidad_senal(datos, t):
    """Recalcular sobre df.iloc[:t+1] da la misma señal e indicadores en t que la serie completa."""
    completa = generar_senales(datos)
    parcial = generar_senales(datos.iloc[: t + 1])
    assert parcial["senal"].iloc[-1] == completa["senal"].iloc[t]
    pd.testing.assert_series_equal(parcial.iloc[-1], completa.iloc[t], check_names=False)


def test_regla_confirmacion_dos_de_tres():
    """Con un solo voto a favor no se abre posición; con dos o más, sí."""
    votos = pd.DataFrame(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [-1, 0, 0], [-1, -1, 0], [1, -1, 0], [1, 1, -1]],
        columns=["volatilidad", "momento", "tendencia"],
    )
    assert regla_confirmacion(votos).tolist() == [0, 1, 1, 0, -1, 0, 1]


def test_posicion_objetivo_solo_a_favor_de_la_tendencia():
    """La señal a favor abre, la señal en contra cierra sin voltear, y el volteo de la tendencia cierra."""
    from src.signals import posicion_objetivo
    senal = pd.Series([1, 0, 0, -1, 0, 1, 0, -1, -1, 0])
    tendencia = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, -1, -1])
    assert posicion_objetivo(senal, tendencia).tolist() == [1, 1, 1, 0, 0, 1, 1, 0, -1, -1]


def test_compra_nocturna_pide_dos_de_tres_votos():
    """A las 15:50: con un solo voto nocturno no hay compra; con dos, sí; fuera de las 15:50, nunca."""
    from src.signals import senal_nocturna
    indice = pd.DatetimeIndex(["2026-03-02 15:50", "2026-03-03 15:50", "2026-03-04 15:45"], tz="America/New_York")
    ind = pd.DataFrame({"dist_vwap": [-0.006, -0.006, -0.006], "pos_rango": [0.5, 0.2, 0.2], "rsi_14": [50, 50, 30]}, index=indice)
    assert senal_nocturna(ind, 0.005)["nocturna"].tolist() == [0, 1, 0]
    assert senal_nocturna(ind, 0.005, "simple")["nocturna"].tolist() == [1, 1, 0]
