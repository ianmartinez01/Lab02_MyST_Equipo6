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
