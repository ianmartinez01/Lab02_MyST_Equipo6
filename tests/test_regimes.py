import pytest

from src.data import cargar_datos, separar_train_test
from src.regimes import ajustar_modelo, calcular_variables_regimen, etiquetar


@pytest.fixture(scope="module")
def modelo_y_datos():
    datos = cargar_datos()
    train, _ = separar_train_test(datos)
    modelo = ajustar_modelo(calcular_variables_regimen(train))
    return modelo, datos


@pytest.mark.parametrize("t", [2000, 6000, 12000])
def test_regimen_no_cambia_con_datos_posteriores(modelo_y_datos, t):
    """La etiqueta de régimen en t no cambia al agregar datos posteriores a t."""
    modelo, datos = modelo_y_datos
    completa = etiquetar(modelo, calcular_variables_regimen(datos))
    parcial = etiquetar(modelo, calcular_variables_regimen(datos.iloc[: t + 1]))
    assert parcial.iloc[-1] == completa.iloc[t]
    assert (parcial == completa.iloc[: t + 1]).all()
