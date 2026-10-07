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


@pytest.mark.parametrize("t", [800, 4321, 9000])
def test_truncamiento_por_columna(modelo_y_datos, t):
    """Cada columna de regime_features en t es la misma con df.iloc[:t+1] que con toda la serie."""
    import numpy as np
    from src.regimes import regime_features
    _, datos = modelo_y_datos
    completo, parcial = regime_features(datos), regime_features(datos.iloc[: t + 1])
    for col in completo.columns:
        a, b = parcial[col].iloc[-1], completo[col].iloc[t]
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-9, abs=1e-12), col


def test_hmm_filtrado_no_cambia_con_datos_posteriores(modelo_y_datos):
    """La etiqueta filtrada del HMM en t solo depende de datos hasta t (Viterbi no cumple esto)."""
    from src.regimes import HMM, regime_features
    _, datos = modelo_y_datos
    train, _ = separar_train_test(datos)
    hmm = HMM().ajustar(regime_features(train))
    completa = hmm.etiquetar(regime_features(datos))
    for t in (5000, 10000):
        assert (hmm.etiquetar(regime_features(datos.iloc[: t + 1])) == completa.iloc[: t + 1]).all()
