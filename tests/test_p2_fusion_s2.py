"""La fusión con Sentinel-2 (pregunta 2, bono del entregable 6).

Lo que se prueba es lo que decide si la comparación entre SAR, óptico y fusión
significa algo:

* que los índices de agua se calculen bien y **no exploten** donde las dos
  bandas valen cero (los bordes sin dato de cada chip);
* que los tres modos sirvan **los mismos chips** — si uno se quedara con más
  chips que otro, la diferencia mediría el conjunto y no la fusión;
* que el bootstrap pareado detecte una mejora real y **no** invente una donde
  los dos modelos son iguales.
"""

from __future__ import annotations

import numpy as np
import pytest

from pregunta2.paso8_fusion_s2.comparar_fusion import (
    bootstrap_pareado, iou_agregada,
)
from pregunta2.paso8_fusion_s2.datos_fusion import (
    BANDAS_S2, BANDAS_USADAS, INDICES, MODOS, canales_de, canales_opticos,
    cobertura_de_nube, indice_normalizado,
)


class TestIndiceNormalizado:
    def test_el_agua_da_ndwi_positivo(self):
        # Verde alto, NIR casi cero: la firma optica del agua.
        verde = np.array([[0.06]], np.float32)
        nir = np.array([[0.01]], np.float32)
        assert indice_normalizado(verde, nir)[0, 0] > 0.5

    def test_la_vegetacion_da_ndwi_negativo(self):
        # NIR mucho mas alto que verde: la firma de la vegetacion sana.
        verde = np.array([[0.05]], np.float32)
        nir = np.array([[0.35]], np.float32)
        assert indice_normalizado(verde, nir)[0, 0] < -0.5

    def test_el_borde_sin_dato_no_divide_por_cero(self):
        """Donde las dos bandas son 0 el índice no existe: debe dar 0, no NaN.

        Un NaN acá envenenaría el gradiente de toda la red, y los chips de
        Sen1Floods11 tienen bordes sin dato de forma rutinaria.
        """
        ceros = np.zeros((4, 4), np.float32)
        r = indice_normalizado(ceros, ceros)
        assert np.all(np.isfinite(r))
        assert np.all(r == 0)

    def test_nunca_sale_del_rango_menos_uno_a_uno(self):
        rng = np.random.default_rng(0)
        a = rng.random((50, 50)).astype(np.float32)
        b = rng.random((50, 50)).astype(np.float32)
        r = indice_normalizado(a, b)
        assert np.all(np.isfinite(r))
        assert r.min() >= -1.0 and r.max() <= 1.0


class TestCanalesOpticos:
    def _s2(self, valor: float = 0.1) -> np.ndarray:
        return np.full((len(BANDAS_S2), 8, 8), valor, np.float32)

    def test_apila_las_bandas_elegidas_mas_los_indices(self):
        c = canales_opticos(self._s2())
        assert c.shape[0] == len(BANDAS_USADAS) + len(INDICES)
        assert c.shape[0] == canales_de("optico")
        assert c.dtype == np.float32

    def test_con_todas_las_bandas_iguales_los_indices_son_cero(self):
        c = canales_opticos(self._s2(0.2))
        assert np.allclose(c[len(BANDAS_USADAS):], 0.0)

    def test_sobrevive_a_un_chip_entero_en_cero(self):
        c = canales_opticos(np.zeros((len(BANDAS_S2), 8, 8), np.float32))
        assert np.all(np.isfinite(c))


class TestCanalesDe:
    def test_la_fusion_suma_los_dos_sensores(self):
        assert canales_de("fusion") == canales_de("sar") + canales_de("optico")

    def test_el_sar_sigue_siendo_dos(self):
        assert canales_de("sar") == 2

    def test_un_modo_desconocido_falla_fuerte(self):
        with pytest.raises(ValueError, match="modo desconocido"):
            canales_de("rgb")

    def test_los_tres_modos_declarados_son_validos(self):
        for m in MODOS:
            assert canales_de(m) > 0


class TestCoberturaDeNube:
    def test_un_chip_oscuro_no_tiene_nube(self):
        s2 = np.full((len(BANDAS_S2), 10, 10), 0.05, np.float32)
        assert cobertura_de_nube(s2) == 0.0

    def test_un_chip_brillante_es_todo_nube(self):
        s2 = np.full((len(BANDAS_S2), 10, 10), 0.6, np.float32)
        assert cobertura_de_nube(s2) == 1.0

    def test_mide_la_fraccion_sobre_la_banda_azul(self):
        s2 = np.full((len(BANDAS_S2), 10, 10), 0.05, np.float32)
        azul = BANDAS_S2.index("B2")
        s2[azul, :3, :] = 0.6          # 30 de 100 pixeles nublados
        assert cobertura_de_nube(s2) == pytest.approx(0.30)
        # Y no se confunde con otra banda brillante.
        s2b = np.full((len(BANDAS_S2), 10, 10), 0.05, np.float32)
        s2b[BANDAS_S2.index("B8"), :, :] = 0.9
        assert cobertura_de_nube(s2b) == 0.0


class TestBootstrapPareado:
    def _conf(self, vp, fp, fn, n):
        """n chips idénticos con esa confusión."""
        return np.tile(np.array([vp, fp, fn, 0.0]), (n, 1))

    def test_iou_agregada_sobre_una_confusion_conocida(self):
        # vp=50, fp=25, fn=25 -> 50 / 100 = 0.5
        assert iou_agregada(self._conf(50, 25, 25, 1)) == pytest.approx(0.5)

    def test_detecta_una_mejora_grande_como_significativa(self):
        peor = self._conf(40, 30, 30, 60)
        mejor = self._conf(80, 10, 10, 60)
        r = bootstrap_pareado(peor, mejor, remuestreos=400, semilla=0)
        assert r["delta"] > 0
        assert r["significativa"] is True
        assert r["ic95"][0] > 0

    def test_no_inventa_diferencia_entre_dos_modelos_iguales(self):
        igual = self._conf(50, 20, 20, 60)
        r = bootstrap_pareado(igual, igual.copy(), remuestreos=400, semilla=0)
        assert r["delta"] == pytest.approx(0.0, abs=1e-9)
        assert r["significativa"] is False

    def test_es_pareado_de_verdad(self):
        """Dos modelos que empatan chip a chip no deben dar diferencia.

        Si el remuestreo no fuera pareado —índices distintos para cada modelo—
        esta comparación daría ruido distinto de cero.
        """
        rng = np.random.default_rng(3)
        a = np.column_stack([rng.integers(10, 500, 80),
                             rng.integers(1, 200, 80),
                             rng.integers(1, 200, 80),
                             np.zeros(80)]).astype(float)
        r = bootstrap_pareado(a, a.copy(), remuestreos=300, semilla=7)
        assert r["ic95"] == [0.0, 0.0]

    def test_sin_chips_no_revienta(self):
        vacio = np.zeros((0, 4))
        assert bootstrap_pareado(vacio, vacio, 100, 0) == {"chips": 0}
