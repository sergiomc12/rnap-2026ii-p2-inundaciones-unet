"""La región peruana (pregunta 2, entregable 4).

Se prueban las dos piezas donde un error sale caro y no se nota al correr:

* **el nombre del trabajo de HyP3.** Es la clave con la que `find_jobs` reconoce
  un RTC ya producido. Si deja de coincidir con la escena, el script encola uno
  nuevo en vez de reusar el existente: 60 créditos de la cuota y 75 minutos, sin
  ningún error a la vista ---el resultado final es correcto, solo que se pagó dos
  veces por él.

* **el Otsu del diagnóstico.** Es lo que sostiene la afirmación «el fallo no es
  del modelo», así que tiene que fallar ruidosamente en vez de devolver un número
  plausible. Lo que se fija acá es que separe de verdad cuando hay dos modas, y
  que un chip sin variación ---que no tiene umbral que elegir--- no desaparezca de
  la medición ni reviente, porque un chip que se cae en silencio cambia el
  denominador del IoU agregado.
"""

from __future__ import annotations

import numpy as np
import pytest

from pregunta2.paso5_lima.descargar_sentinel1 import ESCENAS, nombre_trabajo
from pregunta2.paso5_lima.diagnosticar_etiqueta import otsu_por_chip


class TestNombreTrabajo:
    def test_deriva_la_fecha_del_granule(self):
        assert nombre_trabajo(
            "S1B_IW_GRDH_1SDV_20170317T103823_20170317T103848"
            "_004747_0084AE_040D") == "lima_2017_03_17"

    def test_las_dos_escenas_dan_nombres_distintos(self):
        nombres = {nombre_trabajo(g) for g, _ in ESCENAS.values()}
        assert nombres == {"lima_2017_03_17", "lima_2017_03_05"}

    def test_el_nombre_del_evento_no_cambia(self):
        """El RTC del 17 ya está producido bajo este nombre exacto.

        Si alguien reescribe `nombre_trabajo` y el valor se corre, el script deja
        de encontrar ese trabajo y vuelve a encolarlo.
        """
        granule, sufijo = ESCENAS["evento"]
        assert nombre_trabajo(granule) == "lima_2017_03_17"
        assert sufijo == "S1Lima"

    def test_cada_escena_escribe_en_su_propio_sufijo(self):
        """Dos escenas en el mismo sufijo se pisarían chip por chip."""
        sufijos = [s for _, s in ESCENAS.values()]
        assert len(sufijos) == len(set(sufijos))


class _RasterFalso:
    """Lo mínimo de `rasterio.open` que `otsu_por_chip` usa: `read(1)`."""

    def __init__(self, banda: np.ndarray) -> None:
        self._banda = banda

    def read(self, _indice: int) -> np.ndarray:
        return self._banda

    def __enter__(self):
        return self

    def __exit__(self, *_excepcion) -> bool:
        return False


@pytest.fixture
def abrir_falso(monkeypatch):
    """Sustituye `rasterio.open` por un diccionario ruta -> banda."""
    bandas: dict[str, np.ndarray] = {}

    def _open(ruta, *_args, **_kwargs):
        return _RasterFalso(bandas[str(ruta).replace("\\", "/").split("/")[-1]])

    import rasterio
    monkeypatch.setattr(rasterio, "open", _open)
    return bandas


def _chip_con_dos_modas(lado: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Mitad agua oscura, mitad suelo brillante, bien separadas."""
    vv = np.full((lado, lado), -6.0, dtype=np.float32)
    vv[: lado // 2] = -20.0
    etiqueta = np.zeros((lado, lado), dtype=np.int8)
    etiqueta[: lado // 2] = 1
    return vv, etiqueta


class TestOtsuPorChip:
    def test_separa_dos_modas_bien_definidas(self, abrir_falso, tmp_path):
        vv, etiqueta = _chip_con_dos_modas()
        abrir_falso["c_S1.tif"] = vv
        abrir_falso["c_Lab.tif"] = etiqueta

        confusion = otsu_por_chip(["c"], tmp_path, "S1", "Lab")

        vp, fp, fn, vn = confusion[0]
        assert fp == 0 and fn == 0
        assert vp == 512 and vn == 512

    def test_un_chip_sin_variacion_cuenta_como_suelo_y_no_revienta(
            self, abrir_falso, tmp_path):
        """No tiene umbral que elegir; lo que no puede es desaparecer."""
        vv = np.full((8, 8), -9.0, dtype=np.float32)
        etiqueta = np.zeros((8, 8), dtype=np.int8)
        etiqueta[0] = 1                        # 8 píxeles de agua que se pierden
        abrir_falso["c_S1.tif"] = vv
        abrir_falso["c_Lab.tif"] = etiqueta

        confusion = otsu_por_chip(["c"], tmp_path, "S1", "Lab")

        assert len(confusion) == 1, "el chip no puede caerse de la medición"
        vp, fp, fn, vn = confusion[0]
        assert (vp, fp) == (0, 0)
        assert fn == 8 and vn == 56
        assert vp + fp + fn + vn == 64

    def test_los_pixeles_sin_etiqueta_no_entran_en_la_confusion(
            self, abrir_falso, tmp_path):
        vv, etiqueta = _chip_con_dos_modas()
        etiqueta[:, :16] = -1                  # media imagen sin observar
        abrir_falso["c_S1.tif"] = vv
        abrir_falso["c_Lab.tif"] = etiqueta

        confusion = otsu_por_chip(["c"], tmp_path, "S1", "Lab")

        assert confusion[0].sum() == 32 * 16

    def test_los_nan_de_la_imagen_tampoco_entran(self, abrir_falso, tmp_path):
        vv, etiqueta = _chip_con_dos_modas()
        vv = vv.copy()
        vv[:, :8] = np.nan                     # fuera de la franja de la escena
        abrir_falso["c_S1.tif"] = vv
        abrir_falso["c_Lab.tif"] = etiqueta

        confusion = otsu_por_chip(["c"], tmp_path, "S1", "Lab")

        assert confusion[0].sum() == 32 * 24
