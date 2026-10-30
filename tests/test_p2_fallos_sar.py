"""El análisis de fallos del SAR (pregunta 2, entregable 5).

Lo que se prueba acá es la parte que **decide la conclusión**: dónde queda la
zona ambigua y cómo se reparten los errores entre el sensor y el modelo. Los dos
bugs que tuvo este código al construirlo están cubiertos con un test cada uno,
porque los dos producían números que parecían razonables:

* la zona ambigua se tomaba entre el primer y el último bin ambiguo, y bastaban
  dos bines de ruido en las colas para estirarla hasta el borde del histograma,
  con lo que **cualquier** error caía "en la zona ambigua";
* el error irreducible se calculaba como solape/2, que supone clases
  equiprobables — acá el agua es el 12.8% de los píxeles.
"""

from __future__ import annotations

import numpy as np
import pytest

from pregunta2.paso7_fallos_sar.analizar_fallos import (
    BARRIDO_DENSIDAD, BARRIDO_UMBRAL, BORDES, CENTROS, DENSIDAD_MINIMA_BIN,
    UMBRAL_SOLAPE, _cuantiles, clasificar, sensibilidad, zonas_de_vv,
)


def _gauss(centro: float, sigma: float, n: int) -> np.ndarray:
    """Un histograma gaussiano sobre la grilla de bines del módulo."""
    d = np.exp(-0.5 * ((CENTROS - centro) / sigma) ** 2)
    return np.round(n * d / d.sum()).astype(np.int64)


class TestZonasDeVv:
    def test_clases_bien_separadas_no_dejan_casi_solape(self):
        agua = _gauss(-22.0, 1.0, 1_000_000)
        seco = _gauss(-6.0, 1.0, 1_000_000)
        z = zonas_de_vv(agua, seco)
        assert z["coeficiente_solape"] < 0.01
        assert z["error_bayes_con_solo_vv"] < 0.01

    def test_clases_identicas_solapan_del_todo(self):
        h = _gauss(-12.0, 2.0, 1_000_000)
        z = zonas_de_vv(h, h.copy())
        assert z["coeficiente_solape"] == pytest.approx(1.0, abs=0.02)
        # Con priors iguales y densidades iguales, el error de Bayes es 0.5.
        assert z["error_bayes_con_solo_vv"] == pytest.approx(0.5, abs=0.02)

    def test_el_error_de_bayes_usa_las_proporciones_reales(self):
        """El bug: solape/2 supone 50/50 y acá el agua es minoría.

        Con la misma forma de densidades, un conjunto donde el agua es el 10%
        tiene MENOS error de Bayes que uno 50/50: conviene apostar casi siempre
        a la clase mayoritaria.
        """
        agua = _gauss(-16.0, 3.0, 100_000)
        seco = _gauss(-10.0, 3.0, 100_000)
        parejo = zonas_de_vv(agua, seco)
        desparejo = zonas_de_vv(agua, _gauss(-10.0, 3.0, 900_000))
        assert desparejo["proporcion_de_agua"] == pytest.approx(0.10, abs=0.01)
        assert desparejo["error_bayes_con_solo_vv"] < parejo["error_bayes_con_solo_vv"]
        # Y el solape, que no depende de las proporciones, no se mueve.
        assert desparejo["coeficiente_solape"] == pytest.approx(
            parejo["coeficiente_solape"], abs=0.01)

    def test_el_ruido_de_las_colas_no_estira_la_zona_ambigua(self):
        """El bug que motivó tomar el tramo CONTIGUO más largo.

        Dos clases que se solapan de verdad cerca de -14 dB, más un puñado de
        píxeles sueltos de cada una en un bin lejano a +2 dB. Con el criterio
        viejo —primer y último bin ambiguo— ese bin de ruido estiraba la zona
        hasta +2, con lo que **cualquier** error caía "en la zona ambigua".
        """
        agua = _gauss(-18.0, 3.0, 1_000_000)
        seco = _gauss(-10.0, 3.0, 1_000_000)
        limpio = zonas_de_vv(agua, seco)
        assert limpio["bins_ambiguos_contiguos"] > 0, "el caso base no solapa"

        lejano = int(np.argmin(np.abs(CENTROS - 2.0)))
        agua[lejano] += 3
        seco[lejano] += 3
        con_ruido = zonas_de_vv(agua, seco)

        # El bin de ruido no tiene masa suficiente: la zona no se mueve.
        assert con_ruido["zona_ambigua_db"] == limpio["zona_ambigua_db"]
        bajo, alto = con_ruido["zona_ambigua_db"]
        assert not (bajo <= 2.0 <= alto), "la zona llega hasta el bin de ruido"

    def test_un_tramo_ambiguo_lejano_no_se_une_con_el_principal(self):
        """Dos tramos ambiguos separados: se toma el más largo, no su unión."""
        agua = _gauss(-18.0, 3.0, 1_000_000)
        seco = _gauss(-10.0, 3.0, 1_000_000)
        # Un segundo tramo ambiguo, corto pero con masa de sobra, a +2 dB.
        for d in (-0.25, 0.0, 0.25):
            i = int(np.argmin(np.abs(CENTROS - (2.0 + d))))
            agua[i] += 20_000
            seco[i] += 20_000

        z = zonas_de_vv(agua, seco)
        bajo, alto = z["zona_ambigua_db"]
        assert not (bajo <= 2.0 <= alto), "los dos tramos se unieron"
        assert z["bins_ambiguos_totales"] > z["bins_ambiguos_contiguos"]

    def test_sin_bines_ambiguos_la_zona_queda_indefinida(self):
        # Densidades disjuntas: ningun bin tiene masa de las dos clases.
        agua = np.zeros(len(CENTROS), np.int64)
        seco = np.zeros(len(CENTROS), np.int64)
        agua[10] = 1000
        seco[150] = 1000
        z = zonas_de_vv(agua, seco)
        assert np.isnan(z["zona_ambigua_db"][0])
        assert z["bins_ambiguos_contiguos"] == 0

    def test_histogramas_vacios_no_revientan(self):
        vacio = np.zeros(len(CENTROS), np.int64)
        z = zonas_de_vv(vacio, vacio.copy())
        assert z["coeficiente_solape"] == 0.0
        assert z["bins_ambiguos_contiguos"] == 0


class TestClasificar:
    def test_reparte_los_errores_en_las_tres_zonas(self):
        h = np.zeros(len(CENTROS), np.int64)
        oscuro = int(np.argmin(np.abs(CENTROS - (-25.0))))
        medio = int(np.argmin(np.abs(CENTROS - (-12.0))))
        claro = int(np.argmin(np.abs(CENTROS - (-2.0))))
        h[oscuro], h[medio], h[claro] = 5, 7, 11

        r = clasificar(h, bajo=-18.0, alto=-5.0)
        assert r["total"] == 23
        assert r["zona_agua"] == 5      # mas oscuro que la zona ambigua
        assert r["zona_ambigua"] == 7
        assert r["zona_suelo"] == 11    # mas claro
        assert r["zona_agua"] + r["zona_ambigua"] + r["zona_suelo"] == r["total"]

    def test_las_tres_zonas_siempre_suman_el_total(self):
        rng = np.random.default_rng(0)
        h = rng.integers(0, 50, size=len(CENTROS))
        for bajo, alto in ((-30.0, -20.0), (-18.0, -5.0), (-6.0, 4.0)):
            r = clasificar(h, bajo, alto)
            assert (r["zona_agua"] + r["zona_ambigua"]
                    + r["zona_suelo"]) == r["total"]


class TestCuantiles:
    def test_la_mediana_de_un_histograma_centrado(self):
        h = _gauss(-15.0, 2.0, 100_000)
        q = _cuantiles(h)
        assert q["mediana"] == pytest.approx(-15.0, abs=0.3)
        assert q["p5"] < q["mediana"] < q["p95"]
        assert q["n"] == int(h.sum())

    def test_histograma_vacio(self):
        assert _cuantiles(np.zeros(len(CENTROS), np.int64)) == {"n": 0}


def test_los_bines_cubren_el_rango_de_un_sar_real():
    # Sentinel-1 GRD en dB: el agua baja de -25 y el suelo brillante no pasa
    # de unos pocos dB positivos. Si los bordes se recortan, los cuantiles de
    # las colas quedan pegados al borde sin que nada avise.
    assert BORDES[0] <= -40.0
    assert BORDES[-1] >= 5.0
    assert len(CENTROS) == len(BORDES) - 1

class TestSensibilidad:
    """El barrido de umbrales que reemplazó a una afirmación sin medir.

    El README de este paso decía que mover los umbrales no cambiaba el reparto
    sensor/modelo. Lo cambia. Estos tests fijan que el barrido se haga de verdad
    y que **no deje tocados** los valores globales del módulo: si los dejara
    movidos, el análisis que corre después usaría otros umbrales que los que
    reporta, en silencio.
    """

    def _histogramas(self):
        agua = _gauss(-18.0, 3.0, 1_000_000)
        seco = _gauss(-10.0, 3.0, 1_000_000)
        # Errores repartidos entre las dos zonas.
        fp = _gauss(-15.0, 2.0, 100_000)
        fn = _gauss(-11.0, 2.0, 100_000)
        return agua, seco, fp, fn

    def test_barre_todas_las_combinaciones(self):
        r = sensibilidad(*self._histogramas())
        assert len(r["filas"]) == len(BARRIDO_UMBRAL) * len(BARRIDO_DENSIDAD)

    def test_marca_exactamente_una_fila_como_la_publicada(self):
        r = sensibilidad(*self._histogramas())
        publicadas = [f for f in r["filas"] if f.get("publicado")]
        assert len(publicadas) == 1
        assert publicadas[0]["umbral"] == UMBRAL_SOLAPE
        assert publicadas[0]["densidad_minima"] == DENSIDAD_MINIMA_BIN

    def test_no_deja_movidos_los_umbrales_globales(self):
        """El bug que este test previene: `sensibilidad` escribe las globales.

        Sin el `finally` que las restaura, todo lo que corriera después usaría
        el último valor del barrido (0.30 / 0.002) mientras el JSON seguiría
        diciendo 0.10 / 0.001.
        """
        import pregunta2.paso7_fallos_sar.analizar_fallos as A
        antes = (A.UMBRAL_SOLAPE, A.DENSIDAD_MINIMA_BIN)
        sensibilidad(*self._histogramas())
        assert (A.UMBRAL_SOLAPE, A.DENSIDAD_MINIMA_BIN) == antes

    def test_una_dominancia_mas_exigente_angosta_la_zona(self):
        agua, seco, fp, fn = self._histogramas()
        r = sensibilidad(agua, seco, fp, fn)
        por_umbral = {}
        for f in r["filas"]:
            if f["zona_ambigua_db"] and f["densidad_minima"] == DENSIDAD_MINIMA_BIN:
                b, a = f["zona_ambigua_db"]
                por_umbral[f["umbral"]] = a - b
        anchos = [por_umbral[u] for u in sorted(por_umbral)]
        # Mas exigencia de dominancia -> menos bines ambiguos -> zona mas corta.
        assert anchos == sorted(anchos, reverse=True)

    def test_los_rangos_cubren_lo_que_reportan_las_filas(self):
        r = sensibilidad(*self._histogramas())
        fps = [f["fp_sensor"] for f in r["filas"] if f["zona_ambigua_db"]]
        assert r["rango_fp_sensor"] == [min(fps), max(fps)]
