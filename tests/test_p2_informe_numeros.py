"""Los números del informe tienen que ser los que midieron los scripts.

`paso9_informe/compilar_informe.py` verifica el formato ---rango de páginas,
mínimo de referencias--- pero **no** los valores: el informe lleva sus tablas
escritas a mano en el `.tex`, así que un número puede quedar viejo después de
re-correr una etapa y el PDF compila igual, sin ninguna señal.

Eso ya pasaba antes de estas pruebas y es el modo de fallo más caro que tiene el
entregable: un informe internamente consistente, bien compuesto, que afirma algo
que sus propios artefactos no dicen.

Se cubren los números de la región peruana porque son los que más se movieron, y
porque son los que sostienen la conclusión de que la brecha es de la etiqueta y
no del modelo. Las pruebas se saltan ---no fallan--- si el artefacto todavía no
existe: en una copia recién clonada del repositorio no hay `data/`, y esto no es
una prueba sobre si las etapas corrieron.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

RAIZ = pathlib.Path(__file__).resolve().parents[1]
TEX = RAIZ / "pregunta2/paso9_informe/informe/informe.tex"
EVALUACION = RAIZ / "pregunta2/paso5_lima/evaluacion_lima.json"
DIAGNOSTICO = RAIZ / "pregunta2/paso5_lima/diagnostico_etiqueta.json"
CAMBIO = RAIZ / "pregunta2/paso5_lima/deteccion_cambio.json"


def _cargar(ruta: pathlib.Path) -> dict:
    if not ruta.is_file():
        pytest.skip(f"falta {ruta.name}; correr la etapa que lo produce")
    return json.loads(ruta.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tex() -> str:
    if not TEX.is_file():
        pytest.skip("no está el informe.tex")
    return TEX.read_text(encoding="utf-8")


def _aparece(tex: str, valor: float, decimales: int) -> bool:
    """¿El valor, con ese formato, está escrito en el .tex?

    Se busca el texto formateado y no un regex suelto porque lo que hay que fijar
    es exactamente lo que el lector ve impreso.
    """
    return f"{valor:.{decimales}f}" in tex


class TestBrechaDeLima:
    def test_el_iou_de_lima_es_el_medido(self, tex):
        d = _cargar(EVALUACION)
        assert _aparece(tex, d["lima"]["iou_agua"], 4), \
            f"el informe no dice IoU {d['lima']['iou_agua']:.4f} para Lima"

    def test_el_iou_de_prueba_es_el_medido(self, tex):
        d = _cargar(EVALUACION)
        assert _aparece(tex, d["test"]["iou_agua"], 4)

    def test_la_brecha_y_su_intervalo_son_los_medidos(self, tex):
        b = _cargar(EVALUACION)["brecha_lima_menos_test"]
        bajo, alto = b["ic95"]
        assert _aparece(tex, abs(b["delta"]), 3), "la brecha no coincide"
        assert _aparece(tex, bajo, 3) and _aparece(tex, alto, 3), \
            "el intervalo de la brecha no coincide"

    def test_el_resumen_cita_el_iou_de_lima_redondeado(self, tex):
        """El abstract lo da con 3 decimales; tiene que ser el mismo número."""
        d = _cargar(EVALUACION)
        resumen = tex[tex.index("\\begin{abstract}"):tex.index("\\end{abstract}")]
        assert f"{d['lima']['iou_agua']:.3f}" in resumen


class TestDiagnosticoDeLaEtiqueta:
    def test_los_dos_otsu_son_los_medidos(self, tex):
        d = _cargar(DIAGNOSTICO)["otsu_mismo_codigo"]
        assert _aparece(tex, d["lima"]["iou_agua"], 4)
        assert _aparece(tex, d["bolivia"]["iou_agua"], 4)

    def test_el_otsu_del_conjunto_se_cita_como_tal(self, tex):
        """El 0.6973 del dataset y el 0.3530 propio no son el mismo número.

        Citar los dos sin distinguirlos es justamente lo que haría ver una
        contradicción donde no la hay, así que el informe tiene que traer ambos.
        """
        d = _cargar(DIAGNOSTICO)
        propio = d["otsu_mismo_codigo"]["bolivia"]["iou_agua"]
        del_conjunto = d["otsu_precalculado_del_conjunto"]["bolivia"]["iou_agua"]
        assert propio != pytest.approx(del_conjunto, abs=1e-4)
        assert _aparece(tex, propio, 4) and _aparece(tex, del_conjunto, 4)

    def test_el_contraste_de_oscuros_es_el_medido(self, tex):
        c = _cargar(DIAGNOSTICO)["contraste_oscuro_lima"]
        dentro = c["etiquetado_agua"]["fraccion_oscura"] * 100
        fuera = c["etiquetado_suelo"]["fraccion_oscura"] * 100
        assert f"{dentro:.1f}" in tex and f"{fuera:.1f}" in tex

    def test_la_etiqueta_sigue_anticorrelacionada(self):
        """Si esto deja de cumplirse, la conclusión del informe cambia."""
        c = _cargar(DIAGNOSTICO)["contraste_oscuro_lima"]
        assert c["cociente_dentro_sobre_fuera"] < 1.0, (
            "los pixeles oscuros ya no son mas frecuentes fuera de la etiqueta: "
            "el informe afirma lo contrario"
        )


class TestCuadroDeFirmas:
    #: Cómo se llama cada capa en el `.tex`, que escapa el guión bajo.
    CAPAS = ("Potential_Maximum_flood", "Riverbank", "WatrcrsA", "LakeresA")

    def test_cada_capa_lleva_su_mediana_medida(self, tex):
        capas = _cargar(DIAGNOSTICO)["firma_capas_lima"]
        for nombre in self.CAPAS:
            mediana = capas[nombre]["mediana_vv_db"]
            assert _aparece(tex, abs(mediana), 2), \
                f"la mediana de {nombre} ({mediana:.2f}) no está en el informe"

    def test_cada_capa_lleva_su_conteo_de_pixeles(self, tex):
        capas = _cargar(DIAGNOSTICO)["firma_capas_lima"]
        # El .tex separa los miles con `\,`, igual que el resto del documento.
        sin_espacios = re.sub(r"\\,", "", tex)
        for nombre in self.CAPAS:
            n = capas[nombre]["pixeles"]
            assert str(n) in sin_espacios, \
                f"el conteo de {nombre} ({n}) no está en el informe"

    def test_la_referencia_dorada_es_la_medida(self, tex):
        d = _cargar(DIAGNOSTICO)["firma_sen1floods11_bolivia"]
        for clase in ("agua", "suelo"):
            assert _aparece(tex, abs(d[clase]["mediana_vv_db"]), 2)

    def test_el_cauce_es_mas_brillante_que_la_capa_usada(self):
        """La afirmación central del párrafo «la causa es física»."""
        capas = _cargar(DIAGNOSTICO)["firma_capas_lima"]
        assert (capas["Riverbank"]["mediana_vv_db"]
                > capas["Potential_Maximum_flood"]["mediana_vv_db"])

    def test_la_capa_usada_se_parece_al_suelo_y_no_al_agua(self):
        d = _cargar(DIAGNOSTICO)
        usada = d["firma_capas_lima"]["Potential_Maximum_flood"]["mediana_vv_db"]
        dorada = d["firma_sen1floods11_bolivia"]
        a_suelo = abs(usada - dorada["suelo"]["mediana_vv_db"])
        a_agua = abs(usada - dorada["agua"]["mediana_vv_db"])
        assert a_suelo < a_agua, "la etiqueta ya no se parece mas al suelo"

    def test_la_distancia_al_agua_dorada_es_la_que_cita_el_informe(self, tex):
        d = _cargar(DIAGNOSTICO)
        usada = d["firma_capas_lima"]["Potential_Maximum_flood"]["mediana_vv_db"]
        agua = d["firma_sen1floods11_bolivia"]["agua"]["mediana_vv_db"]
        assert _aparece(tex, usada - agua, 1)


class TestDeteccionDeCambio:
    def test_el_cambio_dentro_y_fuera_son_los_medidos(self, tex):
        vv = _cargar(CAMBIO)["bandas"]["VV"]["mediana_cambio_db"]
        assert _aparece(tex, vv["etiquetado_agua"], 2)
        assert _aparece(tex, abs(vv["etiquetado_suelo"]), 2)

    def test_la_cota_superior_es_la_medida(self, tex):
        cota = _cargar(CAMBIO)["bandas"]["VV"]["cota_superior_barriendo_umbral"]
        assert _aparece(tex, cota["iou_agua"], 4)

    def test_la_zona_etiquetada_se_aclara_en_vez_de_oscurecerse(self):
        """La afirmación que cierra el argumento del entregable.

        Si esto se diera vuelta, la inundación *sí* sería detectable por cambio
        y la conclusión del informe ---que el límite no es el modelo de una sola
        fecha--- dejaría de valer.
        """
        for banda in ("VV", "VH"):
            cambio = _cargar(CAMBIO)["bandas"][banda]["mediana_cambio_db"]
            assert cambio["etiquetado_agua"] > 0, (
                f"{banda}: la zona etiquetada ya no se aclara entre el antes y "
                f"el durante; el informe afirma que si"
            )

    def test_la_cota_superior_sigue_siendo_despreciable(self):
        """El umbral se eligió viendo la verdad; aun así no separa."""
        cota = _cargar(CAMBIO)["bandas"]["VV"]["cota_superior_barriendo_umbral"]
        assert cota["iou_agua"] < 0.05, (
            "la deteccion de cambio empezo a funcionar: revisar la conclusion"
        )

    def test_las_dos_escenas_son_del_mismo_path_y_hora(self):
        """Sin geometría idéntica, la diferencia mezcla ángulo con inundación."""
        d = _cargar(CAMBIO)
        hora = {g.split("_")[4][9:] for g in (d["granule_durante"],
                                              d["granule_antes"])}
        assert hora == {"103823"}, f"las escenas no son de la misma hora: {hora}"
