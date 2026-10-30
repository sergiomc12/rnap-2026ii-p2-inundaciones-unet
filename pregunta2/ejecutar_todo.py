"""Corre la Pregunta 2 de extremo a extremo, con un único comando.

El examen lo exige: *"El código debe ejecutarse de extremo a extremo con un
único comando documentado en el README"*.

Cada etapa se salta si su artefacto ya existe, así que volver a correrlo hace
solo lo que falta. ``--rehacer`` fuerza todo. ``--rapido`` recorta las etapas
caras ---pocas épocas, pocos remuestreos--- para verificar el circuito completo
en minutos en vez de horas.

**Las dos descargas son reanudables.** La de Sen1Floods11 son 7.5 GB y la de
Sentinel-2 ---que solo hace falta para el bono--- 0.95 GB: si se cortan, volver a
lanzar el comando sigue donde quedó.

**Este script vive en la raíz de la pregunta, no dentro de un paso.** Las
preguntas 1 y 3 lo tienen en su paso de informe porque ese paso existe; acá no
hay paso de informe, y meterlo dentro de cualquier otro sugeriría que pertenece a
esa etapa.

Usage:
    python pregunta2/ejecutar_todo.py
    python pregunta2/ejecutar_todo.py --rapido
    python pregunta2/ejecutar_todo.py --desde 4
    python pregunta2/ejecutar_todo.py --listar
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[1]
PYTHON = sys.executable

#: La corrida que produjo los números publicados: entrenada **solo** con los 252
#: chips dorados de entrenamiento, que es el resultado de la ablación (a) del
#: entregable 2. (446 es el total de chips dorados del conjunto; el split de
#: entrenamiento son 252.)
CORRIDA = "pregunta2/paso3_entrenamiento/corridas/a_solo_dorados"


@dataclass(frozen=True)
class Etapa:
    """Una etapa del pipeline y cómo saber que ya corrió."""

    paso: int
    nombre: str
    script: str
    argumentos: tuple[str, ...] = ()
    artefacto: Optional[str] = None
    #: Argumentos que reemplazan a los normales en modo rápido. ``None`` quiere
    #: decir que la etapa es barata y corre igual.
    rapidos: Optional[tuple[str, ...]] = None
    #: Etapas del bono (entregable 6), que es opcional.
    bono: bool = False

    @property
    def ya_corrio(self) -> bool:
        if self.artefacto is None:
            return False
        ruta = _RAIZ / self.artefacto
        if ruta.is_dir():
            return any(ruta.iterdir())
        return ruta.is_file()


ETAPAS = (
    Etapa(1, "Descargar Sen1Floods11 (7.5 GB, reanudable)",
          "pregunta2/paso1_datos/descargar.py",
          artefacto="pregunta2/datos/sen1floods11/v1.1/data/flood_events/"
                    "HandLabeled/S1Hand"),
    Etapa(1, "Medir la normalización sobre el conjunto de entrenamiento",
          "pregunta2/paso1_datos/estadisticas.py",
          artefacto="pregunta2/paso1_datos/estadisticas.json"),
    Etapa(3, "Entrenar la U-Net (entregables 1 y 3)",
          "pregunta2/paso3_entrenamiento/entrenar.py",
          ("--etiqueta", "a_solo_dorados"),
          f"{CORRIDA}/pesos.pt",
          ("--etiqueta", "a_solo_dorados", "--epocas", "3")),
    Etapa(3, "Ablación: dorados vs. débiles, y función de pérdida (entregable 2)",
          "pregunta2/paso3_entrenamiento/ablacion.py",
          artefacto="pregunta2/paso3_entrenamiento/ablacion.json",
          # --solo corre una sola variante; es el recorte que ablacion.py
          # ofrece de verdad (no tiene --epocas).
          rapidos=("--solo", "b_bce")),
    Etapa(4, "Generalización contra Otsu, con bootstrap pareado (entregable 4)",
          "pregunta2/paso4_generalizacion/comparar.py",
          artefacto="pregunta2/paso4_generalizacion/comparacion.json",
          rapidos=("--remuestreos", "200")),
    Etapa(5, "Construir la región de prueba de Lima desde Copernicus EMS",
          "pregunta2/paso5_lima/construir_region.py",
          artefacto="pregunta2/datos/lima/region/chips.json"),
    # Pide un Earthdata Login gratuito en .env (TOKEN_HYP3). Si el producto ya
    # esta bajado, esta etapa solo vuelve a recortar los chips; si no, encola el
    # RTC en HyP3 y espera, que son ~75 min.
    Etapa(5, "Bajar la escena Sentinel-1 de Lima (HyP3) y recortarla a los chips",
          "pregunta2/paso5_lima/descargar_sentinel1.py",
          artefacto="pregunta2/datos/lima/region/S1Lima"),
    Etapa(5, "Evaluar la U-Net sobre Lima y medir la brecha (entregable 4)",
          "pregunta2/paso5_lima/evaluar_lima.py",
          artefacto="pregunta2/paso5_lima/evaluacion_lima.json",
          rapidos=("--remuestreos", "200")),
    Etapa(5, "Medir si la brecha de Lima es del modelo o de la etiqueta",
          "pregunta2/paso5_lima/diagnosticar_etiqueta.py",
          artefacto="pregunta2/paso5_lima/diagnostico_etiqueta.json"),
    Etapa(5, "Figura de la etiqueta de Lima contra la señal de radar",
          "pregunta2/paso5_lima/figura_alineacion.py",
          artefacto="pregunta2/paso9_informe/informe/figuras/etiqueta_lima.png"),
    # La escena pre-evento es otro RTC de HyP3: mismos ~75 min la primera vez.
    Etapa(5, "Bajar la escena pre-evento del 5 de marzo (HyP3)",
          "pregunta2/paso5_lima/descargar_sentinel1.py", ("--escena", "pre"),
          artefacto="pregunta2/datos/lima/region/S1LimaPre"),
    Etapa(5, "¿La inundación se ve comparando el antes con el durante?",
          "pregunta2/paso5_lima/deteccion_cambio.py",
          artefacto="pregunta2/paso5_lima/deteccion_cambio.json",
          rapidos=("--sin-figura",)),
    Etapa(6, "Vistas PNG de los chips, para poder mirarlos",
          "pregunta2/paso6_figuras/previsualizar.py",
          artefacto="pregunta2/paso6_figuras/vistas"),
    Etapa(7, "Fallos característicos del SAR (entregable 5)",
          "pregunta2/paso7_fallos_sar/analizar_fallos.py",
          artefacto="pregunta2/paso7_fallos_sar/fallos.json",
          rapidos=("--sin-figuras",)),
    # --- Bono (entregable 6) -------------------------------------------------
    Etapa(8, "Descargar S2Hand, solo el óptico dorado (0.95 GB)",
          "pregunta2/paso1_datos/descargar.py",
          ("--solo-lo-pedido", "--s2-dorado"),
          "pregunta2/datos/sen1floods11/v1.1/data/flood_events/"
          "HandLabeled/S2Hand",
          bono=True),
    Etapa(8, "Separación de cada banda óptica contra la del radar",
          "pregunta2/paso8_fusion_s2/medir_separacion.py",
          artefacto="pregunta2/paso8_fusion_s2/separacion_optica.json",
          bono=True),
    Etapa(8, "Entrenar SAR, óptico y fusión sobre los mismos chips",
          "pregunta2/paso8_fusion_s2/entrenar_fusion.py",
          artefacto="pregunta2/paso8_fusion_s2/fusion.json",
          rapidos=("--epocas", "3"), bono=True),
    Etapa(8, "Bootstrap pareado de la fusión, separado por nubosidad",
          "pregunta2/paso8_fusion_s2/comparar_fusion.py",
          artefacto="pregunta2/paso8_fusion_s2/comparacion_fusion.json",
          rapidos=("--remuestreos", "200"), bono=True),
    # --- Informe (paso 9) ---------------------------------------------------
    # Las tablas del informe NO se regeneran desde los JSON como en las otras
    # dos preguntas: este informe las lleva escritas en el .tex. Si un numero
    # cambia, hay que actualizarlo ahi; el compilador verifica el rango de
    # paginas y el minimo de referencias, no los valores.
    Etapa(9, "Compilar el informe NeurIPS (PDF)",
          "pregunta2/paso9_informe/compilar_informe.py"),
    Etapa(9, "Exportar el informe a Word",
          "pregunta2/paso9_informe/exportar_word.py"),
)


def correr(etapa: Etapa, rapido: bool) -> int:
    argumentos = (etapa.rapidos if rapido and etapa.rapidos is not None
                  else etapa.argumentos)
    orden = [PYTHON, str(_RAIZ / etapa.script), *argumentos]
    return subprocess.run(orden, cwd=_RAIZ).returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehacer", action="store_true",
                        help="corre todas las etapas aunque ya tengan artefacto")
    parser.add_argument("--rapido", action="store_true",
                        help="recorta las etapas caras, para verificar el "
                             "circuito completo en minutos")
    parser.add_argument("--desde", type=int, default=0,
                        help="empezar en este paso")
    parser.add_argument("--hasta", type=int, default=99)
    parser.add_argument("--sin-bono", action="store_true",
                        help="omitir el entregable 6, que es opcional")
    parser.add_argument("--listar", action="store_true",
                        help="mostrar las etapas y su estado, sin correr nada")
    args = parser.parse_args(argv)

    etapas = [e for e in ETAPAS
              if args.desde <= e.paso <= args.hasta
              and not (args.sin_bono and e.bono)]

    if args.listar:
        print(f"{'paso':>5}  {'estado':<12} etapa")
        print("-" * 78)
        for e in etapas:
            estado = "ya corrió" if e.ya_corrio else "pendiente"
            print(f"{e.paso:>5}  {estado:<12} {e.nombre}"
                  f"{'  [bono]' if e.bono else ''}")
        return 0

    comenzado = time.time()
    corridas = saltadas = 0
    for e in etapas:
        if e.ya_corrio and not args.rehacer:
            print(f"[paso {e.paso}] ya está: {e.nombre}")
            saltadas += 1
            continue
        print(f"\n{'='*78}\n[paso {e.paso}] {e.nombre}\n{'='*78}", flush=True)
        codigo = correr(e, args.rapido)
        if codigo != 0:
            print(f"\nFALLÓ en el paso {e.paso}: {e.nombre} (código {codigo}).")
            print("Se corta acá: las etapas siguientes dependen de esta.")
            return codigo
        corridas += 1

    minutos = (time.time() - comenzado) / 60
    print(f"\n{'='*78}")
    print(f"listo: {corridas} etapas corridas, {saltadas} ya estaban "
          f"({minutos:.1f} min)")
    if args.rapido:
        print("\nOJO: fue en modo --rapido. Los números NO son los del informe;")
        print("sirve para verificar que el circuito corre de punta a punta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
