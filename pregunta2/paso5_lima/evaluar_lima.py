"""Evalúa la U-Net sobre la región peruana y mide la brecha (entregable 4).

Lo que el enunciado pide textualmente: entrenar **solo** con los eventos de
Sen1Floods11 ---sin ningún dato peruano--- y evaluar sobre la región de Lima
construida por el grupo, reportando la brecha frente a la evaluación en
distribución y al menos dos hipótesis sobre su causa.

Acá se mide la brecha. El modelo es el de la corrida publicada
(`a_solo_dorados`, IoU 0.6808 en prueba), cargado tal cual: **no se reentrena ni
se reajusta nada con datos de Lima**, que es justamente lo que haría inválido al
experimento.

**La comparación Lima-vs-prueba no es pareada.** Son chips distintos, sin
correspondencia uno a uno, así que remuestrear no cancela el ruido del conjunto
como sí lo hace el bootstrap pareado de `paso4_generalizacion`. El intervalo que
sale de acá es más ancho y hay que leerlo como tal ---es el mismo criterio con
el que el informe reporta la brecha de Bolivia (+0.0713, intervalo que cruza
cero), en vez de presentarla como si fuera una diferencia medida sobre los
mismos chips.

Uso:
    python pregunta2/paso5_lima/evaluar_lima.py
    python pregunta2/paso5_lima/evaluar_lima.py --corrida c_somera
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

import numpy as np

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

import torch  # noqa: E402

from pregunta2.paso1_datos.sen1floods11 import (  # noqa: E402
    MANO, ChipsSen1Floods11, Normalizador, acumular_confusion, chips_del_split,
    metricas,
)
from pregunta2.paso2_modelo.unet import ConfigUNet, UNet  # noqa: E402
from pregunta2.paso4_generalizacion.comparar import iou_de  # noqa: E402
from pregunta2.paso5_lima.construir_region import (  # noqa: E402
    chips_de_lima,
)

CORRIDAS = _RAIZ / "pregunta2/paso3_entrenamiento/corridas"
REGION = _RAIZ / "data/lima/region"
SALIDA = _RAIZ / "pregunta2/paso5_lima/evaluacion_lima.json"


@torch.no_grad()
def confusion_por_chip(modelo, chips, norm, dispositivo, *, mano, sufijo_imagen,
                       sufijo_etiqueta, umbral=0.5):
    """(n_chips, 4) con los conteos del modelo, un chip por fila.

    Es la misma idea que `paso4_generalizacion.confusion_por_chip`, pero esa
    tiene fijadas las rutas de Sen1Floods11; acá hacen falta las de Lima, que el
    propio `ChipsSen1Floods11` ya acepta como parámetros.
    """
    conjunto = ChipsSen1Floods11(chips, norm, mano=mano,
                                 sufijo_imagen=sufijo_imagen,
                                 sufijo_etiqueta=sufijo_etiqueta)
    filas = []
    modelo.eval()
    for i in range(len(conjunto)):
        x, y, m = conjunto[i]
        logits = modelo(x[None].to(dispositivo))
        filas.append(acumular_confusion(logits.float().cpu(), y[None], m[None],
                                        umbral))
    return np.stack(filas), conjunto.chips


def brecha_no_pareada(a: np.ndarray, b: np.ndarray, remuestreos: int = 10000,
                      semilla: int = 0) -> dict:
    """IoU(a) - IoU(b) remuestreando cada conjunto por separado.

    Separado y no pareado a propósito: no hay correspondencia entre un chip de
    Lima y uno de prueba, así que no existe la diferencia por par que el
    bootstrap pareado promedia.
    """
    rng = np.random.default_rng(semilla)
    observada = iou_de(a) - iou_de(b)
    diferencias = np.empty(remuestreos)
    for i in range(remuestreos):
        ia = rng.integers(0, len(a), len(a))
        ib = rng.integers(0, len(b), len(b))
        diferencias[i] = iou_de(a[ia]) - iou_de(b[ib])
    bajo, alto = np.percentile(diferencias, [2.5, 97.5])
    return {"delta": float(observada), "ic95": [float(bajo), float(alto)],
            "significativa": bool(bajo > 0 or alto < 0),
            "pareado": False}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrida", default="a_solo_dorados",
                        help="la corrida publicada; no se reentrena nada")
    parser.add_argument("--remuestreos", type=int, default=10000)
    parser.add_argument("--dispositivo", default="cuda")
    parser.add_argument("--umbral", type=float, default=0.5)
    parser.add_argument("--region", type=pathlib.Path, default=REGION)
    parser.add_argument("--cobertura-minima", type=float, default=0.01,
                        help="fraccion de pixeles con dato para admitir un chip")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    chips_lima = chips_de_lima(args.region, args.cobertura_minima)
    if not chips_lima:
        print(f"no hay chips con imagen Y etiqueta en {args.region}.\n"
              f"correr antes:\n"
              f"  python pregunta2/paso5_lima/construir_region.py     (etiquetas)\n"
              f"  python pregunta2/paso5_lima/descargar_sentinel1.py  (imagenes)")
        return 1

    pesos = CORRIDAS / args.corrida / "pesos.pt"
    if not pesos.is_file():
        print(f"no esta {pesos}")
        return 1
    guardado = torch.load(pesos, map_location="cpu", weights_only=False)
    dispositivo = torch.device(
        args.dispositivo if torch.cuda.is_available() else "cpu")
    modelo = UNet(ConfigUNet(**guardado["config"])).to(dispositivo)
    modelo.load_state_dict(guardado["pesos"])
    norm = Normalizador.desde_estadisticas()

    print(f"corrida {args.corrida} | {len(chips_lima)} chips de Lima | "
          f"umbral {args.umbral}\n")

    conf_lima, usados = confusion_por_chip(
        modelo, chips_lima, norm, dispositivo, mano=args.region,
        sufijo_imagen="S1Lima", sufijo_etiqueta="LabelLima",
        umbral=args.umbral)
    conf_test, _ = confusion_por_chip(
        modelo, chips_del_split("test"), norm, dispositivo, mano=MANO,
        sufijo_imagen="S1Hand", sufijo_etiqueta="LabelHand",
        umbral=args.umbral)

    m_lima = metricas(conf_lima.sum(axis=0))
    m_test = metricas(conf_test.sum(axis=0))
    brecha = brecha_no_pareada(conf_lima, conf_test, args.remuestreos)

    print(f"{'conjunto':<16} {'IoU':>7} {'F1':>7} {'prec':>7} {'rec':>7} {'chips':>6}")
    print("-" * 56)
    for nombre, m, n in (("Lima (Peru)", m_lima, len(usados)),
                         ("prueba", m_test, len(conf_test))):
        print(f"{nombre:<16} {m['iou_agua']:>7.4f} {m['f1']:>7.4f} "
              f"{m['precision']:>7.4f} {m['exhaustividad']:>7.4f} {n:>6}")
    print(f"\nbrecha Lima - prueba: {brecha['delta']:+.4f}  "
          f"IC95% [{brecha['ic95'][0]:+.4f}, {brecha['ic95'][1]:+.4f}]  "
          f"{'significativa' if brecha['significativa'] else 'NO se distingue del ruido'}")
    print("(intervalo no pareado: son chips distintos, ver docstring)")

    args.salida.write_text(json.dumps({
        "corrida": args.corrida, "umbral": args.umbral,
        "remuestreos": args.remuestreos,
        "lima": {"chips": len(usados), **m_lima},
        "test": {"chips": int(len(conf_test)), **m_test},
        "brecha_lima_menos_test": brecha,
        "granule": "S1B_IW_GRDH_1SDV_20170317T103823_20170317T103848"
                   "_004747_0084AE_040D",
        "limite": ("etiquetas de EMSN038 derivadas de optico WorldView-2 del 18 "
                   "y 21 de marzo contra escena SAR del 17: supervision debil "
                   "propia, no verdad de terreno. Ademas el RTC de HyP3 aplica "
                   "correccion radiometrica de terreno que el S1_GRD de GEE "
                   "usado por Sen1Floods11 no aplica."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
