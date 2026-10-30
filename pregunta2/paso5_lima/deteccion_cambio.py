"""¿La inundación de Lima se ve comparando el antes con el durante?

`diagnosticar_etiqueta.py` dejó demostrado que la etiqueta de EMSN038 no marca
agua oscura. Queda una pregunta que ese diagnóstico no responde y que decide cómo
leer todo el entregable: **¿la inundación es invisible para el sensor, o solo
para un modelo de una sola fecha?**

Un modelo que mira una imagen suelta tiene que decidir «agua» por el valor
absoluto de retrodispersión. La detección de cambio no: compara cada píxel
consigo mismo doce días antes, así que un suelo que se inundó cae varios
decibelios aunque nunca llegue a verse tan oscuro como un lago. Es como Copernicus
EMS mapea inundaciones con SAR de verdad, y es la única vía que quedaba para
rescatar una señal acá.

Las dos escenas son del mismo path 171 descendente y de la misma hora
---10:38:23---, o sea **mismo ángulo de incidencia píxel a píxel**. Sin eso la
diferencia mezclaría geometría con inundación y no significaría nada.

**La diferencia se centra por escena antes de usarla.** Entre dos fechas cambian
la calibración, la humedad del suelo y la vegetación, y eso corre el histograma
entero; lo que importa es el cambio *relativo* al resto de la escena. Sin
centrar, un corrimiento global se confundiría con inundación en todos lados.

**El umbral se elige a favor de la hipótesis que se quiere refutar.** En vez de
fijar un corte y reportar su IoU, se barren todos y se reporta **el mejor**. Ese
número no es alcanzable en la práctica ---elegirlo requiere ver la respuesta---,
y justamente por eso sirve: es una **cota superior** de lo que cualquier detector
de cambio por umbral global podría lograr sobre estos datos. Si ni siquiera el
oráculo separa, no hay umbral que lo haga.

Uso:
    python pregunta2/paso5_lima/deteccion_cambio.py
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

from pregunta2.paso4_generalizacion.comparar import iou_de  # noqa: E402
from pregunta2.paso5_lima.construir_region import (  # noqa: E402
    chips_de_lima,
)

REGION = _RAIZ / "data/lima/region"
SALIDA = _RAIZ / "pregunta2/paso5_lima/deteccion_cambio.json"
FIGURA = _RAIZ / "pregunta2/paso9_informe/informe/figuras/cambio_lima.png"

#: Caída de retrodispersión que la literatura usa como indicio de inundación.
#: Se reporta además del barrido, por ser el valor que alguien elegiría sin
#: mirar la respuesta.
UMBRAL_CAIDA_DB = -3.0

#: Los cortes que se barren para la cota superior, en dB de cambio centrado.
BARRIDO_DB = np.arange(-12.0, 0.01, 0.25)


def _leer(region: pathlib.Path, chip: str, sufijo: str, banda: int = 1):
    import rasterio
    with rasterio.open(region / sufijo / f"{chip}_{sufijo}.tif") as r:
        return r.read(banda)


def diferencia_centrada(region: pathlib.Path, chip: str, banda: int = 1):
    """(delta_centrado, etiqueta, validos) de un chip.

    `validos` son los píxeles observados en **las dos** escenas y con etiqueta:
    un píxel que solo tiene una de las dos fechas no tiene diferencia que medir.
    """
    durante = _leer(region, chip, "S1Lima", banda)
    antes = _leer(region, chip, "S1LimaPre", banda)
    etiqueta = _leer(region, chip, "LabelLima")

    validos = (np.isfinite(durante) & np.isfinite(antes) & (etiqueta != -1))
    delta = np.full(durante.shape, np.nan, dtype=np.float32)
    delta[validos] = durante[validos] - antes[validos]
    if validos.any():
        # Centrado por escena: lo que importa es el cambio relativo al resto.
        delta[validos] -= np.median(delta[validos])
    return delta, etiqueta, validos


def juntar(region: pathlib.Path, chips: Sequence[str], banda: int = 1):
    """Todos los chips apilados en tres vectores planos."""
    deltas, etiquetas = [], []
    for chip in chips:
        delta, etiqueta, validos = diferencia_centrada(region, chip, banda)
        if not validos.any():
            continue
        deltas.append(delta[validos])
        etiquetas.append(etiqueta[validos] == 1)
    return np.concatenate(deltas), np.concatenate(etiquetas)


def confusion_con_umbral(delta: np.ndarray, agua: np.ndarray,
                         umbral: float) -> np.ndarray:
    """[vp, fp, fn, vn] tratando «cayó más que el umbral» como agua."""
    predicho = delta < umbral
    return np.array([
        int(np.count_nonzero(predicho & agua)),
        int(np.count_nonzero(predicho & ~agua)),
        int(np.count_nonzero(~predicho & agua)),
        int(np.count_nonzero(~predicho & ~agua)),
    ], dtype=np.float64)


def cota_superior(delta: np.ndarray, agua: np.ndarray,
                  barrido: np.ndarray = BARRIDO_DB) -> dict:
    """El mejor IoU alcanzable barriendo el umbral, y dónde se alcanza.

    No es un resultado honesto como detector ---el umbral se eligió viendo la
    verdad--- y por eso es útil: acota por arriba a cualquier detector de cambio
    con umbral global sobre estos datos.
    """
    mejor = {"iou_agua": -1.0, "umbral_db": None}
    curva = []
    for umbral in barrido:
        iou = float(iou_de(confusion_con_umbral(delta, agua, umbral)[None]))
        curva.append({"umbral_db": float(umbral), "iou_agua": iou})
        if iou > mejor["iou_agua"]:
            mejor = {"iou_agua": iou, "umbral_db": float(umbral)}
    return {**mejor, "curva": curva}


def figura(delta: np.ndarray, agua: np.ndarray, curva: Sequence[dict],
           salida: pathlib.Path, maximo: int = 2_000_000,
           semilla: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(semilla)

    def _muestra(v):
        return v if v.size <= maximo else v[rng.choice(v.size, maximo, False)]

    dentro, fuera = _muestra(delta[agua]), _muestra(delta[~agua])

    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    bins = np.linspace(-12, 12, 121)
    ax_h.hist(fuera, bins=bins, density=True, histtype="step", linewidth=1.8,
              color="#8a8a8a", linestyle="--",
              label=f"etiquetado suelo (n={fuera.size/1e6:.1f} M)")
    ax_h.hist(dentro, bins=bins, density=True, histtype="step", linewidth=1.8,
              color="#e8442a",
              label=f"etiquetado agua (n={dentro.size/1e3:.0f} k)")
    ax_h.axvline(UMBRAL_CAIDA_DB, color="#1f6fb4", linewidth=1.0,
                 linestyle=":", label=f"{UMBRAL_CAIDA_DB:.0f} dB de caída")
    ax_h.set_xlabel("cambio de VV, 17-mar menos 5-mar (dB, centrado)")
    ax_h.set_ylabel("densidad")
    ax_h.set_title("Si la inundación se viera, el rojo caería a la izquierda",
                   fontsize=9)
    ax_h.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax_h.spines[["top", "right"]].set_visible(False)

    umbrales = [c["umbral_db"] for c in curva]
    ious = [c["iou_agua"] for c in curva]
    ax_c.plot(umbrales, ious, color="#1f6fb4", linewidth=1.8)
    mejor = max(curva, key=lambda c: c["iou_agua"])
    ax_c.plot([mejor["umbral_db"]], [mejor["iou_agua"]], "o", color="#e8442a",
              markersize=5)
    # Abajo-izquierda del pico: arriba choca con el titulo del panel.
    ax_c.annotate(f"mejor: {mejor['iou_agua']:.4f}\nen {mejor['umbral_db']:.2f} dB",
                  (mejor["umbral_db"], mejor["iou_agua"]), fontsize=7.5,
                  xytext=(-14, -30), textcoords="offset points",
                  ha="right", va="top")
    ax_c.set_ylim(0, mejor["iou_agua"] * 1.25)
    ax_c.set_xlabel("umbral de caída (dB)")
    ax_c.set_ylabel("IoU de agua")
    ax_c.set_title("Cota superior de cualquier umbral global", fontsize=9)
    ax_c.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=190, bbox_inches="tight")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=pathlib.Path, default=REGION)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--figura", type=pathlib.Path, default=FIGURA)
    parser.add_argument("--sin-figura", action="store_true")
    args = parser.parse_args(argv)

    pre = args.region / "S1LimaPre"
    if not pre.is_dir():
        print(f"falta {pre}: correr antes\n"
              f"  python pregunta2/paso5_lima/descargar_sentinel1.py --escena pre")
        return 1

    chips = chips_de_lima(args.region)
    print(f"{len(chips)} chips con imagen del evento y etiqueta\n")

    resultados = {}
    for banda, nombre in ((1, "VV"), (2, "VH")):
        delta, agua = juntar(args.region, chips, banda)
        d_dentro, d_fuera = delta[agua], delta[~agua]
        cota = cota_superior(delta, agua)
        conf_fijo = confusion_con_umbral(delta, agua, UMBRAL_CAIDA_DB)

        resultados[nombre] = {
            "pixeles_comparables": int(delta.size),
            "mediana_cambio_db": {
                "etiquetado_agua": float(np.median(d_dentro)),
                "etiquetado_suelo": float(np.median(d_fuera)),
            },
            "fraccion_que_cae_mas_de_3db": {
                "etiquetado_agua": float((d_dentro < UMBRAL_CAIDA_DB).mean()),
                "etiquetado_suelo": float((d_fuera < UMBRAL_CAIDA_DB).mean()),
            },
            "umbral_fijo_3db": {
                "umbral_db": UMBRAL_CAIDA_DB,
                "iou_agua": float(iou_de(conf_fijo[None])),
            },
            "cota_superior_barriendo_umbral": {
                k: v for k, v in cota.items() if k != "curva"},
        }

        print(f"--- {nombre} ---")
        print(f"  mediana del cambio   dentro {np.median(d_dentro):+6.2f} dB   "
              f"fuera {np.median(d_fuera):+6.2f} dB")
        print(f"  cae mas de 3 dB      dentro {(d_dentro < UMBRAL_CAIDA_DB).mean():6.1%}   "
              f"fuera {(d_fuera < UMBRAL_CAIDA_DB).mean():6.1%}")
        print(f"  IoU con umbral fijo de {UMBRAL_CAIDA_DB:.0f} dB : "
              f"{iou_de(conf_fijo[None]):.4f}")
        print(f"  IoU MAXIMO barriendo el umbral    : "
              f"{cota['iou_agua']:.4f}  (en {cota['umbral_db']:.2f} dB)")
        print("  ^ cota superior: el umbral se eligio viendo la verdad\n")

        if nombre == "VV" and not args.sin_figura:
            figura(delta, agua, cota["curva"], args.figura)
            print(f"  figura en {args.figura}\n")

    referencia = {
        "iou_unet_una_fecha": 0.0027841466727741493,
        "nota": ("de evaluacion_lima.json; la comparacion que importa es si la "
                 "deteccion de cambio supera a la U-Net de una sola fecha"),
    }
    args.salida.write_text(json.dumps({
        "granule_durante": "S1B_IW_GRDH_1SDV_20170317T103823_20170317T103848"
                           "_004747_0084AE_040D",
        "granule_antes": "S1B_IW_GRDH_1SDV_20170305T103823_20170305T103848"
                         "_004572_007F86_4C55",
        "chips": len(chips),
        "centrado_por_escena": True,
        "bandas": resultados,
        "referencia": referencia,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
