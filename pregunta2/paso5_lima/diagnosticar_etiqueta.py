"""Mide por qué el modelo falla en Lima, en vez de suponerlo.

`evaluar_lima.py` deja un hecho sin explicar: IoU 0.0028 en Lima contra 0.6808
en prueba. Un número así admite dos lecturas incompatibles ---o el modelo no
generaliza, o la etiqueta no marca lo que el modelo predice--- y de cuál sea
depende qué conclusión escribe el informe. Acá se separan con tres mediciones:

1. **Firma de radar de cada capa candidata del geodatabase**, contra la firma
   del agua dorada de Sen1Floods11. Si la capa que se usó como verdad no es
   oscura en VV, entonces no marca superficie de agua en el sentido en que el
   modelo aprendió «agua», y el IoU bajo no mide generalización.

2. **Otsu con el mismo código sobre Lima y sobre Bolivia.** Otsu no aprende
   nada: elige su umbral dentro de cada imagen. Si también fracasa en Lima y no
   en Bolivia, el fallo no puede ser del modelo ---no hay modelo.

   Deliberadamente **no** se usa el `S1OtsuLabelHand` precalculado que trae el
   dataset para el lado de Bolivia: compararlo contra un Otsu propio en Lima
   mezclaría la diferencia entre regiones con la diferencia entre dos
   implementaciones. Las dos mitades se calculan acá con la misma función.

   Y la diferencia entre implementaciones no es menor, así que se mide también
   el `S1OtsuLabelHand` sobre Bolivia para dejarla a la vista: un Otsu ingenuo
   sobre VV rinde bastante menos que el del conjunto, que es una cadena con
   preprocesamiento propio y no un umbral de una línea. El informe cita las dos
   cosas y no puede dar a entender que son el mismo número.

3. **Fracción de píxeles oscuros dentro y fuera de la etiqueta.** Es la prueba
   directa: si hay proporcionalmente más píxeles oscuros *fuera* de lo
   etiquetado como agua que dentro, la etiqueta está anticorrelacionada con la
   señal, y ningún umbral ni modelo puede recuperarla.

Uso:
    python pregunta2/paso5_lima/diagnosticar_etiqueta.py
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

from pregunta2.paso1_datos.sen1floods11 import (  # noqa: E402
    MANO, chips_del_split,
)
from pregunta2.paso4_generalizacion.comparar import (  # noqa: E402
    confusion_otsu_por_chip, iou_de,
)
from pregunta2.paso5_lima.construir_region import (  # noqa: E402
    chips_de_lima,
)

GDB = _RAIZ / "data/lima/EMSN038_UTM18S_v0.gdb"
REGION = _RAIZ / "data/lima/region"
SALIDA = _RAIZ / "pregunta2/paso5_lima/diagnostico_etiqueta.json"

#: Las capas del geodatabase que podrían hacer de verdad de terreno. La primera
#: es la que `construir_region.py` usa hoy; las otras tres son las candidatas a
#: reemplazarla, de más a menos amplia.
CAPAS_CANDIDATAS = ("Potential_Maximum_flood", "Riverbank", "WatrcrsA",
                    "LakeresA")

#: Umbral de «oscuro en VV». No es un umbral de decisión ---para eso está
#: Otsu---, es solo el punto donde el agua lisa deja de confundirse con suelo en
#: los histogramas de `paso7_fallos_sar` (la zona ambigua empieza en -17.75 dB y
#: el suelo domina por encima de -15).
UMBRAL_OSCURO_DB = -15.0


def _vv_de_chip(ruta: pathlib.Path) -> np.ndarray:
    import rasterio
    with rasterio.open(ruta) as r:
        return r.read(1)


def firma_por_capa(gdb: pathlib.Path, region: pathlib.Path,
                   capas: Sequence[str] = CAPAS_CANDIDATAS) -> dict:
    """Mediana de VV dentro de cada capa, rasterizada a la grilla de los chips.

    Rasterizar y no muestrear puntos: la pregunta es qué retrodispersión tienen
    los píxeles que la capa declara agua, y eso es exactamente el conjunto que
    `rasterize` devuelve con la misma grilla con la que se construyó la etiqueta.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    resultados = {}
    for capa in capas:
        try:
            geo = gpd.read_file(gdb, layer=capa)
        except Exception as exc:                     # capa ausente en el .gdb
            resultados[capa] = {"error": str(exc)[:200]}
            continue

        valores = []
        for ruta_img in sorted((region / "S1Lima").glob("*_S1Lima.tif")):
            with rasterio.open(ruta_img) as r:
                vv = r.read(1)
                geo_chip = geo.to_crs(r.crs)
                if geo_chip.is_empty.all():
                    continue
                dentro = rasterize(
                    [(g, 1) for g in geo_chip.geometry if g is not None
                     and not g.is_empty],
                    out_shape=vv.shape, transform=r.transform,
                    fill=0, dtype="uint8", all_touched=False)
            sel = (dentro == 1) & np.isfinite(vv)
            if sel.any():
                valores.append(vv[sel])

        if not valores:
            resultados[capa] = {"pixeles": 0}
            continue
        v = np.concatenate(valores)
        resultados[capa] = {
            "pixeles": int(v.size),
            "mediana_vv_db": float(np.median(v)),
            "media_vv_db": float(v.mean()),
            "p10_vv_db": float(np.percentile(v, 10)),
            "fraccion_oscura": float((v < UMBRAL_OSCURO_DB).mean()),
        }
    return resultados


def firma_agua_dorada(chips: Sequence[str], mano: pathlib.Path = MANO) -> dict:
    """La misma medición sobre el agua que el modelo sí aprendió.

    Es la referencia contra la que las capas de Lima se vuelven interpretables:
    sin ella, «-9 dB» no dice si es agua o no.
    """
    import rasterio

    agua, suelo = [], []
    for chip in chips:
        with rasterio.open(mano / "S1Hand" / f"{chip}_S1Hand.tif") as r:
            vv = r.read(1)
        with rasterio.open(mano / "LabelHand" / f"{chip}_LabelHand.tif") as r:
            etiqueta = r.read(1)
        observado = (etiqueta != -1) & np.isfinite(vv)
        agua.append(vv[observado & (etiqueta == 1)])
        suelo.append(vv[observado & (etiqueta == 0)])
    a, s = np.concatenate(agua), np.concatenate(suelo)
    return {
        "agua": {"pixeles": int(a.size), "mediana_vv_db": float(np.median(a)),
                 "fraccion_oscura": float((a < UMBRAL_OSCURO_DB).mean())},
        "suelo": {"pixeles": int(s.size), "mediana_vv_db": float(np.median(s)),
                  "fraccion_oscura": float((s < UMBRAL_OSCURO_DB).mean())},
    }


def otsu_por_chip(chips: Sequence[str], mano: pathlib.Path, sufijo_imagen: str,
                  sufijo_etiqueta: str) -> np.ndarray:
    """(n_chips, 4) de confusión umbralizando VV con Otsu dentro de cada chip.

    La misma función para las dos regiones: es lo único que hace comparable el
    resultado de Lima con el de Bolivia.
    """
    import rasterio
    from skimage.filters import threshold_otsu

    filas = []
    for chip in chips:
        with rasterio.open(
                mano / sufijo_imagen / f"{chip}_{sufijo_imagen}.tif") as r:
            vv = r.read(1)
        with rasterio.open(
                mano / sufijo_etiqueta / f"{chip}_{sufijo_etiqueta}.tif") as r:
            etiqueta = r.read(1)

        observado = (etiqueta != -1) & np.isfinite(vv)
        if observado.sum() < 2 or np.ptp(vv[observado]) == 0:
            # Un chip sin variación no tiene umbral que elegir; cuenta como
            # todo-suelo en vez de desaparecer de la medición.
            filas.append([0, 0, int((etiqueta[observado] == 1).sum()),
                          int((etiqueta[observado] == 0).sum())])
            continue

        umbral = threshold_otsu(vv[observado])
        pred = vv < umbral                       # agua = la moda oscura
        p, v = pred[observado], (etiqueta[observado] == 1)
        filas.append([int((p & v).sum()), int((p & ~v).sum()),
                      int((~p & v).sum()), int((~p & ~v).sum())])
    return np.array(filas, dtype=np.float64)


def contraste_oscuro(region: pathlib.Path, chips: Sequence[str]) -> dict:
    """Fracción de píxeles oscuros dentro y fuera de la etiqueta de Lima.

    Si el cociente es menor que 1, la etiqueta está anticorrelacionada con la
    señal que define agua en SAR, y no hay umbral que la recupere.
    """
    import rasterio

    dentro, fuera = [], []
    for chip in chips:
        with rasterio.open(region / "S1Lima" / f"{chip}_S1Lima.tif") as r:
            vv = r.read(1)
        with rasterio.open(
                region / "LabelLima" / f"{chip}_LabelLima.tif") as r:
            etiqueta = r.read(1)
        observado = (etiqueta != -1) & np.isfinite(vv)
        dentro.append(vv[observado & (etiqueta == 1)])
        fuera.append(vv[observado & (etiqueta == 0)])
    d, f = np.concatenate(dentro), np.concatenate(fuera)
    frac_d = float((d < UMBRAL_OSCURO_DB).mean())
    frac_f = float((f < UMBRAL_OSCURO_DB).mean())
    return {
        "umbral_db": UMBRAL_OSCURO_DB,
        "etiquetado_agua": {"pixeles": int(d.size),
                            "mediana_vv_db": float(np.median(d)),
                            "fraccion_oscura": frac_d},
        "etiquetado_suelo": {"pixeles": int(f.size),
                             "mediana_vv_db": float(np.median(f)),
                             "fraccion_oscura": frac_f},
        "cociente_dentro_sobre_fuera": frac_d / frac_f if frac_f else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdb", type=pathlib.Path, default=GDB)
    parser.add_argument("--region", type=pathlib.Path, default=REGION)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    chips_lima = chips_de_lima(args.region)
    chips_bolivia = chips_del_split("bolivia")
    print(f"{len(chips_lima)} chips de Lima | {len(chips_bolivia)} de Bolivia\n")

    print("1. firma de radar por capa del geodatabase")
    capas = firma_por_capa(args.gdb, args.region)
    dorada = firma_agua_dorada(chips_bolivia)
    print(f"{'capa':<26} {'mediana VV':>11} {'oscuro':>8} {'pixeles':>10}")
    print("-" * 58)
    for nombre, d in capas.items():
        if "mediana_vv_db" not in d:
            print(f"{nombre:<26} {'(sin datos)':>11}")
            continue
        print(f"{nombre:<26} {d['mediana_vv_db']:>8.2f} dB "
              f"{d['fraccion_oscura']:>7.1%} {d['pixeles']:>10,}")
    print("-" * 58)
    for clase in ("agua", "suelo"):
        d = dorada[clase]
        print(f"{'Sen1Floods11: ' + clase:<26} {d['mediana_vv_db']:>8.2f} dB "
              f"{d['fraccion_oscura']:>7.1%} {d['pixeles']:>10,}")

    print("\n2. Otsu con el mismo codigo en las dos regiones")
    conf_lima = otsu_por_chip(chips_lima, args.region, "S1Lima", "LabelLima")
    conf_bol = otsu_por_chip(chips_bolivia, MANO, "S1Hand", "LabelHand")
    iou_lima, iou_bol = iou_de(conf_lima), iou_de(conf_bol)
    iou_bol_dataset = iou_de(confusion_otsu_por_chip(chips_bolivia))
    print(f"  Lima    IoU {iou_lima:.4f}  ({len(conf_lima)} chips)")
    print(f"  Bolivia IoU {iou_bol:.4f}  ({len(conf_bol)} chips)")
    print("  Otsu no aprende nada: si tambien falla en Lima, el fallo no es "
          "del modelo.")
    print(f"  (referencia: el S1Otsu precalculado del conjunto rinde "
          f"{iou_bol_dataset:.4f} en Bolivia; es una cadena afinada, no este "
          f"umbral de una linea)")

    print("\n3. contraste de pixeles oscuros dentro vs fuera de la etiqueta")
    contraste = contraste_oscuro(args.region, chips_lima)
    dentro = contraste["etiquetado_agua"]["fraccion_oscura"]
    fuera = contraste["etiquetado_suelo"]["fraccion_oscura"]
    print(f"  dentro de 'agua': {dentro:.1%} oscuro   "
          f"fuera: {fuera:.1%} oscuro   "
          f"cociente {contraste['cociente_dentro_sobre_fuera']:.2f}")

    args.salida.write_text(json.dumps({
        "umbral_oscuro_db": UMBRAL_OSCURO_DB,
        "firma_capas_lima": capas,
        "firma_sen1floods11_bolivia": dorada,
        "otsu_mismo_codigo": {
            "lima": {"iou_agua": float(iou_lima), "chips": len(conf_lima)},
            "bolivia": {"iou_agua": float(iou_bol), "chips": len(conf_bol)},
            "nota": ("umbral de Otsu por chip sobre VV, la misma funcion en "
                     "las dos regiones"),
        },
        "otsu_precalculado_del_conjunto": {
            "bolivia": {"iou_agua": float(iou_bol_dataset)},
            "nota": ("S1OtsuLabelHand que trae Sen1Floods11: cadena con "
                     "preprocesamiento propio, no comparable con la de arriba "
                     "y sin equivalente para Lima"),
        },
        "contraste_oscuro_lima": contraste,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
