"""Cuánto separa cada banda óptica el agua del suelo, y cómo se compara con VV.

Es el número que justifica el bono entero: el paso 7 midió que VV y VH dejan un
**41.2% de solape** entre las dos clases, y la pregunta es si el óptico lo
resuelve. Esto lo contesta sobre los mismos chips dorados, con la misma máscara.

Se mide de dos formas, porque dicen cosas distintas:

* **separación de medianas** — cuánto se corren los dos centros. Es intuitiva
  pero no dice nada sobre el ancho de las distribuciones;
* **coeficiente de solape** — el área común de las dos densidades, con el mismo
  cálculo que usa el paso 7 sobre VV. Es lo que permite poner el óptico y el
  radar en la misma tabla y compararlos de verdad.

**Por qué existe este archivo.** El README de este paso publicaba la tabla de
separación medida con un script suelto que no quedó en ninguna parte: los
números no se podían auditar ni volver a correr.

Usage:
    python pregunta2/paso8_fusion_s2/medir_separacion.py
    python pregunta2/paso8_fusion_s2/medir_separacion.py --chips 40
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

import numpy as np
import rasterio

# Las rutas de abajo se anclan a ESTE archivo, no al directorio de
# trabajo: la pregunta se comparte sola y sus scripts se corren desde
# cualquier lado.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

from pregunta2.paso1_datos.sen1floods11 import MANO, chips_del_split  # noqa: E402
from pregunta2.paso8_fusion_s2.datos_fusion import (  # noqa: E402
    BANDAS_S2, hay_optico, indice_normalizado, leer_s2,
)

SALIDA = _RAIZ / "pregunta2/paso8_fusion_s2/separacion_optica.json"

#: Las capas que se miden, cada una con su rango y el paso del histograma. Las
#: reflectancias viven en [0, 1]; los índices normalizados, en [-1, 1].
CAPAS = (
    ("verde B3", "banda", "B3"),
    ("rojo B4", "banda", "B4"),
    ("NIR B8", "banda", "B8"),
    ("SWIR B11", "banda", "B11"),
    ("NDWI", "indice", ("B3", "B8")),
    ("MNDWI", "indice", ("B3", "B11")),
)

BORDES_REFLECTANCIA = np.linspace(0.0, 1.0, 201)
BORDES_INDICE = np.linspace(-1.0, 1.0, 201)


def capa_de(s2: np.ndarray, tipo: str, cual) -> np.ndarray:
    if tipo == "banda":
        return s2[BANDAS_S2.index(cual)]
    a, b = cual
    return indice_normalizado(s2[BANDAS_S2.index(a)], s2[BANDAS_S2.index(b)])


def coeficiente_solape(h_agua: np.ndarray, h_seco: np.ndarray) -> float:
    """Área común de las dos densidades. Mismo cálculo que el paso 7 sobre VV."""
    pa = h_agua / max(1, h_agua.sum())
    ps = h_seco / max(1, h_seco.sum())
    return float(np.minimum(pa, ps).sum())


def mediana_de(hist: np.ndarray, bordes: np.ndarray) -> Optional[float]:
    n = hist.sum()
    if n == 0:
        return None
    centros = (bordes[:-1] + bordes[1:]) / 2
    acum = np.cumsum(hist) / n
    return float(centros[int(np.searchsorted(acum, 0.5))])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conjuntos", nargs="+", default=["test"])
    parser.add_argument("--chips", type=int, default=0,
                        help="limitar a los primeros N (0 = todos)")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    chips = []
    for c in args.conjuntos:
        chips += [x for x in chips_del_split(c) if hay_optico(x)]
    if args.chips:
        chips = chips[:args.chips]
    if not chips:
        print("no hay chips con S2Hand. Corre:\n"
              "  python pregunta2/paso1_datos/descargar.py "
              "--solo-lo-pedido --s2-dorado")
        return 1
    print(f"{len(chips)} chips de {', '.join(args.conjuntos)}\n")

    H = {n: {"agua": np.zeros(len(BORDES_REFLECTANCIA) - 1, np.int64),
             "seco": np.zeros(len(BORDES_REFLECTANCIA) - 1, np.int64)}
         for n, _, _ in CAPAS}
    for i, chip in enumerate(chips, 1):
        with rasterio.open(MANO / "LabelHand" / f"{chip}_LabelHand.tif") as r:
            etiqueta = r.read(1)
        s2 = leer_s2(chip)
        # Los bordes sin dato de Sentinel-2 vienen en cero en TODAS las bandas;
        # contarlos como "suelo oscuro" inventaria una separacion que no existe.
        valido = (etiqueta != -1) & (s2.sum(0) > 0)
        agua, seco = (etiqueta == 1) & valido, (etiqueta == 0) & valido
        if not (agua.any() or seco.any()):
            continue
        for nombre, tipo, cual in CAPAS:
            v = capa_de(s2, tipo, cual)
            b = BORDES_INDICE if tipo == "indice" else BORDES_REFLECTANCIA
            if agua.any():
                H[nombre]["agua"] += np.histogram(v[agua], bins=b)[0]
            if seco.any():
                H[nombre]["seco"] += np.histogram(v[seco], bins=b)[0]
        if i % 30 == 0 or i == len(chips):
            print(f"  {i}/{len(chips)} chips", flush=True)

    print(f"\n{'capa':<12} {'mediana agua':>13} {'mediana suelo':>14} "
          f"{'separacion':>11} {'solape':>8}")
    print("-" * 64)
    filas = []
    for nombre, tipo, _ in CAPAS:
        b = BORDES_INDICE if tipo == "indice" else BORDES_REFLECTANCIA
        ma = mediana_de(H[nombre]["agua"], b)
        ms = mediana_de(H[nombre]["seco"], b)
        sol = coeficiente_solape(H[nombre]["agua"], H[nombre]["seco"])
        if ma is None or ms is None:
            continue
        filas.append({"capa": nombre, "mediana_agua": round(ma, 4),
                      "mediana_suelo": round(ms, 4),
                      "separacion": round(ma - ms, 4),
                      "coeficiente_solape": round(sol, 4),
                      "pixeles_agua": int(H[nombre]["agua"].sum()),
                      "pixeles_suelo": int(H[nombre]["seco"].sum())})
        print(f"{nombre:<12} {ma:>13.3f} {ms:>14.3f} {ma-ms:>+11.3f} "
              f"{100*sol:>7.1f}%")

    mejor = min(filas, key=lambda f: f["coeficiente_solape"])
    print("-" * 64)
    print(f"la capa que mejor separa es {mejor['capa']}, con "
          f"{100*mejor['coeficiente_solape']:.1f}% de solape.")
    print(f"VV, de referencia (paso 7):                     41.2% de solape.")

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(
        {"conjuntos": args.conjuntos, "chips": len(chips),
         "solape_vv_de_referencia": 0.412, "capas": filas,
         "mejor_capa": mejor["capa"]}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nescrito: {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
