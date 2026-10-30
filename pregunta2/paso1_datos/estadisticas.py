"""Mide las estadísticas de Sen1Floods11 antes de fijar la normalización.

Hay una tentación fuerte de copiar las constantes de ImageNet
(``mean=(0.485, 0.456, 0.406)``) porque están en todos los cuadernos. Acá no
tienen ningún sentido: la entrada no son tres canales de color entre 0 y 1, son
**dos canales de retrodispersión en decibeles**, que son negativos y se mueven
en un rango de decenas. Normalizar con números de fotos dejaría la entrada
descentrada y mal escalada.

Así que se miden. Lo que sale de acá alimenta ``sen1floods11.py``:

* media y desvío por banda (VV, VH) sobre el **split de entrenamiento**, nunca
  sobre validación ni prueba: calcular la normalización con datos de prueba es
  fuga, chica pero fuga.
* los percentiles, para elegir dónde recortar los valores extremos. El SAR tiene
  cola larga hacia abajo (sombra de radar, agua muy calma) y hacia arriba
  (rebote doble en zona urbana inundada).
* la proporción de píxeles ``-1``. El dataset marca con -1 lo que no se puede
  etiquetar, y **la pérdida tiene que ignorarlos**: contarlos como "no agua"
  metería una supervisión falsa en dos tercios de algunos chips.
* la prevalencia de agua, que dice cuán desbalanceada está la tarea.

Usage:
    python pregunta2/paso1_datos/estadisticas.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Las rutas de datos salen de pregunta2/rutas.py, que prefiere
# pregunta2/datos/ (los enlaces) y cae a data/ si no existen.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from pregunta2 import rutas as _rutas
from typing import Optional, Sequence

import numpy as np
import rasterio

# Las rutas de abajo se anclan a ESTE archivo, no al directorio de
# trabajo: la pregunta se comparte sola y sus scripts se corren desde
# cualquier lado.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]

RAIZ = _rutas.V11
MANO = _rutas.MANO
SPLITS = _rutas.SPLITS
SALIDA = _RAIZ / "pregunta2/paso1_datos/estadisticas.json"

IGNORAR = -1
BANDAS = ("VV", "VH")


def chips_del_split(nombre: str) -> list[str]:
    """Los identificadores (``Region_12345``) que lista un CSV de particion."""
    ruta = SPLITS / f"flood_{nombre}_data.csv"
    filas = ruta.read_text(encoding="utf-8").strip().splitlines()
    # El CSV trae el nombre del .tif de S1 en la primera columna.
    return [f.split(",")[0].replace("_S1Hand.tif", "") for f in filas if f.strip()]


def acumular(chips: Sequence[str]) -> dict:
    """Una sola pasada: suma, suma de cuadrados y conteos. Sin cargar todo en RAM."""
    n = np.zeros(len(BANDAS), dtype=np.float64)
    suma = np.zeros(len(BANDAS), dtype=np.float64)
    suma2 = np.zeros(len(BANDAS), dtype=np.float64)
    minimo = np.full(len(BANDAS), np.inf)
    maximo = np.full(len(BANDAS), -np.inf)
    muestras = []                      # para percentiles, submuestreado
    etiquetas = {-1: 0, 0: 0, 1: 0}
    faltantes = []
    rng = np.random.default_rng(0)

    for chip in chips:
        s1 = MANO / "S1Hand" / f"{chip}_S1Hand.tif"
        lbl = MANO / "LabelHand" / f"{chip}_LabelHand.tif"
        if not (s1.is_file() and lbl.is_file()):
            faltantes.append(chip)
            continue
        with rasterio.open(s1) as r:
            a = r.read().astype(np.float64)
        # Los -inf aparecen cuando la retrodispersion es 0 antes del log.
        valido = np.isfinite(a)
        for b in range(len(BANDAS)):
            v = a[b][valido[b]]
            if v.size == 0:
                continue
            n[b] += v.size
            suma[b] += v.sum()
            suma2[b] += (v * v).sum()
            minimo[b] = min(minimo[b], v.min())
            maximo[b] = max(maximo[b], v.max())
        # Submuestreo para los percentiles: 2000 pixeles por chip alcanzan y
        # evitan juntar 100 M de valores en memoria.
        plano = a.reshape(len(BANDAS), -1)
        fin = np.isfinite(plano).all(axis=0)
        idx = np.flatnonzero(fin)
        if idx.size:
            muestras.append(plano[:, rng.choice(idx, min(2000, idx.size),
                                                replace=False)])

        with rasterio.open(lbl) as r:
            m = r.read(1)
        u, c = np.unique(m, return_counts=True)
        for k, v in zip(u.tolist(), c.tolist()):
            etiquetas[k] = etiquetas.get(k, 0) + v

    media = suma / np.maximum(n, 1)
    varianza = suma2 / np.maximum(n, 1) - media ** 2
    return {
        "chips": len(chips), "faltantes": faltantes,
        "media": media, "desvio": np.sqrt(np.maximum(varianza, 0)),
        "min": minimo, "max": maximo,
        "muestras": np.concatenate(muestras, axis=1) if muestras else None,
        "etiquetas": etiquetas,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    particiones = {n: chips_del_split(n)
                   for n in ("train", "valid", "test", "bolivia")}
    print("particiones oficiales del dataset:")
    for n, c in particiones.items():
        regiones = sorted({x.split("_")[0] for x in c})
        print(f"  {n:<9} {len(c):>4} chips | {len(regiones):>2} regiones")
    print()

    r = acumular(particiones["train"])
    if r["faltantes"]:
        print(f"AVISO: {len(r['faltantes'])} chips del split no estan en disco\n")

    print(f"=== ESTADISTICAS sobre el split de ENTRENAMIENTO "
          f"({r['chips']} chips) ===")
    print(f"{'banda':<6} {'media':>9} {'desvio':>9} {'min':>9} {'max':>9}")
    print("-" * 46)
    for b, nombre in enumerate(BANDAS):
        print(f"{nombre:<6} {r['media'][b]:>9.3f} {r['desvio'][b]:>9.3f} "
              f"{r['min'][b]:>9.2f} {r['max'][b]:>9.2f}")

    pcts = [0.1, 1, 2, 5, 50, 95, 98, 99, 99.9]
    q = np.percentile(r["muestras"], pcts, axis=1)
    print(f"\npercentiles (dB)")
    print("       " + "".join(f"{p:>9}" for p in pcts))
    for b, nombre in enumerate(BANDAS):
        print(f"{nombre:<6} " + "".join(f"{q[i, b]:>9.2f}" for i in range(len(pcts))))

    total = sum(r["etiquetas"].values())
    print(f"\n=== ETIQUETAS ===")
    nombres = {-1: "sin dato (IGNORAR)", 0: "no agua", 1: "agua"}
    for k in (-1, 0, 1):
        v = r["etiquetas"].get(k, 0)
        print(f"  {k:>2} {nombres[k]:<20} {v:>13,}  {100*v/total:>6.2f}%")
    observadas = r["etiquetas"].get(0, 0) + r["etiquetas"].get(1, 0)
    print(f"\n  prevalencia de agua entre las observadas: "
          f"{100*r['etiquetas'].get(1,0)/max(1,observadas):.2f}%")
    print(f"  ({100*r['etiquetas'].get(-1,0)/total:.1f}% de los pixeles NO se "
          f"pueden supervisar: la perdida los tiene que enmascarar)")

    # El recorte se toma en percentiles y no en el min/max: un solo pixel de
    # -50 dB estiraria la escala y aplastaria todo lo demas.
    recorte = [[float(q[1, b]), float(q[-2, b])] for b in range(len(BANDAS))]
    args.salida.write_text(json.dumps({
        "fuente": "split de entrenamiento oficial, sin tocar valid/test/bolivia",
        "chips": r["chips"],
        "bandas": list(BANDAS),
        "media": r["media"].tolist(),
        "desvio": r["desvio"].tolist(),
        "min": r["min"].tolist(),
        "max": r["max"].tolist(),
        "percentiles": {str(p): q[i].tolist() for i, p in enumerate(pcts)},
        "recorte_p1_p99": recorte,
        "etiquetas": {str(k): v for k, v in r["etiquetas"].items()},
        "particiones": {n: len(c) for n, c in particiones.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
