"""Baja Sen1Floods11 desde el bucket oficial, solo lo que hace falta.

El dataset completo pesa 32.4 GB, pero **el 77% de eso es óptico Sentinel-2**
(`S2Weak` solo son 23.9 GB) y el óptico aparece una sola vez en el enunciado,
en el entregable 6, que es un bono. Bajando únicamente lo que necesitan los
entregables 1 a 5 son **7.5 GB**.

Medido contra el bucket, y coincide con lo que dice el enunciado:

| carpeta | archivos | GB | para qué |
|---|---:|---:|---|
| `HandLabeled/S1Hand` | 446 | 0.68 | SAR dorado (VV, VH en dB) |
| `HandLabeled/LabelHand` | 446 | 0.004 | **máscara dorada**, anotada a mano |
| `HandLabeled/S1OtsuLabelHand` | 446 | 0.002 | umbralización clásica sobre los mismos chips |
| `HandLabeled/JRCWaterHand` | 446 | 0.001 | agua **permanente** (JRC) |
| `WeaklyLabeled/S1Weak` | 4 384 | 6.82 | SAR débil |
| `WeaklyLabeled/S1OtsuLabelWeak` | 4 384 | 0.02 | máscara débil, por umbralización |

``JRCWaterHand`` entra aunque el enunciado no lo pida: marca el agua que ya
estaba antes de la inundación (ríos, lagos). Sin eso, un modelo que aprenda a
segmentar "agua" se lleva el río Rímac entero como si fuera inundación, y la
discusión de fallos del entregable 5 queda sin una de sus piezas.

``S1OtsuLabelHand`` también: son las etiquetas por umbralización **sobre los
chips dorados**, o sea la línea base clásica contra la que se compara una U-Net
en el mismo conjunto. Pesa 2 MB.

El bucket es público y se lee por HTTP sin credenciales. Se baja con reintentos
y de forma reanudable: un archivo ya presente con el tamaño correcto no se
vuelve a pedir.

Usage:
    python pregunta2/paso1_datos/descargar.py
    python pregunta2/paso1_datos/descargar.py --con-optico   # agrega el bono
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
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Sequence

BUCKET = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
DESCARGA = "https://storage.googleapis.com/sen1floods11"
RAIZ = "v1.1/data/flood_events"
DESTINO = _rutas.SEN1FLOODS11

#: Lo que necesitan los entregables 1 a 5.
CARPETAS_SAR = [
    f"{RAIZ}/HandLabeled/S1Hand",
    f"{RAIZ}/HandLabeled/LabelHand",
    f"{RAIZ}/HandLabeled/S1OtsuLabelHand",
    f"{RAIZ}/HandLabeled/JRCWaterHand",
    f"{RAIZ}/WeaklyLabeled/S1Weak",
    f"{RAIZ}/WeaklyLabeled/S1OtsuLabelWeak",
]
#: Solo para el entregable 6 (bono): fusión SAR + óptico.
CARPETAS_OPTICO = [
    f"{RAIZ}/HandLabeled/S2Hand",
    f"{RAIZ}/WeaklyLabeled/S2Weak",
    f"{RAIZ}/WeaklyLabeled/S2IndexLabelWeak",
]
#: Las particiones oficiales del dataset. Pesan nada y definen qué chips son de
#: Bolivia, que es la región de respaldo del entregable 4.
SPLITS = "v1.1/splits/flood_handlabeled"


def listar(prefijo: str, intentos: int = 5) -> list[tuple[str, int]]:
    """(nombre, tamaño) de cada objeto bajo un prefijo, paginando.

    **Reintenta con espera creciente**, igual que ``bajar_uno``. Sin esto, un
    ``HTTP 503`` del bucket —que aparece en cuanto hay varias descargas
    compitiendo— tiraba la excepción y mataba el script **antes de bajar un
    solo archivo**, sin dejar nada en el log: el inventario es lo primero que
    corre. Bajar 7.5 GB y que lo tumbe una respuesta transitoria del listado no
    tiene sentido.
    """
    objetos, token = [], None
    while True:
        url = f"{BUCKET}?prefix={prefijo}&maxResults=1000"
        if token:
            url += f"&pageToken={token}"
        for intento in range(1, intentos + 1):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    datos = json.load(r)
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if intento == intentos:
                    raise RuntimeError(
                        f"el bucket no respondio al listar {prefijo} despues de "
                        f"{intentos} intentos ({type(e).__name__}). Suele ser "
                        f"limitacion temporal: volve a correrlo en unos minutos, "
                        f"es reanudable.") from e
                espera = 5 * intento
                print(f"    {type(e).__name__} al listar; reintento "
                      f"{intento}/{intentos - 1} en {espera}s", flush=True)
                time.sleep(espera)
        objetos += [(o["name"], int(o.get("size", 0)))
                    for o in datos.get("items", []) if not o["name"].endswith("/")]
        token = datos.get("nextPageToken")
        if not token:
            return objetos


def bajar_uno(nombre: str, tamano: int, destino: pathlib.Path,
              intentos: int = 4) -> str:
    """'ok', 'ya' o 'fallo'. Reanudable: compara el tamaño antes de pedir."""
    ruta = destino / nombre
    if ruta.is_file() and ruta.stat().st_size == tamano:
        return "ya"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    for intento in range(1, intentos + 1):
        try:
            # Al archivo temporal primero: un corte de red no deja un .tif a
            # medias que la proxima corrida tome por completo.
            parcial = ruta.with_suffix(ruta.suffix + ".parcial")
            with urllib.request.urlopen(f"{DESCARGA}/{nombre}", timeout=120) as r, \
                    parcial.open("wb") as mano:
                while True:
                    trozo = r.read(1 << 20)
                    if not trozo:
                        break
                    mano.write(trozo)
            parcial.replace(ruta)
            return "ok"
        except (urllib.error.URLError, TimeoutError, OSError):
            if intento == intentos:
                return "fallo"
            time.sleep(3 * intento)
    return "fallo"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=pathlib.Path, default=DESTINO)
    parser.add_argument("--con-optico", action="store_true",
                        help="agrega TODO Sentinel-2 (+24.9 GB, casi todo debil)")
    parser.add_argument("--s2-dorado", action="store_true",
                        help="solo S2Hand (0.95 GB): los 446 chips opticos que "
                             "emparejan con el conjunto dorado. Es lo unico que "
                             "necesita el bono del entregable 6")
    parser.add_argument("--solo-lo-pedido", action="store_true",
                        help="no baja el SAR, solo las carpetas que se pidan "
                             "con los flags de arriba")
    parser.add_argument("--hilos", type=int, default=8)
    args = parser.parse_args(argv)

    # El bono necesita S2Hand y nada mas: 0.95 GB contra los 24.9 de --con-optico,
    # cuyo peso es casi todo S2Weak (23.9 GB de optico DEBIL, que no empareja con
    # ninguna mascara dorada y por lo tanto no sirve para medir la fusion).
    carpetas = [] if args.solo_lo_pedido else list(CARPETAS_SAR) + [SPLITS]
    if args.con_optico:
        carpetas += CARPETAS_OPTICO
    elif args.s2_dorado:
        carpetas += [c for c in CARPETAS_OPTICO if c.endswith("S2Hand")]
    if not carpetas:
        print("no hay nada que bajar: con --solo-lo-pedido hay que agregar "
              "--s2-dorado o --con-optico")
        return 1

    print("inventariando el bucket...")
    objetos = []
    for c in carpetas:
        lista = listar(c)
        peso = sum(t for _, t in lista)
        print(f"  {c.split('/')[-1]:20} {len(lista):>6,} archivos  {peso/2**30:>6.2f} GB")
        objetos += lista
    total = sum(t for _, t in objetos)
    print(f"\n{len(objetos):,} archivos | {total/2**30:.2f} GB -> {args.destino}\n")

    comenzado = time.time()
    hechos = ya = fallidos = 0
    with ThreadPoolExecutor(max_workers=args.hilos) as piscina:
        futuros = {piscina.submit(bajar_uno, n, t, args.destino): n
                   for n, t in objetos}
        for i, f in enumerate(as_completed(futuros), start=1):
            r = f.result()
            hechos += r == "ok"; ya += r == "ya"; fallidos += r == "fallo"
            if i % 250 == 0 or i == len(objetos):
                seg = time.time() - comenzado
                print(f"  {i:>6,}/{len(objetos):,} | nuevos {hechos:,} "
                      f"| ya estaban {ya:,} | fallidos {fallidos} "
                      f"| {seg/60:.1f} min | ETA {(len(objetos)-i)*seg/i/60:.1f} min",
                      flush=True)

    print(f"\nnuevos {hechos:,} | ya estaban {ya:,} | FALLIDOS {fallidos}")
    print(f"{(time.time()-comenzado)/60:.1f} min")
    if fallidos:
        print("volve a correrlo: es reanudable y solo pide lo que falta")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
