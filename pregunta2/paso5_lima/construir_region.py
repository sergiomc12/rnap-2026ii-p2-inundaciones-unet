"""Construye la región de prueba de Lima: la grilla y las etiquetas.

Esto resuelve la mitad de la región peruana que **no** depende de Sentinel-1: la
grilla de chips y la máscara de referencia. Cuando la imagen SAR esté
disponible se pide exactamente sobre las ventanas que este script deja escritas
en ``chips.json``, y los dos lados quedan alineados píxel a píxel por
construcción en vez de por reproyección posterior.

**De dónde sale la etiqueta.** Del geodatabase de la activación Copernicus EMS
EMSN038, capa ``Potential_Maximum_flood``: 13 polígonos, 13.58 km². Que son la
delineación *observada* y no un escenario hipotético lo confirma la capa
``DamAssess_A``, que evalúa daños sobre esos mismos polígonos vía la columna
``FID_OPTICAL_FloodExtent_OBIA`` —el análisis orientado a objetos sobre las
imágenes ópticas—; no se evalúa daño sobre algo que no pasó.

Se usan **los 13**, no solo los 2 que quedan fuera del cauce. Los otros 11 son
el río a altura máxima (son idénticos a la capa ``Riverbank``, verificado: la
intersección da los 7.24 km² completos). Sen1Floods11 etiqueta **agua**, no
"inundación": en una imagen SAR el cauce también es agua, y su etiqueta dorada
lo incluye. Dejarlo afuera crearía un desacuerdo de criterio con el conjunto de
entrenamiento que se confundiría con un fallo de generalización.

**Dónde se puede afirmar algo y dónde no.** La capa ``Footprints`` trae las 6
huellas de imagen con su fecha. Solo dos son post-evento —**WorldView-2 del
2017-03-18 y del 2017-03-21**, las que menciona el enunciado—. Fuera de esa
unión nadie miró, así que no es "no inundado": es **desconocido**, y va como
``-1``, el mismo valor que Sen1Floods11 usa para lo que no se puede etiquetar.
Escribir 0 ahí inventaría supervisión negativa sobre media Lima y haría ver al
modelo mucho mejor de lo que es.

**Límite que hay que declarar, y el enunciado lo pide explícitamente.** Estas
etiquetas no se generaron con el mismo criterio que las doradas de
Sen1Floods11: vienen de imágenes ópticas de mayor resolución, en fechas que no
son exactamente la de la imagen SAR. Son supervisión débil propia, no una verdad
de terreno perfecta.

Usage:
    python pregunta2/paso5_lima/construir_region.py
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
import warnings
from typing import Optional, Sequence

import numpy as np

warnings.filterwarnings("ignore", message=".*Measured.*")

GDB = _rutas.GDB
DESTINO = _rutas.LIMA

#: Las dos huellas post-evento. El resto son de 2016 y de noviembre: referencia.
FECHAS_POST = ("2017-03-18", "2017-03-21")

#: Tamaño de píxel de Sen1Floods11, en grados. No es un número redondo elegido
#: acá: es el que traen sus GeoTIFF (10 m en EPSG:4326). Se copia exacto para
#: que un chip de Lima sea intercambiable con uno del dataset.
PASO = 8.983152841195215e-05
LADO = 512
IGNORAR = -1


def chips_de_lima(region: pathlib.Path, cobertura_minima: float = 0.01) -> list[str]:
    """Chips con etiqueta Y con imagen que de verdad trae datos.

    El rectángulo de chips se construyó desde la huella de las imágenes ópticas
    de EMSN038, pero la escena Sentinel-1 es una franja con otra geometría: 10
    de los 58 chips caen fuera y quedan enteramente sin dato.

    Esos 10 **sí tienen etiqueta** ---secos, sin un solo píxel de agua---, así
    que si se evaluaran, el `Normalizador` mandaría sus NaN al piso del
    percentil, el modelo predeciría sobre esa entrada inventada, y cada falso
    positivo contaría en contra sin que hubiera habido imagen que mirar. Peor:
    los aciertos triviales sobre 2.6 millones de píxeles secos inflarían la
    exactitud. Se descartan antes de medir nada.
    """
    import rasterio

    etiquetas = {p.stem.replace("_LabelLima", "")
                 for p in (region / "LabelLima").glob("*_LabelLima.tif")}
    usables = []
    for ruta in sorted((region / "S1Lima").glob("*_S1Lima.tif")):
        chip = ruta.stem.replace("_S1Lima", "")
        if chip not in etiquetas:
            continue
        with rasterio.open(ruta) as r:
            cobertura = float(np.isfinite(r.read(1)).mean())
        if cobertura >= cobertura_minima:
            usables.append(chip)
    descartados = len(etiquetas) - len(usables)
    if descartados:
        print(f"{descartados} chips sin cobertura de la escena Sentinel-1, "
              f"descartados antes de medir")
    return usables


def huella_post(gdb: pathlib.Path):
    """Unión de las huellas de imagen posteriores al evento, en EPSG:4326."""
    import geopandas as gpd
    import pandas as pd

    f = gpd.read_file(gdb, layer="Footprints")
    fechas = pd.to_datetime(f["SDV"], utc=True).dt.strftime("%Y-%m-%d")
    post = f[fechas.isin(FECHAS_POST)]
    if post.empty:
        raise ValueError(f"ninguna huella con fecha en {FECHAS_POST}; "
                         f"hay {sorted(fechas.unique())}")
    print(f"huellas post-evento: {len(post)} "
          f"({', '.join(sorted(post['SDP'].unique()))}) "
          f"| {post.geometry.area.sum()/1e6:.0f} km2")
    return post.to_crs(4326).geometry.union_all()


def poligonos_agua(gdb: pathlib.Path):
    import geopandas as gpd

    a = gpd.read_file(gdb, layer="Potential_Maximum_flood")
    print(f"agua delineada: {len(a)} poligonos | "
          f"{a.geometry.area.sum()/1e6:.2f} km2")
    for fuente, sub in a.groupby("source"):
        print(f"    {fuente:<16} {len(sub):>3} poligonos  "
              f"{sub.geometry.area.sum()/1e6:>6.2f} km2")
    return a.to_crs(4326).geometry.union_all()


def grilla(limites, paso: float, lado: int) -> list[tuple[int, int, float, float]]:
    """Esquinas de chips que cubren ``limites``, alineadas a una grilla global.

    La grilla se ancla en múltiplos exactos de ``paso`` desde (0, 0) y no en la
    esquina del AOI. Así dos corridas sobre recortes distintos producen chips que
    caen en las mismas coordenadas, y un chip de Lima se puede comparar con otro
    sin reproyectar nada.
    """
    oeste, sur, este, norte = limites
    i0 = int(np.floor(oeste / (paso * lado)))
    j0 = int(np.floor(sur / (paso * lado)))
    i1 = int(np.ceil(este / (paso * lado)))
    j1 = int(np.ceil(norte / (paso * lado)))
    return [(i, j, i * paso * lado, j * paso * lado)
            for i in range(i0, i1) for j in range(j0, j1)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdb", type=pathlib.Path, default=GDB)
    parser.add_argument("--destino", type=pathlib.Path, default=DESTINO)
    parser.add_argument("--cobertura-minima", type=float, default=0.30,
                        help="fraccion minima del chip dentro de la huella "
                             "post-evento para conservarlo")
    args = parser.parse_args(argv)

    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely.geometry import box

    if not args.gdb.is_dir():
        print(f"no existe {args.gdb}; descomprimi primero el geodatabase EMSN038")
        return 1

    agua = poligonos_agua(args.gdb)
    huella = huella_post(args.gdb)
    print()

    celdas = grilla(huella.bounds, PASO, LADO)
    print(f"grilla candidata: {len(celdas)} chips de {LADO}x{LADO} "
          f"({LADO*PASO*111.32:.1f} km de lado)")

    dir_etiquetas = args.destino / "LabelLima"
    dir_etiquetas.mkdir(parents=True, exist_ok=True)

    manifiesto, conservados = [], 0
    total_agua = total_observado = total_pixeles = 0
    for i, j, oeste, sur in celdas:
        caja = box(oeste, sur, oeste + LADO * PASO, sur + LADO * PASO)
        if not caja.intersects(huella):
            continue
        cobertura = caja.intersection(huella).area / caja.area
        if cobertura < args.cobertura_minima:
            continue

        # from_origin toma la esquina SUPERIOR izquierda y el paso positivo:
        # el eje Y de un raster crece hacia abajo.
        transformacion = from_origin(oeste, sur + LADO * PASO, PASO, PASO)
        dentro = rasterize([(caja.intersection(huella), 1)], out_shape=(LADO, LADO),
                           transform=transformacion, fill=0, dtype="uint8")
        mojado = rasterize([(agua, 1)], out_shape=(LADO, LADO),
                           transform=transformacion, fill=0, dtype="uint8",
                           all_touched=False)
        # Primero todo desconocido, despues se abre lo que la huella cubre.
        etiqueta = np.full((LADO, LADO), IGNORAR, dtype=np.int8)
        etiqueta[dentro == 1] = 0
        etiqueta[(dentro == 1) & (mojado == 1)] = 1

        nombre = f"Lima_{i}_{j}"
        with rasterio.open(
                dir_etiquetas / f"{nombre}_LabelLima.tif", "w", driver="GTiff",
                height=LADO, width=LADO, count=1, dtype="int8",
                crs="EPSG:4326", transform=transformacion,
                nodata=IGNORAR, compress="deflate") as r:
            r.write(etiqueta, 1)

        n_agua = int((etiqueta == 1).sum())
        n_obs = int((etiqueta != IGNORAR).sum())
        total_agua += n_agua; total_observado += n_obs; total_pixeles += LADO * LADO
        conservados += 1
        manifiesto.append({
            "chip": nombre, "i": i, "j": j,
            "oeste": oeste, "sur": sur,
            "este": oeste + LADO * PASO, "norte": sur + LADO * PASO,
            "cobertura_huella": round(cobertura, 4),
            "pixeles_agua": n_agua, "pixeles_observados": n_obs,
        })

    if not conservados:
        print("ningun chip supero la cobertura minima; bajala con --cobertura-minima")
        return 1

    con_agua = sum(1 for m in manifiesto if m["pixeles_agua"] > 0)
    print(f"\n{conservados} chips conservados (cobertura >= "
          f"{args.cobertura_minima:.0%} de la huella post-evento)")
    print(f"  con al menos un pixel de agua : {con_agua}")
    print(f"  completamente secos           : {conservados - con_agua}")
    print(f"  pixeles observados            : {100*total_observado/total_pixeles:.1f}%")
    print(f"  agua entre los observados     : "
          f"{100*total_agua/max(1,total_observado):.2f}%")
    print(f"  (Sen1Floods11 entrenamiento tiene 9.51% de agua: "
          f"Lima es mucho mas seca)")

    (args.destino / "chips.json").write_text(json.dumps({
        "fuente_etiquetas": "Copernicus EMS EMSN038, capa Potential_Maximum_flood",
        "fuente_huella": f"capa Footprints, SDV en {list(FECHAS_POST)}",
        "crs": "EPSG:4326", "paso_grados": PASO, "lado": LADO,
        "cobertura_minima": args.cobertura_minima,
        "chips": manifiesto,
        "limite_declarado":
            "etiquetas derivadas de imagenes opticas WorldView-2 del 18 y 21 de "
            "marzo de 2017, con criterio distinto al de las etiquetas doradas de "
            "Sen1Floods11 y en fecha no identica a la imagen SAR: supervision "
            "debil propia, no verdad de terreno",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito en {args.destino}")
    print(f"  LabelLima/   {conservados} GeoTIFF de etiquetas")
    print(f"  chips.json   las ventanas exactas que hay que pedirle a Sentinel-1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
