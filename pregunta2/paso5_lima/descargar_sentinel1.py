"""Baja la escena Sentinel-1 de Lima desde HyP3 y la recorta a los 58 chips.

Es la mitad del entregable 4 que faltaba. `construir_region.py` ya dejó la
grilla y las etiquetas; esto trae el lado SAR y lo deja en el mismo formato que
`S1Hand` de Sen1Floods11, para que el modelo entrenado no note la diferencia.

**Por qué HyP3 y no el GRD crudo.** Sen1Floods11 se construyó con
`COPERNICUS/S1_GRD` de Google Earth Engine, que entrega sigma0 en decibelios ya
calibrado y corregido por terreno. Procesar el GRD crudo con otra cadena metería
un confundidor de preprocesamiento justo en el entregable que mide
generalización. HyP3 (NASA/ASF) produce esos mismos tres parámetros bajo demanda
---`radiometry='sigma0'`, `scale='decibel'`, `resolution=10`--- así que el
producto es comparable sin tener que montar la cadena a mano.

**La diferencia que queda, y hay que declararla.** HyP3 aplica corrección
radiométrica de terreno (aplana el relieve con un DEM); el `S1_GRD` de GEE aplica
corrección geométrica y calibración a sigma0, sin ese aplanado. En terreno plano
la diferencia es despreciable; en el valle del Rímac, con relieve, algo pesa.
Es mucho menor que procesar el GRD crudo, pero no es cero.

**La alineación no se calcula, se hereda.** Cada chip se reproyecta a la grilla
que YA tiene su etiqueta correspondiente en `LabelLima/`: se lee el `transform`,
el `crs` y la forma del GeoTIFF de la etiqueta y se reproyecta la escena a eso
exacto. Así la correspondencia píxel a píxel entre imagen y etiqueta queda
garantizada por construcción, en vez de depender de que dos cálculos
independientes de la grilla coincidan.

Uso:
    python pregunta2/paso5_lima/descargar_sentinel1.py
    python pregunta2/paso5_lima/descargar_sentinel1.py --escena pre --solo-lanzar
    python pregunta2/paso5_lima/descargar_sentinel1.py --escena pre
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import zipfile

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

#: Las dos adquisiciones del mismo path 171 descendente, VV+VH, ambas a las
#: 10:38:23 ---o sea geometría idéntica, no solo órbita parecida: mismo ángulo de
#: incidencia píxel a píxel, que es lo que hace comparable un par pre/durante.
#:
#: `evento` es la de durante: la delineación óptica de EMSN038 es del 18 y 21 de
#: marzo, así que el desfase es de un día. `pre` es 12 días antes, un ciclo
#: completo de repetición de Sentinel-1B. La tercera de la serie (29 de marzo) es
#: post-evento y no se usa: para detección de cambio hace falta el antes, no el
#: después.
ESCENAS = {
    "evento": ("S1B_IW_GRDH_1SDV_20170317T103823_20170317T103848"
               "_004747_0084AE_040D", "S1Lima"),
    "pre": ("S1B_IW_GRDH_1SDV_20170305T103823_20170305T103848"
            "_004572_007F86_4C55", "S1LimaPre"),
}

#: Los tres que alinean el producto con Sen1Floods11. Ninguno es el default de
#: HyP3 (que son gamma0, power y 30 m), así que van explícitos.
RADIOMETRIA = "sigma0"
ESCALA = "decibel"
RESOLUCION = 10


def leer_token(raiz: pathlib.Path) -> str:
    """El token de Earthdata, del .env. Nunca se imprime ni se pasa por CLI."""
    env = raiz / ".env"
    if not env.exists():
        raise SystemExit(f"falta {env} con TOKEN_HYP3")
    for linea in env.read_text(encoding="utf-8", errors="replace").splitlines():
        if linea.startswith("TOKEN_HYP3="):
            return linea.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("TOKEN_HYP3 no esta en .env")


def nombre_trabajo(granule: str) -> str:
    """`lima_2017_03_17` a partir del granule.

    Derivado y no escrito a mano para que el nombre siga a la escena: es la
    clave con la que `find_jobs` reconoce un trabajo ya corrido, así que si los
    dos se separan, el script relanza algo que ya está hecho y gasta créditos.
    """
    fecha = granule[17:25]
    return f"lima_{fecha[:4]}_{fecha[4:6]}_{fecha[6:8]}"


def obtener_producto(destino: pathlib.Path, token: str, granule: str,
                     *, esperar: bool = True) -> pathlib.Path | None:
    """El .zip del RTC, lanzando el trabajo si todavia no existe.

    Con `esperar=False` devuelve `None` si el trabajo está en curso, en vez de
    bloquear los ~75 min que tarda HyP3.
    """
    import hyp3_sdk

    nombre = nombre_trabajo(granule)
    hyp3 = hyp3_sdk.HyP3(token=token)
    trabajos = [t for t in hyp3.find_jobs(name=nombre)
                if t.status_code != "FAILED"]
    if not trabajos:
        print(f"lanzando RTC sobre {granule}")
        lote = hyp3.submit_rtc_job(
            granule=granule, name=nombre, resolution=RESOLUCION,
            radiometry=RADIOMETRIA, scale=ESCALA)
        trabajos = lote.jobs

    trabajo = trabajos[0]
    if trabajo.status_code != "SUCCEEDED":
        if not esperar:
            print(f"{nombre}: en curso (estado {trabajo.status_code}), "
                  f"volver a correr esto cuando termine")
            return None
        print(f"esperando a HyP3 (estado {trabajo.status_code})...")
        trabajo = hyp3.watch(trabajo)

    destino.mkdir(parents=True, exist_ok=True)
    existente = list(destino.glob("*.zip"))
    if existente:
        print(f"ya estaba descargado: {existente[0].name}")
        return existente[0]
    print("descargando el producto...")
    return trabajo.download_files(destino)[0]


def bandas_del_zip(ruta_zip: pathlib.Path) -> dict:
    """{polarizacion: ruta} de los GeoTIFF que trae el producto."""
    carpeta = ruta_zip.parent
    with zipfile.ZipFile(ruta_zip) as z:
        z.extractall(carpeta)
    bandas = {}
    for tif in carpeta.rglob("*.tif"):
        for pol in ("VV", "VH"):
            # El RTC trae mas capas (DEM, mascara, angulo); solo interesan las
            # dos polarizaciones, y se reconocen por el sufijo del nombre.
            if tif.stem.endswith(f"_{pol}"):
                bandas[pol] = tif
    faltan = {"VV", "VH"} - set(bandas)
    if faltan:
        raise SystemExit(f"el producto no trae {faltan}: {sorted(carpeta.rglob('*.tif'))}")
    return bandas


def recortar(bandas: dict, etiquetas: pathlib.Path, salida: pathlib.Path,
             sufijo: str = "S1Lima") -> int:
    """Un GeoTIFF de 2 bandas por chip, en la grilla exacta de su etiqueta.

    El sufijo del nombre de archivo es el que después lee `ChipsSen1Floods11`
    vía `sufijo_imagen`, así que coincide con el de la carpeta de salida.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    salida.mkdir(parents=True, exist_ok=True)
    fuentes = {pol: rasterio.open(ruta) for pol, ruta in bandas.items()}
    escritos = 0
    try:
        for ruta_etiqueta in sorted(etiquetas.glob("*_LabelLima.tif")):
            with rasterio.open(ruta_etiqueta) as ref:
                perfil_destino = {"crs": ref.crs, "transform": ref.transform,
                                  "width": ref.width, "height": ref.height}

            pila = np.full((2, perfil_destino["height"], perfil_destino["width"]),
                           np.nan, dtype=np.float32)
            for i, pol in enumerate(("VV", "VH")):
                fuente = fuentes[pol]
                reproject(
                    source=rasterio.band(fuente, 1),
                    destination=pila[i],
                    src_transform=fuente.transform, src_crs=fuente.crs,
                    dst_transform=perfil_destino["transform"],
                    dst_crs=perfil_destino["crs"],
                    # Bilineal y no vecino mas cercano: la escena viene a 10 m y
                    # el destino tambien, asi que el remuestreo es casi una
                    # traslacion; bilineal evita el aliasing del submuestreo sin
                    # suavizar de mas.
                    resampling=Resampling.bilinear,
                    src_nodata=fuente.nodata, dst_nodata=np.nan)

            # HyP3 marca el vacio con 0 en vez de nodata en algunos productos:
            # un 0 exacto en dB seria 1.0 en potencia, fisicamente improbable
            # sobre agua o suelo, asi que se trata como sin dato.
            pila[pila == 0] = np.nan

            nombre = ruta_etiqueta.stem.replace("_LabelLima", "")
            with rasterio.open(
                    salida / f"{nombre}_{sufijo}.tif", "w", driver="GTiff",
                    height=perfil_destino["height"], width=perfil_destino["width"],
                    count=2, dtype="float32", crs=perfil_destino["crs"],
                    transform=perfil_destino["transform"], nodata=np.nan,
                    compress="deflate") as destino:
                destino.write(pila)
            escritos += 1
    finally:
        for f in fuentes.values():
            f.close()
    return escritos


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=pathlib.Path, default=RAIZ)
    parser.add_argument("--escena", choices=sorted(ESCENAS), default="evento",
                        help="'evento' (17-mar) o 'pre' (5-mar, 12 dias antes)")
    parser.add_argument("--solo-lanzar", action="store_true",
                        help="encolar en HyP3 y salir, sin esperar los ~75 min")
    parser.add_argument("--solo-descargar", action="store_true",
                        help="traer el producto sin recortar los chips")
    args = parser.parse_args(argv)

    granule, sufijo = ESCENAS[args.escena]
    region = args.project_dir / "data" / "lima" / "region"
    etiquetas = region / "LabelLima"
    if not etiquetas.is_dir():
        raise SystemExit(f"falta {etiquetas}: correr antes construir_region.py")

    # Un subdirectorio por escena: `obtener_producto` localiza el .zip ya
    # descargado con un glob, asi que dos escenas en la misma carpeta harian que
    # la segunda encontrara el .zip de la primera y recortara la escena errada.
    ruta_zip = obtener_producto(region / "sentinel1" / args.escena,
                                leer_token(args.project_dir), granule,
                                esperar=not args.solo_lanzar)
    if ruta_zip is None:
        return 0
    print(f"producto: {ruta_zip.name}")
    if args.solo_descargar:
        return 0

    bandas = bandas_del_zip(ruta_zip)
    print(f"bandas: {', '.join(f'{k}={v.name}' for k, v in bandas.items())}")
    n = recortar(bandas, etiquetas, region / sufijo, sufijo)
    print(f"{n} chips escritos en {region / sufijo}")

    esperados = len(list(etiquetas.glob("*_LabelLima.tif")))
    if n != esperados:
        print(f"ATENCION: {esperados} etiquetas pero {n} chips de imagen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
