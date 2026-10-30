"""Genera PNG navegables de los chips, para poder mirarlos sin un SIG.

Los datos de este proyecto no se abren en un visor de imágenes común:

* los chips SAR son GeoTIFF **float32 en decibeles**, con valores negativos
  (VV va de -30 a +3 dB). Un visor los interpreta como 0..255 y muestra negro.
* las etiquetas son GeoTIFF **int8 con tres valores**: 1 agua, 0 no agua y -1
  "sin dato". Un visor muestra casi negro y el -1 se confunde con el 0, que es
  justo la distinción que importa.

Así que acá se convierten a PNG legibles: el SAR se estira a los percentiles 2
y 98 de **cada banda** (no al min/max: un píxel de -40 dB aplastaría todo lo
demás) y la etiqueta usa tres colores bien separados.

Se arma además un ``indice.html`` para recorrerlos en el navegador.

Usage:
    python pregunta2/paso6_figuras/previsualizar.py
    python pregunta2/paso6_figuras/previsualizar.py --que lima
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import sys

# Las rutas de datos salen de pregunta2/rutas.py, que prefiere
# pregunta2/datos/ (los enlaces) y cae a data/ si no existen.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from pregunta2 import rutas as _rutas
from typing import Optional, Sequence

import numpy as np

# Las rutas de abajo se anclan a ESTE archivo, no al directorio de
# trabajo: la pregunta se comparte sola y sus scripts se corren desde
# cualquier lado.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]

SALIDA = _RAIZ / "pregunta2/paso6_figuras/vistas"
MANO = _rutas.MANO
LIMA = _rutas.LIMA

#: 1 agua (azul), 0 no agua (gris oscuro), -1 sin dato (gris claro).
COLORES = {1: (40, 130, 255), 0: (52, 62, 74), -1: (214, 214, 214)}


def sar_a_rgb(a: np.ndarray) -> np.ndarray:
    """(2, alto, ancho) en dB -> RGB. R=VV, G=VH, B=VV-VH.

    La tercera banda es la diferencia porque es la que separa agua de tierra:
    el agua calma devuelve poco en las dos polarizaciones, pero la relación
    VV/VH (que en dB es una resta) cambia según el mecanismo de dispersión.
    Darle un canal propio hace visible a simple vista lo que el modelo usa.
    """
    def estirar(v):
        v = np.where(np.isfinite(v), v, np.nan)
        lo, hi = np.nanpercentile(v, [2, 98])
        v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
        return np.nan_to_num(v)

    vv, vh = estirar(a[0]), estirar(a[1])
    return (np.stack([vv, vh, estirar(a[0] - a[1])], axis=-1) * 255).astype(np.uint8)


def etiqueta_a_rgb(m: np.ndarray) -> np.ndarray:
    salida = np.zeros((*m.shape, 3), dtype=np.uint8)
    for valor, color in COLORES.items():
        salida[m == valor] = color
    return salida


def guardar(arr: np.ndarray, ruta: pathlib.Path) -> None:
    from PIL import Image
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(ruta)


def superponer(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """SAR con el agua marcada encima, que es como de verdad se revisa un chip."""
    salida = rgb.copy()
    agua = m == 1
    salida[agua] = (0.35 * salida[agua] +
                    0.65 * np.array(COLORES[1])).astype(np.uint8)
    sin = m == -1
    salida[sin] = (0.55 * salida[sin] + 0.45 * 255).astype(np.uint8)
    return salida


def hacer_sen1floods(destino: pathlib.Path, splits: Sequence[str]) -> list[dict]:
    import rasterio
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from pregunta2.paso1_datos.sen1floods11 import chips_del_split

    filas = []
    for split in splits:
        for chip in chips_del_split(split):
            s1 = MANO / "S1Hand" / f"{chip}_S1Hand.tif"
            lbl = MANO / "LabelHand" / f"{chip}_LabelHand.tif"
            if not (s1.is_file() and lbl.is_file()):
                continue
            with rasterio.open(s1) as r:
                a = r.read().astype(np.float32)
            with rasterio.open(lbl) as r:
                m = r.read(1)
            base = destino / split
            guardar(sar_a_rgb(a), base / f"{chip}_sar.png")
            guardar(etiqueta_a_rgb(m), base / f"{chip}_etiqueta.png")
            guardar(superponer(sar_a_rgb(a), m), base / f"{chip}_super.png")
            obs = int((m != -1).sum())
            filas.append({
                "grupo": split, "chip": chip,
                "region": chip.split("_")[0],
                "agua": int((m == 1).sum()), "observados": obs,
                "pct_agua": 100 * int((m == 1).sum()) / max(1, obs),
                "sar": f"{split}/{chip}_sar.png",
                "etiqueta": f"{split}/{chip}_etiqueta.png",
                "super": f"{split}/{chip}_super.png"})
    return filas


def hacer_lima(destino: pathlib.Path) -> list[dict]:
    import rasterio

    manifiesto = json.loads((LIMA / "chips.json").read_text(encoding="utf-8"))
    filas = []
    for c in manifiesto["chips"]:
        ruta = LIMA / "LabelLima" / f"{c['chip']}_LabelLima.tif"
        if not ruta.is_file():
            continue
        with rasterio.open(ruta) as r:
            m = r.read(1)
        guardar(etiqueta_a_rgb(m), destino / "lima" / f"{c['chip']}_etiqueta.png")
        filas.append({
            "grupo": "lima", "chip": c["chip"], "region": "Lima",
            "agua": c["pixeles_agua"], "observados": c["pixeles_observados"],
            "pct_agua": 100 * c["pixeles_agua"] / max(1, c["pixeles_observados"]),
            "sar": None,     # todavia no hay imagen SAR de Lima
            "etiqueta": f"lima/{c['chip']}_etiqueta.png", "super": None})
    return filas


def escribir_indice(filas: list[dict], destino: pathlib.Path) -> None:
    """Una pagina estatica: abrirla con doble clic, sin servidor ni nada."""
    por_grupo: dict[str, list[dict]] = {}
    for f in filas:
        por_grupo.setdefault(f["grupo"], []).append(f)

    partes = ["""<!doctype html><meta charset="utf-8">
<title>Chips</title><style>
body{font:14px system-ui;margin:0;background:#11151a;color:#e6e6e6}
header{padding:16px 20px;background:#0b0e12;position:sticky;top:0;border-bottom:1px solid #222}
h2{margin:28px 20px 8px;font-size:16px}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px;padding:0 20px}
.c{background:#1a2027;border-radius:8px;padding:8px;border:1px solid #242c35}
.c img{width:100%;display:block;border-radius:4px;image-rendering:pixelated}
.r{display:flex;gap:4px}.r>div{flex:1;min-width:0}
.n{font:12px ui-monospace;margin:6px 0 2px;overflow:hidden;text-overflow:ellipsis}
.m{font-size:11px;color:#8aa}
small{color:#8aa;font-weight:normal}
</style><header><b>Chips de la Pregunta 2</b> &mdash;
<small>izquierda: SAR (R=VV, G=VH, B=VV&minus;VH) &nbsp;|&nbsp;
centro: etiqueta (<span style="color:#288dff">azul</span> agua,
<span style="color:#9aa">gris oscuro</span> no agua,
<span style="color:#d6d6d6">gris claro</span> sin dato) &nbsp;|&nbsp;
derecha: superpuestas</small></header>"""]
    titulos = {"test": "PRUEBA en distribucion (entregable 3) &mdash; 90 chips",
               "bolivia": "BOLIVIA &mdash; region retenida de respaldo (entregable 4)",
               "lima": "LIMA &mdash; region principal del entregable 4 "
                       "(etiquetas listas; falta la imagen SAR)",
               "train": "entrenamiento", "valid": "validacion"}
    for grupo in ("test", "bolivia", "lima", "valid", "train"):
        filas_g = por_grupo.get(grupo)
        if not filas_g:
            continue
        filas_g.sort(key=lambda f: -f["pct_agua"])
        partes.append(f"<h2>{titulos.get(grupo, grupo)} "
                      f"<small>({len(filas_g)} chips, ordenados por % de agua)"
                      f"</small></h2><div class=g>")
        for f in filas_g:
            imgs = "".join(
                f"<div><img src='{html.escape(f[k])}' loading=lazy></div>"
                for k in ("sar", "etiqueta", "super") if f[k])
            partes.append(
                f"<div class=c><div class=r>{imgs}</div>"
                f"<div class=n>{html.escape(f['chip'])}</div>"
                f"<div class=m>{f['agua']:,} px agua &middot; "
                f"{f['pct_agua']:.2f}% de lo observado</div></div>")
        partes.append("</div>")
    (destino / "indice.html").write_text("\n".join(partes), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--que", default="test,bolivia,lima",
                        help="grupos separados por coma: test, bolivia, lima, "
                             "valid, train")
    parser.add_argument("--destino", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    querides = [q.strip() for q in args.que.split(",") if q.strip()]
    args.destino.mkdir(parents=True, exist_ok=True)

    filas = []
    splits = [q for q in querides if q != "lima"]
    if splits:
        filas += hacer_sen1floods(args.destino, splits)
    if "lima" in querides and (LIMA / "chips.json").is_file():
        filas += hacer_lima(args.destino)

    escribir_indice(filas, args.destino)
    por_grupo: dict[str, int] = {}
    for f in filas:
        por_grupo[f["grupo"]] = por_grupo.get(f["grupo"], 0) + 1
    for g, n in por_grupo.items():
        print(f"  {g:<10} {n:>4} chips")
    print(f"\nabri: {args.destino.resolve() / 'indice.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
