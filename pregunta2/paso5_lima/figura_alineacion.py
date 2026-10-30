"""La figura que sostiene la conclusión sobre Lima.

El informe afirma dos cosas que un lector no tiene por qué creer sin verlas: que
la imagen y la etiqueta están bien alineadas ---es decir, que el IoU de 0.0028
no es un error de georreferenciación--- y que lo que la etiqueta marca no es
agua oscura en VV. Las dos se ven en una sola figura de dos paneles:

- **Izquierda**: VV en decibelios del chip con más agua etiquetada, con el
  contorno de la etiqueta encima. Si el contorno sigue el cauce visible, la
  alineación es correcta; si además el interior del contorno no es más oscuro
  que el exterior, la etiqueta no marca superficie de agua.

- **Derecha**: histogramas de VV. Los dos de Lima (dentro y fuera de la
  etiqueta) contra el agua dorada de Sen1Floods11. Es la versión cuantitativa
  del panel izquierdo: si las dos curvas de Lima se superponen entre sí y
  ninguna cae donde está el agua dorada, no hay umbral ni modelo que las separe.

Uso:
    python pregunta2/paso5_lima/figura_alineacion.py
"""

from __future__ import annotations

import argparse
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
from pregunta2.paso5_lima.construir_region import (  # noqa: E402
    chips_de_lima,
)

REGION = _RAIZ / "data/lima/region"
SALIDA = _RAIZ / "pregunta2/paso9_informe/informe/figuras/etiqueta_lima.png"

#: Rango de VV para el panel de imagen. Fijo y no por percentiles de la escena:
#: así el gris de esta figura significa los mismos decibelios que el de
#: `fallo_especular.png`, y las dos se pueden comparar a ojo.
VMIN_DB, VMAX_DB = -25.0, 5.0

#: El mismo umbral de «oscuro» que mide `diagnosticar_etiqueta.py`. Importado
#: y no copiado: la figura ilustra esa medicion, asi que si el numero cambia
#: ahi, tiene que cambiar aca.
from pregunta2.paso5_lima.diagnosticar_etiqueta import (  # noqa: E402
    UMBRAL_OSCURO_DB,
)


def chip_mas_mojado(region: pathlib.Path, chips: Sequence[str]) -> str:
    """El chip con mayor fracción de agua etiquetada entre los observados.

    Elegido por una regla y no a mano: la figura es evidencia, y un chip
    seleccionado a ojo invita a la sospecha de haber buscado el que convenía.
    """
    import rasterio

    mejor, mejor_frac = chips[0], -1.0
    for chip in chips:
        with rasterio.open(
                region / "LabelLima" / f"{chip}_LabelLima.tif") as r:
            etiqueta = r.read(1)
        observado = etiqueta != -1
        if not observado.any():
            continue
        frac = float((etiqueta[observado] == 1).mean())
        if frac > mejor_frac:
            mejor, mejor_frac = chip, frac
    return mejor


def _vv_y_etiqueta(region: pathlib.Path, chip: str):
    import rasterio
    with rasterio.open(region / "S1Lima" / f"{chip}_S1Lima.tif") as r:
        vv = r.read(1)
    with rasterio.open(region / "LabelLima" / f"{chip}_LabelLima.tif") as r:
        etiqueta = r.read(1)
    return vv, etiqueta


def vv_agua_dorada(chips: Sequence[str], mano: pathlib.Path,
                   maximo: int = 3_000_000, semilla: int = 0) -> np.ndarray:
    """VV de los píxeles de agua dorada, submuestreados para el histograma."""
    import rasterio

    rng = np.random.default_rng(semilla)
    partes = []
    for chip in chips:
        with rasterio.open(mano / "S1Hand" / f"{chip}_S1Hand.tif") as r:
            vv = r.read(1)
        with rasterio.open(mano / "LabelHand" / f"{chip}_LabelHand.tif") as r:
            etiqueta = r.read(1)
        sel = (etiqueta == 1) & np.isfinite(vv)
        if sel.any():
            partes.append(vv[sel])
    v = np.concatenate(partes)
    if v.size > maximo:
        v = v[rng.choice(v.size, maximo, replace=False)]
    return v


def main(argv: Optional[Sequence[str]] = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=pathlib.Path, default=REGION)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    chips_lima = chips_de_lima(args.region)
    chip = chip_mas_mojado(args.region, chips_lima)
    print(f"chip elegido por mayor fraccion de agua: {chip}")

    vv, etiqueta = _vv_y_etiqueta(args.region, chip)
    observado = (etiqueta != -1) & np.isfinite(vv)
    dentro = vv[observado & (etiqueta == 1)]
    fuera = vv[observado & (etiqueta == 0)]
    dorada = vv_agua_dorada(chips_del_split("bolivia"), MANO)

    fig, (ax_img, ax_hist) = plt.subplots(
        1, 2, figsize=(11.0, 4.5),
        gridspec_kw={"width_ratios": [1.0, 1.25]})

    # --- panel izquierdo: la imagen con el contorno de la etiqueta ---
    im = ax_img.imshow(np.where(np.isfinite(vv), vv, np.nan), cmap="gray",
                       vmin=VMIN_DB, vmax=VMAX_DB, interpolation="nearest")

    # Los pixeles oscuros, pintados donde estan. Es la version visual del
    # cociente 0.53 que mide `diagnosticar_etiqueta.py`: si el azul cayera
    # dentro del contorno rojo, la etiqueta marcaria agua.
    oscuro = np.isfinite(vv) & (vv < UMBRAL_OSCURO_DB)
    capa = np.zeros((*vv.shape, 4))
    capa[oscuro] = (0.11, 0.44, 0.71, 0.85)
    ax_img.imshow(capa, interpolation="nearest")

    # `contour` sobre la mascara binaria dibuja el borde sin tapar el interior,
    # que es justamente lo que hay que poder mirar.
    ax_img.contour((etiqueta == 1).astype(float), levels=[0.5],
                   colors="#e8442a", linewidths=1.4)
    ax_img.set_title(f"VV (dB), contorno de la etiqueta (rojo)\n"
                     f"y pixeles bajo {UMBRAL_OSCURO_DB:.0f} dB (azul)",
                     fontsize=9)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.03, label="VV (dB)")

    # --- panel derecho: los histogramas ---
    bins = np.linspace(-30, 10, 121)
    for datos, etiq, color, estilo in (
            (dorada, f"agua dorada Sen1Floods11 (n={dorada.size/1e6:.1f} M)",
             "#1f6fb4", "-"),
            (dentro, f"Lima, etiquetado agua (n={dentro.size/1e3:.0f} k)",
             "#e8442a", "-"),
            (fuera, f"Lima, etiquetado suelo (n={fuera.size/1e6:.1f} M)",
             "#8a8a8a", "--")):
        ax_hist.hist(datos, bins=bins, density=True, histtype="step",
                     color=color, linestyle=estilo, linewidth=1.8, label=etiq)
    for datos, color in ((dorada, "#1f6fb4"), (dentro, "#e8442a")):
        ax_hist.axvline(np.median(datos), color=color, linewidth=0.9,
                        alpha=0.55, linestyle=":")
    ax_hist.set_xlabel("VV (dB)")
    ax_hist.set_ylabel("densidad")
    ax_hist.set_title("Distribucion de VV: Lima contra el agua dorada",
                      fontsize=9)
    ax_hist.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax_hist.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.salida, dpi=190, bbox_inches="tight")
    print(f"escrita {args.salida}")
    print(f"  mediana VV dentro de la etiqueta : {np.median(dentro):+7.2f} dB")
    print(f"  mediana VV fuera de la etiqueta  : {np.median(fuera):+7.2f} dB")
    print(f"  mediana VV del agua dorada       : {np.median(dorada):+7.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
