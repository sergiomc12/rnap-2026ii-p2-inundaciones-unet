"""Las figuras del entregable 5: el error, al lado de la señal que lo explica.

Una figura de segmentación que muestra solo *verdad contra predicción* deja al
lector sin lo único que decide si el fallo es del sensor: **cómo se veía el
píxel**. Cada panel lleva el VV en decibelios junto al mapa de errores, y los
histogramas llevan marcada la **zona ambigua** —el tramo de VV donde agua y
suelo conviven—, de modo que un error que cae ahí se puede ver, no solo leer en
una tabla.

No se importa matplotlib a nivel de módulo: `analizar_fallos.py` corre bien con
`--sin-figuras` y en ese caso no hay por qué pagar el import.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import rasterio

from pregunta2.paso1_datos.sen1floods11 import MANO

#: Los dB en que se dibuja VV. Fuera de este rango la imagen se satura: el p1 de
#: una escena ronda -25 dB y el p99 -4, así que estirar más solo agrega ruido.
RANGO_VV = (-25.0, -4.0)

#: Rojo = el modelo vio agua donde no había; azul = la perdió.
COLOR_FP = (0.95, 0.25, 0.20)
COLOR_FN = (0.15, 0.45, 0.95)
COLOR_VP = (0.30, 0.85, 0.45)
COLOR_AGUA = (0.20, 0.50, 0.95)
COLOR_SECO = (0.65, 0.55, 0.40)


def _leer(chip: str, sufijo: str, mano: pathlib.Path = MANO):
    ruta = mano / sufijo / f"{chip}_{sufijo}.tif"
    if not ruta.is_file():
        return None
    with rasterio.open(ruta) as r:
        return r.read() if sufijo == "S1Hand" else r.read(1)


def _a_gris(vv: np.ndarray) -> np.ndarray:
    bajo, alto = RANGO_VV
    g = np.clip((np.nan_to_num(vv, nan=bajo) - bajo) / (alto - bajo), 0, 1)
    return np.repeat(g[:, :, None], 3, axis=2)


def _superponer(vv: np.ndarray, vp, fp, fn) -> np.ndarray:
    vista = _a_gris(vv)
    for mascara, color in ((vp, COLOR_VP), (fn, COLOR_FN), (fp, COLOR_FP)):
        if mascara is not None and mascara.any():
            vista[mascara] = vista[mascara] * 0.35 + np.array(color) * 0.65
    return np.clip(vista, 0, 1)


def _predecir(modelo, norm, x, dispositivo):
    import torch
    with torch.no_grad():
        t = torch.from_numpy(norm(x.copy()))[None].to(dispositivo)
        return (torch.sigmoid(modelo(t))[0, 0].cpu().numpy() > 0.5)


def _marcar_zona(ax, zona, etiqueta: bool = True) -> None:
    bajo, alto = zona
    if not (np.isfinite(bajo) and np.isfinite(alto)):
        return
    ax.axvspan(bajo, alto, color="0.45", alpha=0.20, zorder=0,
               label="zona ambigua" if etiqueta else None)


def panel_de_chip(chip: str, modelo, norm, dispositivo, destino: pathlib.Path,
                  zona: Optional[tuple] = None) -> Optional[pathlib.Path]:
    """Cuatro vistas de un chip: VV, verdad, errores, y dónde caen en VV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    x = _leer(chip, "S1Hand")
    etiqueta = _leer(chip, "LabelHand")
    if x is None or etiqueta is None:
        return None
    pred = _predecir(modelo, norm, x, dispositivo)
    valido = (etiqueta != -1) & np.isfinite(x[0]) & np.isfinite(x[1])
    verdad = (etiqueta == 1) & valido
    vp, fp, fn = pred & verdad, pred & ~verdad & valido, ~pred & verdad & valido

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    ax[0].imshow(_a_gris(x[0]))
    ax[0].set_title(f"VV (dB, {RANGO_VV[0]:.0f} a {RANGO_VV[1]:.0f})", fontsize=9)

    v = np.full(etiqueta.shape + (3,), 0.15)
    v[verdad] = COLOR_AGUA
    v[~valido] = 0.45          # gris: sin etiqueta utilizable
    ax[1].imshow(v)
    ax[1].set_title(f"verdad — agua {100*verdad.sum()/max(1,valido.sum()):.1f}% "
                    f"de lo observado", fontsize=9)

    ax[2].imshow(_superponer(x[0], vp, fp, fn))
    inter = int(vp.sum())
    union = int((pred & valido).sum() + verdad.sum() - inter)
    iou = inter / union if union else float("nan")
    ax[2].set_title(f"errores — IoU {iou:.3f}  FP {int(fp.sum()):,}  "
                    f"FN {int(fn.sum()):,}", fontsize=9)
    ax[2].legend(handles=[mp.Patch(color=COLOR_VP, label="acierto agua"),
                          mp.Patch(color=COLOR_FP, label="falso positivo"),
                          mp.Patch(color=COLOR_FN, label="falso negativo")],
                 loc="lower right", fontsize=7, framealpha=0.85)

    # El cuarto panel es el que responde la pregunta del entregable.
    bordes = np.linspace(-32, 0, 65)
    _marcar_zona(ax[3], zona or (np.nan, np.nan))
    for mascara, color, nombre in ((vp, COLOR_VP, "acierto agua"),
                                   (fp, COLOR_FP, "falso positivo"),
                                   (fn, COLOR_FN, "falso negativo")):
        if mascara.any():
            ax[3].hist(x[0][mascara], bins=bordes, density=True, alpha=0.55,
                       color=color, label=nombre)
    ax[3].set_xlabel("VV (dB)", fontsize=8)
    ax[3].set_ylabel("densidad", fontsize=8)
    ax[3].set_title("¿el error cae en la zona ambigua?", fontsize=9)
    ax[3].legend(fontsize=7)
    ax[3].tick_params(labelsize=7)

    for a in ax[:3]:
        a.axis("off")
    fig.suptitle(chip, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    destino.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destino, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return destino


def figura_separabilidad(resultado: dict, destino: pathlib.Path) -> pathlib.Path:
    """Las densidades de VV de agua y suelo, su solape, y dónde caen los errores.

    Es la figura central del entregable: muestra que el solape existe en los
    datos, antes de cualquier modelo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = resultado["histogramas_vv"]
    bordes = np.asarray(h["bordes"], float)
    centros = (bordes[:-1] + bordes[1:]) / 2
    z = resultado["separabilidad_vv"]
    zona = z["zona_ambigua_db"]

    def dens(clave):
        a = np.asarray(h[clave], float)
        s = a.sum()
        return a / s if s else a

    agua, seco = dens("vv_agua"), dens("vv_seco")
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={"height_ratios": [1.25, 1]})

    _marcar_zona(ax[0], zona)
    ax[0].fill_between(centros, agua, color=COLOR_AGUA, alpha=0.55,
                       label="agua (verdad)")
    ax[0].fill_between(centros, seco, color=COLOR_SECO, alpha=0.55,
                       label="suelo (verdad)")
    ax[0].fill_between(centros, np.minimum(agua, seco), color="0.15",
                       alpha=0.75,
                       label=f"solape = {100*z['coeficiente_solape']:.1f}%")
    ax[0].set_ylabel("densidad", fontsize=9)
    ax[0].set_title(
        f"Lo que el sensor no puede separar: solape "
        f"{100*z['coeficiente_solape']:.1f}% de las densidades de VV\n"
        f"zona ambigua {zona[0]:.1f} a {zona[1]:.1f} dB · error mínimo con solo "
        f"VV: {100*z['error_bayes_con_solo_vv']:.1f}%", fontsize=11)
    ax[0].legend(fontsize=8)

    _marcar_zona(ax[1], zona, etiqueta=False)
    ax[1].fill_between(centros, dens("vv_fp"), color=COLOR_FP, alpha=0.6,
                       label="falso positivo")
    ax[1].fill_between(centros, dens("vv_fn"), color=COLOR_FN, alpha=0.6,
                       label="falso negativo")
    ax[1].set_xlabel("VV (dB)", fontsize=9)
    ax[1].set_ylabel("densidad", fontsize=9)
    fp = resultado["reparto_fp_por_zona"]
    fn = resultado["reparto_fn_por_zona"]
    ax[1].set_title(
        f"Y dónde caen los errores: {100*fp['zona_ambigua']/max(1,fp['total']):.0f}% "
        f"de los FP y {100*fn['zona_ambigua']/max(1,fn['total']):.0f}% de los FN "
        f"están en la zona ambigua", fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].set_xlim(-32, 0)

    plt.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destino, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return destino


def dibujar_todo(resultado: dict, modelo, norm, dispositivo,
                 carpeta: pathlib.Path) -> list[pathlib.Path]:
    carpeta = pathlib.Path(carpeta)
    zona = tuple(resultado["separabilidad_vv"]["zona_ambigua_db"])
    escritas = [figura_separabilidad(resultado, carpeta / "separabilidad_vv.png")]
    vistos = set()
    for modo, chips in resultado["ejemplos"].items():
        for chip in chips:
            if chip in vistos:
                continue
            vistos.add(chip)
            r = panel_de_chip(chip, modelo, norm, dispositivo,
                              carpeta / f"{modo}__{chip}.png", zona)
            if r:
                escritas.append(r)
    print(f"\n{len(escritas)} figuras en {carpeta}")
    for f in escritas:
        print(f"  {f.name}")
    return escritas
