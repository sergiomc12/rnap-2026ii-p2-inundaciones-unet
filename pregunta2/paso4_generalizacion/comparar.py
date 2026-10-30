"""¿La U-Net pierde de verdad en la región retenida, o son 15 chips y ruido?

La corrida base dejó un resultado incómodo y por eso interesante:

| conjunto | U-Net | Otsu | diferencia |
|---|---:|---:|---:|
| prueba (10 regiones) | 0.6808 | 0.5458 | **+0.135** |
| Bolivia (retenida)   | 0.6096 | 0.6973 | **-0.088** |

Leído crudo: lo aprendido le gana al método clásico dentro de la distribución y
**lo pierde fuera**. Es justo lo que pregunta el entregable 4, pero Bolivia son
**15 chips**, así que antes de escribir esa conclusión hay que medir si esa
diferencia se distingue del ruido.

Se usa **bootstrap pareado sobre chips**, no intervalos sueltos. Los dos métodos
se evalúan sobre los mismos chips, así que remuestrear los chips y mirar la
*diferencia* cancela el ruido del conjunto; con intervalos sueltos, dos
intervalos anchos que se solapan no dicen nada aunque la diferencia sea estable.
Es el mismo instrumento que en la Pregunta 1 detectó un +0.050 real cuyos
intervalos sueltos se solapaban por completo.

Detalle que importa: se remuestrean **chips enteros**, nunca píxeles. Los píxeles
de un mismo chip están fuertemente correlacionados —el agua viene en manchas, no
salpicada—, así que tratarlos como independientes daría intervalos
artificialmente angostos y haría ver como significativo casi cualquier cosa.

Usage:
    python pregunta2/paso4_generalizacion/comparar.py
    python pregunta2/paso4_generalizacion/comparar.py --corrida base
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

import numpy as np

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

import rasterio  # noqa: E402
import torch  # noqa: E402

from pregunta2.paso1_datos.sen1floods11 import (  # noqa: E402
    MANO, ChipsSen1Floods11, Normalizador, acumular_confusion, chips_del_split,
)
from pregunta2.paso2_modelo.unet import UNet, ConfigUNet  # noqa: E402

CORRIDAS = _RAIZ / "pregunta2/paso3_entrenamiento/corridas"
SALIDA = _RAIZ / "pregunta2/paso4_generalizacion/comparacion.json"


def iou_de(conf: np.ndarray) -> float:
    """IoU de agua a partir de ``[vp, fp, fn, vn]`` sumados."""
    vp, fp, fn, _ = conf.sum(axis=0) if conf.ndim == 2 else conf
    union = vp + fp + fn
    return vp / union if union > 0 else float("nan")


@torch.no_grad()
def confusion_por_chip(modelo, chips, norm, dispositivo, umbral=0.5) -> np.ndarray:
    """(n_chips, 4) con los conteos de la U-Net, un chip por fila."""
    conjunto = ChipsSen1Floods11(chips, norm)
    filas = []
    modelo.eval()
    for i in range(len(conjunto)):
        x, y, m = conjunto[i]
        logits = modelo(x[None].to(dispositivo))
        filas.append(acumular_confusion(logits.float().cpu(), y[None], m[None], umbral))
    return np.stack(filas), conjunto.chips


def confusion_otsu_por_chip(chips) -> np.ndarray:
    """Lo mismo para la umbralización clásica que trae el dataset."""
    filas = []
    for chip in chips:
        ruta_otsu = MANO / "S1OtsuLabelHand" / f"{chip}_S1OtsuLabelHand.tif"
        ruta_real = MANO / "LabelHand" / f"{chip}_LabelHand.tif"
        with rasterio.open(ruta_otsu) as r:
            pred = r.read(1)
        with rasterio.open(ruta_real) as r:
            real = r.read(1)
        # Mismos pixeles observados que la U-Net: si no, no es una comparacion
        # entre pares sino entre dos conjuntos distintos.
        m = (real != -1) & (pred != -1)
        p, v = (pred[m] == 1), (real[m] == 1)
        filas.append([(p & v).sum(), (p & ~v).sum(), (~p & v).sum(), (~p & ~v).sum()])
    return np.array(filas, dtype=np.float64)


def bootstrap_pareado(a: np.ndarray, b: np.ndarray, remuestreos: int = 10000,
                      semilla: int = 0) -> dict:
    """Diferencia de IoU (a - b) remuestreando los mismos chips para ambos."""
    rng = np.random.default_rng(semilla)
    n = len(a)
    observada = iou_de(a) - iou_de(b)
    diferencias = np.empty(remuestreos)
    for i in range(remuestreos):
        idx = rng.integers(0, n, n)
        diferencias[i] = iou_de(a[idx]) - iou_de(b[idx])
    bajo, alto = np.percentile(diferencias, [2.5, 97.5])
    return {"delta": float(observada), "ic95": [float(bajo), float(alto)],
            "significativa": bool(bajo > 0 or alto < 0), "chips": int(n)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # El default era "base", que es el nombre de la ARQUITECTURA (el campo
    # 'variante' de resultados.json), no el de ninguna carpeta de corrida: el
    # comando documentado fallaba con un FileNotFoundError de torch.load. La
    # corrida que produjo los numeros publicados (IoU 0.6808 en prueba) es
    # 'a_solo_dorados' -- entrenada solo con las etiquetas doradas, que es
    # justamente el resultado de la ablacion (a).
    parser.add_argument("--corrida", default="a_solo_dorados")
    parser.add_argument("--remuestreos", type=int, default=10000)
    parser.add_argument("--dispositivo", default="cuda")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    carpeta = CORRIDAS / args.corrida
    pesos = carpeta / "pesos.pt"
    if not pesos.is_file():
        disponibles = sorted(d.name for d in CORRIDAS.iterdir()
                             if (d / "pesos.pt").is_file()) if CORRIDAS.is_dir() else []
        print(f"no esta {pesos}.\n"
              f"corridas con pesos: {', '.join(disponibles) or '(ninguna)'}\n"
              f"entrena primero: python pregunta2/paso3_entrenamiento/entrenar.py")
        return 1
    guardado = torch.load(pesos, map_location="cpu", weights_only=False)
    dispositivo = torch.device(
        args.dispositivo if torch.cuda.is_available() else "cpu")
    modelo = UNet(ConfigUNet(**guardado["config"])).to(dispositivo)
    modelo.load_state_dict(guardado["pesos"])
    norm = Normalizador.desde_estadisticas()

    print(f"corrida {args.corrida} | variante {guardado['variante']} | "
          f"{args.remuestreos:,} remuestreos\n")

    salida = {"corrida": args.corrida, "remuestreos": args.remuestreos,
              "conjuntos": {}}
    print(f"{'conjunto':<22} {'U-Net':>7} {'Otsu':>7} {'delta':>8} "
          f"{'IC95% de la diferencia':>24} {'sig':>4}")
    print("-" * 78)
    for nombre in ("test", "bolivia"):
        chips = chips_del_split(nombre)
        unet, usados = confusion_por_chip(modelo, chips, norm, dispositivo)
        otsu = confusion_otsu_por_chip(usados)
        c = bootstrap_pareado(unet, otsu, args.remuestreos)
        print(f"{nombre:<22} {iou_de(unet):>7.4f} {iou_de(otsu):>7.4f} "
              f"{c['delta']:>+8.4f}  [{c['ic95'][0]:+.4f}, {c['ic95'][1]:+.4f}]"
              f" {'SI' if c['significativa'] else 'no':>7}")
        salida["conjuntos"][nombre] = {
            "chips": len(usados), "iou_unet": iou_de(unet),
            "iou_otsu": iou_de(otsu), **c}

    t, b = salida["conjuntos"]["test"], salida["conjuntos"]["bolivia"]
    print(f"\nlectura:")
    if t["significativa"] and t["delta"] > 0:
        print(f"  dentro de distribucion la U-Net le gana a Otsu por "
              f"{t['delta']:+.3f} y la diferencia ES significativa")
    else:
        print(f"  dentro de distribucion la diferencia ({t['delta']:+.3f}) NO se "
              f"distingue del ruido")
    if b["significativa"] and b["delta"] < 0:
        print(f"  en la region retenida la PIERDE por {b['delta']:+.3f}, y esa "
              f"diferencia TAMBIEN es significativa")
        print(f"  -> lo aprendido no se transfiere; el metodo clasico es mas robusto "
              f"fuera de distribucion")
    elif not b["significativa"]:
        print(f"  en la region retenida la diferencia es {b['delta']:+.3f} pero con "
              f"{b['chips']} chips NO se distingue del ruido")
        print(f"  -> se puede decir que la ventaja DESAPARECE, no que se invierta")

    # Cuanto cae la propia U-Net al cambiar de region, con su propio intervalo.
    chips_t, chips_b = chips_del_split("test"), chips_del_split("bolivia")
    ut, _ = confusion_por_chip(modelo, chips_t, norm, dispositivo)
    ub, _ = confusion_por_chip(modelo, chips_b, norm, dispositivo)
    rng = np.random.default_rng(1)
    # Acá NO es pareado: son chips distintos. Se remuestrea cada conjunto por
    # separado, que es lo correcto cuando no hay correspondencia uno a uno.
    d = np.array([iou_de(ut[rng.integers(0, len(ut), len(ut))]) -
                  iou_de(ub[rng.integers(0, len(ub), len(ub))])
                  for _ in range(args.remuestreos)])
    bajo, alto = np.percentile(d, [2.5, 97.5])
    print(f"\ncaida de la U-Net al pasar de prueba a Bolivia: "
          f"{iou_de(ut) - iou_de(ub):+.4f} [{bajo:+.4f}, {alto:+.4f}]")
    salida["caida_unet_test_a_bolivia"] = {
        "delta": float(iou_de(ut) - iou_de(ub)),
        "ic95": [float(bajo), float(alto)],
        "significativa": bool(bajo > 0 or alto < 0),
        "nota": "no pareado: son chips distintos, no hay correspondencia 1 a 1",
    }

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nescrito en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
