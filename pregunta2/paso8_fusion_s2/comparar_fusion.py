"""¿La ganancia de fusionar Sentinel-2 sobrevive a tener 90 chips de prueba?

`entrenar_fusion.py` deja tres IoU. Esa diferencia suelta no alcanza para
afirmar nada: con 90 chips, el intervalo de un IoU mide varias centésimas y dos
modelos distintos se solapan casi seguro. Lo que decide es el **bootstrap
pareado sobre chips** —los mismos chips remuestreados para los dos modelos, y se
mira la distribución de la **diferencia**—, que es el mismo procedimiento del
paso 4 y por la misma razón: el ruido del conjunto se cancela en vez de sumarse
dos veces.

**Se remuestrea sobre chips, nunca sobre píxeles.** Los píxeles vecinos de un
chip están correlacionados: un río cubre miles de píxeles que son una sola
observación. Remuestrear píxeles daría intervalos absurdamente angostos.

La pregunta que este script contesta y el promedio no
-----------------------------------------------------
La fusión puede subir el número total y aun así no servir, porque Sentinel-2 no
atraviesa nubes y una inundación viene con tormenta. Así que la diferencia se
mide **tres veces**: sobre todos los chips, solo sobre los despejados, y solo
sobre los nublados. Si el intervalo cruza cero en los nublados, la fusión ayuda
donde el SAR no hacía falta.

Usage:
    python pregunta2/paso8_fusion_s2/comparar_fusion.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

import numpy as np

# Las rutas de abajo se anclan a ESTE archivo, no al directorio de
# trabajo: la pregunta se comparte sola y sus scripts se corren desde
# cualquier lado.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

RESUMEN = _RAIZ / "pregunta2/paso8_fusion_s2/fusion.json"
SALIDA = _RAIZ / "pregunta2/paso8_fusion_s2/comparacion_fusion.json"


def iou_agregada(conf: np.ndarray) -> float:
    """IoU de agua sumando los conteos de los chips elegidos."""
    if conf.size == 0:
        return float("nan")
    vp, fp, fn = conf[:, 0].sum(), conf[:, 1].sum(), conf[:, 2].sum()
    union = vp + fp + fn
    return float(vp / union) if union > 0 else float("nan")


def bootstrap_pareado(a: np.ndarray, b: np.ndarray, remuestreos: int,
                      semilla: int) -> dict:
    """IoU de ``b`` menos la de ``a``, remuestreando **los mismos** chips."""
    n = len(a)
    if n == 0:
        return {"chips": 0}
    rng = np.random.default_rng(semilla)
    obs = iou_agregada(b) - iou_agregada(a)
    diffs = np.empty(remuestreos)
    for i in range(remuestreos):
        idx = rng.integers(0, n, n)
        diffs[i] = iou_agregada(b[idx]) - iou_agregada(a[idx])
    bajo, alto = np.percentile(diffs, [2.5, 97.5])
    return {"chips": n,
            "iou_a": round(iou_agregada(a), 4),
            "iou_b": round(iou_agregada(b), 4),
            "delta": round(float(obs), 4),
            "ic95": [round(float(bajo), 4), round(float(alto), 4)],
            "p_b_mejor": round(float((diffs > 0).mean()), 4),
            "significativa": bool(bajo > 0 or alto < 0)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resumen", type=pathlib.Path, default=RESUMEN)
    parser.add_argument("--remuestreos", type=int, default=10000)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    if not args.resumen.is_file():
        print(f"falta {args.resumen}. Corré primero:\n"
              f"  python pregunta2/paso8_fusion_s2/entrenar_fusion.py")
        return 1
    r = json.loads(args.resumen.read_text(encoding="utf-8"))
    conf = {m: np.array(v, dtype=np.float64)
            for m, v in r["confusion_test_por_chip"].items()}
    chips = r["chips_test"]
    nubes = r["nubosidad_por_chip"]
    umbral = r["umbral_chip_nublado"]
    nublado = np.array([nubes.get(c, 0.0) > umbral for c in chips])

    grupos = {"todos": np.ones(len(chips), bool),
              "despejados": ~nublado,
              "nublados": nublado}
    print(f"{len(chips)} chips de prueba | nublados {int(nublado.sum())} "
          f"| despejados {int((~nublado).sum())} "
          f"| umbral {100*umbral:.0f}% de pixeles\n")

    pares = [("sar", "fusion"), ("sar", "optico"), ("optico", "fusion")]
    pares = [(a, b) for a, b in pares if a in conf and b in conf]

    resultado = {"remuestreos": args.remuestreos, "semilla": args.semilla,
                 "umbral_chip_nublado": umbral, "comparaciones": {}}
    for a, b in pares:
        print(f"=== {b} contra {a} ===")
        print(f"{'grupo':<12} {'IoU '+a:>10} {'IoU '+b:>10} {'delta':>9} "
              f"{'IC95% de la diferencia':>26} {'sig':>4}")
        print("-" * 76)
        resultado["comparaciones"][f"{b}_vs_{a}"] = {}
        for grupo, m in grupos.items():
            if not m.any():
                continue
            s = bootstrap_pareado(conf[a][m], conf[b][m],
                                  args.remuestreos, args.semilla)
            resultado["comparaciones"][f"{b}_vs_{a}"][grupo] = s
            ic = f"[{s['ic95'][0]:+.4f}, {s['ic95'][1]:+.4f}]"
            print(f"{grupo:<12} {s['iou_a']:>10.4f} {s['iou_b']:>10.4f} "
                  f"{s['delta']:>+9.4f} {ic:>26} "
                  f"{'SI' if s['significativa'] else 'no':>4}")
        print()

    # La lectura, dicha sin adornos.
    clave = "fusion_vs_sar"
    if clave in resultado["comparaciones"]:
        c = resultado["comparaciones"][clave]
        print("=" * 76)
        print("LECTURA")
        print("=" * 76)
        for grupo in ("despejados", "nublados"):
            if grupo not in c:
                continue
            s = c[grupo]
            if s["significativa"] and s["delta"] > 0:
                print(f"  con cielo {grupo:<12}: la fusion GANA "
                      f"{s['delta']:+.4f} y es significativa")
            elif s["significativa"]:
                print(f"  con cielo {grupo:<12}: la fusion PIERDE "
                      f"{s['delta']:+.4f} y es significativa")
            else:
                print(f"  con cielo {grupo:<12}: {s['delta']:+.4f}, el "
                      f"intervalo cruza cero -> no se puede afirmar")
        d, n = c.get("despejados"), c.get("nublados")
        if d and n and d["significativa"] and not n["significativa"]:
            print("\n  => la fusion ayuda SOLO donde el optico ve, que es donde")
            print("     el SAR no hacia falta. Para la emergencia real --nublada--")
            print("     este trabajo no muestra ninguna mejora.")
        elif d and n and d["significativa"] and n["significativa"] and n["delta"] > 0:
            print("\n  => la fusion ayuda TAMBIEN con nubes, que es el caso que")
            print("     importa. Vale la pena aunque el optico se degrade.")

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nescrito: {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
