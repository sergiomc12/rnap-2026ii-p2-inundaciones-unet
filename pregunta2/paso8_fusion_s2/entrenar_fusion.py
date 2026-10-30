"""Bono (entregable 6): ¿fusionar Sentinel-2 con el SAR mejora algo?

Entrena **la misma U-Net tres veces** —solo SAR, solo óptico, y los dos
fusionados— sobre los mismos chips, la misma partición, la misma semilla y el
mismo presupuesto de épocas. Lo único que cambia son los canales de entrada, así
que la diferencia mide la fusión y no otra cosa.

Por qué este bono dejó de ser un extra
--------------------------------------
El paso 7 midió que el **99.8% de los falsos positivos** y el **96.5% de los
falsos negativos** del modelo SAR caen donde las dos clases comparten la misma
retrodispersión. Con VV y VH no hay nada más que sacar. El infrarrojo cercano
sí separa esas clases —el agua absorbe casi toda la radiación ahí— así que la
fusión no es "más datos": es la única dimensión nueva disponible.

La pregunta incómoda que este paso tiene que contestar
------------------------------------------------------
Sentinel-2 **no ve a través de las nubes**, y una inundación viene con tormenta.
Si la fusión mejora el número promedio pero esa mejora desaparece en los chips
nublados, entonces mejora justo donde no hace falta. Por eso cada corrida se
evalúa además **separando los chips por nubosidad**, y la conclusión se saca de
ahí y no del promedio.

Usage:
    python pregunta2/paso8_fusion_s2/entrenar_fusion.py
    python pregunta2/paso8_fusion_s2/entrenar_fusion.py --epocas 80
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Optional, Sequence

import numpy as np

# Las rutas de abajo se anclan a ESTE archivo, no al directorio de
# trabajo: la pregunta se comparte sola y sus scripts se corren desde
# cualquier lado.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pregunta2.paso1_datos.sen1floods11 import (  # noqa: E402
    MANO, Normalizador, acumular_confusion, chips_del_split, iou_por_chip,
    metricas, perdida_enmascarada,
)
from pregunta2.paso2_modelo.unet import ConfigUNet, UNet  # noqa: E402
from pregunta2.paso8_fusion_s2.datos_fusion import (  # noqa: E402
    MODOS, ChipsFusion, NormalizadorOptico, canales_de, cobertura_de_nube,
    hay_optico, leer_s2,
)

SALIDA = _RAIZ / "pregunta2/paso8_fusion_s2/corridas"
RESUMEN = _RAIZ / "pregunta2/paso8_fusion_s2/fusion.json"

#: Un chip cuenta como nublado cuando más de este porcentaje de sus píxeles
#: supera el umbral de brillo en azul. Con 10% ya hay nube suficiente para que
#: el óptico pierda una parte visible de la escena.
UMBRAL_CHIP_NUBLADO = 0.10


def sembrar(semilla: int) -> None:
    import random
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


@torch.no_grad()
def evaluar(modelo, cargador, dispositivo) -> tuple[dict, list]:
    modelo.eval()
    total = np.zeros(4, np.float64)
    porchip = []
    for x, y, m in cargador:
        x, y, m = x.to(dispositivo), y.to(dispositivo), m.to(dispositivo)
        logits = modelo(x)
        for i in range(len(x)):
            c = acumular_confusion(logits[i:i+1], y[i:i+1], m[i:i+1])
            porchip.append(c)
            total += c
    return metricas(total), porchip


def entrenar_una(modo: str, chips, norm_sar, norm_opt, args, dispositivo) -> dict:
    sembrar(args.semilla)
    ds = {p: ChipsFusion(chips[p], modo, norm_sar, norm_opt,
                         aumentar=(p == "train"), semilla=args.semilla)
          for p in ("train", "valid", "test")}
    cargadores = {p: DataLoader(d, batch_size=args.lote,
                                shuffle=(p == "train"), num_workers=0)
                  for p, d in ds.items()}

    modelo = UNet(ConfigUNet(canales_entrada=canales_de(modo))).to(dispositivo)
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epocas)

    print(f"\n{'='*70}\nmodo {modo.upper()} | {canales_de(modo)} canales | "
          f"{sum(p.numel() for p in modelo.parameters())/1e6:.2f} M parametros")
    print(f"{'='*70}")
    mejor, mejor_estado, mejor_epoca = -1.0, None, 0
    t0 = time.time()
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        suma = n = 0
        for x, y, m in cargadores["train"]:
            x, y, m = x.to(dispositivo), y.to(dispositivo), m.to(dispositivo)
            opt.zero_grad(set_to_none=True)
            perdida = perdida_enmascarada(modelo(x), y, m)
            perdida.backward()
            opt.step()
            # .detach(): sin esto, float() sobre un tensor con grad
            # dispara un UserWarning de PyTorch en cada lote.
            suma += perdida.detach().item() * len(x)
            n += len(x)
        plan.step()
        met, _ = evaluar(modelo, cargadores["valid"], dispositivo)
        if met["iou_agua"] > mejor:
            mejor, mejor_epoca = met["iou_agua"], epoca
            mejor_estado = {k: v.detach().cpu().clone()
                            for k, v in modelo.state_dict().items()}
        if epoca % 10 == 0 or epoca == 1:
            print(f"  epoca {epoca:>3}  perdida {suma/max(1,n):.4f}  "
                  f"IoU valid {met['iou_agua']:.4f}", flush=True)

    modelo.load_state_dict(mejor_estado)
    carpeta = SALIDA / modo
    carpeta.mkdir(parents=True, exist_ok=True)
    torch.save({"pesos": modelo.state_dict(), "modo": modo,
                "config": modelo.config.__dict__}, carpeta / "pesos.pt")

    resultado = {"modo": modo, "canales": canales_de(modo),
                 "mejor_epoca": mejor_epoca, "iou_validacion": round(mejor, 4),
                 "minutos": round((time.time() - t0) / 60, 2), "conjuntos": {}}
    por_chip_test = None
    for parte in ("valid", "test"):
        met, porchip = evaluar(modelo, cargadores[parte], dispositivo)
        met["iou_promedio_por_chip"] = round(
            iou_por_chip(np.array(porchip)), 4)
        resultado["conjuntos"][parte] = {k: (round(v, 4) if isinstance(v, float)
                                             else v) for k, v in met.items()}
        if parte == "test":
            por_chip_test = porchip
    resultado["_chips_test"] = ds["test"].chips
    resultado["_confusion_test"] = [list(map(int, c)) for c in por_chip_test]
    print(f"  -> IoU prueba {resultado['conjuntos']['test']['iou_agua']:.4f} "
          f"({resultado['minutos']:.1f} min)")
    return resultado


def nubosidad_de(chips: Sequence[str], mano: pathlib.Path = MANO) -> dict:
    """Fracción de píxeles probablemente nublados, por chip."""
    return {c: round(cobertura_de_nube(leer_s2(c, mano)), 4)
            for c in chips if hay_optico(c, mano)}


def iou_de(confusiones: np.ndarray) -> float:
    vp, fp, fn = confusiones[:, 0].sum(), confusiones[:, 1].sum(), confusiones[:, 2].sum()
    return float(vp / (vp + fp + fn)) if (vp + fp + fn) else float("nan")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modos", nargs="+", default=list(MODOS))
    parser.add_argument("--epocas", type=int, default=60)
    parser.add_argument("--lote", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--dispositivo", default="cuda")
    parser.add_argument("--region-retenida", default="bolivia")
    parser.add_argument("--salida", type=pathlib.Path, default=RESUMEN)
    args = parser.parse_args(argv)

    chips = {p: chips_del_split(p) for p in ("train", "valid", "test")}
    con_optico = {p: [c for c in v if hay_optico(c)] for p, v in chips.items()}
    faltan = {p: len(chips[p]) - len(con_optico[p]) for p in chips}
    if not any(con_optico.values()):
        print("no hay ningun chip con S2Hand. Corre primero:\n"
              "  python pregunta2/paso1_datos/descargar.py "
              "--solo-lo-pedido --s2-dorado")
        return 1
    print("chips por particion (solo los que tienen las tres piezas):")
    for p in ("train", "valid", "test"):
        print(f"  {p:<7} {len(con_optico[p]):>4} de {len(chips[p]):>4} "
              f"({faltan[p]} sin optico)")

    dispositivo = torch.device(args.dispositivo if torch.cuda.is_available()
                               else "cpu")
    norm_sar = Normalizador.desde_estadisticas()
    print("\nmidiendo la normalizacion optica sobre ENTRENAMIENTO...")
    norm_opt = NormalizadorOptico.desde_chips(con_optico["train"])

    corridas = {}
    for modo in args.modos:
        corridas[modo] = entrenar_una(modo, con_optico, norm_sar, norm_opt,
                                      args, dispositivo)

    # La pregunta incomoda: la ganancia, separada por nubosidad.
    print(f"\n{'='*70}\nLA GANANCIA, SEPARADA POR NUBOSIDAD\n{'='*70}")
    nubes = nubosidad_de(corridas[args.modos[0]]["_chips_test"])
    chips_test = corridas[args.modos[0]]["_chips_test"]
    nublado = np.array([nubes.get(c, 0.0) > UMBRAL_CHIP_NUBLADO
                        for c in chips_test])
    print(f"chips de prueba: {len(chips_test)} | nublados "
          f"(> {100*UMBRAL_CHIP_NUBLADO:.0f}% de pixeles): {int(nublado.sum())} "
          f"| despejados: {int((~nublado).sum())}")

    por_nubosidad = {}
    print(f"\n{'modo':<10} {'IoU todos':>10} {'IoU despejados':>15} "
          f"{'IoU nublados':>13}")
    print("-" * 52)
    for modo, r in corridas.items():
        conf = np.array(r["_confusion_test"])
        fila = {"todos": round(iou_de(conf), 4),
                "despejados": (round(iou_de(conf[~nublado]), 4)
                               if (~nublado).any() else None),
                "nublados": (round(iou_de(conf[nublado]), 4)
                             if nublado.any() else None)}
        por_nubosidad[modo] = fila
        d = "—" if fila["despejados"] is None else f"{fila['despejados']:.4f}"
        n = "—" if fila["nublados"] is None else f"{fila['nublados']:.4f}"
        print(f"{modo:<10} {fila['todos']:>10.4f} {d:>15} {n:>13}")

    if "sar" in por_nubosidad and "fusion" in por_nubosidad:
        print(f"\nganancia de la fusion sobre el SAR solo:")
        for k in ("todos", "despejados", "nublados"):
            a, b = por_nubosidad["sar"][k], por_nubosidad["fusion"][k]
            if a is not None and b is not None:
                print(f"  {k:<12} {b - a:+.4f}")
        print("\n(la significancia se mide en comparar_fusion.py, con bootstrap"
              "\n pareado sobre chips: estas diferencias sueltas no alcanzan)")

    resumen = {
        "epocas": args.epocas, "semilla": args.semilla,
        "umbral_chip_nublado": UMBRAL_CHIP_NUBLADO,
        "chips": {p: len(v) for p, v in con_optico.items()},
        "chips_sin_optico": faltan,
        "corridas": {m: {k: v for k, v in r.items() if not k.startswith("_")}
                     for m, r in corridas.items()},
        "por_nubosidad": por_nubosidad,
        "nubosidad_por_chip": nubes,
        "confusion_test_por_chip": {m: r["_confusion_test"]
                                    for m, r in corridas.items()},
        "chips_test": chips_test,
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nescrito: {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
