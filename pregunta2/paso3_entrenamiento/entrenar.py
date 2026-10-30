"""Entrena la U-Net sobre Sen1Floods11 y la evalúa donde corresponde.

Cubre los entregables 1 y 3: la red propia de ``paso2_modelo/unet.py`` entrenada
sobre los 252 chips etiquetados a mano, elegida por validación (89 chips) y
reportada sobre prueba (90) y sobre **Bolivia** (15), que es la región que el
propio dataset deja fuera de las tres particiones.

Cuatro decisiones que valen la pena justificar:

**Se elige por IoU de validación, no por pérdida.** La pérdida mezcla BCE y
Dice y mejora aunque la IoU no se mueva; la métrica del enunciado es la IoU, así
que esa es la que decide qué época se guarda.

**Los conteos de confusión se acumulan y la IoU se calcula al final.** Promediar
la IoU chip por chip le daría el mismo peso a un chip con un río entero que a uno
con treinta píxeles de agua, donde la IoU es casi ruido.

**Se compara contra la umbralización de Otsu, no contra nada.** El dataset trae
``S1OtsuLabelHand``: el método clásico sobre los mismos chips. Decir "la U-Net
saca IoU 0.7" no significa nada suelto; lo que importa es si le gana al método
que no necesita entrenamiento.

**Bolivia se mira una sola vez, al final.** Es la región retenida: si se la usa
para elegir hiperparámetros deja de medir generalización y pasa a ser otra
validación.

Usage:
    python pregunta2/paso3_entrenamiento/entrenar.py
    python pregunta2/paso3_entrenamiento/entrenar.py --variante angosta --etiqueta angosta
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Optional, Sequence

import numpy as np

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

import rasterio  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pregunta2.paso1_datos.sen1floods11 import (  # noqa: E402
    DEBIL, MANO, ChipsSen1Floods11, Normalizador, acumular_confusion,
    chips_debiles, chips_del_split, iou_por_chip, metricas, perdida_enmascarada,
)
from pregunta2.paso2_modelo.unet import construir_unet  # noqa: E402

SALIDA = _RAIZ / "pregunta2/paso3_entrenamiento/corridas"


def fijar_semilla(s: int) -> None:
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluar(modelo, cargador, dispositivo, umbral: float = 0.5):
    """Métricas agregadas **y** la IoU promedio por chip.

    Las dos, porque miden cosas distintas y el enunciado pide la segunda: los
    autores de Sen1Floods11 reportan "IoU promedio sobre los chips". La agregada
    pesa más los chips con mucha agua; la promedio por chip pesa todos igual.
    Reportar solo una deja la comparación con el paper a medias.
    """
    modelo.eval()
    conf = np.zeros(4); por_chip = []; suma = vistos = 0
    for x, y, m in cargador:
        x, y, m = x.to(dispositivo), y.to(dispositivo), m.to(dispositivo)
        with torch.amp.autocast("cuda", enabled=dispositivo.type == "cuda"):
            logits = modelo(x)
        suma += perdida_enmascarada(logits.float(), y, m).item() * len(x)
        vistos += len(x)
        conf += acumular_confusion(logits.float(), y, m, umbral)
        for j in range(len(x)):
            por_chip.append(acumular_confusion(
                logits[j:j+1].float(), y[j:j+1], m[j:j+1], umbral))
    r = metricas(conf)
    r["iou_promedio_por_chip"] = iou_por_chip(np.stack(por_chip))
    r["perdida"] = suma / max(1, vistos)
    r["confusion"] = {"vp": int(conf[0]), "fp": int(conf[1]),
                      "fn": int(conf[2]), "vn": int(conf[3])}
    return r


def linea_base_otsu(chips: Sequence[str]) -> dict:
    """La umbralización clásica que el dataset ya trae calculada.

    Es la referencia honesta: no se entrena con nada, así que cualquier IoU que
    la U-Net no supere significa que el aprendizaje no aportó.
    """
    conf = np.zeros(4)
    for chip in chips:
        ruta_otsu = MANO / "S1OtsuLabelHand" / f"{chip}_S1OtsuLabelHand.tif"
        ruta_real = MANO / "LabelHand" / f"{chip}_LabelHand.tif"
        if not (ruta_otsu.is_file() and ruta_real.is_file()):
            continue
        with rasterio.open(ruta_otsu) as r:
            pred = r.read(1)
        with rasterio.open(ruta_real) as r:
            real = r.read(1)
        # Se evalua sobre los MISMOS pixeles observados que la U-Net: si Otsu se
        # midiera sobre todos, la comparacion no seria entre pares.
        m = (real != -1) & (pred != -1)
        p = (pred[m] == 1); v = (real[m] == 1)
        conf += np.array([(p & v).sum(), (p & ~v).sum(),
                          (~p & v).sum(), (~p & ~v).sum()], dtype=np.float64)
    return metricas(conf)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variante", default="base")
    parser.add_argument("--epocas", type=int, default=60)
    parser.add_argument("--lote", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--peso-dice", type=float, default=0.5)
    parser.add_argument("--paciencia", type=int, default=15)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--dispositivo", default="cuda")
    parser.add_argument("--sin-aumentar", action="store_true")
    parser.add_argument("--debiles", type=int, default=0,
                        help="cuantos chips de etiqueta debil agregar "
                             "(0 = solo los dorados; -1 = todos los que haya)")
    parser.add_argument("--region-retenida", default="bolivia",
                        help="region que NUNCA entra al entrenamiento, ni "
                             "siquiera por el conjunto debil")
    parser.add_argument("--etiqueta")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    args = parser.parse_args(argv)

    fijar_semilla(args.semilla)
    dispositivo = torch.device(
        args.dispositivo if torch.cuda.is_available() else "cpu")
    norm = Normalizador.desde_estadisticas()

    particiones = {n: chips_del_split(n)
                   for n in ("train", "valid", "test", "bolivia")}
    entrenamiento = ChipsSen1Floods11(particiones["train"], norm,
                                      aumentar=not args.sin_aumentar,
                                      semilla=args.semilla)
    if args.debiles:
        # Se excluye la region retenida del conjunto DEBIL tambien. Es el punto
        # facil de pasar por alto: los 4 384 chips debiles incluyen 224 de
        # Bolivia, y meterlos "porque son de otro conjunto" le mostraria al
        # modelo justo la region sobre la que despues se mide generalizacion.
        candidatos = chips_debiles(excluir_regiones=[args.region_retenida])
        rng_sel = np.random.default_rng(args.semilla)
        if 0 < args.debiles < len(candidatos):
            candidatos = sorted(rng_sel.choice(candidatos, args.debiles,
                                               replace=False).tolist())
        debiles = ChipsSen1Floods11(
            candidatos, norm, aumentar=not args.sin_aumentar,
            semilla=args.semilla + 1, mano=DEBIL,
            sufijo_imagen="S1Weak", sufijo_etiqueta="S1OtsuLabelWeak")
        print(f"  + {len(debiles)} chips DEBILES "
              f"(excluida la region retenida '{args.region_retenida}')")
        entrenamiento = torch.utils.data.ConcatDataset([entrenamiento, debiles])

    conjuntos = {
        "train": entrenamiento,
        "valid": ChipsSen1Floods11(particiones["valid"], norm),
        "test": ChipsSen1Floods11(particiones["test"], norm),
        "bolivia": ChipsSen1Floods11(particiones["bolivia"], norm),
    }
    cargadores = {n: DataLoader(c, batch_size=args.lote,
                                shuffle=(n == "train"), num_workers=0,
                                pin_memory=dispositivo.type == "cuda")
                  for n, c in conjuntos.items()}

    modelo = construir_unet(args.variante, canales_entrada=2, clases=1).to(dispositivo)
    criterio = lambda lo, y, m: perdida_enmascarada(lo, y, m, args.peso_dice)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizador, T_max=args.epocas, eta_min=1e-6)
    escalador = torch.amp.GradScaler("cuda", enabled=dispositivo.type == "cuda")

    print(f"variante {args.variante} | {modelo.parametros_millones:.1f} M parametros")
    print(f"dispositivo {dispositivo} | aumentacion "
          f"{'NO' if args.sin_aumentar else 'si'}")
    print("  ".join(f"{n} {len(c)}" for n, c in conjuntos.items()) + "\n")

    mejor, mejor_epoca, sin_mejorar, mejores = -1.0, 0, 0, None
    historia, comenzado = [], time.time()
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        suma = vistos = 0
        for x, y, m in cargadores["train"]:
            x, y, m = x.to(dispositivo), y.to(dispositivo), m.to(dispositivo)
            optimizador.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dispositivo.type == "cuda"):
                logits = modelo(x)
            perdida = criterio(logits.float(), y, m)
            escalador.scale(perdida).backward()
            escalador.step(optimizador)
            escalador.update()
            suma += perdida.detach().item() * len(x); vistos += len(x)
        planificador.step()

        v = evaluar(modelo, cargadores["valid"], dispositivo)
        historia.append({"epoca": epoca, "perdida_train": suma / max(1, vistos),
                         **{f"val_{k}": val for k, val in v.items()}})
        marca = ""
        # Se elige por IoU, que es la metrica del enunciado, no por perdida.
        if v["iou_agua"] > mejor:
            mejor, mejor_epoca, sin_mejorar = v["iou_agua"], epoca, 0
            mejores = {k: t.detach().cpu().clone()
                       for k, t in modelo.state_dict().items()}
            marca = "  <- mejor"
        else:
            sin_mejorar += 1
        if epoca % 5 == 0 or epoca == 1 or marca:
            print(f"  epoca {epoca:>3}/{args.epocas} | train {suma/max(1,vistos):.4f} "
                  f"| val IoU {v['iou_agua']:.4f} F1 {v['f1']:.4f}{marca}", flush=True)
        if sin_mejorar >= args.paciencia:
            print(f"  parada temprana en la epoca {epoca}")
            break

    modelo.load_state_dict(mejores)
    minutos = (time.time() - comenzado) / 60
    print(f"\nmejor epoca {mejor_epoca} | IoU validacion {mejor:.4f} | {minutos:.1f} min")

    resultados = {n: evaluar(modelo, cargadores[n], dispositivo)
                  for n in ("valid", "test", "bolivia")}
    otsu = {n: linea_base_otsu(particiones[n]) for n in ("test", "bolivia")}

    # Las dos IoU: la agregada (un solo cociente sobre todos los pixeles) y la
    # promedio por chip, que es la que reportan los autores de Sen1Floods11.
    print(f"\n{'conjunto':<24} {'IoU agreg':>9} {'IoU/chip':>9} {'F1':>7} "
          f"{'prec':>7} {'exh':>7}")
    print("-" * 68)
    for n in ("valid", "test", "bolivia"):
        r = resultados[n]
        nombre = n + (" (RETENIDA)" if n == "bolivia" else "")
        print(f"{nombre:<24} {r['iou_agua']:>9.4f} "
              f"{r['iou_promedio_por_chip']:>9.4f} {r['f1']:>7.4f} "
              f"{r['precision']:>7.4f} {r['exhaustividad']:>7.4f}")
    print()
    for n in ("test", "bolivia"):
        o = otsu[n]
        print(f"{'Otsu (sin entrenar) ' + n:<26} {o['iou_agua']:>9.4f} "
              f"{o['f1']:>7.4f} {o['precision']:>7.4f} {o['exhaustividad']:>7.4f}")

    c = resultados["test"]["confusion"]
    tot = sum(c.values())
    print(f"\nmatriz de confusion sobre PRUEBA ({tot:,} pixeles observados):")
    print(f"{'':>14} {'pred agua':>14} {'pred no agua':>14}")
    print(f"{'real agua':>14} {c['vp']:>14,} {c['fn']:>14,}")
    print(f"{'real no agua':>14} {c['fp']:>14,} {c['vn']:>14,}")
    print(f"  omision  (agua real que se perdio) : "
          f"{100*c['fn']/max(1,c['vp']+c['fn']):.2f}%")
    print(f"  comision (agua predicha que no era): "
          f"{100*c['fp']/max(1,c['vp']+c['fp']):.2f}%")

    delta_t = resultados["test"]["iou_agua"] - otsu["test"]["iou_agua"]
    delta_b = resultados["bolivia"]["iou_agua"] - otsu["bolivia"]["iou_agua"]
    caida = resultados["test"]["iou_agua"] - resultados["bolivia"]["iou_agua"]
    print(f"\nU-Net vs Otsu    en prueba : {delta_t:+.4f}")
    print(f"U-Net vs Otsu    en Bolivia: {delta_b:+.4f}")
    print(f"caida de prueba -> Bolivia  : {caida:+.4f}  "
          f"(cuanto cuesta cambiar de region)")

    nombre = args.etiqueta or args.variante
    carpeta = args.salida / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    torch.save({"pesos": modelo.state_dict(), "variante": args.variante,
                "config": modelo.config.__dict__}, carpeta / "pesos.pt")
    (carpeta / "resultados.json").write_text(json.dumps({
        "variante": args.variante,
        "parametros_millones": round(modelo.parametros_millones, 3),
        "epocas": args.epocas, "lote": args.lote, "lr": args.lr,
        "weight_decay": args.weight_decay, "peso_dice": args.peso_dice,
        "aumentacion": not args.sin_aumentar, "semilla": args.semilla,
        "chips_debiles": args.debiles,
        "region_retenida": args.region_retenida,
        "chips_entrenamiento": len(conjuntos["train"]),
        "mejor_epoca": mejor_epoca, "iou_validacion": mejor,
        "minutos": round(minutos, 2),
        "resultados": resultados, "otsu": otsu,
        "delta_vs_otsu": {"test": delta_t, "bolivia": delta_b},
        "caida_test_a_bolivia": caida,
        "historia": historia,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito en {carpeta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
