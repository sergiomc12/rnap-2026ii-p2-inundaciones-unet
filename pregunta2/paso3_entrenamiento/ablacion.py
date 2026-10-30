"""Entregable 2: ablación sobre los factores que pide el enunciado.

El enunciado exige comparar **al menos dos decisiones**:

* **(a) fuente de supervisión** — entrenar solo con los 446 chips de etiqueta
  dorada contra agregar los de etiqueta débil. Es el factor más interesante del
  examen entero, porque las etiquetas débiles salen de una umbralización de Otsu:
  agregarlas es pedirle a la red que imite al método clásico en 4 000 chips más.
* **(b) función de pérdida** — BCE sola, Dice sola, o la combinación.

Se agrega un tercero, la **profundidad**, que el enunciado admite como
alternativa a (b) y acá entra como extra.

Dos cuidados que no son decorativos:

1. **La región retenida se excluye también del conjunto débil.** Los 4 384 chips
   débiles incluyen 224 de Bolivia. Sumarlos "porque son del otro conjunto" le
   mostraría al modelo la región sobre la que después se mide la generalización.
2. **Presupuesto de pasos, no de épocas.** Una época sobre 4 412 chips no es
   comparable con una sobre 252: serían 17 veces más actualizaciones. Las épocas
   de cada corrida se fijan para que el número de chips vistos quede del mismo
   orden, así la comparación es entre *qué datos* y no entre *cuánto entrenó*.

Usage:
    python pregunta2/paso3_entrenamiento/ablacion.py --listar
    python pregunta2/paso3_entrenamiento/ablacion.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Optional, Sequence

_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

CORRIDAS = _RAIZ / "pregunta2/paso3_entrenamiento/corridas"
RESUMEN = _RAIZ / "pregunta2/paso3_entrenamiento/ablacion.json"

#: (nombre, factor, argumentos). ~15 000 chips vistos en cada una.
CORRIDAS_ABLACION = [
    # --- (a) fuente de supervision -----------------------------------------
    ("a_solo_dorados", "supervision", ["--epocas", "60", "--debiles", "0"]),
    ("a_debiles_1000", "supervision", ["--epocas", "12", "--debiles", "1000"]),
    ("a_debiles_todos", "supervision", ["--epocas", "4", "--debiles", "-1"]),
    # --- (b) funcion de perdida (siempre solo dorados, para aislar el factor) -
    ("b_bce", "perdida", ["--epocas", "60", "--peso-dice", "0.0"]),
    ("b_dice", "perdida", ["--epocas", "60", "--peso-dice", "1.0"]),
    # (la combinada 0.5 es a_solo_dorados: no se repite)
    # --- (c) profundidad -----------------------------------------------------
    ("c_somera", "profundidad", ["--epocas", "60", "--variante", "somera"]),
    ("c_profunda", "profundidad", ["--epocas", "60", "--variante", "profunda"]),
]


def correr(nombre: str, extra: Sequence[str], salida: pathlib.Path) -> bool:
    carpeta = salida / nombre
    if (carpeta / "resultados.json").is_file():
        print(f"  {nombre}: ya existe, se saltea")
        return True
    print(f"  {nombre}: entrenando...", flush=True)
    t = time.time()
    r = subprocess.run(
        [sys.executable, "pregunta2/paso3_entrenamiento/entrenar.py",
         "--etiqueta", nombre, "--salida", str(salida), *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"  {nombre}: FALLO\n{(r.stderr or '')[-800:]}")
        return False
    print(f"  {nombre}: listo en {(time.time()-t)/60:.1f} min")
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listar", action="store_true")
    parser.add_argument("--solo", help="nombres separados por coma")
    parser.add_argument("--salida", type=pathlib.Path, default=CORRIDAS)
    args = parser.parse_args(argv)

    corridas = list(CORRIDAS_ABLACION)
    if args.solo:
        querides = {s.strip() for s in args.solo.split(",")}
        corridas = [c for c in corridas if c[0] in querides]
    if args.listar:
        for n, f, e in corridas:
            print(f"  {n:<18} [{f:<12}] {' '.join(e)}")
        return 0

    args.salida.mkdir(parents=True, exist_ok=True)
    for nombre, _, extra in corridas:
        correr(nombre, extra, args.salida)

    # --- tabla comparativa ---------------------------------------------------
    filas = []
    for nombre, factor, _ in corridas:
        ruta = args.salida / nombre / "resultados.json"
        if not ruta.is_file():
            continue
        d = json.loads(ruta.read_text(encoding="utf-8"))
        filas.append({"corrida": nombre, "factor": factor,
                      "chips_entrenamiento": d.get("chips_entrenamiento"),
                      "epocas": d["epocas"], "minutos": d["minutos"],
                      "iou_test": d["resultados"]["test"]["iou_agua"],
                      "iou_test_por_chip":
                          d["resultados"]["test"].get("iou_promedio_por_chip"),
                      "iou_bolivia": d["resultados"]["bolivia"]["iou_agua"],
                      "iou_bolivia_por_chip":
                          d["resultados"]["bolivia"].get("iou_promedio_por_chip"),
                      "otsu_test": d["otsu"]["test"]["iou_agua"],
                      "otsu_bolivia": d["otsu"]["bolivia"]["iou_agua"]})

    if not filas:
        print("\nninguna corrida produjo resultados")
        return 1

    print(f"\n{'corrida':<18} {'factor':<13} {'chips':>6} {'ep':>3} "
          f"{'IoU test':>9} {'IoU Bol':>9} {'test-Bol':>9}")
    print("-" * 74)
    for f in filas:
        print(f"{f['corrida']:<18} {f['factor']:<13} "
              f"{f['chips_entrenamiento'] or 0:>6} {f['epocas']:>3} "
              f"{f['iou_test']:>9.4f} {f['iou_bolivia']:>9.4f} "
              f"{f['iou_test'] - f['iou_bolivia']:>+9.4f}")
    otsu = filas[0]
    print(f"{'Otsu (sin entrenar)':<18} {'-':<13} {'-':>6} {'-':>3} "
          f"{otsu['otsu_test']:>9.4f} {otsu['otsu_bolivia']:>9.4f} "
          f"{otsu['otsu_test'] - otsu['otsu_bolivia']:>+9.4f}")

    RESUMEN.write_text(json.dumps(filas, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\nescrito en {RESUMEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
