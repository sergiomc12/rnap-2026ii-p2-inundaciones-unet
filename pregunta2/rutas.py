"""Dónde están los datos de la Pregunta 2.

**Todas las rutas quedan dentro de ``pregunta2/``.** No es una preferencia de
orden: esta carpeta se comparte por separado con el equipo, y cualquier ruta que
salga —``../data/...``— no existiría en la máquina de quien la reciba.

Son 12 GB (Sen1Floods11 7.6 GB y Lima 4.4 GB), demasiado para copiarlos, así que
``alinear_datos.py`` los pone bajo ``pregunta2/datos/`` como **enlaces de
directorio** (junctions de Windows). No duplican un solo byte, y al comprimir o
copiar la carpeta Windows sigue el enlace e incluye el contenido, así que la
carpeta se comparte igual.

Usage:
    from pregunta2.rutas import MANO, DEBIL, SPLITS, LIMA
"""

from __future__ import annotations

import pathlib

#: La carpeta de esta pregunta, sin importar desde donde se invoque.
AQUI = pathlib.Path(__file__).resolve().parent

#: Todo lo que esta pregunta lee vive acá adentro. **No hay ruta de respaldo
#: fuera del directorio**: esta carpeta se comparte por separado, y una ruta como
#: ``../data/...`` no existiría en la máquina de quien la reciba.
DATOS = AQUI / "datos"

SEN1FLOODS11 = DATOS / "sen1floods11"
LIMA_RAIZ = DATOS / "lima"

#: Dentro de Sen1Floods11. La versión va en la ruta porque el dataset la publica
#: así, y el día que salga una v1.2 conviene que el cambio sea visible.
V11 = SEN1FLOODS11 / "v1.1"
MANO = V11 / "data/flood_events/HandLabeled"
DEBIL = V11 / "data/flood_events/WeaklyLabeled"
SPLITS = V11 / "splits/flood_handlabeled"

#: Lo de Lima, que esta pregunta construye en vez de descargar.
LIMA = LIMA_RAIZ / "region"
GDB = LIMA_RAIZ / "EMSN038_UTM18S_v0.gdb"


def verificar() -> list[str]:
    """Qué falta, si falta algo. Lista vacía = todo en su lugar."""
    problemas = [nombre for nombre, ruta in
                 (("sen1floods11/", SEN1FLOODS11), ("lima/", LIMA_RAIZ))
                 if not ruta.is_dir()]
    if problemas:
        problemas = [f"falta {p}" for p in problemas]
        problemas.append("corre: python alinear_datos.py")
    return problemas


def describir() -> str:
    faltan = verificar()
    return ("datos de la Pregunta 2 en pregunta2/datos/" if not faltan
            else "datos de la Pregunta 2 INCOMPLETOS: " + "; ".join(faltan))


if __name__ == "__main__":
    print(describir())
    for nombre, ruta in (("S1Hand", MANO / "S1Hand"),
                         ("LabelHand", MANO / "LabelHand"),
                         ("S1Weak", DEBIL / "S1Weak"),
                         ("splits", SPLITS),
                         ("region de Lima", LIMA),
                         ("geodatabase EMSN038", GDB)):
        if ruta.is_dir():
            n = sum(1 for _ in ruta.iterdir())
            print(f"  {nombre:<22} OK        {n:>6} entradas   {ruta}")
        else:
            print(f"  {nombre:<22} NO EXISTE            {ruta}")
