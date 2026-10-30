"""Convierte el informe LaTeX a Word (.docx), junto al PDF.

El enunciado pide el informe en PDF; esta versión en Word existe para poder
revisarlo y comentarlo con el equipo sin tener LaTeX instalado.

**Por qué no se le pasa el ``.tex`` directamente a pandoc.** Se probó y
**se cuelga**: el preámbulo carga ``neurips.sty``, que pandoc intenta
interpretar y no termina nunca (verificado con un ``timeout`` de 110 s contra un
archivo mínimo que sí convierte en un segundo). Así que este script:

1. recorta el cuerpo entre ``\\begin{document}`` y ``\\end{document}``, que es lo
   único que pandoc necesita;
2. **resuelve los ``\\input{tablas/...}``** pegando el contenido, porque pandoc
   no sigue esas inclusiones y las tablas desaparecerían del Word sin ningún
   aviso;
3. le antepone un preámbulo mínimo con los paquetes que el cuerpo usa de verdad
   (``booktabs`` para las reglas de las tablas, ``graphicx`` para las figuras);
4. llama a pandoc con la bibliografía, para que las citas salgan resueltas y no
   como ``[?]``.

**Este archivo está duplicado en las tres preguntas a propósito.** Cada carpeta
``preguntaN/`` se comparte sola con el equipo, y un script compartido en la raíz
del proyecto dejaría de existir del otro lado.

Usage:
    python pregunta2/paso9_informe/exportar_word.py
    python pregunta2/paso9_informe/exportar_word.py --pandoc C:/ruta/pandoc.exe
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
from typing import Optional, Sequence

#: La carpeta del informe, relativa a ESTE archivo: el script se corre desde
#: cualquier directorio de trabajo.
INFORME = pathlib.Path(__file__).resolve().parent / "informe"

#: Dónde buscar pandoc si no está en el PATH. En Windows el instalador de
#: winget lo deja en AppData, que no siempre entra al PATH de la sesión.
CANDIDATOS_PANDOC = (
    "pandoc",
    r"C:\Users\%USERNAME%\AppData\Local\Pandoc\pandoc.exe",
    r"C:\Program Files\Pandoc\pandoc.exe",
)


def encontrar_pandoc(explicito: Optional[str] = None) -> Optional[str]:
    if explicito:
        return explicito if pathlib.Path(explicito).is_file() else None
    for c in CANDIDATOS_PANDOC:
        ruta = pathlib.Path(pathlib.os.path.expandvars(c))
        if ruta.is_file():
            return str(ruta)
        hallado = shutil.which(c)
        if hallado:
            return hallado
    return None


def _llave(t: str, orden: str) -> str:
    """El contenido de ``\\orden{...}``, contando llaves anidadas.

    Un ``re.search`` con ``[^}]*`` se corta en la primera llave de cierre, y el
    bloque de autores tiene llaves adentro (``\\texttt{...}``, ``\\\\``).
    """
    i = t.find("\\" + orden + "{")
    if i < 0:
        return ""
    i += len(orden) + 2
    nivel, j = 1, i
    while j < len(t) and nivel:
        nivel += (t[j] == "{") - (t[j] == "}")
        j += 1
    crudo = t[i:j - 1]
    # Se limpia lo que es puramente de formato LaTeX.
    for a, b in (("\\\\", " "), ("~", " "), ("\\quad", "   "),
                 ("\\qquad", "    "), ("\\and", "   "), ("\\texttt{", ""),
                 ("\\textbf{", ""), ("\\thanks{", ""), ("%", ""), ("}", "")):
        crudo = crudo.replace(a, b)
    return " ".join(crudo.split())


def metadatos(tex: pathlib.Path) -> list:
    """``-M titulo -M autor`` para pandoc, sacados del preámbulo."""
    t = tex.read_text(encoding="utf-8")
    salida = []
    for orden, clave in (("title", "title"), ("author", "author")):
        valor = _llave(t, orden)
        if valor:
            salida += ["-M", f"{clave}={valor}"]
    return salida


def cuerpo_plano(tex: pathlib.Path) -> str:
    """El cuerpo del informe, con los ``\\input`` ya resueltos."""
    t = tex.read_text(encoding="utf-8")
    try:
        i = t.index(r"\begin{document}") + len(r"\begin{document}")
        j = t.index(r"\end{document}")
    except ValueError as e:
        raise ValueError(f"{tex.name} no parece un documento LaTeX completo") from e
    cuerpo = t[i:j]

    def pegar(m: "re.Match") -> str:
        f = tex.parent / f"{m.group(1)}.tex"
        if not f.is_file():
            # Se avisa en vez de dejar un hueco: una tabla que desaparece del
            # Word sin mensaje es peor que un error.
            print(f"  AVISO: falta {f}, esa tabla no va a estar en el Word")
            return ""
        return f.read_text(encoding="utf-8")

    return re.sub(r"\\input\{([^}]+)\}", pegar, cuerpo)


#: Preámbulo mínimo. No incluye neurips.sty a propósito (ver el docstring).
PREAMBULO = "\n".join([
    r"\documentclass{article}",
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage{booktabs}",
    r"\usepackage{graphicx}",
    r"\usepackage{amsmath}",
    r"\usepackage{natbib}",
    r"\begin{document}",
    "",
])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--informe", type=pathlib.Path, default=INFORME)
    parser.add_argument("--pandoc", help="ruta a pandoc.exe si no está en PATH")
    parser.add_argument("--salida", type=pathlib.Path,
                        help="por defecto, informe.docx junto al PDF")
    args = parser.parse_args(argv)

    tex = args.informe / "informe.tex"
    if not tex.is_file():
        print(f"no está {tex}")
        return 1

    pandoc = encontrar_pandoc(args.pandoc)
    if pandoc is None:
        print("no se encontró pandoc. Instalalo con:\n"
              "  winget install --id JohnMacFarlane.Pandoc\n"
              "o pasá su ruta con --pandoc. El PDF se genera igual sin esto.")
        return 1

    destino = args.salida or (args.informe / "informe.docx")
    plano = args.informe / "informe_plano.tex"
    plano.write_text(PREAMBULO + cuerpo_plano(tex) + "\n\\end{document}\n",
                     encoding="utf-8")

    orden = [pandoc, plano.name, "-o", destino.name,
             "--from", "latex", "--to", "docx",
             # El titulo y los autores viven en el preambulo, que este script
             # descarta: sin esto el Word arranca directo en el resumen.
             *metadatos(tex)]
    bib = args.informe / "referencias.bib"
    if bib.is_file():
        orden += ["--citeproc", "--bibliography", bib.name]

    print(f"  pandoc: {pandoc}")
    r = subprocess.run(orden, cwd=args.informe, capture_output=True, text=True)
    plano.unlink(missing_ok=True)      # era temporal, no un artefacto

    if r.returncode != 0:
        print(f"  pandoc falló ({r.returncode}):\n{(r.stderr or '')[:600]}")
        return r.returncode
    if not destino.is_file():
        print("  pandoc terminó bien pero no escribió el archivo")
        return 1
    print(f"  escrito: {destino}  ({destino.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
