"""Compila el informe en formato NeurIPS y verifica que cumpla el enunciado.

El examen pide "un informe de 6 a 10 páginas en formato NeurIPS, con un mínimo
de 5 referencias". Las dos condiciones se verifican acá en vez de confiar en la
vista previa.

El límite es sobre el **cuerpo**, y las referencias no cuentan, así que no basta
con las páginas del PDF: se mide en qué página y a qué altura arranca la entrada
``[1]`` de la bibliografía (``paginas_de_cuerpo``). El conteo de referencias sale
del ``.bbl`` ya resuelto — no del ``.bib``, que puede tener más entradas de las
que el texto cita, y el mínimo es sobre las citadas.

Se corre ``latexmk`` si está disponible, porque resuelve por sí solo cuántas
pasadas hacen falta para que las citas y las referencias cruzadas converjan.
Sin él, se cae a la secuencia clásica pdflatex / bibtex / pdflatex / pdflatex.

Usage:
    python pregunta2/paso9_informe/compilar_informe.py
    python pregunta2/paso9_informe/compilar_informe.py --limpiar
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
from typing import Optional, Sequence

INFORME = pathlib.Path(__file__).resolve().parent / "informe"
MINIMO_PAGINAS, MAXIMO_PAGINAS = 6, 10
MINIMO_REFERENCIAS = 5


def _correr(orden: Sequence[str], directorio: pathlib.Path) -> int:
    return subprocess.run(orden, cwd=directorio, capture_output=True,
                          text=True, errors="replace").returncode


def compilar(carpeta: pathlib.Path, nombre: str = "informe") -> bool:
    """Compila y devuelve si el PDF quedó escrito."""
    if shutil.which("latexmk"):
        _correr(["latexmk", "-pdf", "-interaction=nonstopmode",
                 "-halt-on-error", f"{nombre}.tex"], carpeta)
    elif shutil.which("pdflatex"):
        # Tres pasadas: la primera genera los .aux, bibtex resuelve las citas,
        # las dos siguientes fijan los números de referencia y el índice.
        _correr(["pdflatex", "-interaction=nonstopmode", f"{nombre}.tex"], carpeta)
        if shutil.which("bibtex"):
            _correr(["bibtex", nombre], carpeta)
        for _ in range(2):
            _correr(["pdflatex", "-interaction=nonstopmode", f"{nombre}.tex"],
                    carpeta)
    else:
        print("no hay latexmk ni pdflatex en el PATH; instalá MiKTeX o TeX Live")
        return False
    return (carpeta / f"{nombre}.pdf").is_file()


def paginas_del_log(carpeta: pathlib.Path, nombre: str = "informe") -> Optional[int]:
    """Lee del log de LaTeX cuántas páginas escribió."""
    log = carpeta / f"{nombre}.log"
    if not log.is_file():
        return None
    texto = log.read_text(encoding="utf-8", errors="replace")
    encontrado = re.search(r"Output written on .*?\((\d+) page", texto)
    return int(encontrado.group(1)) if encontrado else None


def referencias_del_bbl(carpeta: pathlib.Path,
                        nombre: str = "informe") -> Optional[int]:
    """Cuenta las referencias que el texto realmente cita."""
    bbl = carpeta / f"{nombre}.bbl"
    if not bbl.is_file():
        return None
    return len(re.findall(r"\\bibitem", bbl.read_text(encoding="utf-8",
                                                      errors="replace")))


def paginas_de_cuerpo(carpeta: pathlib.Path,
                      nombre: str = "informe") -> Optional[float]:
    """Donde termina el cuerpo: las referencias no cuentan para el limite.

    El enunciado limita el cuerpo, no el PDF. Contar paginas totales obliga a
    suponer cuanto ocupa la bibliografia, asi que se mide: se busca la pagina
    donde arranca la entrada ``[1]`` y se devuelve su altura relativa dentro de
    la caja de texto. Un cuerpo que termina a media pagina 10 da ``9.5``.

    Devuelve ``None`` si no esta ``pypdf`` o si el PDF no deja extraer texto;
    quien llama cae entonces al conteo de paginas totales.
    """
    pdf = carpeta / f"{nombre}.pdf"
    if not pdf.is_file():
        return None
    try:
        import pypdf
    except ImportError:
        return None

    try:
        lector = pypdf.PdfReader(str(pdf))
    except Exception:
        return None

    for numero, pagina in enumerate(lector.pages, 1):
        try:
            texto = pagina.extract_text() or ""
        except Exception:
            return None
        # Se exige ver tambien "[2]": una cita suelta "[1]" en medio del cuerpo
        # no es el arranque de la bibliografia.
        marca = re.search(r"(?m)^\[1\]\s", texto)
        if not marca or not re.search(r"(?m)^\[2\]\s", texto[marca.end():]):
            continue
        return (numero - 1) + _altura_relativa(pagina)
    return float(len(lector.pages))


TITULOS = ("References", "Referencias", "Bibliograf")


def _altura_relativa(pagina) -> float:
    """Que fraccion de la pagina ocupa el cuerpo antes de la bibliografia.

    El corte es el TITULO de la bibliografia, no su primera entrada: si la
    pagina arranca con "References" y despues ``[1]``, el cuerpo termino en la
    pagina anterior y esta no aporta nada —medir desde ``[1]`` daria 0.03
    paginas de cuerpo que no existen, y con el limite justo eso alcanza para
    reprobar un informe que cumple—.
    """
    lineas: list = []

    def visitar(texto, cm, tm, fuente, tamano):
        lineas.append((tm[5], texto))

    try:
        pagina.extract_text(visitor_text=visitar)
    except Exception:
        return 0.0
    alturas = [y for y, _ in lineas]
    if not alturas or max(alturas) == min(alturas):
        return 0.0
    arriba, abajo = max(alturas), min(alturas)
    corte = None
    for y, texto in lineas:
        limpio = texto.strip()
        if corte is None and limpio.startswith(TITULOS):
            corte = y
            break
        if limpio.startswith("[1]"):
            corte = y          # sin titulo visible, la primera entrada sirve
            break
    if corte is None:
        return 0.0
    return max(0.0, (arriba - corte) / (arriba - abajo))

def errores_del_log(carpeta: pathlib.Path,
                    nombre: str = "informe") -> list[str]:
    """Las líneas de error de LaTeX, para no tener que abrir el log."""
    log = carpeta / f"{nombre}.log"
    if not log.is_file():
        return []
    lineas = log.read_text(encoding="utf-8", errors="replace").splitlines()
    return [l for l in lineas if l.startswith("!")][:10]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", type=pathlib.Path, default=INFORME)
    parser.add_argument("--nombre", default="informe")
    parser.add_argument("--limpiar", action="store_true",
                        help="borrar los auxiliares antes de compilar")
    args = parser.parse_args(argv)

    fuente = args.carpeta / f"{args.nombre}.tex"
    if not fuente.is_file():
        print(f"falta {fuente}")
        return 1

    if args.limpiar:
        for extension in (".aux", ".bbl", ".blg", ".log", ".out", ".fls",
                          ".fdb_latexmk"):
            (args.carpeta / f"{args.nombre}{extension}").unlink(missing_ok=True)

    if not compilar(args.carpeta, args.nombre):
        for linea in errores_del_log(args.carpeta, args.nombre):
            print(linea)
        print("la compilación no produjo PDF")
        return 1

    paginas = paginas_del_log(args.carpeta, args.nombre)
    referencias = referencias_del_bbl(args.carpeta, args.nombre)
    pdf = args.carpeta / f"{args.nombre}.pdf"
    print(f"{pdf} ({pdf.stat().st_size / 1024:.0f} KB)")

    problemas = []
    cuerpo = paginas_de_cuerpo(args.carpeta, args.nombre)
    if cuerpo is not None:
        print(f"páginas: {paginas} en el PDF, {cuerpo:.2f} de cuerpo "
              f"(el enunciado pide {MINIMO_PAGINAS}–{MAXIMO_PAGINAS} de cuerpo)")
        if cuerpo < MINIMO_PAGINAS:
            problemas.append(f"{cuerpo:.2f} páginas de cuerpo, menos que el mínimo")
        elif cuerpo > MAXIMO_PAGINAS:
            problemas.append(f"{cuerpo:.2f} páginas de cuerpo, excede el máximo "
                             f"de {MAXIMO_PAGINAS}")
    elif paginas is None:
        problemas.append("no se pudo leer el número de páginas del log")
    else:
        # Sin pypdf no se puede medir donde termina el cuerpo, asi que se cae al
        # conteo total suponiendo que la bibliografia ocupa a lo sumo una pagina.
        # Es un proxy: se avisa en vez de presentarlo como una verificacion.
        print(f"páginas: {paginas} en el PDF (sin pypdf no se puede medir el "
              f"cuerpo; límite {MINIMO_PAGINAS}–{MAXIMO_PAGINAS})")
        if paginas < MINIMO_PAGINAS:
            problemas.append(f"{paginas} páginas, menos que el mínimo")
        elif paginas > MAXIMO_PAGINAS + 1:
            problemas.append(f"{paginas} páginas, el cuerpo excede el máximo")

    if referencias is None:
        problemas.append("no hay .bbl: ¿corrió bibtex?")
    else:
        print(f"referencias citadas: {referencias} "
              f"(mínimo {MINIMO_REFERENCIAS})")
        if referencias < MINIMO_REFERENCIAS:
            problemas.append(f"{referencias} referencias, menos que el mínimo")

    errores = errores_del_log(args.carpeta, args.nombre)
    if errores:
        print("\nerrores de LaTeX:")
        for linea in errores:
            print(f"  {linea}")

    if problemas:
        print("\nel informe NO cumple:")
        for problema in problemas:
            print(f"  - {problema}")
        return 1

    print("\nel informe cumple el formato que pide el enunciado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
