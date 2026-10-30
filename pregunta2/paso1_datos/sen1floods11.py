"""Lector de Sen1Floods11 y la pérdida enmascarada que exige.

Tres decisiones de este módulo, todas medidas y no supuestas (los números salen
de ``estadisticas.py`` sobre el split de entrenamiento oficial):

**1. La normalización no puede ser la de ImageNet.** La entrada son dos canales
de retrodispersión en decibeles, no colores entre 0 y 1. Medido::

    VV  media -10.39 dB  desvio 4.04   p0.1 -29.3   p99.9  +3.1
    VH  media -17.24 dB  desvio 4.75   p0.1 -47.0   p99.9  -5.1

Se recorta a los percentiles 0.1/99.9 y después se estandariza. El recorte no es
cosmético: el mínimo crudo es -65 dB en VV y -79 dB en VH, que es esencialmente
``log(0)`` —píxeles donde la retrodispersión dio cero—, y un solo valor así
estira la escala y aplasta todo lo demás.

**2. La pérdida tiene que ignorar los -1.** El dataset usa tres valores: ``1``
agua, ``0`` no agua y ``-1`` "no se pudo etiquetar". Medido sobre entrenamiento,
los ``-1`` son el **13.26%** de los píxeles, y en algunos chips pasan del 60%.
Tratarlos como "no agua" metería supervisión falsa en uno de cada siete píxeles.
Es el mismo problema que ya resolvimos en la Pregunta 1 con las etiquetas
inciertas de CheXpert, y la solución es la misma: una máscara que entra en el
divisor de la pérdida.

**3. El agua es el 9.5% de lo observado.** Con esa prevalencia, la exactitud por
píxel es una métrica inútil —predecir "nunca hay agua" da 90.5%—. Por eso el
enunciado pide **IoU**, y por eso la pérdida combina BCE con Dice: la BCE sola
tiende a la clase mayoritaria cuando el desbalance es así de fuerte.
"""

from __future__ import annotations

import json
import pathlib
import sys

# Las rutas de datos salen de pregunta2/rutas.py, que prefiere
# pregunta2/datos/ (los enlaces) y cae a data/ si no existen.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from pregunta2 import rutas as _rutas
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Las rutas de abajo se anclan a ESTE archivo, no al directorio de
# trabajo: la pregunta se comparte sola y sus scripts se corren desde
# cualquier lado.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]

RAIZ = _rutas.V11
MANO = _rutas.MANO
DEBIL = _rutas.DEBIL
SPLITS = _rutas.SPLITS
ESTADISTICAS = _RAIZ / "pregunta2/paso1_datos/estadisticas.json"

IGNORAR = -1


def cargar_estadisticas(ruta: pathlib.Path = ESTADISTICAS) -> dict:
    if not ruta.is_file():
        raise FileNotFoundError(
            f"falta {ruta}; corre pregunta2/paso1_datos/estadisticas.py")
    return json.loads(ruta.read_text(encoding="utf-8"))


def chips_del_split(nombre: str, splits: pathlib.Path = SPLITS) -> list[str]:
    """``train`` | ``valid`` | ``test`` | ``bolivia`` -> ids ``Region_12345``.

    ``bolivia`` es la partición que el propio dataset deja **fuera** de
    entrenamiento, validación y prueba: es su región de generalización. Lima
    (entregable 4) va a ser una segunda, externa al dataset entero.
    """
    ruta = splits / f"flood_{nombre}_data.csv"
    filas = ruta.read_text(encoding="utf-8").strip().splitlines()
    return [f.split(",")[0].replace("_S1Hand.tif", "") for f in filas if f.strip()]


def chips_debiles(excluir_regiones: Sequence[str] = (),
                  debil: pathlib.Path = DEBIL) -> list[str]:
    """Los 4 384 chips con etiqueta débil, menos las regiones que se excluyan.

    ``excluir_regiones`` **no es opcional en la práctica**: el conjunto débil
    incluye 224 chips de **Bolivia** y 534 de Colombia. Bolivia es la región que
    el entregable 4 retiene para medir generalización, así que agregarla al
    entrenamiento por venir "del otro conjunto" sería una fuga silenciosa: el
    modelo habría visto la región sobre la que después se lo evalúa, y la brecha
    medida no significaría nada.

    No hace falta excluir los chips dorados: se verificó que los identificadores
    de ambos conjuntos no se solapan en ninguno de los 4 384.
    """
    debil = pathlib.Path(debil)
    excluidas = {r.lower() for r in excluir_regiones}
    chips = sorted(p.name.replace("_S1Weak.tif", "")
                   for p in (debil / "S1Weak").glob("*_S1Weak.tif"))
    return [c for c in chips if c.split("_")[0].lower() not in excluidas]


def iou_por_chip(confusiones: np.ndarray) -> float:
    """IoU **promedio sobre los chips**, que es la métrica de los autores.

    No es lo mismo que la IoU agregada (sumar los conteos de todos los chips y
    dividir una sola vez). La agregada le da más peso a los chips con más agua;
    el promedio por chip le da el mismo peso a cada chip, y por eso es más
    sensible a los chips casi secos, donde la IoU es inestable.

    Un chip sin un solo píxel de agua ni predicho ni real tiene unión cero: la
    IoU no está definida ahí. Se lo **excluye** del promedio en vez de contarlo
    como 0 (castigaría un acierto) o como 1 (premiaría no hacer nada).
    """
    vp, fp, fn = confusiones[:, 0], confusiones[:, 1], confusiones[:, 2]
    union = vp + fp + fn
    definidas = union > 0
    if not definidas.any():
        return float("nan")
    return float((vp[definidas] / union[definidas]).mean())


@dataclass(frozen=True)
class Normalizador:
    """Recorta a percentiles medidos y estandariza. Congelado a propósito."""

    media: tuple[float, ...]
    desvio: tuple[float, ...]
    piso: tuple[float, ...]
    techo: tuple[float, ...]

    @classmethod
    def desde_estadisticas(cls, est: Optional[dict] = None) -> "Normalizador":
        est = est or cargar_estadisticas()
        p = est["percentiles"]
        return cls(media=tuple(est["media"]), desvio=tuple(est["desvio"]),
                   piso=tuple(p["0.1"]), techo=tuple(p["99.9"]))

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """(bandas, alto, ancho) en dB -> estandarizado, sin no-finitos."""
        salida = np.empty_like(a, dtype=np.float32)
        for b in range(a.shape[0]):
            v = a[b]
            # Los -inf de log(0) se mandan al piso en vez de propagarse como NaN
            # por toda la red: un NaN en la entrada envenena todo el gradiente.
            v = np.where(np.isfinite(v), v, self.piso[b])
            v = np.clip(v, self.piso[b], self.techo[b])
            salida[b] = (v - self.media[b]) / max(self.desvio[b], 1e-6)
        return salida


def _voltear(x: np.ndarray, y: np.ndarray, m: np.ndarray, rng) -> tuple:
    """Volteos y giros de 90 grados. Nada de recortes ni escalas.

    Un recorte aleatorio rompería la alineación con la grilla georreferenciada,
    y una rotación que no sea múltiplo de 90 grados interpolaría la máscara e
    inventaría píxeles de agua en los bordes. Con 252 chips la aumentación hace
    falta, pero no a costa de ensuciar la etiqueta.
    """
    if rng.random() < 0.5:
        x, y, m = x[:, :, ::-1], y[:, ::-1], m[:, ::-1]
    if rng.random() < 0.5:
        x, y, m = x[:, ::-1, :], y[::-1, :], m[::-1, :]
    k = rng.integers(0, 4)
    if k:
        x = np.rot90(x, k, axes=(1, 2))
        y, m = np.rot90(y, k), np.rot90(m, k)
    return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(m)


class ChipsSen1Floods11(Dataset):
    """Sirve ``(x, y, mascara)`` por chip.

    * ``x``: (2, 512, 512) float32, VV y VH estandarizados.
    * ``y``: (1, 512, 512) float32 con 1 = agua, 0 = no agua. Donde la etiqueta
      original era -1 se escribe 0, pero la máscara lo apaga: el valor que queda
      ahí no importa porque no entra en la pérdida.
    * ``mascara``: (1, 512, 512) float32, 1 donde hay etiqueta utilizable.
    """

    def __init__(self, chips: Sequence[str], normalizador: Normalizador,
                 aumentar: bool = False, semilla: int = 0,
                 mano: pathlib.Path = MANO,
                 sufijo_imagen: str = "S1Hand",
                 sufijo_etiqueta: str = "LabelHand") -> None:
        self.mano = pathlib.Path(mano)
        self.sufijo_imagen = sufijo_imagen
        self.sufijo_etiqueta = sufijo_etiqueta
        self.chips = [c for c in chips if self._rutas(c)[0].is_file()
                      and self._rutas(c)[1].is_file()]
        faltan = len(chips) - len(self.chips)
        if faltan:
            print(f"  aviso: {faltan} chips del split no estan en disco, se omiten")
        if not self.chips:
            raise FileNotFoundError(f"ningun chip legible bajo {self.mano}")
        self.normalizador = normalizador
        self.aumentar = aumentar
        self.semilla = semilla

    def _rutas(self, chip: str) -> tuple[pathlib.Path, pathlib.Path]:
        return (self.mano / self.sufijo_imagen / f"{chip}_{self.sufijo_imagen}.tif",
                self.mano / self.sufijo_etiqueta / f"{chip}_{self.sufijo_etiqueta}.tif")

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, i: int):
        chip = self.chips[i]
        ruta_x, ruta_y = self._rutas(chip)
        with rasterio.open(ruta_x) as r:
            x = r.read().astype(np.float32)
        with rasterio.open(ruta_y) as r:
            etiqueta = r.read(1)

        x = self.normalizador(x)
        mascara = (etiqueta != IGNORAR).astype(np.float32)
        y = (etiqueta == 1).astype(np.float32)

        if self.aumentar:
            # Semilla por indice y por epoca implicita: distinto en cada lectura,
            # pero reproducible si se fija la semilla global.
            rng = np.random.default_rng(self.semilla + i * 7919 +
                                        int(torch.randint(0, 2**20, (1,)).item()))
            x, y, mascara = _voltear(x, y, mascara, rng)

        return (torch.from_numpy(x),
                torch.from_numpy(y)[None],
                torch.from_numpy(mascara)[None])


def perdida_enmascarada(logits: torch.Tensor, objetivo: torch.Tensor,
                        mascara: torch.Tensor, peso_dice: float = 0.5,
                        pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """BCE + Dice, ambas calculadas **solo sobre los píxeles etiquetados**.

    El divisor es la cantidad de píxeles observados, no el total: dividir por el
    total haría que un chip con muchos -1 pese menos que otro igual de informativo
    solo por tener más huecos.

    Si un lote no tiene ni un píxel observado se devuelve cero **con gradiente**
    (``logits * 0``) y no el escalar 0: un ``torch.tensor(0.)`` suelto corta el
    grafo y ``backward`` falla.
    """
    observadas = mascara.sum()
    if observadas == 0:
        return (logits * 0.0).sum()

    celdas = F.binary_cross_entropy_with_logits(
        logits, objetivo, reduction="none", pos_weight=pos_weight)
    bce = (celdas * mascara).sum() / observadas

    if peso_dice <= 0:
        return bce

    prob = torch.sigmoid(logits) * mascara
    verdad = objetivo * mascara
    interseccion = (prob * verdad).sum()
    union = prob.sum() + verdad.sum()
    dice = 1 - (2 * interseccion + 1.0) / (union + 1.0)
    return (1 - peso_dice) * bce + peso_dice * dice


@torch.no_grad()
def acumular_confusion(logits: torch.Tensor, objetivo: torch.Tensor,
                       mascara: torch.Tensor, umbral: float = 0.5) -> np.ndarray:
    """``[vp, fp, fn, vn]`` sobre los píxeles observados.

    Se acumulan los conteos y la IoU se calcula **al final**, sobre el total. El
    promedio de las IoU por chip no es lo mismo: un chip casi sin agua da una IoU
    muy ruidosa y pesaría igual que uno con un río entero.
    """
    pred = ((torch.sigmoid(logits) > umbral).float() * mascara)
    verdad = objetivo * mascara
    vp = (pred * verdad).sum().item()
    fp = (pred * (1 - verdad) * mascara).sum().item()
    fn = ((1 - pred) * verdad * mascara).sum().item()
    vn = ((1 - pred) * (1 - verdad) * mascara).sum().item()
    return np.array([vp, fp, fn, vn], dtype=np.float64)


def metricas(conf: np.ndarray) -> dict:
    """IoU de agua, F1, precisión, exhaustividad y exactitud."""
    vp, fp, fn, vn = conf
    union = vp + fp + fn
    return {
        "iou_agua": vp / union if union > 0 else float("nan"),
        "f1": 2 * vp / (2 * vp + fp + fn) if (2 * vp + fp + fn) > 0 else float("nan"),
        "precision": vp / (vp + fp) if (vp + fp) > 0 else float("nan"),
        "exhaustividad": vp / (vp + fn) if (vp + fn) > 0 else float("nan"),
        "exactitud": (vp + vn) / conf.sum() if conf.sum() > 0 else float("nan"),
        "pixeles_observados": int(conf.sum()),
    }
