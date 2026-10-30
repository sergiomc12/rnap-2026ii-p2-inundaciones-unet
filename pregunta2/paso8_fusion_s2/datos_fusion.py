"""El lector que sirve SAR, óptico o los dos, sobre los mismos chips dorados.

El bono del entregable 6 pide fusionar Sentinel-2 con el SAR. Para que la
comparación signifique algo, las tres configuraciones —solo SAR, solo óptico y
fusión— tienen que correr sobre **los mismos chips, con la misma máscara y la
misma partición**; si no, la diferencia mide el conjunto y no la fusión. Por eso
hay un solo `Dataset` con un parámetro de modo, en vez de tres lectores.

Qué bandas de Sentinel-2, y por qué
-----------------------------------
`S2Hand` trae las 13 bandas L1C. Meterlas todas sería desperdiciar capacidad en
un conjunto de 252 chips de entrenamiento, así que se usan las cuatro donde el
agua se separa de verdad:

* **verde (B3)** y **NIR (B8)** — el agua refleja algo en verde y **absorbe casi
  todo** en infrarrojo cercano. Esa diferencia es el NDWI, el índice clásico de
  agua;
* **SWIR (B11)** — la absorción es todavía más fuerte, y además distingue agua de
  sombra de nube, que en NDWI se parecen;
* **rojo (B4)** — para que la red pueda formar un índice de vegetación si le
  sirve, en vez de obligarla a inferirlo.

Se agregan **NDWI** y **MNDWI** ya calculados. No aportan información nueva —son
funciones de bandas que ya están— pero le ahorran a la red tener que descubrir un
cociente con 252 chips, y el cociente normaliza las diferencias de iluminación
entre escenas, que es justo lo que hace que el óptico generalice mal.

La trampa que este módulo evita
-------------------------------
Sentinel-2 **no ve a través de las nubes**, que es la razón entera por la que
esta tarea usa SAR. Un chip con nubes tiene reflectancias altísimas y planas, y
un modelo que fusione sin más aprenderá a confiar en el óptico incluso ahí. Por
eso `cobertura_de_nube()` mide, por chip, la fracción de píxeles probablemente
nublados, y `entrenar_fusion.py` reporta la ganancia **separada por nubosidad**:
si la fusión solo ayuda con cielo despejado, eso es un resultado, no un detalle.

Usage:
    from pregunta2.paso8_fusion_s2.datos_fusion import ChipsFusion, MODOS
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from pregunta2.paso1_datos.sen1floods11 import (
    IGNORAR, MANO, Normalizador, _voltear,
)

#: Orden de las 13 bandas L1C en `S2Hand`, tal como las escribe Sen1Floods11.
BANDAS_S2 = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
             "B9", "B10", "B11", "B12")

#: Las que se usan. Ver el docstring del módulo para el por qué de cada una.
BANDAS_USADAS = ("B3", "B4", "B8", "B11")

#: Índices que se agregan como canales derivados.
#: NDWI  = (verde − NIR) / (verde + NIR)      — McFeeters, 1996
#: MNDWI = (verde − SWIR) / (verde + SWIR)    — Xu, 2006
INDICES = ("NDWI", "MNDWI")

MODOS = ("sar", "optico", "fusion")

#: Umbral de reflectancia sobre el que un píxel se considera probablemente
#: nublado. Las nubes son brillantes en todo el visible; 0.25 de reflectancia en
#: azul es el criterio grueso que usan los filtros rápidos de Sentinel-2. No es
#: una máscara de nube seria —para eso está la banda de calidad, que S2Hand no
#: trae— y por eso se reporta como "probablemente nublado", no como verdad.
UMBRAL_NUBE_AZUL = 0.25

#: Sen1Floods11 guarda la reflectancia L1C escalada por 10 000, que es el formato
#: nativo de Sentinel-2. Se divide para trabajar en reflectancia [0, 1].
ESCALA_S2 = 10000.0


def ruta_s2(chip: str, mano: pathlib.Path = MANO) -> pathlib.Path:
    return mano / "S2Hand" / f"{chip}_S2Hand.tif"


def hay_optico(chip: str, mano: pathlib.Path = MANO) -> bool:
    return ruta_s2(chip, mano).is_file()


def leer_s2(chip: str, mano: pathlib.Path = MANO) -> np.ndarray:
    """Las 13 bandas en reflectancia [0, 1], con los no-finitos en cero."""
    with rasterio.open(ruta_s2(chip, mano)) as r:
        a = r.read().astype(np.float32) / ESCALA_S2
    return np.where(np.isfinite(a), a, 0.0)


def indice_normalizado(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a − b) / (a + b), con el denominador protegido.

    Donde las dos bandas son cero —borde sin dato— el índice no está definido;
    devolver 0 ahí es lo correcto: 0 es el valor que separa agua de no-agua en
    NDWI, o sea la respuesta más neutra posible.
    """
    den = a + b
    return np.where(np.abs(den) > 1e-6, (a - b) / np.where(den == 0, 1, den), 0.0)


def cobertura_de_nube(s2: np.ndarray) -> float:
    """Fracción de píxeles probablemente nublados, por brillo en azul (B2)."""
    azul = s2[BANDAS_S2.index("B2")]
    return float((azul > UMBRAL_NUBE_AZUL).mean())


def canales_opticos(s2: np.ndarray) -> np.ndarray:
    """Las bandas elegidas más NDWI y MNDWI, apiladas."""
    idx = {b: BANDAS_S2.index(b) for b in BANDAS_USADAS}
    verde, nir, swir = (s2[BANDAS_S2.index("B3")],
                        s2[BANDAS_S2.index("B8")],
                        s2[BANDAS_S2.index("B11")])
    capas = [s2[idx[b]] for b in BANDAS_USADAS]
    capas.append(indice_normalizado(verde, nir))    # NDWI
    capas.append(indice_normalizado(verde, swir))   # MNDWI
    return np.stack(capas).astype(np.float32)


def canales_de(modo: str) -> int:
    if modo == "sar":
        return 2
    if modo == "optico":
        return len(BANDAS_USADAS) + len(INDICES)
    if modo == "fusion":
        return 2 + len(BANDAS_USADAS) + len(INDICES)
    raise ValueError(f"modo desconocido: {modo!r}; use uno de {MODOS}")


@dataclass(frozen=True)
class NormalizadorOptico:
    """Estandariza los canales ópticos con medias y desvíos medidos.

    Se mide **solo sobre los chips de entrenamiento**: usar el conjunto entero
    filtraría información de prueba en la normalización, que es la forma más
    silenciosa de inflar un resultado.
    """

    media: tuple[float, ...]
    desvio: tuple[float, ...]

    @classmethod
    def desde_chips(cls, chips: Sequence[str],
                    mano: pathlib.Path = MANO) -> "NormalizadorOptico":
        sumas = conteos = sumas2 = None
        for c in chips:
            if not hay_optico(c, mano):
                continue
            v = canales_opticos(leer_s2(c, mano)).reshape(canales_de("optico"), -1)
            if sumas is None:
                sumas = v.sum(1).astype(np.float64)
                sumas2 = (v.astype(np.float64) ** 2).sum(1)
                conteos = v.shape[1]
            else:
                sumas += v.sum(1)
                sumas2 += (v.astype(np.float64) ** 2).sum(1)
                conteos += v.shape[1]
        if sumas is None:
            raise FileNotFoundError(
                "ningun chip de entrenamiento tiene S2Hand. Corre:\n"
                "  python pregunta2/paso1_datos/descargar.py "
                "--solo-lo-pedido --s2-dorado")
        media = sumas / conteos
        var = np.maximum(sumas2 / conteos - media ** 2, 1e-12)
        return cls(media=tuple(media.tolist()),
                   desvio=tuple(np.sqrt(var).tolist()))

    def __call__(self, a: np.ndarray) -> np.ndarray:
        salida = np.empty_like(a, dtype=np.float32)
        for b in range(a.shape[0]):
            salida[b] = (a[b] - self.media[b]) / max(self.desvio[b], 1e-6)
        return salida


class ChipsFusion(Dataset):
    """Sirve ``(x, y, mascara)`` en el modo pedido, sobre los chips dorados.

    ``x`` tiene ``canales_de(modo)`` canales. ``y`` y ``mascara`` son idénticos
    en los tres modos: la verdad no cambia porque cambien las entradas, y eso es
    lo que hace comparables las tres corridas.
    """

    def __init__(self, chips: Sequence[str], modo: str,
                 norm_sar: Normalizador,
                 norm_optico: Optional[NormalizadorOptico] = None,
                 aumentar: bool = False, semilla: int = 0,
                 mano: pathlib.Path = MANO) -> None:
        if modo not in MODOS:
            raise ValueError(f"modo desconocido: {modo!r}; use uno de {MODOS}")
        self.modo = modo
        self.mano = pathlib.Path(mano)
        self.norm_sar = norm_sar
        self.norm_optico = norm_optico
        self.aumentar = aumentar
        self.semilla = semilla
        # Un chip sin optico NO entra en ningun modo, ni siquiera en 'sar':
        # comparar sobre conjuntos distintos mediria el conjunto.
        self.chips = [c for c in chips
                      if (self.mano / "S1Hand" / f"{c}_S1Hand.tif").is_file()
                      and (self.mano / "LabelHand" / f"{c}_LabelHand.tif").is_file()
                      and hay_optico(c, self.mano)]
        faltan = len(chips) - len(self.chips)
        if faltan:
            print(f"  aviso: {faltan} chips sin S1, etiqueta u optico, se omiten")
        if not self.chips:
            raise FileNotFoundError(
                f"ningun chip con las tres piezas bajo {self.mano}")
        if modo in ("optico", "fusion") and norm_optico is None:
            raise ValueError("el modo optico necesita un NormalizadorOptico")

    def __len__(self) -> int:
        return len(self.chips)

    def entrada_de(self, chip: str) -> np.ndarray:
        partes = []
        if self.modo in ("sar", "fusion"):
            with rasterio.open(self.mano / "S1Hand" / f"{chip}_S1Hand.tif") as r:
                partes.append(self.norm_sar(r.read().astype(np.float32)))
        if self.modo in ("optico", "fusion"):
            partes.append(self.norm_optico(canales_opticos(
                leer_s2(chip, self.mano))))
        return np.concatenate(partes, axis=0).astype(np.float32)

    def __getitem__(self, i: int):
        chip = self.chips[i]
        x = self.entrada_de(chip)
        with rasterio.open(self.mano / "LabelHand" / f"{chip}_LabelHand.tif") as r:
            etiqueta = r.read(1)
        mascara = (etiqueta != IGNORAR).astype(np.float32)
        y = (etiqueta == 1).astype(np.float32)

        if self.aumentar:
            rng = np.random.default_rng(self.semilla + i * 7919 +
                                        int(torch.randint(0, 2**20, (1,)).item()))
            x, y, mascara = _voltear(x, y, mascara, rng)

        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y))[None],
                torch.from_numpy(np.ascontiguousarray(mascara))[None])
