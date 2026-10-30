"""U-Net propia, siguiendo Ronneberger et al. (2015) y las notas de clase.

El entregable 1 pide implementación propia **sin copiar código de bibliotecas
de segmentación**. Esta es esa implementación; lo único de terceros que se usa
son las primitivas de PyTorch (``Conv2d``, ``MaxPool2d``, ``ConvTranspose2d``).

La estructura sigue las notas de clase del curso:

* **Codificador**: bloques de dos convoluciones 3x3 + ReLU, seguidos de
  max-pooling 2x2, duplicando canales al reducir resolución
  (64 -> 128 -> 256 -> 512), con cuello de botella de 1024.
* **Decodificador**: up-convolución 2x2 que duplica resolución y reduce
  canales a la mitad, y el resultado se **concatena** con el mapa del
  codificador del mismo nivel.
* **Capa final**: convolución 1x1 a K clases. Acá K=1 con sigmoide, porque
  "agua / no agua" es binario.

Tres decisiones donde esta implementación se aparta del artículo original, y
por qué:

1. **Relleno "same" en vez de convoluciones sin relleno.** El artículo usa
   convoluciones sin relleno, por lo que el mapa del codificador sale más
   grande que el del decodificador y hay que **recortarlo** antes de
   concatenar (572 -> 568 -> ... en el diagrama de las notas). Las notas dicen
   explícitamente que las implementaciones modernas usan relleno para evitar
   ese recorte "simplificando la implementación sin cambiar la idea central".
   Acá importa además por una razón del dominio: con relleno la salida tiene
   **exactamente** el tamaño del chip de entrada (512x512), así que la máscara
   predicha se alinea píxel a píxel con la grilla georreferenciada del GeoTIFF.
   Sin eso habría que reproyectar la predicción para compararla con la
   delineación de Copernicus, y ese paso introduce su propio error.

2. **BatchNorm después de cada convolución.** No está en el artículo de 2015
   (BatchNorm es de 2015 también y aún no era estándar), pero el artículo
   entrena con lotes de tamaño 1 y momentum 0.99 justamente para compensar la
   falta de normalización. Con chips de 512x512 y una GPU moderna entran lotes
   de 8-16, así que se usa BatchNorm y un optimizador normal. Se deja
   ``normalizacion="ninguna"`` para poder ablacionar esta decisión.

3. **Profundidad configurable.** El entregable 2(b) admite "otra decisión
   arquitectónica (profundidad, diseño de las conexiones de salto)" como
   segundo factor de ablación, así que la profundidad es un parámetro y no una
   constante.

La entrada son **2 canales** (VV y VH en dB), no 3 como una imagen RGB: el
SAR no tiene canales de color. Es un parámetro porque el bono del entregable 6
agrega bandas ópticas de Sentinel-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(nombre: str, canales: int) -> nn.Module:
    if nombre == "batch":
        return nn.BatchNorm2d(canales)
    if nombre == "grupo":
        # GroupNorm sirve cuando el lote es muy chico, que es el caso si se
        # entrena a 512x512 sin recortar en una GPU modesta.
        return nn.GroupNorm(min(8, canales), canales)
    if nombre == "ninguna":
        return nn.Identity()
    raise ValueError(f"normalizacion desconocida: {nombre}")


class DobleConv(nn.Module):
    """Las dos convoluciones 3x3 + ReLU que forman cada nivel de la U."""

    def __init__(self, entrada: int, salida: int, normalizacion: str = "batch",
                 dropout: float = 0.0) -> None:
        super().__init__()
        capas: list[nn.Module] = []
        for dentro, fuera in ((entrada, salida), (salida, salida)):
            capas += [
                # bias=False cuando hay normalizacion: el termino de sesgo es
                # redundante con el desplazamiento que aprende la norma.
                nn.Conv2d(dentro, fuera, 3, padding=1,
                          bias=normalizacion == "ninguna"),
                _norm(normalizacion, fuera),
                nn.ReLU(inplace=True),
            ]
        if dropout > 0:
            capas.append(nn.Dropout2d(dropout))
        self.bloque = nn.Sequential(*capas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bloque(x)


@dataclass(frozen=True)
class ConfigUNet:
    """Todo lo que define la red, congelado para que una corrida no se cambie a sí misma."""

    canales_entrada: int = 2          # VV, VH
    clases: int = 1                   # agua / no agua
    base: int = 64                    # canales del primer nivel, como el articulo
    profundidad: int = 4              # niveles de bajada (4 -> cuello de 1024)
    normalizacion: str = "batch"
    dropout_cuello: float = 0.0
    bilineal: bool = False            # True usa interpolacion en vez de ConvTranspose

    def __post_init__(self) -> None:
        if self.canales_entrada < 1:
            raise ValueError("canales_entrada debe ser >= 1")
        if self.profundidad < 1:
            raise ValueError("profundidad debe ser >= 1")
        if self.base < 1:
            raise ValueError("base debe ser >= 1")

    @property
    def canales(self) -> list[int]:
        """Canales de cada nivel: 64, 128, 256, 512, ... y el cuello."""
        return [self.base * 2 ** i for i in range(self.profundidad + 1)]

    @property
    def multiplo_requerido(self) -> int:
        """El lado de la entrada tiene que ser multiplo de esto.

        Cada nivel divide la resolucion por 2. Si el lado no es multiplo de
        2**profundidad, el max-pooling trunca y la up-convolucion devuelve un
        mapa de tamano distinto al del salto. Se valida en el forward en vez de
        dejar que falle con un error de formas en la concatenacion.
        """
        return 2 ** self.profundidad


class UNet(nn.Module):
    """Codificador-decodificador simétrico con conexiones de salto por concatenación."""

    def __init__(self, config: Optional[ConfigUNet] = None, **kwargs) -> None:
        super().__init__()
        self.config = config or ConfigUNet(**kwargs)
        c = self.config.canales

        # --- camino de contraccion ---
        self.bajadas = nn.ModuleList()
        dentro = self.config.canales_entrada
        for fuera in c[:-1]:
            self.bajadas.append(DobleConv(dentro, fuera, self.config.normalizacion))
            dentro = fuera
        self.pool = nn.MaxPool2d(2)

        # --- cuello de botella ---
        self.cuello = DobleConv(dentro, c[-1], self.config.normalizacion,
                                dropout=self.config.dropout_cuello)

        # --- camino de expansion ---
        self.subidas = nn.ModuleList()
        self.fusiones = nn.ModuleList()
        dentro = c[-1]
        for fuera in reversed(c[:-1]):
            if self.config.bilineal:
                # Alternativa sin parametros: interpolar y arreglar canales con
                # una conv 1x1. Evita los artefactos de tablero de ajedrez que
                # a veces deja ConvTranspose.
                self.subidas.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(dentro, fuera, 1, bias=False)))
            else:
                self.subidas.append(nn.ConvTranspose2d(dentro, fuera, 2, stride=2))
            # fuera*2 porque entra la concatenacion: el mapa que sube mas el salto
            self.fusiones.append(DobleConv(fuera * 2, fuera, self.config.normalizacion))
            dentro = fuera

        self.salida = nn.Conv2d(dentro, self.config.clases, 1)
        self._inicializar()

    def _inicializar(self) -> None:
        """Inicializacion de He, que es la que piden las notas de clase.

        El articulo la describe como "gaussiana con desviacion sqrt(2/N)", que
        es exactamente ``kaiming_normal_`` con no-linealidad ReLU.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(lote, canales, alto, ancho) -> (lote, clases, alto, ancho).

        Devuelve **logits**, no probabilidades: la sigmoide vive en la pérdida
        (``BCEWithLogitsLoss`` es numéricamente estable) y en la evaluación.
        """
        m = self.config.multiplo_requerido
        if x.shape[-1] % m or x.shape[-2] % m:
            raise ValueError(
                f"el lado debe ser multiplo de {m} para profundidad "
                f"{self.config.profundidad}; llego {tuple(x.shape[-2:])}")

        saltos = []
        for bajada in self.bajadas:
            x = bajada(x)
            saltos.append(x)
            x = self.pool(x)

        x = self.cuello(x)

        for subida, fusion, salto in zip(self.subidas, self.fusiones,
                                         reversed(saltos)):
            x = subida(x)
            if x.shape[-2:] != salto.shape[-2:]:
                # Con relleno "same" y lado multiplo de 2**profundidad esto no
                # deberia pasar nunca. Si pasa, se ajusta en vez de romper: un
                # error de un pixel por redondeo no justifica perder la corrida.
                x = F.interpolate(x, size=salto.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = fusion(torch.cat([salto, x], dim=1))

        return self.salida(x)

    @property
    def parametros_millones(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e6


def construir_unet(variante: str = "base", **kwargs) -> UNet:
    """Las variantes que usa la ablación del entregable 2(b)."""
    variantes = {
        "base": {},                                   # 4 niveles, 64 base
        "angosta": {"base": 32},                      # mitad de canales
        "profunda": {"profundidad": 5},               # un nivel mas
        "somera": {"profundidad": 3},                 # un nivel menos
        "bilineal": {"bilineal": True},               # sin ConvTranspose
        "sin_norm": {"normalizacion": "ninguna"},     # como el articulo de 2015
    }
    if variante not in variantes:
        raise ValueError(f"variante desconocida: {variante}; "
                         f"hay {sorted(variantes)}")
    return UNet(ConfigUNet(**{**variantes[variante], **kwargs}))
