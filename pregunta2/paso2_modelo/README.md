# Paso 2 — La U-Net propia (entregable 1)

El entregable 1 pide una U-Net implementada **sin copiar código de bibliotecas de
segmentación existentes**. Esta es esa implementación: lo único de terceros que
se usa son las primitivas de PyTorch (`Conv2d`, `MaxPool2d`, `ConvTranspose2d`).

## Archivos de este paso

| archivo | qué hace |
|---|---|
| `unet.py` | `ConfigUNet`, `DobleConv`, `UNet` y `construir_unet()` |

## Comandos

```python
from pregunta2.paso2_modelo.unet import construir_unet, UNet, ConfigUNet

modelo = construir_unet("base", canales_entrada=2, clases=1)   # 31.0 M
modelo = construir_unet("angosta")                             # 7.8 M
modelo = UNet(ConfigUNet(profundidad=5, base=64))              # 124.4 M
```

```bash
# Las variantes, verificadas
python -c "
import sys; sys.path.insert(0,'.')
from pregunta2.paso2_modelo.unet import construir_unet
for v in ('base','angosta','profunda','somera','bilineal','sin_norm'):
    m = construir_unet(v)
    print(f'{v:<10} {m.parametros_millones:>7.1f} M  {m.config.canales}')
"
```

## Lo que se hizo

### La estructura, como la piden las notas de clase

* **Codificador**: bloques de dos convoluciones 3×3 + ReLU, seguidos de
  max-pooling 2×2, duplicando canales al reducir resolución
  (64 → 128 → 256 → 512), con cuello de botella de 1024.
* **Decodificador**: up-convolución 2×2 que duplica la resolución y reduce los
  canales a la mitad; el resultado se **concatena** —no se suma— con el mapa del
  codificador del mismo nivel.
* **Capa final**: convolución 1×1 a K clases. Acá K=1 con sigmoide, porque
  "agua / no agua" es binario.
* **Inicialización de He**, que es lo que el artículo describe como "gaussiana
  con desviación √(2/N)": exactamente `kaiming_normal_` con no-linealidad ReLU.

La red devuelve **logits, no probabilidades**: la sigmoide vive en la pérdida
(`binary_cross_entropy_with_logits` es numéricamente estable) y en la evaluación.

### Las variantes

| variante | parámetros | niveles | canales |
|---|---:|---:|---|
| `base` | 31.0 M | 4 | `[64, 128, 256, 512, 1024]` |
| `angosta` | 7.8 M | 4 | `[32, 64, 128, 256, 512]` |
| `profunda` | 124.4 M | 5 | `[64, …, 2048]` |
| `somera` | 7.7 M | 3 | `[64, 128, 256, 512]` |
| `bilineal` | 28.9 M | 4 | sin `ConvTranspose`, interpola |
| `sin_norm` | 31.0 M | 4 | como el artículo de 2015, sin BatchNorm |

Verificado: entrada `(2, 2, 512, 512)` → salida `(2, 1, 512, 512)`, o sea el
**mismo tamaño espacial**.

### Tres apartamientos del artículo de 2015, y por qué

**1. Relleno "same" en vez de convoluciones sin relleno.** El artículo usa
convoluciones sin relleno, por lo que el mapa del codificador sale más grande que
el del decodificador y hay que **recortarlo** antes de concatenar (572 → 568 → …
en el diagrama de las notas). Las notas de clase dicen explícitamente que las
implementaciones modernas usan relleno para evitar ese recorte "simplificando la
implementación sin cambiar la idea central". Acá importa además por una razón del
dominio: con relleno la salida tiene **exactamente** el tamaño del chip de
entrada (512×512), así que la máscara predicha se alinea píxel a píxel con la
grilla georreferenciada del GeoTIFF. Sin eso habría que reproyectar la predicción
para compararla con la delineación de Copernicus, y ese paso introduce su propio
error.

**2. BatchNorm después de cada convolución.** No está en el artículo, que entrena
con lotes de tamaño 1 y momentum 0.99 justamente para compensar la falta de
normalización. Con chips de 512×512 y una GPU moderna entran lotes de 8–16, así
que se usa BatchNorm y un optimizador normal. Queda `normalizacion="ninguna"`
para poder ablacionar esta decisión.

**3. Profundidad configurable.** El entregable 2(b) admite "otra decisión
arquitectónica (profundidad, diseño de las conexiones de salto)" como segundo
factor de ablación, así que la profundidad es un parámetro y no una constante.

### Dos detalles que evitan fallos silenciosos

**Se valida que el lado sea múltiplo de 2^profundidad.** Cada nivel divide la
resolución por 2; si el lado no es múltiplo, el max-pooling trunca y la
up-convolución devuelve un mapa de tamaño distinto al del salto. Se valida en el
`forward` con un mensaje claro, en vez de dejar que falle con un error de formas
dentro de la concatenación.

**`bias=False` cuando hay normalización.** El término de sesgo es redundante con
el desplazamiento que aprende la norma; dejarlo es parámetros que no hacen nada.

## La entrada son 2 canales, no 3

El SAR no tiene canales de color: VV y VH son dos polarizaciones, en decibeles.
Es un parámetro (`canales_entrada`) y no una constante porque el bono del
entregable 6 agrega bandas ópticas de Sentinel-2.
