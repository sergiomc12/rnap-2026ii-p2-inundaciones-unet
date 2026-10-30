# Paso 6 — Vistas para poder mirar los chips

Los datos de esta pregunta **no se abren en un visor de imágenes común**, y eso
es un problema práctico: buena parte de los errores de este trabajo se
encontraron mirando, no leyendo métricas.

* Los chips SAR son GeoTIFF **float32 en decibeles**, con valores negativos (VV
  va de −30 a +3 dB). Un visor los interpreta como 0..255 y muestra negro.
* Las etiquetas son GeoTIFF **int8 con tres valores**: `1` agua, `0` no agua y
  `-1` sin dato. Un visor muestra casi negro, y el `-1` se confunde con el `0`
  —que es justo la distinción que importa—.

## Archivos de este paso

| archivo | qué hace |
|---|---|
| `previsualizar.py` | convierte los chips a PNG y arma un `indice.html` navegable |

## Comandos

```bash
python pregunta2/paso6_figuras/previsualizar.py
python pregunta2/paso6_figuras/previsualizar.py --que lima
python pregunta2/paso6_figuras/previsualizar.py --que test,bolivia,lima,valid,train
```

Después se abre `pregunta2/paso6_figuras/vistas/indice.html` con doble clic. No
hace falta servidor.

## Lo que se hizo

### El SAR se estira por percentiles, no por mínimo y máximo

Se recorta a los percentiles 2 y 98 **de cada banda**. Con mínimo y máximo, un
solo píxel de −40 dB aplastaría todo el resto del rango y la imagen saldría
uniformemente gris.

### La tercera banda es VV − VH, y no es decorativa

El PNG usa R = VV, G = VH, B = VV − VH. Esa resta es lo que separa agua de
tierra: el agua calma devuelve poco en las dos polarizaciones, pero la **relación**
VV/VH (que en decibeles es una resta) cambia según el mecanismo de dispersión.
Darle un canal propio hace visible a simple vista lo que el modelo usa.

### Tres vistas por chip

| vista | para qué |
|---|---|
| `_sar.png` | la entrada como la ve el modelo |
| `_etiqueta.png` | azul agua, gris oscuro no agua, **gris claro sin dato** |
| `_super.png` | las dos mezcladas, que es como de verdad se revisa un chip |

El `indice.html` ordena cada grupo **por porcentaje de agua**, así los casos
interesantes quedan arriba en vez de perdidos en el medio.

## Lo que produce

| carpeta | contenido |
|---|---|
| `vistas/test/` | 90 chips × 3 vistas = 270 PNG |
| `vistas/bolivia/` | 15 × 3 = 45 PNG |
| `vistas/lima/` | 58 etiquetas (sin SAR todavía) |
| `vistas/indice.html` | la página para recorrerlos |

## Un caso que ya sirve para el entregable 5

En `test/Sri-Lanka_534068` la etiqueta es casi toda agua pero el SAR se ve
**brillante y moteado**, no oscuro. Eso es vegetación inundada: el rebote doble
entre la lámina de agua y los troncos devuelve mucha más energía que el agua
abierta. Es uno de los tres modos de fallo característicos que pide el entregable
5, y está ahí para mirarlo.
