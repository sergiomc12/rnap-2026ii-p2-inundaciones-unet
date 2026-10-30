# Paso 1 — Los datos, sus estadísticas y la pérdida que exigen

Este paso resuelve tres cosas que después usa todo el resto: **traer**
Sen1Floods11, **medir** cómo está distribuido, y **leerlo** de una forma que no
mienta.

## Archivos de este paso

| archivo | qué hace |
|---|---|
| `descargar.py` | baja Sen1Floods11 del bucket oficial, reanudable |
| `descargar_emsn038.sh` | reanuda la descarga del geodatabase de Copernicus EMS (~2.2 GB a ~180 KB/s) |
| `estadisticas.py` | mide medias, desvíos, percentiles y prevalencia sobre el split de entrenamiento |
| `sen1floods11.py` | el `Dataset`, el normalizador, la pérdida enmascarada y las métricas |

## Comandos

```bash
# Solo SAR: 7.5 GB. Es lo que necesitan los entregables 1 a 5.
python pregunta2/paso1_datos/descargar.py

# El bono (entregable 6) necesita SOLO S2Hand: 0.95 GB, no los 24.9 de todo
# el optico. Ver abajo por que.
python pregunta2/paso1_datos/descargar.py --solo-lo-pedido --s2-dorado

# Todo el optico, si alguna vez hiciera falta el debil tambien (+24.9 GB)
python pregunta2/paso1_datos/descargar.py --con-optico

# Mide la normalización. Escribe estadisticas.json, que lee sen1floods11.py.
python pregunta2/paso1_datos/estadisticas.py

# El geodatabase de Lima (solo si se va a hacer el entregable 4 con Perú)
bash pregunta2/paso1_datos/descargar_emsn038.sh
```

## Lo que se hizo

### Por qué se bajan 7.5 GB y no 32.4

El dataset completo pesa 32.4 GB, pero **el 77% de eso es óptico Sentinel-2**
(`S2Weak` solo son 23.9 GB) y el óptico aparece una sola vez en el enunciado, en
el entregable 6, que es un bono. Bajando únicamente lo que necesitan los
entregables 1 a 5 son 7.5 GB. Medido contra el bucket:

| carpeta | archivos | GB |
|---|---:|---:|
| `HandLabeled/S1Hand` | 446 | 0.68 |
| `HandLabeled/LabelHand` | 446 | 0.004 |
| `HandLabeled/S1OtsuLabelHand` | 446 | 0.002 |
| `HandLabeled/JRCWaterHand` | 446 | 0.001 |
| `WeaklyLabeled/S1Weak` | 4 384 | 6.82 |
| `WeaklyLabeled/S1OtsuLabelWeak` | 4 384 | 0.02 |

### Y por qué el bono solo necesita 0.95 GB, no 24.9

`--con-optico` baja las tres carpetas ópticas, pero **23.9 de sus 24.9 GB son
`S2Weak`**: óptico sobre los chips de etiqueta *débil*, que no emparejan con
ninguna máscara anotada a mano. Para medir si fusionar Sentinel-2 mejora algo
hace falta comparar contra la misma verdad que usan los entregables 1 a 5, y esa
verdad son los **446 chips dorados**:

| carpeta | archivos | GB | ¿sirve para el bono? |
|---|---:|---:|---|
| **`HandLabeled/S2Hand`** | **446** | **0.95** | **sí** — empareja uno a uno con `S1Hand` y `LabelHand` |
| `WeaklyLabeled/S2Weak` | 4 384 | 23.93 | no — su etiqueta es la débil, no la dorada |
| `WeaklyLabeled/S2IndexLabelWeak` | 4 385 | 0.03 | no — es una máscara derivada de índices ópticos, no una verdad |

De ahí `--solo-lo-pedido --s2-dorado`: baja exactamente los 446 chips ópticos
que se pueden evaluar contra la máscara dorada, y nada más.

Dos carpetas entran aunque el enunciado no las pida. `JRCWaterHand` marca el agua
que ya estaba **antes** de la inundación: sin eso, un modelo que aprenda a
segmentar "agua" se lleva el río entero como si fuera inundación, y la discusión
de fallos del entregable 5 queda sin una de sus piezas. `S1OtsuLabelHand` son las
etiquetas por umbralización **sobre los chips dorados**, o sea la línea base
clásica contra la que se compara una U-Net; pesa 2 MB y es lo que sostiene todo
el entregable 4.

### Las estadísticas no se suponen, se miden

Hay una tentación fuerte de copiar las constantes de ImageNet porque están en
todos los cuadernos. Acá no tienen ningún sentido: la entrada no son tres canales
de color entre 0 y 1, son **dos canales de retrodispersión en decibeles**, que
son negativos y se mueven en un rango de decenas.

Medido sobre el **split de entrenamiento oficial** (nunca sobre validación ni
prueba: calcular la normalización con datos de prueba es fuga, chica pero fuga):

| banda | media | desvío | p0.1 | p99.9 | mín. crudo |
|---|---:|---:|---:|---:|---:|
| VV | −10.393 dB | 4.039 | −29.3 | +3.1 | −65.45 |
| VH | −17.241 dB | 4.754 | −47.0 | −5.1 | −79.16 |

El recorte se toma en percentiles y no en el mínimo: ese −79 dB es esencialmente
`log(0)` —píxeles donde la retrodispersión dio cero— y un solo valor así estira
la escala y aplasta todo lo demás.

### Las etiquetas tienen tres valores, no dos

| valor | significado | % de píxeles |
|---:|---|---:|
| `1` | agua | 8.25% |
| `0` | no agua | 78.50% |
| `-1` | **no se pudo etiquetar** | **13.26%** |

Los `-1` son el 13.26% del total y en algunos chips pasan del 60%. Tratarlos como
"no agua" metería supervisión falsa en uno de cada siete píxeles. Es el mismo
problema que resolvimos en la Pregunta 1 con las etiquetas inciertas de CheXpert,
y la solución es la misma: una **máscara que entra en el divisor de la pérdida**.

Entre lo observado, el agua es el **9.51%**. Con esa prevalencia la exactitud por
píxel es inútil —predecir "nunca hay agua" ya da 90.5%— y por eso el enunciado
pide IoU, y por eso la pérdida combina BCE con Dice: la BCE sola tiende a la
clase mayoritaria cuando el desbalance es así de fuerte.

### Tres detalles del lector que evitan errores silenciosos

**Los no-finitos se mandan al piso, no se propagan.** Un `-inf` de `log(0)` que
entra a la red envenena todo el gradiente con NaN y la corrida "termina" sin
aprender nada.

**La aumentación es solo volteos y giros de 90 grados.** Un recorte aleatorio
rompería la alineación con la grilla georreferenciada, y una rotación que no sea
múltiplo de 90 interpolaría la máscara e inventaría píxeles de agua en los
bordes. Con 252 chips la aumentación hace falta, pero no a costa de ensuciar la
etiqueta.

**Un lote sin un solo píxel observado devuelve cero *con gradiente*.** Devolver
un `torch.tensor(0.)` suelto corta el grafo y `backward` falla; se devuelve
`(logits * 0).sum()`, que es cero pero sigue conectado. Verificado: el gradiente
existe y no tiene NaN.

### Dos IoU, no una

`metricas()` devuelve la IoU **agregada** (se suman los conteos de todos los
chips y se divide una sola vez) y `iou_por_chip()` la **promedio por chip**, que
es la que reportan los autores de Sen1Floods11. No son lo mismo: la agregada le
da más peso a los chips con más agua, la promedio por chip pesa todos igual. Un
chip sin un solo píxel de agua ni predicho ni real tiene unión cero y la IoU no
está definida ahí: se lo **excluye** del promedio en vez de contarlo como 0
(castigaría un acierto) o como 1 (premiaría no hacer nada).

## Lo que produce

| archivo | contenido |
|---|---|
| `estadisticas.json` | medias, desvíos, percentiles, conteo de etiquetas y tamaño de cada partición |
| `data/sen1floods11/` | 10 556 archivos, 7.6 GB |

## Advertencia

`chips_debiles()` acepta `excluir_regiones` y **en la práctica no es opcional**:
el conjunto débil incluye **224 chips de Bolivia** y 534 de Colombia. Bolivia es
la región que el entregable 4 retiene para medir generalización, así que
agregarla al entrenamiento por venir "del otro conjunto" sería una fuga
silenciosa: el modelo habría visto la región sobre la que después se lo evalúa, y
la brecha medida no significaría nada.
