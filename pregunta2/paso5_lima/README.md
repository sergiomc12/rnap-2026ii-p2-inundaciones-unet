# Paso 5 — La región de prueba peruana (entregable 4)

El enunciado nombra Lima explícitamente y pide **construir la región de prueba**
combinando dos fuentes independientes: imágenes Sentinel-1 GRD de marzo de 2017
sobre Lima metropolitana, y la delineación oficial de la activación Copernicus
EMS **EMSN038**. Bolivia es el **respaldo**, no el camino principal.

Este paso construye la región completa —grilla, máscara de referencia e imagen
Sentinel-1—, la evalúa, y **mide por qué el resultado no sirve como cifra de
generalización**. Ese último punto es el hallazgo principal del paso.

## Archivos de este paso

| archivo | qué hace |
|---|---|
| `construir_region.py` | rasteriza la delineación de EMSN038 a la grilla de Sen1Floods11 |
| `descargar_sentinel1.py` | encola el RTC en ASF HyP3, lo baja y lo recorta a los 58 chips |
| `evaluar_lima.py` | corre la U-Net ya entrenada sobre Lima y mide la brecha |
| `diagnosticar_etiqueta.py` | decide si la brecha es del modelo o de la etiqueta |
| `figura_alineacion.py` | la figura de evidencia que va al informe |
| `deteccion_cambio.py` | compara el antes con el durante: ¿se ve la inundación? |

## Comandos

```bash
# Primero el geodatabase (paso 1):  bash pregunta2/paso1_datos/descargar_emsn038.sh
python pregunta2/paso5_lima/construir_region.py

# La escena SAR. Necesita TOKEN_HYP3 en .env (Earthdata Login, gratuito).
# Si el producto ya esta bajado solo vuelve a recortar; si no, encola y espera ~75 min.
python pregunta2/paso5_lima/descargar_sentinel1.py
python pregunta2/paso5_lima/descargar_sentinel1.py --escena pre --solo-lanzar

python pregunta2/paso5_lima/evaluar_lima.py
python pregunta2/paso5_lima/diagnosticar_etiqueta.py
python pregunta2/paso5_lima/figura_alineacion.py

# Detección de cambio: necesita tambien la escena pre-evento.
python pregunta2/paso5_lima/descargar_sentinel1.py --escena pre
python pregunta2/paso5_lima/deteccion_cambio.py
```

## Lo que se hizo

### De dónde sale la etiqueta

Del geodatabase de EMSN038, capa `Potential_Maximum_flood`: **13 polígonos,
13.58 km²**. Que son la delineación *observada* y no un escenario hipotético lo
confirma la capa `DamAssess_A`, que evalúa daños sobre esos mismos polígonos vía
la columna `FID_OPTICAL_FloodExtent_OBIA` —el análisis orientado a objetos sobre
las imágenes ópticas—: no se evalúa daño sobre algo que no pasó.

Se usan **los 13**, no solo los 2 que quedan fuera del cauce. Los otros 11 son el
río a altura máxima (verificado: son idénticos a la capa `Riverbank`, la
intersección da los 7.24 km² completos). Sen1Floods11 etiqueta **agua**, no
"inundación", y dejar el cauce afuera crearía un desacuerdo de criterio con el
entrenamiento que después se confundiría con un fallo de generalización.

> **La premisa de que el cauce *se ve* como agua en SAR resultó falsa.** Medido
> después (ver "Por qué el resultado no mide generalización"): `Riverbank` tiene
> mediana VV de **−6.97 dB**, la capa *más brillante* de todo el geodatabase, con
> apenas 1.4 % de píxeles bajo −15 dB. El razonamiento de incluir los 13 sigue
> siendo correcto; lo que falla es que ninguna de las dos opciones corresponde a
> agua oscura.

### Dónde se puede afirmar algo y dónde no

La capa `Footprints` trae las 6 huellas de imagen con su fecha. Solo dos son
post-evento: **WorldView-2 del 2017-03-18 y del 2017-03-21**, las que menciona el
enunciado. Las otras cuatro son de 2016 y de noviembre: referencia.

Fuera de esa unión **nadie miró**, así que no es "no inundado": es *desconocido*,
y va como `-1`, el mismo valor que Sen1Floods11 usa para lo que no se puede
etiquetar. Escribir 0 ahí inventaría supervisión negativa sobre media Lima y
haría ver al modelo mucho mejor de lo que es.

### La grilla se ancla a múltiplos exactos

El paso de píxel es `8.983152841195215e-05` grados, **copiado exacto** de los
GeoTIFF de Sen1Floods11 (10 m en EPSG:4326), para que un chip de Lima sea
intercambiable con uno del dataset. La grilla se ancla en múltiplos de ese paso
desde (0, 0) y no en la esquina del área de interés: así dos corridas sobre
recortes distintos producen chips en las mismas coordenadas.

## Resultados medidos

| | |
|---|---:|
| chips de 512×512 conservados | **58** |
| con al menos un píxel de agua | 18 |
| completamente secos | 40 |
| píxeles observados (dentro de la huella) | 89.7% |
| **agua entre los observados** | **1.01%** |
| agua en el entrenamiento de Sen1Floods11 | 9.51% |

Esa última fila es **una de las dos hipótesis que pide el entregable 4**: Lima es
casi diez veces más seca que lo que el modelo vio entrenando. Medida, no
supuesta —aunque resultó no ser la que manda; ver más abajo—.

## Límite que hay que declarar, y el enunciado lo pide

Estas etiquetas **no se generaron con el mismo criterio** que las doradas de
Sen1Floods11: vienen de imágenes ópticas de mayor resolución, en fechas que no
son exactamente la de la imagen SAR. Son supervisión débil propia, no una verdad
de terreno perfecta. El desajuste temporal es chico —la delineación es del 18 y
21 de marzo, y la escena Sentinel-1 disponible es del **17 de marzo**— pero
existe.

Al medirlo resultó ser mucho más grave que un desajuste de fechas: no es que la
etiqueta esté corrida un día, es que **marca otra cosa**. Ver "El resultado, y
por qué no mide generalización".

## La escena Sentinel-1

Se bajó de **ASF HyP3** (NASA/Alaska Satellite Facility), no de Google Earth
Engine. HyP3 produce el RTC bajo demanda con los tres parámetros que alinean el
producto con Sen1Floods11 —`radiometry=sigma0`, `scale=decibel`,
`resolution=10`— y solo pide un Earthdata Login gratuito.

| fecha | órbita | rol | estado |
|---|---|---|---|
| 2017-03-05 | path 171, descendente | pre-inundación | **bajada y usada** (detección de cambio) |
| **2017-03-17** | **path 171, descendente** | **durante** | **bajada y usada** |
| 2017-03-29 | path 171, descendente | post | no se usa |

Las tres son a las 10:38:23, o sea **geometría idéntica**: mismo ángulo de
incidencia píxel a píxel, que es lo que hace comparable un par pre/durante.

**La alineación no se calcula, se hereda.** Cada chip se reproyecta a la grilla
que *ya* tiene su etiqueta en `LabelLima/` —se lee su `transform`, su CRS y su
forma—, así que la correspondencia píxel a píxel queda garantizada por
construcción en vez de depender de que dos cálculos independientes coincidan.

**Diferencia que queda declarada:** HyP3 aplica corrección radiométrica de
terreno (aplana el relieve con un DEM) y el `COPERNICUS/S1_GRD` de GEE usado para
construir Sen1Floods11 no. En terreno plano es despreciable; en el valle del
Rímac algo pesa. Es mucho menor que procesar el GRD crudo con otra cadena, pero
no es cero.

## El resultado, y por qué no mide generalización

| conjunto | IoU | F1 | precisión | recall | chips |
|---|---:|---:|---:|---:|---:|
| Lima | **0.0028** | 0.0056 | 0.0029 | 0.0484 | 48 |
| prueba | 0.6808 | 0.8101 | 0.8111 | 0.8091 | 90 |

Brecha −0.6780, IC95 % [−0.7610, −0.5702], no pareado. Diez de los 58 chips caen
fuera de la franja de la escena y se descartan antes de medir: evaluarlos sumaría
aciertos triviales sobre píxeles que nadie observó.

Un desplome así admite dos lecturas incompatibles —o el modelo no generaliza, o
la etiqueta no marca lo que el modelo predice—. `diagnosticar_etiqueta.py` las
separa con tres mediciones en vez de elegir la cómoda:

1. **La alineación es correcta.** Se hereda de la etiqueta (arriba) y se ve en
   `etiqueta_lima.png`: el contorno sigue el cauce del Rímac. El 0.0028 no es un
   error de georreferenciación.
2. **Otsu también fracasa, y Otsu no aprende nada.** Con la misma función sobre
   las dos regiones: **0.0108** en Lima contra **0.3530** en Bolivia. Como Otsu
   recalibra su umbral dentro de cada imagen, su fallo no puede atribuirse al
   modelo —no hay modelo—.
   *(Ese 0.3530 es un Otsu ingenuo sobre VV, no el `S1OtsuLabelHand` del dataset,
   que rinde 0.6973 en Bolivia por traer preprocesamiento propio. Se usa el
   ingenuo porque es el único que existe para Lima, y se aplica igual a las dos
   regiones para no mezclar regiones con implementaciones.)*
3. **La etiqueta está anticorrelacionada con la señal.** Dentro de lo etiquetado
   agua, el 8.7 % de los píxeles está bajo −15 dB; **fuera, el 16.3 %**. Los
   píxeles oscuros son casi el doble de frecuentes fuera del agua declarada que
   dentro.

### La causa es física, y el cauce la delata

Firma de radar medida por capa del geodatabase:

| capa | mediana VV | % bajo −15 dB | píxeles |
|---|---:|---:|---:|
| `Riverbank` (el cauce) | −6.97 dB | 1.4 % | 74 079 |
| `Potential_Maximum_flood` (la usada) | −9.16 dB | 8.7 % | 136 104 |
| `WatrcrsA` | −11.70 dB | 19.4 % | 40 172 |
| `LakeresA` | −15.11 dB | 51.0 % | 1 788 |
| Sen1Floods11, **suelo** | −8.36 dB | 2.5 % | 2 372 533 |
| Sen1Floods11, **agua** | **−16.47 dB** | **65.4 %** | 451 901 |

El cauce es lo **más brillante** de todas, no lo más oscuro. El Rímac de marzo de
2017 era un torrente de montaña turbulento y cargado de sedimento en un cañón
urbano, y una superficie rugosa retrodispersa como suelo; lo que el radar ve
negro es el agua **lisa**. La confirmación está en `LakeresA` —lagunas, agua
quieta—, la única que se acerca al agua dorada: donde sí hay agua lisa, el sensor
la ve oscura. **El sensor funciona; la inundación no era agua lisa.**

En el vocabulario de `paso7_fallos_sar/`, Lima es el **modo 3** —agua brillante
que se ve como suelo— extendido a un evento entero.

### Tampoco la rescata la detección de cambio

Quedaba una salida posible: que la inundación fuera invisible para un modelo de
*una sola fecha* pero no para uno que compare con el antes —que es como
Copernicus EMS mapea inundaciones con SAR de verdad—. Se bajó la escena del
**5 de marzo de 2017** (mismo path 171 descendente, misma hora 10:38:23, doce
días antes: **mismo ángulo de incidencia píxel a píxel**) y se midió con
`deteccion_cambio.py`.

| banda | cambio dentro de «agua» | cambio fuera | IoU máx. barriendo umbral |
|---|---:|---:|---:|
| VV | **+0.78 dB** | −0.01 dB | 0.0242 (en −4.50 dB) |
| VH | **+0.61 dB** | −0.01 dB | 0.0138 (en −4.25 dB) |

La diferencia se centra por escena antes de usarla: entre dos fechas cambian la
calibración, la humedad y la vegetación, y eso corre el histograma entero; lo que
importa es el cambio *relativo* al resto de la escena.

El umbral se elige **a favor de la hipótesis que se quiere refutar**: en vez de
fijar un corte, se barren todos y se reporta el mejor. Ese número no es alcanzable
en la práctica —elegirlo exige ver la respuesta— y justamente por eso sirve: es
una **cota superior** de lo que cualquier detector de cambio por umbral global
podría lograr. Con un umbral fijo de −3 dB, el valor que alguien elegiría sin
mirar, da 0.0196.

**El área etiquetada se aclaró, no se oscureció.** Una inundación vista por SAR
oscurece; acá sube casi un decibelio. El límite no es el modelo de una sola fecha
ni la falta de un método de cambio: es que esa agua nunca se vio como agua.

### Qué queda en pie

**No queda en pie** leer el 0.0028 como medida de generalización: la etiqueta
delimita la *llanura potencialmente inundable* —evaluada después por
`DamAssess_A` para estimar daños—, no la superficie de agua del 17 de marzo, que
es lo que el modelo predice. Comparar ambas mide el desacuerdo entre dos
definiciones. Bolivia sigue siendo el experimento de generalización válido.

**Sí queda en pie**, y es el límite que el enunciado pide documentar, que un
producto de mapeo rápido de Copernicus no sirve como verdad de terreno para SAR
aunque describa la misma inundación, la misma semana y el mismo lugar: la
pregunta que responde —¿hasta dónde pudo llegar el agua?— no es la que responde
una etiqueta de Sen1Floods11 —¿qué píxel es agua?—. La distancia entre ambas está
medida acá en **7.3 dB**.

### Lo que sigue faltando

Una delineación de agua trazada **sobre la propia escena SAR del 17 de marzo**,
que EMSN038 no provee. Sin ella no se puede saber cuánto cae el modelo en Lima
*de verdad*.

## Lo que produce

| archivo | contenido |
|---|---|
| `data/lima/region/LabelLima/*.tif` | 58 máscaras int8 (`1` agua, `0` no agua, `-1` sin dato) |
| `data/lima/region/chips.json` | las ventanas exactas, la cobertura y el límite declarado |
| `data/lima/region/S1Lima/*.tif` | 58 GeoTIFF de 2 bandas (VV, VH) en dB, en la grilla de la etiqueta |
| `evaluacion_lima.json` | IoU/F1/precisión/recall en Lima y en prueba, y la brecha con su IC |
| `diagnostico_etiqueta.json` | las tres mediciones que atribuyen la brecha a la etiqueta |
| `deteccion_cambio.json` | el cambio 5-mar → 17-mar y la cota superior por umbral |
| `data/lima/region/S1LimaPre/*.tif` | 58 GeoTIFF de la escena pre-evento, misma grilla |
| `../paso9_informe/informe/figuras/etiqueta_lima.png` | la figura de evidencia del informe |
| `../paso9_informe/informe/figuras/cambio_lima.png` | histograma del cambio y curva de la cota |
