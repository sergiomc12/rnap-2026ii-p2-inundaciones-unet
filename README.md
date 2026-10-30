# Pregunta 2 — Mapeo de Inundaciones con Sentinel-1 (U-Net)

## El problema en una frase

Durante una inundación hay que saber **dónde está el agua** en las horas
siguientes, y las nubes que acompañan a la tormenta dejan ciego al satélite
óptico justo cuando más se lo necesita; el radar de apertura sintética atraviesa
la nube y ve de noche, así que la tarea es segmentar agua píxel a píxel sobre
imágenes SAR de Sentinel-1.

## Los seis entregables del examen

| # | qué pide | estado |
|---|---|---|
| 1 | U-Net propia en PyTorch, **sin copiar código de bibliotecas de segmentación**, con las dos bandas SAR (VV, VH) | **hecho** — [`paso2_modelo/`](paso2_modelo/) |
| 2 | Ablación de ≥2 decisiones: (a) solo dorados vs. agregar débiles, (b) función de pérdida u otra decisión arquitectónica | **hecho** — 7 corridas, [`paso3_entrenamiento/`](paso3_entrenamiento/) |
| 3 | Evaluación en distribución: IoU, F1, precisión, recall de "agua" + matriz de confusión comentada | **hecho** — [`paso3_entrenamiento/`](paso3_entrenamiento/) |
| 4 | Generalización al Perú: entrenar sin datos peruanos y evaluar sobre una región de Lima construida por el grupo | **hecho** — [`paso4_generalizacion/`](paso4_generalizacion/) y [`paso5_lima/`](paso5_lima/); la cifra válida es Bolivia, y por qué lo es está medido |
| 5 | Análisis cualitativo de ≥3 fallos característicos del SAR | **hecho** — [`paso7_fallos_sar/`](paso7_fallos_sar/) |
| 6 | Bono: fusión con bandas ópticas de Sentinel-2 | **hecho** — [`paso8_fusion_s2/`](paso8_fusion_s2/), IoU 0.6808 → **0.8498** |

## Los nueve pasos

| paso | qué hace |
|---|---|
| [`paso1_datos/`](paso1_datos/) | descarga Sen1Floods11, mide sus estadísticas, y define el lector y la pérdida enmascarada |
| [`paso2_modelo/`](paso2_modelo/) | la U-Net propia y sus variantes de ablación |
| [`paso3_entrenamiento/`](paso3_entrenamiento/) | entrena, evalúa en distribución y corre la ablación |
| [`paso4_generalizacion/`](paso4_generalizacion/) | compara contra Otsu con bootstrap pareado sobre chips |
| [`paso5_lima/`](paso5_lima/) | construye la región de prueba peruana desde Copernicus EMS |
| [`paso6_figuras/`](paso6_figuras/) | convierte los chips a PNG navegables para poder mirarlos |
| [`paso7_fallos_sar/`](paso7_fallos_sar/) | separa los fallos del **sensor** de los del **modelo**, midiendo el solape de las clases en VV |
| [`paso8_fusion_s2/`](paso8_fusion_s2/) | fusiona Sentinel-2 y mide la ganancia **separada por nubosidad** |
| [`paso9_informe/`](paso9_informe/) | el informe NeurIPS, en PDF y Word |

### El comando único

Todo se corre con uno solo, que es lo que pide el enunciado:

```bash
python pregunta2/ejecutar_todo.py
```

Encadena las **20 etapas** —desde la descarga de los 7.5 GB hasta la compilación
del informe— y **cada una se salta si su artefacto ya existe**, así que volver a
lanzarlo hace solo lo que falta. Las dos descargas son reanudables.

Necesita un entorno con las dependencias de [`requirements.txt`](requirements.txt):

```bash
python -m venv .venv
.venv\Scripts\activate          # en Linux/Mac: source .venv/bin/activate
pip install -r pregunta2/requirements.txt
python pregunta2/ejecutar_todo.py
```

Dos etapas dependen de servicios externos gratuitos: construir la región de Lima
baja la escena Sentinel-1 de **ASF HyP3** (NASA/Alaska Satellite Facility, pide
un Earthdata Login en `.env` como `TOKEN_HYP3`) y produce el RTC bajo demanda
(~75 min la primera vez). Sin ese token, `--sin-bono` y saltar los pasos de Lima
dejan corriendo el resto del circuito.

```bash
python pregunta2/ejecutar_todo.py --listar     # qué hay hecho y qué falta
python pregunta2/ejecutar_todo.py --rapido     # el circuito entero en minutos
python pregunta2/ejecutar_todo.py --sin-bono   # omitir el entregable 6
python pregunta2/ejecutar_todo.py --desde 7    # retomar desde un paso
```

`--rapido` recorta las etapas caras (pocas épocas, pocos remuestreos) para
verificar que el circuito corre de punta a punta; **los números que produce no
son los del informe**, y el propio script lo advierte al terminar.

### Y las etapas por separado, si hace falta

```bash
python pregunta2/paso1_datos/descargar.py            # 7.5 GB, reanudable
python pregunta2/paso1_datos/estadisticas.py         # mide la normalización
python pregunta2/paso3_entrenamiento/entrenar.py     # ~10 min
python pregunta2/paso4_generalizacion/comparar.py    # bootstrap pareado
python pregunta2/paso6_figuras/previsualizar.py      # vistas para revisar
python pregunta2/paso7_fallos_sar/analizar_fallos.py # los fallos del SAR
python pregunta2/paso1_datos/descargar.py --solo-lo-pedido --s2-dorado
python pregunta2/paso8_fusion_s2/entrenar_fusion.py  # 3 corridas, ~75 min
python pregunta2/paso8_fusion_s2/comparar_fusion.py  # bootstrap pareado
```

## Resultados medidos

### En distribución (entregable 3)

| conjunto | IoU agregada | IoU promedio por chip | F1 | precisión | recall |
|---|---:|---:|---:|---:|---:|
| validación (89) | 0.6521 | 0.4095 | 0.7894 | 0.8019 | 0.7773 |
| **prueba (90)** | **0.6808** | **0.4178** | **0.8101** | 0.8111 | 0.8091 |
| Bolivia (15, retenida) | 0.6096 | 0.3758 | 0.7574 | 0.7849 | 0.7319 |

Se reportan **las dos IoU** porque miden cosas distintas y el enunciado pide la
segunda: los autores de Sen1Floods11 usan "IoU promedio sobre los chips". La
agregada pesa más los chips con mucha agua; la promedio por chip pesa todos
igual, y por eso es mucho más baja —hay chips casi secos donde la IoU es
inestable—. Reportar solo la agregada dejaría la comparación con el paper a
medias.

### Los fallos del SAR (entregable 5)

| | |
|---|---:|
| solape de las densidades de VV (agua vs. suelo) | **41.2%** |
| error del mejor clasificador posible con **solo** VV | 7.81% |
| error que comete esta U-Net | **4.83%** |
| falsos positivos atribuibles al **sensor** | **99.8%** |
| falsos negativos atribuibles al **sensor** | **96.5%** |

**Casi todo el error restante es físicamente irresoluble con VV y VH**: los
falsos positivos tienen la retrodispersión del agua (mediana −14.6 dB) y los
falsos negativos la del suelo (−10.6 dB). Detalle y figuras en
[`paso7_fallos_sar/`](paso7_fallos_sar/).

### La fusión con Sentinel-2 (bono, entregable 6)

| modo | canales | IoU prueba |
|---|---:|---:|
| SAR (VV, VH) | 2 | 0.6808 |
| **óptico (S2)** | 6 | **0.8511** |
| fusión | 8 | 0.8498 |

**+0.169 de IoU, significativo** — la mejora más grande de toda la pregunta, y
exactamente lo que predijo el paso 7. Pero la fusión **no** le gana al óptico
solo (−0.0013, no significativo): lo que carga la mejora es el óptico. El SAR
solo vuelve a pesar **bajo nube**, donde la ganancia del óptico deja de ser
significativa y la de la fusión no — con 16 chips nublados eso **no alcanza para
afirmarlo**. Y ninguno de los 90 chips pasa del 42.2% de nubosidad, así que el caso
operativo real (tormenta encima) queda fuera de lo medido. Detalle en
[`paso8_fusion_s2/`](paso8_fusion_s2/).

### El resultado principal (entregable 4)

Con **bootstrap pareado sobre chips enteros** (10 000 remuestreos):

| conjunto | U-Net | Otsu (sin entrenar) | diferencia | IC 95% | ¿significativa? |
|---|---:|---:|---:|---|---|
| prueba (10 regiones) | 0.6808 | 0.5458 | **+0.135** | [+0.071, +0.209] | **sí** |
| Bolivia (retenida) | 0.6096 | 0.6973 | **−0.088** | [−0.166, −0.022] | **sí** |

**El signo se da vuelta, y las dos diferencias son significativas**: lo aprendido
le gana al método clásico dentro de la distribución y se lo pierde fuera.

Un matiz que no hay que pasar por alto: la caída de la propia U-Net entre prueba
y Bolivia es +0.071 con IC [−0.089, +0.309], o sea **no se distingue del ruido**.
La lectura correcta no es "la U-Net se derrumba en Bolivia" sino **"Otsu mejora
mucho ahí (0.546 → 0.697) y la U-Net no lo acompaña"**.

### La ablación (entregable 2)

Siete corridas, misma partición y misma semilla, cambiando una cosa por vez:

| corrida | factor | IoU prueba | IoU Bolivia |
|---|---|---:|---:|
| solo dorados (252) | supervisión | **0.6808** | 0.6096 |
| Dice sola | pérdida | 0.6749 | 0.6043 |
| profunda (5 niveles) | arquitectura | 0.6711 | 0.6526 |
| BCE sola | pérdida | 0.6692 | 0.6343 |
| somera (3 niveles) | arquitectura | 0.6540 | **0.6761** |
| + 1 000 débiles | supervisión | 0.5987 | 0.6678 |
| + todos los débiles | supervisión | 0.5871 | 0.6672 |
| **Otsu** (sin entrenar) | — | 0.5458 | **0.6973** |

La correlación de rangos entre las dos columnas es **−0.786**: casi todo lo que
mejora dentro de distribución empeora fuera. Y **ninguna de las siete
configuraciones le gana a Otsu en Bolivia** — la mejor se queda 0.021 por debajo.

Sobre el factor (a) que pide el enunciado: agregar etiquetas débiles **empeora
dentro de distribución (0.681 → 0.587) y mejora fuera (0.610 → 0.667)**. Tiene
una explicación directa: las débiles salen de una umbralización de Otsu, así que
entrenar con ellas empuja al modelo hacia el comportamiento del método clásico.

## La regla metodológica que sostiene todo el trabajo

**Hay que revisar los resultados, no las suposiciones.** Cada decisión de este
directorio se midió antes de adoptarla, y varias se descartaron por eso. Lo que
se midió y salió en contra está escrito igual que lo que salió a favor.

## Lo que ya sabemos del dataset (verificado, no supuesto)

* **Formato**: 512×512 píxeles, 2 bandas (VV, VH) en decibeles, float32,
  EPSG:4326, ~10 m de resolución.
* **Estadísticas del split de entrenamiento**: VV −10.39 ± 4.04 dB,
  VH −17.24 ± 4.75 dB. Normalizar con las constantes de ImageNet sería un
  disparate: la entrada no son colores entre 0 y 1.
* **Etiquetas**: `1` agua, `0` no agua, `-1` sin dato. Los `-1` son el
  **13.26%** de los píxeles y en algunos chips pasan del 60%: **la pérdida los
  tiene que enmascarar**.
* **Prevalencia**: el agua es el **9.51%** de los píxeles observados. Con ese
  desbalance la exactitud por píxel es inútil —predecir "nunca hay agua" da
  90.5%— y por eso la métrica es IoU.
* **Particiones oficiales**: train 252 / valid 89 / test 90 sobre 10 regiones, y
  **Bolivia (15) retenida por completo** como región de generalización.
* **Procedencia**: los chips `S1Hand` salen de `COPERNICUS/S1_GRD` de Google
  Earth Engine, que ya aplica órbitas precisas, quita de ruido, calibración
  radiométrica y corrección de terreno, y entrega decibeles.

## Dónde están los datos

| qué | ruta |
|---|---|
| SAR dorado (VV+VH en dB) | `data/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/` |
| Etiquetas doradas | `.../HandLabeled/LabelHand/` |
| Línea base de Otsu | `.../HandLabeled/S1OtsuLabelHand/` |
| Agua permanente (JRC) | `.../HandLabeled/JRCWaterHand/` |
| SAR y etiquetas débiles | `.../WeaklyLabeled/` |
| Particiones oficiales | `data/sen1floods11/v1.1/splits/flood_handlabeled/*.csv` |
| Región de Lima | `data/lima/region/` |
| Geodatabase EMSN038 | `data/lima/EMSN038_UTM18S_v0.gdb` |

## Lo que falta

Ningún entregable. Lo que falta es una **verdad de terreno de agua** para la
región peruana, y es una limitación de los datos disponibles, no una tarea
pendiente de este trabajo.

### El entregable 4, cerrado

La grilla y las etiquetas de Lima estaban listas desde el principio (58 chips
rasterizados de EMSN038); faltaba la imagen Sentinel-1. Se resolvió **sin**
Google Earth Engine, por **ASF HyP3** (NASA/Alaska Satellite Facility), que
produce el RTC bajo demanda con los tres parámetros que alinean el producto con
Sen1Floods11: `radiometry=sigma0`, `scale=decibel`, `resolution=10`. Solo pide
un Earthdata Login gratuito.

La escena es `S1B_IW_GRDH_1SDV_20170317T103823…` —17-mar-2017, path 171
descendente, VV+VH—, un día antes de la delineación óptica del 18.

**Diferencia que queda declarada:** HyP3 aplica corrección radiométrica de
terreno (aplana el relieve con un DEM) y el `COPERNICUS/S1_GRD` de GEE usado
para construir Sen1Floods11 no. En terreno plano es despreciable; en el valle
del Rímac algo pesa. Es mucho menor que procesar el GRD crudo con otra cadena,
pero no es cero.

### El resultado de Lima, y por qué no es una cifra de generalización

La evaluación da **IoU 0.0028** contra 0.6808 en prueba. Ese número **no** mide
generalización, y `paso5_lima/diagnosticar_etiqueta.py` lo demuestra con tres
mediciones en vez de suponerlo:

1. **La alineación es correcta** —la imagen se reproyecta a la grilla que ya
   tiene su etiqueta, y la figura `etiqueta_lima.png` lo muestra—, así que el
   0.0028 no es un error de georreferenciación.
2. **Otsu también fracasa**, con la misma función sobre las dos regiones: 0.0108
   en Lima contra 0.3530 en Bolivia. Otsu no aprende nada, así que su fallo no
   puede atribuirse al modelo.
3. **La etiqueta está anticorrelacionada con la señal**: dentro de lo etiquetado
   agua el 8.7 % de los píxeles está bajo −15 dB; fuera, el 16.3 %.

La causa es física y la delata el cauce. Firma de radar medida por capa:

| capa | mediana VV | % bajo −15 dB | píxeles |
|---|---:|---:|---:|
| `Riverbank` (el cauce) | −6.97 dB | 1.4 % | 74 079 |
| `Potential_Maximum_flood` (la usada) | −9.16 dB | 8.7 % | 136 104 |
| `WatrcrsA` | −11.70 dB | 19.4 % | 40 172 |
| `LakeresA` | −15.11 dB | 51.0 % | 1 788 |
| Sen1Floods11, **suelo** | −8.36 dB | 2.5 % | 2 372 533 |
| Sen1Floods11, **agua** | **−16.47 dB** | **65.4 %** | 451 901 |

El cauce es lo **más brillante** de todas, no lo más oscuro: el Rímac de marzo
de 2017 era un torrente turbulento y cargado de sedimento, y una superficie
rugosa retrodispersa como suelo. La confirmación está en `LakeresA` —lagunas,
agua quieta—, la única que se acerca al agua dorada: donde sí hay agua lisa, el
sensor la ve oscura. El sensor funciona; la inundación no era agua lisa.

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

**La conclusión que sí queda**, y que el enunciado pide documentar: un producto
de mapeo rápido de Copernicus no sirve como verdad de terreno para SAR aunque
describa la misma inundación, la misma semana y el mismo lugar. La pregunta que
responde —¿hasta dónde pudo llegar el agua?— no es la que responde una etiqueta
de Sen1Floods11 —¿qué píxel es agua?—. Bolivia sigue siendo el experimento de
generalización válido del informe.

## Pesos preentrenados

Los pesos de la U-Net entrenada estan publicados como *release* (superan el limite de 100 MB/archivo de GitHub):

**https://github.com/sergiomc12/rnap-2026ii-p2-inundaciones-unet/releases/tag/v1.0-pesos**

Cada asset `<corrida>_pesos.pt` va en `pregunta2/paso3_entrenamiento/corridas/<corrida>/pesos.pt` (y los de fusion/optico/sar en `pregunta2/paso8_fusion_s2/corridas/<corrida>/pesos.pt`). No hacen falta para leer el informe ni correr el notebook; solo para reusar un modelo sin reentrenar. El dataset Sen1Floods11 se descarga con los scripts del repo.
