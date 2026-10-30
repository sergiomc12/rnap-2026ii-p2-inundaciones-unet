# Paso 8 — Fusión con Sentinel-2 (bono del entregable 6)

> El bono dejó de ser un extra cuando el [paso 7](../paso7_fallos_sar/) midió que
> el **99.8% de los falsos positivos** y el **96.5% de los falsos negativos** del
> modelo SAR son físicamente irresolubles con VV y VH. El óptico no agrega "más
> datos": agrega la única dimensión donde las dos clases sí se separan.

## El resultado en tres líneas

**Agregar Sentinel-2 sube la IoU de 0.6808 a 0.8498: +0.169, significativo.**
Pero lo que carga esa mejora es el óptico, no la fusión: el óptico **solo** da
0.8511, indistinguible de la fusión. Lo único que el SAR todavía aporta aparece
**bajo nube**, y con 16 chips nublados no alcanza para afirmarlo.

## Archivos de este paso

| archivo | qué hace |
|---|---|
| [`datos_fusion.py`](datos_fusion.py) | el lector que sirve SAR, óptico o los dos sobre **los mismos** chips |
| [`entrenar_fusion.py`](entrenar_fusion.py) | entrena la misma U-Net tres veces, cambiando solo los canales de entrada |
| [`comparar_fusion.py`](comparar_fusion.py) | bootstrap **pareado sobre chips** de las tres comparaciones |
| [`medir_separacion.py`](medir_separacion.py) | cuánto separa cada banda óptica, con el mismo solape que el paso 7 usa sobre VV |

```bash
# 0.95 GB, no los 24.9 de --con-optico (ver paso1_datos/README.md)
python pregunta2/paso1_datos/descargar.py --solo-lo-pedido --s2-dorado

python pregunta2/paso8_fusion_s2/entrenar_fusion.py   # ~75 min, 3 corridas
python pregunta2/paso8_fusion_s2/comparar_fusion.py
```

## Qué se fusiona, y por qué esas bandas

`S2Hand` trae las 13 bandas L1C. Meterlas todas sería gastar capacidad en un
conjunto de 252 chips de entrenamiento, así que entran cuatro bandas más dos
índices, **6 canales**:

| canal | por qué |
|---|---|
| **verde (B3)** | el agua todavía refleja algo acá |
| **rojo (B4)** | para que la red pueda formar un índice de vegetación si le sirve |
| **NIR (B8)** | el agua **absorbe casi todo**: es la banda que separa |
| **SWIR (B11)** | absorbe aún más, y distingue agua de sombra de nube |
| **NDWI** = (verde − NIR)/(verde + NIR) | el índice de agua clásico, ya calculado |
| **MNDWI** = (verde − SWIR)/(verde + SWIR) | ídem, más robusto en zona urbana |

Los dos índices no agregan información —son funciones de bandas que ya están—
pero le ahorran a la red descubrir un cociente con 252 chips, y el cociente
cancela las diferencias de iluminación entre escenas.

### La separación óptica contra la del radar, medida

```bash
python pregunta2/paso8_fusion_s2/medir_separacion.py
```

Sobre **los 90 chips de prueba**, con el mismo coeficiente de solape que el
paso 7 usa sobre VV — por eso las dos se pueden poner en la misma tabla:

| capa | mediana agua | mediana suelo | separación | **solape** |
|---|---:|---:|---:|---:|
| rojo B4 | 0.107 | 0.083 | +0.025 | 80.0% |
| verde B3 | 0.128 | 0.107 | +0.020 | 70.9% |
| NIR B8 | 0.092 | 0.273 | −0.180 | 15.2% |
| SWIR B11 | 0.038 | 0.198 | −0.160 | 13.9% |
| NDWI | 0.165 | −0.395 | +0.560 | 11.3% |
| **MNDWI** | **0.495** | **−0.285** | **+0.780** | **9.4%** |
| *VV en dB (paso 7)* | *−17.4* | *−9.4* | | ***41.2%*** |

**MNDWI deja un 9.4% de solape donde VV deja un 41.2%: lo reduce 4.4 veces.**
Esa fila es toda la explicación del +0.169 de IoU que viene después.

Dos observaciones que la tabla deja ver y conviene no esconder:

* **El verde y el rojo casi no separan por sí solos** (70.9% y 80.0% de solape,
  peor que VV). Entran porque son los ingredientes de los índices y porque le dan
  a la red material para formar otros, no porque aporten separación propia.
* **El salto está en el infrarrojo**, exactamente donde la física lo pone: NIR y
  SWIR bajan el solape de 70-80% a 14-15% de una sola vez.

## Cómo se compara, para que la diferencia signifique algo

Las tres configuraciones corren sobre **los mismos chips, la misma partición, la
misma semilla, la misma arquitectura y el mismo presupuesto de 60 épocas**. Solo
cambian los canales de entrada. Un chip sin óptico no entra en **ninguno** de los
tres modos, ni siquiera en el de SAR: comparar sobre conjuntos distintos mediría
el conjunto. Resultó que los 446 chips dorados tienen las tres piezas, así que no
se perdió ninguno (252 / 89 / 90).

**La corrida de SAR reproduce exactamente el 0.6808 del paso 4.** No es
casualidad buscada: es el control que muestra que este pipeline y el del paso 3
miden lo mismo.

## Resultados

| modo | canales | IoU prueba | minutos |
|---|---:|---:|---:|
| SAR (VV, VH) | 2 | 0.6808 | 13.9 |
| **óptico (S2)** | 6 | **0.8511** | 29.0 |
| fusión (SAR + S2) | 8 | 0.8498 | 32.0 |

### Bootstrap pareado sobre chips, 10 000 remuestreos

| comparación | grupo | delta | IC95% | ¿significativa? |
|---|---|---:|---|---|
| **fusión vs. SAR** | todos | **+0.1690** | [+0.0910, +0.2538] | **sí** |
| | despejados (74) | **+0.1768** | [+0.0913, +0.2690] | **sí** |
| | nublados (16) | **+0.1220** | [+0.0186, +0.3405] | **sí** |
| **óptico vs. SAR** | todos | +0.1704 | [+0.0876, +0.2614] | sí |
| | despejados | +0.1832 | [+0.0937, +0.2808] | sí |
| | **nublados** | +0.0938 | **[−0.0327, +0.3346]** | **no** |
| **fusión vs. óptico** | todos | −0.0013 | [−0.0122, +0.0093] | no |
| | despejados | −0.0063 | [−0.0168, +0.0007] | no |
| | nublados | +0.0282 | [−0.0020, +0.0567] | no |

### Lo que estas tres tablas dicen, en orden

1. **El bono funciona, y en grande.** +0.169 de IoU es la mejora más grande que
   produjo cualquier cosa en esta pregunta, y confirma exactamente lo que el
   paso 7 predijo midiendo el solape del radar.

2. **La fusión no le gana al óptico solo.** −0.0013 en total, con el intervalo
   cruzando cero. Si el cielo está despejado, el SAR no aporta nada que el
   óptico no tenga.

3. **Pero bajo nube se invierte el signo, y solo la fusión aguanta.** La ganancia
   del **óptico solo** sobre el SAR deja de ser significativa con nubes
   ([−0.0327, +0.3346] cruza cero); la de la **fusión** sigue siéndolo
   ([+0.0186, +0.3405]). Y fusión contra óptico pasa de −0.0063 a **+0.0282**,
   con el intervalo rozando el cero por dos milésimas.

   **Eso es exactamente donde la física dice que el SAR tiene que aportar**, y es
   el argumento para fusionar en vez de tirar el radar. Pero con **16 chips
   nublados no se puede afirmar**: el signo y la teoría apuntan al mismo lado, los
   datos no lo cierran.

## El límite que invalida cualquier lectura optimista

**Estos chips casi no tienen nubes.** Medido sobre los 90 de prueba:

| | |
|---|---:|
| nubosidad mediana | **0.2%** |
| chips totalmente despejados | 23 de 90 |
| chips con más del 10% nublado | 16 |
| chips con más del 25% nublado | **2** |
| chips con más del 50% nublado | **0** |
| máximo observado | 42% |

El conjunto dorado de Sen1Floods11 **se eligió donde el óptico era etiquetable a
mano**, así que está sesgado hacia cielo despejado por construcción. El grupo que
este paso llama "nublado" tiene una cuarta parte del chip tapada, no una tormenta.

**Una inundación real viene con la tormenta que la causó.** Con cobertura del
80 o el 100%, el óptico no entrega nada y el SAR queda solo — un régimen que
**este experimento no contiene ni un solo chip para medir**. Así que el resultado
correcto de leer es: *la fusión domina en el rango de nubosidad que Sen1Floods11
cubre, y ese rango no incluye el caso operativo que motivó usar radar.*

Decir "fusionar resuelve el problema" sería ir mucho más allá de lo medido.

## Lo que este paso NO hace

- **No mide nubes de verdad.** `cobertura_de_nube()` usa el brillo en azul
  (B2 > 0.25), que es un criterio grueso. Sentinel-2 trae una banda de calidad
  con máscara de nube propia, pero `S2Hand` no la incluye. Por eso el campo se
  llama "probablemente nublado" y el umbral está declarado como constante.
- **No prueba fusión tardía ni atención entre sensores.** Los canales se
  concatenan a la entrada, que es la fusión más simple. Con los dos sensores
  dando resultados tan distintos según la nube, una arquitectura que aprenda
  *cuándo* confiar en cada uno es la continuación obvia — y es justamente lo que
  los 16 chips nublados no alcanzan para entrenar.
- **No toca la Pregunta 4 (Lima).** Lima necesita sus Sentinel-1 vía Earth
  Engine, que sigue pendiente de credenciales.
