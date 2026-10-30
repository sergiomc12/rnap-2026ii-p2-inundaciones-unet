# Paso 3 — Entrenamiento, evaluación en distribución y ablación

Cubre los **entregables 2 y 3**: entrena la U-Net del paso 2, la evalúa con las
métricas que pide el enunciado, y corre la ablación sobre los dos factores
exigidos.

## Archivos de este paso

| archivo | qué hace |
|---|---|
| `entrenar.py` | entrena una configuración, evalúa y compara contra Otsu |
| `ablacion.py` | corre las 7 configuraciones de la ablación y arma la tabla |

## Comandos

```bash
# Una corrida (la mejor configuración)
python pregunta2/paso3_entrenamiento/entrenar.py

# Variantes
python pregunta2/paso3_entrenamiento/entrenar.py --variante angosta --etiqueta angosta
python pregunta2/paso3_entrenamiento/entrenar.py --peso-dice 0.0     # solo BCE
python pregunta2/paso3_entrenamiento/entrenar.py --debiles 1000      # + débiles
python pregunta2/paso3_entrenamiento/entrenar.py --sin-aumentar

# La ablación completa (~1 h)
python pregunta2/paso3_entrenamiento/ablacion.py --listar
python pregunta2/paso3_entrenamiento/ablacion.py
python pregunta2/paso3_entrenamiento/ablacion.py --solo a_solo_dorados,b_bce
```

## Lo que se hizo

### Cuatro decisiones del entrenamiento

**Se elige la época por IoU de validación, no por pérdida.** La pérdida mezcla
BCE y Dice y mejora aunque la IoU no se mueva; la métrica del enunciado es la
IoU, así que esa decide qué pesos se guardan.

**Los conteos de confusión se acumulan y la IoU se calcula al final.** Promediar
la IoU chip por chip le daría el mismo peso a un chip con un río entero que a uno
con treinta píxeles de agua, donde la IoU es casi ruido. Se reportan las dos.

**Se compara contra la umbralización de Otsu, no contra nada.** Decir "la U-Net
saca IoU 0.68" no significa nada suelto; lo que importa es si le gana al método
que no necesita entrenamiento. El dataset trae `S1OtsuLabelHand` justamente para
eso, y se evalúa sobre **los mismos píxeles observados** que la U-Net —si Otsu se
midiera sobre todos, la comparación no sería entre pares—.

**Bolivia se mira una sola vez, al final.** Es la región retenida: usarla para
elegir hiperparámetros la convertiría en otra validación y dejaría de medir
generalización.

### El resultado del entregable 3

Mejor configuración: `base`, 31.0 M parámetros, 60 épocas (mejor en la 43),
lote 8, lr 1e-3, `peso_dice` 0.5, 10.5 min.

| conjunto | IoU agregada | IoU por chip | F1 | precisión | recall |
|---|---:|---:|---:|---:|---:|
| validación (89) | 0.6521 | 0.4095 | 0.7894 | 0.8019 | 0.7773 |
| **prueba (90)** | **0.6808** | **0.4178** | **0.8101** | 0.8111 | 0.8091 |
| Bolivia (15) | 0.6096 | 0.3758 | 0.7574 | 0.7849 | 0.7319 |

**Matriz de confusión sobre prueba** (20 517 367 píxeles observados):

| | predicho agua | predicho no agua |
|---|---:|---:|
| **real agua** | 2 076 313 | 489 788 |
| **real no agua** | 483 649 | 17 467 617 |

* **Omisión** (agua real que se perdió): **19.09%**
* **Comisión** (agua predicha que no era): **18.89%**

Los dos errores están casi balanceados, que es lo que uno quiere de un umbral en
0.5 — el modelo no está sesgado a sobre-detectar ni a sub-detectar. Para respuesta
a emergencias convendría mover el umbral hacia menos omisión: perder agua real es
más caro que una falsa alarma.

### La ablación del entregable 2

El enunciado exige comparar **al menos dos decisiones**. Se corren tres factores:

| corrida | factor | chips | épocas | IoU prueba | IoU Bolivia |
|---|---|---:|---:|---:|---:|
| `a_solo_dorados` | supervisión | 252 | 60 | **0.6808** | 0.6096 |
| `a_debiles_1000` | supervisión | 1 252 | 12 | 0.5987 | 0.6678 |
| `a_debiles_todos` | supervisión | 4 412 | 4 | 0.5871 | 0.6672 |
| `b_bce` | pérdida | 252 | 60 | 0.6692 | 0.6343 |
| `b_dice` | pérdida | 252 | 60 | 0.6749 | 0.6043 |
| `c_somera` | profundidad | 252 | 60 | 0.6540 | **0.6761** |
| `c_profunda` | profundidad | 252 | 60 | 0.6711 | 0.6526 |
| **Otsu** | — | — | — | 0.5458 | **0.6973** |

**El hallazgo del factor (a)**: agregar etiquetas débiles **empeora dentro de
distribución (0.681 → 0.587) y mejora fuera (0.610 → 0.667)**. Tiene explicación
directa: las débiles salen de una umbralización de Otsu, así que entrenar con
ellas empuja al modelo hacia el comportamiento del método clásico —peor donde ya
conoce el terreno, más robusto en una región nueva—.

**El patrón general**: la correlación de rangos entre las dos columnas es
**−0.786**. Casi todo lo que mejora dentro de distribución empeora fuera, y
**ninguna configuración le gana a Otsu en Bolivia**.

### Dos cuidados de la ablación que no son decorativos

**La región retenida se excluye también del conjunto débil.** Los 4 384 chips
débiles incluyen 224 de Bolivia. Sumarlos "porque son del otro conjunto" le
mostraría al modelo la región sobre la que después se mide la generalización.

**Presupuesto de pasos, no de épocas.** Una época sobre 4 412 chips no es
comparable con una sobre 252: serían 17 veces más actualizaciones. Las épocas de
cada corrida se fijan para que la cantidad de chips vistos quede del mismo orden,
así la comparación es entre *qué datos* y no entre *cuánto entrenó*.

## Lo que produce

| archivo | contenido |
|---|---|
| `corridas/<nombre>/pesos.pt` | pesos + variante + config |
| `corridas/<nombre>/resultados.json` | métricas de los 3 conjuntos, Otsu, historia por época |
| `ablacion.json` | la tabla comparativa de las 7 corridas |
