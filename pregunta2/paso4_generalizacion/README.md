# Paso 4 — ¿La diferencia es real o son 15 chips? (entregable 4)

La corrida base dejó un resultado incómodo y por eso interesante:

| conjunto | U-Net | Otsu | diferencia |
|---|---:|---:|---:|
| prueba (10 regiones) | 0.6808 | 0.5458 | **+0.135** |
| Bolivia (retenida) | 0.6096 | 0.6973 | **−0.088** |

Leído crudo: lo aprendido le gana al método clásico dentro de la distribución y
lo pierde fuera. Es justo lo que pregunta el entregable 4, pero Bolivia son
**15 chips**, así que antes de escribir esa conclusión hay que medir si la
diferencia se distingue del ruido.

## Archivos de este paso

| archivo | qué hace |
|---|---|
| `comparar.py` | bootstrap pareado sobre chips entre la U-Net y Otsu |

## Comandos

```bash
python pregunta2/paso4_generalizacion/comparar.py
python pregunta2/paso4_generalizacion/comparar.py --corrida c_somera
python pregunta2/paso4_generalizacion/comparar.py --remuestreos 50000
```

## Lo que se hizo

### Por qué pareado y no intervalos sueltos

Los dos métodos se evalúan sobre **los mismos chips**, así que remuestrear los
chips y mirar la *diferencia* cancela el ruido del conjunto. Con intervalos
sueltos, dos intervalos anchos que se solapan no dicen nada aunque la diferencia
sea perfectamente estable. Es el mismo instrumento que en la Pregunta 1 detectó
un +0.050 real cuyos intervalos sueltos se solapaban por completo.

### Se remuestrean chips enteros, nunca píxeles

Los píxeles de un mismo chip están fuertemente correlacionados —el agua viene en
manchas, no salpicada—, así que tratarlos como independientes daría intervalos
artificialmente angostos y haría ver como significativo casi cualquier cosa.

## Resultados medidos

Con 10 000 remuestreos:

| conjunto | U-Net | Otsu | diferencia | IC 95% | ¿significativa? |
|---|---:|---:|---:|---|---|
| prueba (90 chips) | 0.6808 | 0.5458 | **+0.1350** | [+0.0705, +0.2091] | **sí** |
| Bolivia (15 chips) | 0.6096 | 0.6973 | **−0.0877** | [−0.1655, −0.0220] | **sí** |

**Las dos son significativas y el signo se da vuelta.**

### El matiz que cambia la conclusión

La caída de la **propia U-Net** entre prueba y Bolivia es +0.0713, con IC 95%
[−0.0888, +0.3085]: **no se distingue del ruido**. (Acá el bootstrap no es
pareado, porque son chips distintos y no hay correspondencia uno a uno.)

Entonces la lectura correcta **no** es "la U-Net se derrumba en Bolivia", sino:

> **Otsu mejora mucho en Bolivia (0.546 → 0.697) y la U-Net no lo acompaña.**

Bolivia es una región fácil de umbralizar —agua abierta, contraste limpio— y el
modelo aprendido no aprovecha eso. Decirlo de la otra forma sería sobreinterpretar
15 chips.

### Dos hipótesis sobre la causa (lo que pide el enunciado)

1. **Tipo de cobertura.** Las 10 regiones de entrenamiento son mayormente llanura
   de inundación rural y fluvial. Cada región tiene su propia textura de
   retrodispersión según la vegetación y la rugosidad del suelo, y el modelo
   puede estar aprendiendo esas texturas en vez de la física del agua —que es lo
   que Otsu usa y por eso transfiere—.
2. **Prevalencia.** La proporción de agua cambia entre regiones, y un modelo
   entrenado con una prevalencia calibra su umbral implícito para esa. Otsu
   recalcula su umbral **en cada imagen**, así que se adapta solo.

Las dos son verificables con más medición; ninguna está confirmada todavía.

## Lo que produce

| archivo | contenido |
|---|---|
| `comparacion.json` | deltas, intervalos, significancia y la caída no pareada |
