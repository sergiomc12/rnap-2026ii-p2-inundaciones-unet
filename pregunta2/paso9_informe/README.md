# Paso 9 — El informe (requisito transversal)

> El enunciado lo pide para **cada** pregunta: *"Cada pregunta requiere un
> informe escrito (6–10 páginas) y un repositorio de código reproducible... Los
> informes deben incluir: motivación, trabajos relacionados (mínimo 5
> referencias), metodología, experimentos, resultados y una discusión crítica de
> las limitaciones."*

## Archivos de este paso

| archivo | qué hace |
|---|---|
| [`informe/informe.tex`](informe/informe.tex) | el informe, en formato NeurIPS |
| [`informe/referencias.bib`](informe/referencias.bib) | 12 referencias; el enunciado pide 5 |
| [`compilar_informe.py`](compilar_informe.py) | compila y **verifica** páginas y referencias |
| [`exportar_word.py`](exportar_word.py) | genera el `.docx` desde el mismo `.tex` |

```bash
python pregunta2/paso9_informe/compilar_informe.py     # informe.pdf
python pregunta2/paso9_informe/exportar_word.py        # informe.docx
```

Las dos etapas están en el [comando único](../ejecutar_todo.py), así que una
corrida completa deja el PDF y el Word al día sin tener que acordarse.

## Lo que verifica el compilador, en vez de confiar en la vista previa

| | |
|---|---|
| páginas de **cuerpo** | entre 6 y 10, midiendo dónde arranca la bibliografía |
| referencias | del `.bbl` ya resuelto, **no** del `.bib` |

El límite del enunciado es sobre el cuerpo, y las referencias no cuentan. Antes
el chequeo miraba las páginas del PDF y toleraba una de más, suponiendo que la
bibliografía cabía ahí: un informe con 11 páginas de cuerpo clavadas pasaba
igual. Ahora se busca la página donde empieza la bibliografía y se mide a qué
altura está su título, así que el número es el cuerpo y no una estimación.

La distinción del `.bbl` importa: el `.bib` puede tener más entradas de las que
el texto cita, y el enunciado pide un mínimo de referencias **en el informe**, no
en el archivo de bibliografía.

## Por qué el Word no sale de pasarle el `.tex` a pandoc

Se probó y **se cuelga**: el preámbulo carga `neurips.sty`, que pandoc intenta
interpretar y no termina nunca —verificado con un `timeout` de 110 s contra un
archivo mínimo que sí convierte en un segundo—. `exportar_word.py` recorta el
cuerpo, **resuelve los `\input{}`** —que pandoc no sigue, y sin eso las tablas
desaparecerían del Word sin ningún aviso— y le antepone un preámbulo mínimo.

El título y los autores viven en el preámbulo que se descarta, así que se
extraen y se pasan como metadatos; sin eso el Word arrancaba directo en el
resumen.

## Una diferencia con las preguntas 1 y 3

Allá las tablas del informe **se generan desde los JSON** con `generar_tablas.py`,
de modo que un número que cambia en una corrida se propaga solo. Acá las tablas
están escritas en el `.tex`.

**Es una deuda, no una decisión.** Si una corrida cambia un número, hay que
actualizarlo a mano, y el compilador no lo va a notar: verifica el formato, no
los valores. Mientras tanto, cada cifra del informe está en el README del paso
que la produjo, con el JSON del que sale.

## Estado

| | |
|---|---:|
| cuerpo | **8.81 páginas** (límite 6–10) |
| páginas del PDF | 10, con las referencias aparte |
| referencias citadas | **12** (mínimo 5) |
| errores de LaTeX / citas sin resolver | 0 / 0 |
| tablas / figuras | 7 / 2 |
| Word | `informe.docx`, 5 315 palabras |
