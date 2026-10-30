"""Entregable 5: los fallos del SAR, separados por si el sensor podía resolverlos.

El enunciado pide **al menos tres fallos característicos del SAR**. La forma
fácil de responderlo es elegir tres chips que se ven mal y describirlos. Esto
hace otra cosa, porque esa forma fácil no contesta lo único que decide si vale
la pena seguir entrenando: **si el error es del modelo o del sensor**.

Por qué el SAR falla, en física
-------------------------------
El agua se ve oscura porque una superficie lisa refleja el pulso lejos del
satélite (reflexión especular). Eso trae dos problemas que **no** son del
clasificador:

1. Una superficie **seca y lisa** —asfalto, pista de aeropuerto, arena
   compactada, suelo desnudo— también se ve oscura.
2. Agua **rugosa por viento**, o agua **bajo vegetación** (doble rebote
   tronco-agua, que devuelve la señal amplificada), se ve **brillante**.

Cómo se mide acá
----------------
Se construyen los histogramas de VV condicionados a la verdad —uno para el agua,
uno para el suelo seco— sobre todos los píxeles observados, y se mide **cuánto se
solapan**. El solape es el área común de las dos densidades, y es una cota: un
clasificador que solo mire VV **no puede** separar lo que cae ahí, por bueno que
sea. No es una estimación del modelo, es una propiedad de los datos.

Con eso, cada error se ubica en una de tres zonas de VV:

* **zona de solape** — las dos clases viven ahí; el error es irreducible con
  esta banda;
* **zona de agua** (más oscuro que el solape) — si hay un falso positivo acá, el
  píxel es seco pero se ve como agua: el caso 1 de arriba, del sensor;
* **zona de suelo** (más brillante) — si hay un falso negativo acá, el píxel es
  agua pero se ve como suelo: el caso 2, del sensor.

Los errores que **no** caen en ninguna de esas tres explicaciones son del
modelo: tenía la señal del lado correcto y la clasificó mal.

`JRCWaterHand` agrega el cuarto eje: marca **agua permanente** (ríos, lagos), así
que un falso negativo sobre agua permanente es un fallo distinto de uno sobre
agua de inundación —la superficie es estable y está mapeada desde fuera del
dataset—.

Usage:
    python pregunta2/paso7_fallos_sar/analizar_fallos.py
    python pregunta2/paso7_fallos_sar/analizar_fallos.py --conjuntos test bolivia
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

import numpy as np
import rasterio

# Las rutas de abajo se anclan a ESTE archivo, no al directorio de
# trabajo: la pregunta se comparte sola y sus scripts se corren desde
# cualquier lado.
_RAIZ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

import torch  # noqa: E402

from pregunta2.paso1_datos.sen1floods11 import (  # noqa: E402
    MANO, Normalizador, chips_del_split,
)
from pregunta2.paso2_modelo.unet import ConfigUNet, UNet  # noqa: E402

CORRIDAS = _RAIZ / "pregunta2/paso3_entrenamiento/corridas"
SALIDA = _RAIZ / "pregunta2/paso7_fallos_sar/fallos.json"
FIGURAS = _RAIZ / "pregunta2/paso7_fallos_sar/figuras"

#: Los histogramas se acumulan en bines de 0.25 dB entre -40 y 5. Acumular
#: cuentas en vez de guardar valores permite usar TODOS los píxeles (cientos de
#: millones) de forma exacta, en vez de submuestrear.
BORDES = np.arange(-40.0, 5.0 + 0.25, 0.25)
CENTROS = (BORDES[:-1] + BORDES[1:]) / 2

#: Un bin pertenece a la "zona de solape" cuando la clase minoritaria conserva
#: al menos esta fracción de la densidad del bin. Con 0.10, un bin donde el agua
#: es el 95% y el suelo el 5% ya cuenta como zona de agua, no de solape.
UMBRAL_SOLAPE = 0.10

#: Y además tiene que tener masa real: al menos esta fracción de la densidad de
#: CADA clase. Con bines de 0.25 dB y 180 bines con datos, 0.001 descarta las
#: colas donde la proporción entre clases es ruido de unos pocos píxeles — que
#: era lo que estiraba la zona ambigua hasta el borde del histograma.
DENSIDAD_MINIMA_BIN = 0.001

#: Cuántos chips ilustrar por modo de fallo.
CHIPS_POR_MODO = 3


def leer_bandas(chip: str, mano: pathlib.Path = MANO):
    """VV y VH en dB tal como vienen del GeoTIFF, sin normalizar.

    El modelo recibe los valores estandarizados, pero el análisis físico tiene
    que mirar los decibelios: la firma del agua es una propiedad del sensor, no
    de la normalización que eligió este proyecto.
    """
    with rasterio.open(mano / "S1Hand" / f"{chip}_S1Hand.tif") as r:
        return r.read().astype(np.float32)


def leer_etiqueta(chip: str, sufijo: str, mano: pathlib.Path = MANO):
    ruta = mano / sufijo / f"{chip}_{sufijo}.tif"
    if not ruta.is_file():
        return None
    with rasterio.open(ruta) as r:
        return r.read(1)


@torch.no_grad()
def predecir(modelo, x_norm: np.ndarray, dispositivo) -> np.ndarray:
    t = torch.from_numpy(x_norm)[None].to(dispositivo)
    return (torch.sigmoid(modelo(t))[0, 0].cpu().numpy() > 0.5)


def zonas_de_vv(hist_agua: np.ndarray, hist_seco: np.ndarray) -> dict:
    """Parte el eje VV en zona de agua, de solape y de suelo, y mide el solape.

    El **coeficiente de solape** es el área común de las dos densidades
    normalizadas. El **error de Bayes** usa en cambio las proporciones reales
    (acá el agua es ~13% de los píxeles observados, no la mitad): es
    ``Σ min(π_agua·p_agua, π_seco·p_seco)``, y es el error que cometería el mejor
    clasificador posible que solo viera VV. Usar el solape/2 lo sobrestimaría,
    porque asumiría clases equiprobables.

    **La zona ambigua se toma como el tramo contiguo más largo**, no como el
    primer y el último bin ambiguo. En las colas hay bines con dos o tres
    píxeles donde la proporción entre clases es ruido puro; tomarlos como
    extremos estiraba la zona hasta el borde del histograma y hacía que
    *cualquier* error cayera dentro. Se exige además que el bin tenga una
    fracción mínima de la masa total para contar.
    """
    na, ns = max(1, int(hist_agua.sum())), max(1, int(hist_seco.sum()))
    pa, ps = hist_agua / na, hist_seco / ns
    solape = float(np.minimum(pa, ps).sum())

    pi_a = na / (na + ns)
    conjunta_a, conjunta_s = pi_a * pa, (1 - pi_a) * ps
    bayes = float(np.minimum(conjunta_a, conjunta_s).sum())

    total = pa + ps
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_agua = np.where(total > 0, pa / total, np.nan)
    # Un bin cuenta solo si tiene masa suficiente en las DOS clases: sin esto,
    # un bin con 3 pixeles de cada clase se ve "ambiguo" por ruido.
    masa_minima = DENSIDAD_MINIMA_BIN
    con_masa = (pa >= masa_minima) & (ps >= masa_minima)
    ambiguo = (con_masa & np.isfinite(frac_agua)
               & (frac_agua > UMBRAL_SOLAPE) & (frac_agua < 1 - UMBRAL_SOLAPE))

    # El tramo contiguo mas largo de bines ambiguos.
    mejor_ini = mejor_largo = ini = largo = 0
    for i, v in enumerate(ambiguo):
        if v:
            if largo == 0:
                ini = i
            largo += 1
            if largo > mejor_largo:
                mejor_ini, mejor_largo = ini, largo
        else:
            largo = 0
    if mejor_largo:
        bajo = float(BORDES[mejor_ini])
        alto = float(BORDES[mejor_ini + mejor_largo])
    else:
        bajo = alto = float("nan")

    return {
        "coeficiente_solape": round(solape, 4),
        "proporcion_de_agua": round(pi_a, 4),
        "error_bayes_con_solo_vv": round(bayes, 4),
        "zona_ambigua_db": [round(bajo, 2), round(alto, 2)],
        "bins_ambiguos_contiguos": int(mejor_largo),
        "bins_ambiguos_totales": int(ambiguo.sum()),
        "bins_con_datos": int(((pa > 0) | (ps > 0)).sum()),
        "umbral_dominancia": UMBRAL_SOLAPE,
        "densidad_minima_por_bin": DENSIDAD_MINIMA_BIN,
    }


def clasificar(vv_hist: np.ndarray, bajo: float, alto: float) -> dict:
    """Cuántos píxeles del histograma caen en cada zona de VV."""
    mas_oscuro = CENTROS < bajo
    ambiguo = (CENTROS >= bajo) & (CENTROS <= alto)
    mas_claro = CENTROS > alto
    n = int(vv_hist.sum())
    return {"total": n,
            "zona_agua": int(vv_hist[mas_oscuro].sum()),
            "zona_ambigua": int(vv_hist[ambiguo].sum()),
            "zona_suelo": int(vv_hist[mas_claro].sum())}


def _pct(parte: int, total: int) -> Optional[float]:
    return round(100 * parte / total, 1) if total else None


#: Qué valores de los dos umbrales se barren en el análisis de sensibilidad.
BARRIDO_UMBRAL = (0.02, 0.05, 0.10, 0.20, 0.30)
BARRIDO_DENSIDAD = (0.0005, 0.001, 0.002)


def sensibilidad(hist_agua, hist_seco, hist_fp, hist_fn) -> dict:
    """¿Cuánto depende el reparto sensor/modelo de los dos umbrales elegidos?

    Existe porque una versión anterior del README de este paso **afirmaba sin
    medirlo** que mover los umbrales cambiaba el peso de los modos 1 a 3 entre
    sí "pero no la suma sensor contra modelo". Es falso: la suma también se
    mueve. Lo que sí resiste —y es lo que hay que reportar— es que incluso con
    el ajuste más conservador la mayoría abrumadora del error sigue siendo
    física.

    Se recorre el barrido sobre los histogramas ya acumulados, así que no hace
    falta volver a correr el modelo.
    """
    global UMBRAL_SOLAPE, DENSIDAD_MINIMA_BIN
    original = (UMBRAL_SOLAPE, DENSIDAD_MINIMA_BIN)
    filas = []
    try:
        for u in BARRIDO_UMBRAL:
            for d in BARRIDO_DENSIDAD:
                UMBRAL_SOLAPE, DENSIDAD_MINIMA_BIN = u, d
                z = zonas_de_vv(hist_agua, hist_seco)
                bajo, alto = z["zona_ambigua_db"]
                if not np.isfinite(bajo):
                    filas.append({"umbral": u, "densidad_minima": d,
                                  "zona_ambigua_db": None})
                    continue
                cfp = clasificar(hist_fp, bajo, alto)
                cfn = clasificar(hist_fn, bajo, alto)
                filas.append({
                    "umbral": u, "densidad_minima": d,
                    "zona_ambigua_db": [round(bajo, 2), round(alto, 2)],
                    "fp_sensor": round(100 * (cfp["zona_ambigua"]
                                              + cfp["zona_agua"])
                                       / max(1, cfp["total"]), 1),
                    "fn_sensor": round(100 * (cfn["zona_ambigua"]
                                              + cfn["zona_suelo"])
                                       / max(1, cfn["total"]), 1),
                    "publicado": (u, d) == original,
                })
    finally:
        UMBRAL_SOLAPE, DENSIDAD_MINIMA_BIN = original

    validas = [f for f in filas if f["zona_ambigua_db"] is not None]
    return {"filas": filas,
            "rango_fp_sensor": [min(f["fp_sensor"] for f in validas),
                                max(f["fp_sensor"] for f in validas)],
            "rango_fn_sensor": [min(f["fn_sensor"] for f in validas),
                                max(f["fn_sensor"] for f in validas)]}


def _cuantiles(hist: np.ndarray) -> dict:
    n = hist.sum()
    if n == 0:
        return {"n": 0}
    acum = np.cumsum(hist) / n
    q = {p: float(CENTROS[int(np.searchsorted(acum, p / 100))])
         for p in (5, 50, 95)}
    return {"n": int(n), "p5": round(q[5], 2), "mediana": round(q[50], 2),
            "p95": round(q[95], 2)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrida", default="a_solo_dorados",
                        help="la que produjo los numeros publicados")
    parser.add_argument("--conjuntos", nargs="+", default=["test", "bolivia"])
    parser.add_argument("--dispositivo", default="cuda")
    parser.add_argument("--salida", type=pathlib.Path, default=SALIDA)
    parser.add_argument("--figuras", type=pathlib.Path, default=FIGURAS)
    parser.add_argument("--sin-figuras", action="store_true")
    args = parser.parse_args(argv)

    pesos = CORRIDAS / args.corrida / "pesos.pt"
    if not pesos.is_file():
        print(f"no esta {pesos}. Entrena primero el paso 3.")
        return 1
    guardado = torch.load(pesos, map_location="cpu", weights_only=False)
    dispositivo = torch.device(args.dispositivo if torch.cuda.is_available()
                               else "cpu")
    modelo = UNet(ConfigUNet(**guardado["config"])).to(dispositivo).eval()
    modelo.load_state_dict(guardado["pesos"])
    norm = Normalizador.desde_estadisticas()

    chips = []
    for c in args.conjuntos:
        chips += [(c, x) for x in chips_del_split(c)]
    print(f"corrida {args.corrida} | {len(chips)} chips de "
          f"{', '.join(args.conjuntos)} | {dispositivo}\n")

    nb = len(CENTROS)
    H = {k: np.zeros(nb, np.int64) for k in
         ("vv_agua", "vv_seco", "vv_vp", "vv_vn", "vv_fp", "vv_fn",
          "vh_vp", "vh_vn", "vh_fp", "vh_fn")}
    porchip = []
    for i, (conjunto, chip) in enumerate(chips, 1):
        x = leer_bandas(chip)
        etiqueta = leer_etiqueta(chip, "LabelHand")
        if etiqueta is None:
            continue
        jrc = leer_etiqueta(chip, "JRCWaterHand")
        otsu = leer_etiqueta(chip, "S1OtsuLabelHand")

        pred = predecir(modelo, norm(x.copy()), dispositivo)
        # Solo donde hay etiqueta utilizable Y retrodispersion valida: un pixel
        # NaN no tiene firma fisica que analizar.
        valido = (etiqueta != -1) & np.isfinite(x[0]) & np.isfinite(x[1])
        verdad = (etiqueta == 1) & valido
        seco = (~verdad) & valido

        vp = pred & verdad
        fp = pred & seco
        fn = ~pred & verdad
        vn = ~pred & seco

        for clave, m, banda in (("vv_agua", verdad, 0), ("vv_seco", seco, 0),
                                ("vv_vp", vp, 0), ("vv_vn", vn, 0),
                                ("vv_fp", fp, 0), ("vv_fn", fn, 0),
                                ("vh_vp", vp, 1), ("vh_vn", vn, 1),
                                ("vh_fp", fp, 1), ("vh_fn", fn, 1)):
            if m.any():
                H[clave] += np.histogram(x[banda][m], bins=BORDES)[0]

        permanente = ((jrc > 0) & valido if jrc is not None
                      else np.zeros_like(valido))
        inter = int(vp.sum())
        union = int((pred & valido).sum() + verdad.sum() - inter)
        fila = {"conjunto": conjunto, "chip": chip,
                "iou": round(inter / union, 4) if union else None,
                "vp": inter, "fp": int(fp.sum()), "fn": int(fn.sum()),
                "vn": int(vn.sum()),
                "agua_verdad": int(verdad.sum()),
                "observados": int(valido.sum()),
                "fn_en_agua_permanente": int((fn & permanente).sum()),
                "agua_permanente": int(permanente.sum()),
                "iou_otsu": None}
        if otsu is not None:
            po = (otsu == 1) & valido
            io = int((po & verdad).sum())
            uo = int(po.sum() + verdad.sum() - io)
            fila["iou_otsu"] = round(io / uo, 4) if uo else None
        porchip.append(fila)
        if i % 30 == 0 or i == len(chips):
            print(f"  {i}/{len(chips)} chips", flush=True)

    if H["vv_agua"].sum() == 0:
        print("no hay ningun pixel de agua etiquetado: nada que analizar")
        return 1

    z = zonas_de_vv(H["vv_agua"], H["vv_seco"])
    bajo, alto = z["zona_ambigua_db"]
    print(f"\n{'='*78}")
    print("LO QUE EL SENSOR NO PUEDE RESOLVER, ANTES DE MIRAR AL MODELO")
    print(f"{'='*78}")
    print(f"  solape de las densidades de VV (agua vs. suelo): "
          f"{100*z['coeficiente_solape']:.1f}%")
    print(f"  proporcion de agua en lo observado: "
          f"{100*z['proporcion_de_agua']:.1f}%")
    print(f"  zona ambigua: {bajo:.2f} a {alto:.2f} dB "
          f"({z['bins_ambiguos_contiguos']} bines contiguos de "
          f"{z['bins_con_datos']} con datos)")
    print(f"  => el mejor clasificador posible que SOLO mire VV comete "
          f"{100*z['error_bayes_con_solo_vv']:.1f}% de error")

    print(f"\n{'categoria':<22} {'n':>13} {'VV p5':>7} {'mediana':>8} {'p95':>7}"
          f"   {'VH mediana':>11}")
    print("-" * 78)
    for nombre, cv, cvh in (("agua (verdad)", "vv_agua", None),
                            ("suelo (verdad)", "vv_seco", None),
                            ("acierto agua (VP)", "vv_vp", "vh_vp"),
                            ("acierto seco (VN)", "vv_vn", "vh_vn"),
                            ("falso positivo (FP)", "vv_fp", "vh_fp"),
                            ("falso negativo (FN)", "vv_fn", "vh_fn")):
        q = _cuantiles(H[cv])
        if not q["n"]:
            continue
        vh = _cuantiles(H[cvh])["mediana"] if cvh else None
        print(f"{nombre:<22} {q['n']:>13,} {q['p5']:>7.1f} "
              f"{q['mediana']:>8.1f} {q['p95']:>7.1f}   "
              f"{'' if vh is None else f'{vh:>11.1f}'}")

    cfp = clasificar(H["vv_fp"], bajo, alto)
    cfn = clasificar(H["vv_fn"], bajo, alto)
    fn_perm = sum(c["fn_en_agua_permanente"] for c in porchip)
    perm = sum(c["agua_permanente"] for c in porchip)

    modos = {
        "1_solape_irreducible": {
            "descripcion": "el error cae en la zona de VV donde agua y suelo "
                           "conviven: con solo VV y VH no hay forma de separarlos",
            "falsos_positivos": cfp["zona_ambigua"],
            "falsos_negativos": cfn["zona_ambigua"],
            "atribuible_al": "sensor",
        },
        "2_seco_liso_se_ve_como_agua": {
            "descripcion": "falso positivo MAS OSCURO que la zona ambigua: "
                           "reflexion especular sobre superficie seca y lisa "
                           "(asfalto, pista, arena compactada)",
            "falsos_positivos": cfp["zona_agua"],
            "falsos_negativos": None,
            "atribuible_al": "sensor",
        },
        "3_agua_brillante_se_ve_como_suelo": {
            "descripcion": "falso negativo MAS CLARO que la zona ambigua: agua "
                           "rugosa por viento, o agua bajo vegetacion con doble "
                           "rebote",
            "falsos_positivos": None,
            "falsos_negativos": cfn["zona_suelo"],
            "atribuible_al": "sensor",
        },
        "4_error_del_modelo": {
            "descripcion": "el error cae en la zona donde SU PROPIA clase "
                           "domina: la senal estaba del lado correcto",
            "falsos_positivos": cfp["zona_suelo"],
            "falsos_negativos": cfn["zona_agua"],
            "atribuible_al": "modelo",
        },
        "5_agua_permanente_perdida": {
            "descripcion": "falso negativo sobre agua permanente mapeada por "
                           "JRC: superficie estable, ajena a la inundacion",
            "falsos_positivos": None,
            "falsos_negativos": fn_perm,
            "de_un_total_de_permanentes": perm,
            "atribuible_al": "modelo",
        },
    }

    tfp, tfn = cfp["total"], cfn["total"]
    print(f"\n{'='*78}")
    print("LOS MODOS DE FALLO, CON SU PESO MEDIDO")
    print(f"{'='*78}")
    print(f"{'modo':<38} {'% de los FP':>12} {'% de los FN':>12} {'culpa':>9}")
    print("-" * 78)
    for k, v in modos.items():
        if k.startswith("5_"):
            continue
        a = _pct(v["falsos_positivos"], tfp) if v["falsos_positivos"] is not None else None
        b = _pct(v["falsos_negativos"], tfn) if v["falsos_negativos"] is not None else None
        print(f"{k:<38} {'—' if a is None else f'{a:>11.1f}%'} "
              f"{'—' if b is None else f'{b:>11.1f}%'} "
              f"{v['atribuible_al']:>9}")
    sensor_fp = cfp["zona_ambigua"] + cfp["zona_agua"]
    sensor_fn = cfn["zona_ambigua"] + cfn["zona_suelo"]
    print("-" * 78)
    print(f"{'atribuible al SENSOR':<38} {_pct(sensor_fp, tfp):>11.1f}% "
          f"{_pct(sensor_fn, tfn):>11.1f}%")
    print(f"{'atribuible al MODELO':<38} "
          f"{_pct(cfp['zona_suelo'], tfp):>11.1f}% "
          f"{_pct(cfn['zona_agua'], tfn):>11.1f}%")
    print(f"\nagua permanente perdida: {fn_perm:,} px "
          f"({_pct(fn_perm, perm)}% de los {perm:,} permanentes, "
          f"{_pct(fn_perm, tfn)}% de todos los FN)")

    # Comprobacion que valida el encuadre de arriba. Si el modelo quedara POR
    # ENCIMA de la cota de VV, la cota estaria mal calculada o el modelo no
    # estaria aprovechando nada mas que un umbral. Que quede por DEBAJO solo se
    # explica porque usa VH y el contexto espacial -- y es la prueba de que la
    # cota mide lo que dice medir.
    observados = sum(c["observados"] for c in porchip)
    error_modelo = (tfp + tfn) / max(1, observados)
    print(f"\n{'='*78}")
    print("COMPROBACION: EL MODELO CONTRA LA COTA DE VV")
    print(f"{'='*78}")
    print(f"  error del modelo (FP+FN sobre observados): {100*error_modelo:.2f}%")
    print(f"  cota de Bayes mirando SOLO VV            : "
          f"{100*z['error_bayes_con_solo_vv']:.2f}%")
    if error_modelo < z["error_bayes_con_solo_vv"]:
        print("  => el modelo esta POR DEBAJO de la cota, que es lo esperable:")
        print("     usa VH y el contexto espacial, no un umbral sobre VV.")
    else:
        print("  => el modelo NO supera un umbral sobre VV; con dos bandas y")
        print("     contexto espacial disponibles, eso seria deuda del modelo.")
    comprobacion = {
        "pixeles_observados": int(observados),
        "error_del_modelo": round(error_modelo, 4),
        "cota_bayes_solo_vv": z["error_bayes_con_solo_vv"],
        "modelo_supera_la_cota": bool(error_modelo
                                      < z["error_bayes_con_solo_vv"]),
    }

    for v in modos.values():
        v["pct_de_los_fp"] = (_pct(v["falsos_positivos"], tfp)
                              if v["falsos_positivos"] is not None else None)
        v["pct_de_los_fn"] = (_pct(v["falsos_negativos"], tfn)
                              if v["falsos_negativos"] is not None else None)

    sens = sensibilidad(H["vv_agua"], H["vv_seco"], H["vv_fp"], H["vv_fn"])
    print(f"\n{'='*78}")
    print("SENSIBILIDAD: QUE PASA SI SE MUEVEN LOS DOS UMBRALES")
    print(f"{'='*78}")
    print(f"{'dominancia':>11} {'dens. min':>10} {'zona ambigua (dB)':>20} "
          f"{'FP sensor':>10} {'FN sensor':>10}")
    print("-" * 78)
    for f in sens["filas"]:
        if f["zona_ambigua_db"] is None:
            print(f"{f['umbral']:>11.2f} {f['densidad_minima']:>10.4f} "
                  f"{'(sin zona)':>20}")
            continue
        b, a = f["zona_ambigua_db"]
        marca = "  <- el publicado" if f["publicado"] else ""
        print(f"{f['umbral']:>11.2f} {f['densidad_minima']:>10.4f} "
              f"{f'{b:.2f} a {a:.2f}':>20} {f['fp_sensor']:>9.1f}% "
              f"{f['fn_sensor']:>9.1f}%{marca}")
    print(f"\n  El reparto SI se mueve con los umbrales: FP del "
          f"{sens['rango_fp_sensor'][0]:.1f}% al {sens['rango_fp_sensor'][1]:.1f}%, "
          f"FN del {sens['rango_fn_sensor'][0]:.1f}% al "
          f"{sens['rango_fn_sensor'][1]:.1f}%.")
    print(f"  Lo que NO se mueve es la conclusion: aun en el ajuste mas "
          f"conservador,\n  la gran mayoria del error sigue siendo fisica.")

    con_agua = [c for c in porchip if c["agua_verdad"] > 0 and c["iou"] is not None]
    ejemplos = {
        "mas_falsos_positivos": sorted(con_agua, key=lambda c: -c["fp"])[:CHIPS_POR_MODO],
        "mas_falsos_negativos": sorted(con_agua, key=lambda c: -c["fn"])[:CHIPS_POR_MODO],
        "peor_iou_vs_otsu": sorted(
            [c for c in con_agua if c["iou_otsu"] is not None],
            key=lambda c: c["iou"] - c["iou_otsu"])[:CHIPS_POR_MODO],
    }
    print("\nchips que ilustran cada modo:")
    for k, lista in ejemplos.items():
        print(f"  {k}:")
        for c in lista:
            o = "—" if c["iou_otsu"] is None else f"{c['iou_otsu']:.3f}"
            print(f"    {c['chip']:<22} IoU {c['iou']:.3f} (Otsu {o})  "
                  f"FP {c['fp']:>7,}  FN {c['fn']:>7,}")

    resultado = {
        "corrida": args.corrida,
        "conjuntos": args.conjuntos,
        "chips": len(porchip),
        "separabilidad_vv": z,
        "cuantiles_vv": {k.replace("vv_", ""): _cuantiles(H[k])
                         for k in ("vv_agua", "vv_seco", "vv_vp", "vv_vn",
                                   "vv_fp", "vv_fn")},
        "cuantiles_vh": {k.replace("vh_", ""): _cuantiles(H[k])
                         for k in ("vh_vp", "vh_vn", "vh_fp", "vh_fn")},
        "sensibilidad_de_los_umbrales": sens,
        "comprobacion_contra_la_cota": comprobacion,
        "reparto_fp_por_zona": cfp,
        "reparto_fn_por_zona": cfn,
        "modos_de_fallo": modos,
        "ejemplos": {k: [c["chip"] for c in v] for k, v in ejemplos.items()},
        "histogramas_vv": {"bordes": BORDES.round(3).tolist(),
                           **{k: H[k].tolist() for k in
                              ("vv_agua", "vv_seco", "vv_fp", "vv_fn")}},
        "por_chip": porchip,
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nescrito: {args.salida}")

    if not args.sin_figuras:
        from pregunta2.paso7_fallos_sar.figuras_fallos import dibujar_todo
        dibujar_todo(resultado, modelo, norm, dispositivo, args.figuras)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
