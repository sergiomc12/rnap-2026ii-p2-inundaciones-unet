#!/bin/sh
# Reanuda la descarga del geodatabase EMSN038 (~2.2 GB a ~180 KB/s) hasta
# terminarla. Reanudable: cada corrida pide solo lo que falta.
u="https://mapping.emergency.copernicus.eu/download/EMSN038/EMSN038_Geodatabase.zip"
d="data/lima/EMSN038_Geodatabase.zip"
for i in 1 2 3 4 5 6 7 8; do
  curl -sL -C - --retry 3 --retry-delay 5 --speed-time 120 --speed-limit 1000 -o "$d" "$u"
  if python -c "import zipfile,sys; zipfile.ZipFile('$d'); sys.exit(0)" 2>/dev/null; then
    echo "ZIP COMPLETO Y VALIDO tras $i intento(s): $(stat -c%s "$d") bytes"; exit 0
  fi
  echo "  intento $i: $(stat -c%s "$d" 2>/dev/null) bytes, sigue incompleto"
done
echo "no termino en 8 intentos; volve a correr el script, es reanudable"; exit 1
