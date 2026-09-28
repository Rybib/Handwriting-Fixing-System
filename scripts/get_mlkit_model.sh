#!/bin/bash
# Put ML Kit's English handwriting model (13 MB) into the app, so it works
# offline from the first launch instead of downloading it then. These are the
# exact files ML Kit itself downloads, named by their SHA-1 as it stores them;
# InkReader copies them into ML Kit's model folder on first launch.
# (Google's files, so they're fetched here rather than committed.)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HERE/HandwritingMagic/MLKitEnglishModel.bundle/FileData"
mkdir -p "$DEST"
while read -r SHA URL; do
  if [ -f "$DEST/$SHA" ] && [ "$(shasum "$DEST/$SHA" | cut -d' ' -f1)" = "$SHA" ]; then continue; fi
  echo "downloading $(basename "$URL") ..."
  curl -fL --progress-bar -o "$DEST/$SHA" "$URL"
  [ "$(shasum "$DEST/$SHA" | cut -d' ' -f1)" = "$SHA" ] || { echo "checksum mismatch: $URL"; rm -f "$DEST/$SHA"; exit 1; }
done <<'LIST'
dcc0fcc19e8801e30d8c32309d762d0b5926cf12 https://dl.google.com/handwriting/models/qrnn.en_us.reco_20200318.fst_20191208.recospec.zip
27cfb1b5a7e5a4dee7ba6c07aaea9c3343b34ee4 https://dl.google.com/handwriting/models/indy_lstm.latin.6x216.tflite.20191208.zip
9f1c95b8e0604cdf72bd326b3a913102626bb854 https://dl.google.com/handwriting/models/en_us.20191208.compact.fst.zip
LIST
du -sh "$DEST"
