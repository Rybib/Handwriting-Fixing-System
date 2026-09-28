#!/bin/bash
# Put the AI model into the app as HandwritingMagic/GemmaMLXModel.bundle: the
# same Gemma 3 4B (mlx-community/gemma-3-4b-it-qat-4bit, 4-bit MLX) that
# Rytability carries, WITH its vision tower, so it can look at the ink.
# About 3 GB, too big for git, so it is not committed.
#
#   text weights + tokenizer: Rytability's GemmaMLXModel.bundle (../Rytability)
#   vision tower: split out of the full model (Rytability's backup copy, your
#     Hugging Face cache, or a download) into model-vision.safetensors
set -euo pipefail
REPO="mlx-community/gemma-3-4b-it-qat-4bit"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HERE/HandwritingMagic/GemmaMLXModel.bundle"
RYT="${RYTABILITY:-$HERE/../Rytability}/Rytability/GemmaMLXModel.bundle"
BACKUP="$HERE/../Files/App Work/Legacy dependencies & Backup Models/GemmaMLX-backup/model.safetensors"
CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}/models--${REPO//\//--}/snapshots"
FILES="config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json
  added_tokens.json chat_template.json preprocessor_config.json processor_config.json"
mkdir -p "$DEST"
SNAP="$(ls -d "$CACHE"/*/ 2>/dev/null | head -1 || true)"

fetch() {   # fetch <file in the repo> <dest>
  if [ -n "$SNAP" ] && [ -e "$SNAP$1" ]; then cp -L "$SNAP$1" "$2"
  else echo "downloading $1 ..."; curl -fL --progress-bar -o "$2" "https://huggingface.co/$REPO/resolve/main/$1"; fi
}

for f in $FILES; do
  if [ -e "$RYT/$f" ]; then cp "$RYT/$f" "$DEST/$f"; else fetch "$f" "$DEST/$f"; fi
done

FULL=""
if [ -e "$BACKUP" ]; then FULL="$BACKUP"
elif [ -n "$SNAP" ] && [ -e "${SNAP}model.safetensors" ]; then FULL="${SNAP}model.safetensors"
elif [ ! -e "$RYT/model-text.safetensors" ] || [ ! -e "$DEST/model-vision.safetensors" ]; then
  FULL="$(mktemp -d)/model.safetensors"; fetch model.safetensors "$FULL"
fi

if [ -e "$RYT/model-text.safetensors" ]; then
  [ -e "$DEST/model-text.safetensors" ] || cp "$RYT/model-text.safetensors" "$DEST/"
  [ -e "$DEST/model-vision.safetensors" ] || python3 "$HERE/scripts/split_vision.py" "$FULL" "$DEST/model-vision.safetensors"
else
  # no Rytability checkout: the full model file has text and vision in one
  [ -e "$DEST/model.safetensors" ] || cp -L "$FULL" "$DEST/model.safetensors"
fi
chmod u+w "$DEST"/*
rm -rf "$HERE/HandwritingMagic/Qwen3VLModel.bundle"     # the old reader
du -sh "$DEST"
