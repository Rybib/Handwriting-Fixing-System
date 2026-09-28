#!/bin/bash
# Put the handwriting reader, Qwen3-VL-2B-Instruct (4-bit MLX, 1.7 GB), into the
# app as HandwritingMagic/Qwen3VLModel.bundle. It is too big for git, so it is
# not committed. Uses your Hugging Face cache if it's there, else downloads it.
set -euo pipefail
REPO="mlx-community/Qwen3-VL-2B-Instruct-4bit"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HERE/HandwritingMagic/Qwen3VLModel.bundle"
FILES="config.json generation_config.json model.safetensors model.safetensors.index.json
  tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json vocab.json merges.txt
  chat_template.json chat_template.jinja preprocessor_config.json"
CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}/models--${REPO//\//--}/snapshots"
mkdir -p "$DEST"
SNAP="$(ls -d "$CACHE"/*/ 2>/dev/null | head -1 || true)"
for f in $FILES; do
  if [ -n "$SNAP" ] && [ -e "$SNAP$f" ]; then
    cp -L "$SNAP$f" "$DEST/$f"
  else
    echo "downloading $f ..."
    curl -fL --progress-bar -o "$DEST/$f" "https://huggingface.co/$REPO/resolve/main/$f"
  fi
done
chmod u+w "$DEST"/*
du -sh "$DEST"
