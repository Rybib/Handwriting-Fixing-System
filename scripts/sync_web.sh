#!/bin/bash
# Copy the page from the Mac demo into the app, so both run the same web/.
# usage: scripts/sync_web.sh [path/to/HandwritingMagic mac repo]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MAC="${1:-$HERE/../HandwritingMagic}"
DEST="$HERE/HandwritingMagic/Web.bundle"
rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$MAC/web/." "$DEST/"
find "$DEST" -name .DS_Store -delete
echo "copied $MAC/web -> $DEST"
