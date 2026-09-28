"""Export the handwriting synthesiser for the app: weights + the 13 built-in styles.

The Mac demo keeps them as numpy files (models/hand_synth.npz, models/styles/).
The app reads one flat little-endian float32 file plus a JSON index instead.

usage: python scripts/export_synth.py [path/to/HandwritingMagic mac repo]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
mac = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "HandwritingMagic")
out = os.path.join(HERE, "..", "HandwritingMagic", "Resources")

w = np.load(os.path.join(mac, "models", "hand_synth.npz"))
index, blobs, offset = {}, [], 0
for name in sorted(w.files):
    a = np.ascontiguousarray(w[name], dtype="<f4")
    index[name] = {"offset": offset, "shape": list(a.shape)}
    blobs.append(a.tobytes())
    offset += a.size
with open(os.path.join(out, "hand_synth.bin"), "wb") as f:
    f.write(b"".join(blobs))

styles = []
for i in range(13):
    strokes = np.load(os.path.join(mac, "models", "styles", f"style-{i}-strokes.npy")).astype(np.float32)
    text = np.load(os.path.join(mac, "models", "styles", f"style-{i}-chars.npy")).tobytes().decode("utf-8")
    styles.append({"text": text, "offsets": [round(float(v), 5) for v in strokes.ravel()]})
with open(os.path.join(out, "hand_synth.json"), "w") as f:
    json.dump({"tensors": index, "styles": styles}, f)
print(f"exported {offset} weights and {len(styles)} styles to {os.path.normpath(out)}")
