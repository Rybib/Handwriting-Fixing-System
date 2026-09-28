"""Fetch the pretrained handwriting-synthesis checkpoint and convert it to .npz.

The checkpoint is Sean Vasquez's TF1 implementation of Graves (2013),
https://github.com/sjvasquez/handwriting-synthesis, trained on IAM-OnDB.
We do NOT need TensorFlow: the variables are stored as raw little-endian
float32 blobs inside the .data file, so we slice them out at known offsets
(pinned to an exact commit + sha256, so the offsets can never drift).
"""

import hashlib
import os
import sys
import urllib.request

import numpy as np

COMMIT = "5f58984c3bc793bd8930bad7012cd2008d4a4a66"
BASE = f"https://raw.githubusercontent.com/sjvasquez/handwriting-synthesis/{COMMIT}"
DATA_URL = f"{BASE}/checkpoints/model-17900.data-00000-of-00001"
DATA_SHA256 = "046f30989097e75968b5f37ef98f25ba096d7a6e418d25c5481a648970929889"
STYLE_IDS = list(range(13))

# name -> (byte offset, shape) inside model-17900.data-00000-of-00001
LAYOUT = {
    "attn_b": (12, (30,)),
    "attn_W": (372, (476, 30)),
    "lstm1_b": (171732, (1600,)),
    "lstm1_W": (190932, (476, 1600)),
    "lstm2_b": (9330132, (1600,)),
    "lstm2_W": (9349332, (876, 1600)),
    "lstm3_b": (26168532, (1600,)),
    "lstm3_W": (26187732, (876, 1600)),
    "gmm_b": (43006932, (121,)),
    "gmm_W": (43008384, (400, 121)),
}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "hand_synth.npz")
STYLES_DIR = os.path.join(MODEL_DIR, "styles")


def _download(url):
    print(f"  downloading {url.rsplit('/', 1)[-1]} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def ensure_weights():
    """Download + convert on first run. Returns the path to the .npz."""
    os.makedirs(STYLES_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        print("[weights] fetching pretrained handwriting-synthesis model (~43 MB, one time)")
        data = _download(DATA_URL)
        digest = hashlib.sha256(data).hexdigest()
        if digest != DATA_SHA256:
            raise RuntimeError(f"checkpoint hash mismatch: {digest}")
        arrays = {}
        for name, (offset, shape) in LAYOUT.items():
            n = int(np.prod(shape))
            arrays[name] = np.frombuffer(data, dtype="<f4", count=n, offset=offset).reshape(shape).copy()
        np.savez(WEIGHTS_PATH, **arrays)
        print(f"[weights] saved {WEIGHTS_PATH}")
    for i in STYLE_IDS:
        for kind in ("strokes", "chars"):
            path = os.path.join(STYLES_DIR, f"style-{i}-{kind}.npy")
            if not os.path.exists(path):
                blob = _download(f"{BASE}/styles/style-{i}-{kind}.npy")
                with open(path, "wb") as f:
                    f.write(blob)
    return WEIGHTS_PATH


if __name__ == "__main__":
    ensure_weights()
    sys.exit(0)
