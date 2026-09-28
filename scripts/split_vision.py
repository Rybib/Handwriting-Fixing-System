#!/usr/bin/env python3
"""Copy the vision tower (vision_tower.* and multi_modal_projector.*) out of a
full Gemma 3 MLX safetensors file into a file of its own.

Rytability ships Gemma's text weights (model-text.safetensors) and the vision
tower separately; MLX loads every *.safetensors in the model folder, so the two
files together are the whole model. Standard library only: the tensors are
copied byte for byte.

usage: split_vision.py <full model.safetensors> <out model-vision.safetensors>
"""
import json
import struct
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    base = 8 + n
    keys = sorted(k for k in header if k.startswith(("vision_tower.", "multi_modal_projector.")))
    if not keys:
        sys.exit(f"{src} has no vision tower")
    out, blobs, off = {"__metadata__": header.get("__metadata__", {"format": "mlx"})}, [], 0
    for k in keys:
        s, e = header[k]["data_offsets"]
        f.seek(base + s)
        blobs.append(f.read(e - s))
        out[k] = {"dtype": header[k]["dtype"], "shape": header[k]["shape"], "data_offsets": [off, off + e - s]}
        off += e - s
head = json.dumps(out, separators=(",", ":")).encode()
head += b" " * (-len(head) % 8)
with open(dst, "wb") as g:
    g.write(struct.pack("<Q", len(head)))
    g.write(head)
    for b in blobs:
        g.write(b)
print(f"{len(keys)} vision tensors, {off / 1e6:.0f} MB -> {dst}")
