"""Ink geometry helpers shared by the server: rendering and measuring."""

import numpy as np
from PIL import Image, ImageDraw


def render_strokes(strokes, target_height=96, pad=24, width_px=None, stroke_px=None):
    """Rasterise canvas strokes (list of [N,2], y down) to a white RGB image.

    Scaled so the ink is ~target_height px tall, which suits both the OCR
    models (TrOCR / Vision / VLMs) we feed it to.
    """
    pts = np.vstack([np.asarray(s, float) for s in strokes if len(s)])
    lo, hi = pts.min(0), pts.max(0)
    h = max(hi[1] - lo[1], 1.0)
    scale = target_height / h
    if width_px is not None:
        scale = min(scale, (width_px - 2 * pad) / max(hi[0] - lo[0], 1.0))
    W = int((hi[0] - lo[0]) * scale + 2 * pad)
    Hh = int(h * scale + 2 * pad)
    img = Image.new("RGB", (max(W, 32), max(Hh, 32)), "white")
    d = ImageDraw.Draw(img)
    lw = stroke_px or max(2, int(round(target_height / 22)))
    for s in strokes:
        s = (np.asarray(s, float) - lo) * scale + pad
        if len(s) == 1:
            x, y = s[0]
            d.ellipse([x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2], fill="black")
        else:
            d.line([tuple(p) for p in s], fill="black", width=lw, joint="curve")
            for x, y in (s[0], s[-1]):
                d.ellipse([x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2], fill="black")
    return img


def body_metrics(points_y_down):
    """Robust (baseline, core height) of a blob of ink, y pointing down.

    Uses the same percentiles for user ink and synthesised ink so the two can
    be matched to each other without caring what either absolute scale is.
    """
    y = np.asarray(points_y_down, float)
    lo, hi = np.percentile(y, 20), np.percentile(y, 85)
    return hi, max(hi - lo, 1e-3)
