"""Rytability handwriting-fixing demo server.

Serves the web app (../web) and one JSON endpoint:

POST /api/rewrite
  {
    "lines": [{"strokes": [[[x, y], ...], ...],        # the ink to read
               "prime":   [[[x, y], ...], ...]}],     # (tidied) ink to copy the style of
    "style": "mine" | 0..12,     # "mine" = the user's own handwriting
    "bias": 1.5,                 # neatness
    "fix_spelling": true
  }
-> {"lines": [{"written", "meant", "text", "corrections", "strokes", "style_used", "timing"}]}

Returned strokes are normalised: left edge x=0, baseline y=0, core (x-)height 1,
y pointing down. The browser scales them onto the canvas.

Everything runs locally. Nothing leaves the machine except the one-time model
downloads.
"""

import argparse
import re
import json
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ink import body_metrics, render_strokes  # noqa: E402
from recognize import build_reader, cer, word_corrections  # noqa: E402
from synth import MAX_CHARS, HandwritingSynth, sanitize, user_strokes_to_prime  # noqa: E402
from weights import STYLES_DIR, ensure_weights  # noqa: E402

WEB_DIR = os.path.join(HERE, "..", "web")
FALLBACK_STYLE = 9
WORK_LOCK = threading.Lock()

STATE = {"synth": None, "reader": None, "reader_desc": None, "reader_error": None, "loading": True}


def load_models(reader_kind):
    try:
        STATE["synth"] = HandwritingSynth(ensure_weights(), STYLES_DIR)
        print("[models] handwriting synthesis ready")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        STATE["reader_error"] = f"synthesis model failed: {e}"
    if reader_kind != "none":
        try:
            print(f"[models] loading reader '{reader_kind}' (first run downloads ~4 GB) ...")
            t = time.time()
            STATE["reader"], STATE["reader_desc"] = build_reader(reader_kind)
            print(f"[models] reader ready: {STATE['reader_desc']} ({time.time() - t:.0f}s)")
            # warm up so the first real request is fast
            from PIL import Image
            STATE["reader"].read(Image.new("RGB", (64, 32), "white"))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            STATE["reader_error"] = f"{type(e).__name__}: {e}"
    STATE["loading"] = False
    print("[models] all loaded - go write something!")


def wrap_text(text, limit=MAX_CHARS - 5):
    """The model writes at most 75 chars per sequence; split long text on spaces."""
    words, chunks, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > limit:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def to_strokes(coords):
    """coords (y up, eos) -> list of [N,2] strokes with y down."""
    c = coords.copy()
    c[:, 1] = -c[:, 1]
    return [s[:, :2] for s in np.split(c, np.where(c[:, 2] == 1)[0] + 1) if len(s)]


def normalise_output(coords):
    """coords (y up) -> list of strokes, y down, baseline 0, core height 1, left edge 0."""
    strokes = to_strokes(coords)
    allp = np.vstack(strokes)
    base, core = body_metrics(allp[:, 1])
    x0 = allp[:, 0].min()
    return [((s - [x0, base]) / core) for s in strokes]


def rewrite(req):
    synth, reader = STATE["synth"], STATE["reader"]
    if synth is None:
        raise RuntimeError(STATE["reader_error"] or "models still loading")
    style = req.get("style", "mine")
    bias = float(req.get("bias", 1.5))
    fix = bool(req.get("fix_spelling", True))
    n_cand = int(req.get("candidates", 3))
    verify = bool(req.get("verify", True))
    out = []
    for line in req["lines"]:
        strokes = [np.asarray(s, float) for s in line["strokes"] if len(s)]
        t0 = time.time()
        if "text" in line:                   # caller already knows the text (tests / typing)
            written = meant = line["text"]
        else:
            if reader is None:
                raise RuntimeError(STATE["reader_error"] or "handwriting reader still loading")
            written, meant = reader.read(render_strokes(strokes))
        t_read = time.time() - t0
        target = meant if fix else written
        target = sanitize(target)
        written_s = sanitize(written)
        if not re.search(r"[A-Za-z0-9]", target):     # a doodle, a dash, a dot: leave it alone
            out.append({"written": written, "meant": meant, "text": "", "corrections": [], "strokes": []})
            continue

        chunks = wrap_text(target)
        n = len(chunks)
        safe_prime = synth.style_prime(FALLBACK_STYLE if style == "mine" else int(style))
        mine_prime = None
        if style == "mine" and written_s:
            prime_ink = [np.asarray(s, float) for s in line.get("prime", line["strokes"]) if len(s)]
            mine_prime = user_strokes_to_prime(prime_ink, written_s[:MAX_CHARS])

        t1 = time.time()
        if mine_prime is None:
            coords = synth.write(chunks, bias=bias, primes=[safe_prime] * n, n_candidates=n_cand)
            style_used = FALLBACK_STYLE if style == "mine" else style
        else:
            # One batch: the user's own style AND a clean built-in style as a safety net.
            coords_all, info = synth.write(chunks + chunks, bias=bias, primes=[mine_prime] * n + [safe_prime] * n,
                                           n_candidates=n_cand, return_info=True)
            coords, style_used = coords_all[:n], "mine"
            for i in range(n):
                # Copying the user's style only works if their ink is legible and the
                # network could line it up with its transcript; otherwise it faithfully
                # copies the mess. Check the attention, then have the reader proofread,
                # and fall back to the clean built-in style when it reads worse.
                ok = info[i]["prime_aligned"] and info[i]["score"] < 6
                if ok and verify and reader is not None:
                    e_mine = cer(reader.read_literal(render_strokes(to_strokes(coords[i]))), chunks[i])
                    if e_mine > 0.05:
                        e_safe = cer(reader.read_literal(render_strokes(to_strokes(coords_all[n + i]))), chunks[i])
                        ok = e_mine <= e_safe
                        print(f"[verify] {chunks[i]!r}: mine cer={e_mine:.2f} safe cer={e_safe:.2f}")
                if not ok:
                    coords[i], style_used = coords_all[n + i], FALLBACK_STYLE
        t_synth = time.time() - t1

        # stitch chunks into one long line; the browser wraps it to the page width
        strokes_out, x_off = [], 0.0
        for c in coords:
            norm = normalise_output(c)
            w = max(s[:, 0].max() for s in norm)
            strokes_out += [(s + [x_off, 0]).round(4).tolist() for s in norm]
            x_off += w + 0.9
        out.append({
            "written": written, "meant": meant, "text": target,
            "corrections": word_corrections(written, target) if fix else [],
            "strokes": strokes_out, "style_used": style_used,
            "timing": {"read": round(t_read, 2), "synth": round(t_synth, 2)},
        })
        print(f"[rewrite] read {t_read:.1f}s synth {t_synth:.1f}s  {written!r} -> {target!r}")
    return {"lines": out}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB_DIR, **kw)

    def log_message(self, fmt, *args):  # keep the console readable
        if args and "/api/" in str(args[0]):
            return
        super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            return self._json(200, {
                "loading": STATE["loading"],
                "synth": STATE["synth"] is not None,
                "reader": STATE["reader_desc"],
                "error": STATE["reader_error"],
            })
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/rewrite"):
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            with WORK_LOCK:     # one at a time: the models share the CPU/GPU
                result = rewrite(req)
            return self._json(200, result)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reader", default=os.environ.get("HWFIX_READER", "vlm"), choices=["vlm", "apple", "none"])
    ap.add_argument("--lan", action="store_true", help="listen on the network so an iPad can connect")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    threading.Thread(target=load_models, args=(args.reader,), daemon=True).start()
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    httpd = ThreadingHTTPServer((host, args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n  Rytability handwriting demo: {url}")
    if args.lan and lan_ip():
        print(f"  On your iPad (same Wi-Fi):   http://{lan_ip()}:{args.port}")
    print()
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
