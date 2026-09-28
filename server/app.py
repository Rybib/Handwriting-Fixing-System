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
from recognize import build_proofreader, build_reader, cer, default_vlm, word_corrections  # noqa: E402
from synth import MAX_CHARS, HandwritingSynth, sanitize, user_strokes_to_prime  # noqa: E402
from weights import STYLES_DIR, ensure_weights  # noqa: E402

WEB_DIR = os.path.join(HERE, "..", "web")
FALLBACK_STYLE = 9
WORK_LOCK = threading.Lock()

STATE = {"synth": None, "reader": None, "reader_desc": None, "reader_error": None, "loading": True,
         "phase": "Starting", "dl": None, "proof": None, "proof_fast": False}


def _dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def byte_counter(dl):
    """A tqdm class for snapshot_download that records bytes instead of drawing bars.

    huggingface_hub hides its own download bars when stderr is not a terminal, and
    run.sh pipes everything into logs/last-run.log, so a first-run download of
    several GB used to print nothing at all and looked like a hang.
    """
    from tqdm.auto import tqdm

    sink = open(os.devnull, "w")

    class ByteCounter(tqdm):
        def __init__(self, *a, **kw):
            kw.update(disable=False, file=sink)
            super().__init__(*a, **kw)

        def update(self, n=1):
            shown = super().update(n)
            if self.unit == "B":
                dl["bars"][id(self)] = self.n
            return shown

    return ByteCounter


def download_reader(model_id):
    """Fetch the reader's files up front, reporting progress to the page and the Terminal."""
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = os.path.join(HF_HUB_CACHE, "models--" + model_id.replace("/", "--"))
    dl = {"dir": cache_dir, "total": 4.3e9, "bars": {}, "last": -1, "last_t": time.time()}
    try:
        info = HfApi().model_info(model_id, files_metadata=True)
        dl["total"] = float(sum((f.size or 0) for f in info.siblings)) or dl["total"]
    except Exception as e:  # noqa: BLE001  (offline: fine if it is already cached)
        print(f"[models] could not query model size ({e})")
    STATE["dl"] = dl
    STATE["phase"] = "Downloading the handwriting reader"
    done, t0 = threading.Event(), time.time()

    def report():      # silent if the files are already cached (that takes ~1 s)
        while not done.wait(5):
            text, frac = status_phase()
            print(f"[models] {model_id}: {text} ({frac or 0:.0%})")

    threading.Thread(target=report, daemon=True).start()
    try:
        snapshot_download(model_id, tqdm_class=byte_counter(dl))
        if time.time() - t0 > 5:
            print(f"[models] download finished in {time.time() - t0:.0f}s")
    except Exception as e:  # noqa: BLE001
        print(f"[models] download problem ({e}); trying whatever is cached")
    finally:
        done.set()
        STATE["dl"] = None


def status_phase():
    """Human-readable loading phase, with download progress and stall detection."""
    dl = STATE["dl"]
    if not dl:
        return STATE["phase"], None
    # bytes from huggingface_hub's progress callbacks; the folder size is a fallback
    # for versions that only report whole files
    have = max([_dir_bytes(dl["dir"])] + list(dl["bars"].values()))
    now = time.time()
    if have != dl["last"]:
        dl["last"], dl["last_t"] = have, now
    stalled = now - dl["last_t"] > 90
    frac = min(0.999, have / dl["total"]) if dl["total"] else None
    text = f"{STATE['phase']}: {have / 1e9:.1f} of {dl['total'] / 1e9:.1f} GB"
    if stalled:
        text += " (no progress for a while - check your internet / the Terminal window)"
    return text, frac


def load_models(reader_kind):
    try:
        STATE["phase"] = "Getting the handwriting synthesiser"
        STATE["synth"] = HandwritingSynth(ensure_weights(), STYLES_DIR)
        print("[models] handwriting synthesis ready")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        STATE["reader_error"] = f"synthesis model failed: {e}"
    if reader_kind != "none":
        try:
            download_reader(default_vlm())
            STATE["phase"] = "Loading the handwriting reader into memory"
            print(f"[models] loading reader '{reader_kind}' ...")
            t = time.time()
            STATE["reader"], desc = build_reader(reader_kind)
            print(f"[models] reader loaded: {desc} ({time.time() - t:.0f}s)")
            STATE["phase"] = "Warming up the handwriting reader"
            from PIL import Image
            t = time.time()
            STATE["reader"].read(Image.new("RGB", (64, 32), "white"))
            print(f"[models] warm-up read took {time.time() - t:.1f}s")
            STATE["reader_desc"] = desc
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            STATE["reader"] = None
            STATE["reader_error"] = f"{type(e).__name__}: {e}"
        STATE["proof"], STATE["proof_fast"] = build_proofreader(STATE["reader"])
        if STATE["proof_fast"]:
            hint = getattr(STATE["reader"], "name", "") == "vlm"
            if hint:
                STATE["reader"].ocr = STATE["proof"]
            print("[models] macOS Vision proofreads the rewrites" + (" and gives the reader a second opinion" if hint else ""))
    STATE["loading"] = False
    STATE["phase"] = "Ready"
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


def pick(cands, text, proof, limit):
    """The candidate that reads back best (then the best attention score).

    Candidates come sorted by attention score, so the first one that reads back
    perfectly wins. Without a proofreader, a poor attention score (a skipped
    letter, or the pen never lifting at the end) counts as a misreading.
    """
    best = None
    for c in cands[:limit]:
        c["cer"] = cer(proof(render_strokes(to_strokes(c["coords"]))), text) if proof else float(c["score"] >= 6)
        if best is None or c["cer"] < best["cer"]:
            best = c
        if c["cer"] == 0:
            break
    return best


def rewrite(req):
    synth, reader = STATE["synth"], STATE["reader"]
    if synth is None:
        raise RuntimeError(STATE["reader_error"] or "models still loading")
    style = req.get("style", "mine")
    bias = float(req.get("bias", 1.5))
    fix = bool(req.get("fix_spelling", True))
    n_cand = int(req.get("candidates", 8))
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
        proof = STATE["proof"] if verify else None
        limit = n_cand if STATE["proof_fast"] else 2      # a VLM proofreader is slow
        if mine_prime is None:
            _, info = synth.write(chunks, bias=bias, primes=[safe_prime] * n, n_candidates=n_cand, return_info=True)
            coords = [pick(inf["candidates"], c, proof, limit)["coords"] for inf, c in zip(info, chunks)]
            style_used = FALLBACK_STYLE if style == "mine" else style
        else:
            # One batch: the user's own style AND a clean built-in style as a safety net.
            _, info = synth.write(chunks + chunks, bias=bias, primes=[mine_prime] * n + [safe_prime] * n,
                                  n_candidates=n_cand, return_info=True)
            coords, style_used = [], "mine"
            for i, chunk in enumerate(chunks):
                # Copying the user's style only works if their ink is legible and the
                # network could line it up with its transcript; otherwise it faithfully
                # copies the mess. So the versions are proofread, and the clean
                # built-in style is used unless the user's own reads at least as well.
                mine = pick(info[i]["candidates"], chunk, proof, limit) if info[i]["prime_aligned"] else None
                if mine is None or mine["cer"] > 0:
                    safe = pick(info[n + i]["candidates"], chunk, proof, limit)
                    print(f"[verify] {chunk!r}: mine cer={mine['cer'] if mine else 1:.2f} safe cer={safe['cer']:.2f}")
                    if mine is None or safe["cer"] < mine["cer"]:
                        mine, style_used = safe, FALLBACK_STYLE
                coords.append(mine["coords"])
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
            phase, progress = status_phase() if STATE["loading"] else (STATE["phase"], None)
            return self._json(200, {
                "loading": STATE["loading"],
                "phase": phase,
                "progress": progress,
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


def is_our_demo(url):
    try:
        import urllib.request
        with urllib.request.urlopen(f"{url}/api/status", timeout=2) as r:
            return "synth" in json.loads(r.read())
    except Exception:  # noqa: BLE001
        return False


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
    # run.sh pipes the output through tee; without this, print() is block-buffered
    # and logs/last-run.log stops minutes behind what the server is doing
    sys.stdout.reconfigure(line_buffering=True)

    host ="0.0.0.0" if args.lan else "127.0.0.1"
    httpd = None
    for port in range(args.port, args.port + 20):
        url = f"http://127.0.0.1:{port}"
        try:
            httpd = ThreadingHTTPServer((host, port), Handler)
            break
        except OSError:
            if is_our_demo(url):     # launcher double-clicked twice: just show the running one
                print(f"Handwriting Magic is already running at {url} - opening it.")
                if not args.no_browser:
                    webbrowser.open(url)
                return
            print(f"Port {port} is used by another program, trying {port + 1} ...")
    if httpd is None:
        print(f"Ports {args.port}-{args.port + 19} are all busy. Try:  ./run.sh --port 9000")
        sys.exit(1)
    args.port = port
    threading.Thread(target=load_models, args=(args.reader,), daemon=True).start()
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
