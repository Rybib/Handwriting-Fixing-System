"""Benchmark the whole Magic pipeline the way the browser calls it.

For every eval case: tidy the ink for priming (web/js/tidy.js, via node), run
the server's own rewrite() (reader, best-of-N synthesis and the self-check
fallback), then have an independent judge read the result back.

Per configuration it reports:
  meant ok  the reader's MEANT was the intended sentence
  legible   CER of the judge's reading of the output vs the text it was asked to write
  overall   CER of the judge's reading vs the intended sentence (right AND legible)
  perfect   share of lines the judge read exactly as intended
  mine      share of lines written in the user's own style (the rest fell back)
  time      seconds per line (read + write + self-check)

The judge should be independent of the server's proofreader (macOS Vision on
a Mac), so by default it is Qwen3-VL-4B reading literally; --judge vision uses
macOS Vision. Output images and a contact sheet per configuration go to <out>/.

usage: python tests/eval_pipeline.py <evalset_dir> <out> [neatness,candidates,style ...]
       e.g. 60,3,mine 75,4,mine 75,4,9    (neatness is the page's 0-100 slider)
"""
import argparse
import contextlib
import glob
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
import app  # noqa: E402
from ink import render_strokes  # noqa: E402
from recognize import AppleVisionReader, VLMReader, build_proofreader, cer  # noqa: E402
from synth import HandwritingSynth  # noqa: E402
from weights import STYLES_DIR, ensure_weights  # noqa: E402

def tidied_primes(cases, strength):
    """The browser primes with tidyLine()'s output, so do the same (needs node)."""
    js = (f"import {{ tidyLine }} from {json.dumps(os.path.join(ROOT, 'web', 'js', 'tidy.js'))};\n"
          "import fs from 'node:fs';\n"
          "const cases = JSON.parse(fs.readFileSync(0, 'utf8'));\n"
          f"console.log(JSON.stringify(cases.map((c) => tidyLine(c.map((s) => s.map(([x, y]) => ({{ x, y }}))), "
          f"{{ strength: {strength} }}).strokes.map((s) => s.map((p) => [p.x, p.y])))));")
    out = subprocess.run(["node", "--input-type=module", "-e", js], input=json.dumps([c["strokes"] for c in cases]),
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def draw_output(strokes, h=64, pad=16):
    """Normalised output strokes (baseline 0, core height 1, y down) -> image."""
    if not strokes:
        return Image.new("RGB", (200, h + 2 * pad), "white")
    return render_strokes([np.asarray(s) for s in strokes], target_height=h, pad=pad)


def contact_sheet(rows, path, width=1500):
    """rows: [(input image, output image, caption)] -> one PNG."""
    tiles = []
    for a, b, cap in rows:
        a = a.copy()
        a.thumbnail((width // 2 - 20, 120))
        b = b.copy()
        b.thumbnail((width // 2 - 20, 120))
        tile = Image.new("RGB", (width, max(a.height, b.height) + 26), "white")
        tile.paste(a, (0, 22))
        tile.paste(b, (width // 2, 22))
        ImageDraw.Draw(tile).text((6, 4), cap, fill=(90, 90, 90))
        tiles.append(tile)
    sheet = Image.new("RGB", (width, sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        ImageDraw.Draw(sheet).line([(0, y), (width, y)], fill=(225, 225, 225))
        y += t.height
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evalset")
    ap.add_argument("out")
    ap.add_argument("configs", nargs="*", default=["60,3,mine"])
    ap.add_argument("--seeds", type=int, default=2, help="repeat every line with this many seeds")
    ap.add_argument("--judge", default="Qwen/Qwen3-VL-4B-Instruct", help="a Qwen3-VL model id, or 'vision'")
    ap.add_argument("--proof", default="auto", choices=["auto", "vlm", "none"],
                    help="the server's proofreader: its default, the reader VLM, or none")
    ap.add_argument("--no-fix", action="store_true", help="Fix spelling off")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cases = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.evalset, "*.json")))]
    synth = HandwritingSynth(ensure_weights(), STYLES_DIR)
    reader = VLMReader()
    if args.judge == "vision":
        judge = AppleVisionReader().recognize
    else:
        judge = (reader if args.judge == reader.model_id else VLMReader(args.judge)).read_literal
    proof, fast = {"auto": lambda: build_proofreader(reader), "vlm": lambda: (reader.read_literal, False),
                   "none": lambda: (None, False)}[args.proof]()
    app.STATE.update(synth=synth, reader=reader, proof=proof, proof_fast=fast)
    if fast:
        reader.ocr = proof          # as the server does
    print(f"reader {reader.model_id} on {reader.device}; proofreader {args.proof}; judge {args.judge}\n", flush=True)

    # the reader is deterministic (greedy), so read every case once
    reads, read_t = {}, {}
    orig_read = reader.read
    for i, c in enumerate(cases):
        t = time.time()
        reads[i] = orig_read(render_strokes([np.asarray(s, float) for s in c["strokes"]]))
        read_t[i] = time.time() - t
    ok = [app.sanitize(reads[i][1]).lower().strip(" .") == c["intended"].lower().strip(" .") for i, c in enumerate(cases)]
    print(f"reader: MEANT right on {sum(ok)}/{len(cases)}, {np.mean(list(read_t.values())):.2f}s per line")
    for i, c in enumerate(cases):
        if not ok[i]:
            print(f"   {c['intended']!r:40} read as {reads[i][0]!r} -> {reads[i][1]!r}")
    print(flush=True)

    orig_write = synth.write
    summary = []
    for cfg in args.configs:
        neat, K, style = cfg.split(",")
        neat, K = float(neat), int(K)
        style = style if style == "mine" else int(style)
        primes = tidied_primes(cases, 0.45 + 0.55 * neat / 100)
        errs_leg, errs_all, mine, times, rows = [], [], [], [], []
        for s in range(args.seeds):
            for i, c in enumerate(cases):
                synth.write = lambda *a, _seed=1000 * s + i, **kw: orig_write(*a, seed=_seed, **kw)
                reader.read = lambda img, _i=i: reads[_i]
                req = {"lines": [{"strokes": c["strokes"], "prime": primes[i]}], "style": style,
                       "bias": 0.3 + neat / 100 * 2.2, "fix_spelling": not args.no_fix, "candidates": K}
                t = time.time()
                with contextlib.redirect_stdout(io.StringIO()):
                    line = app.rewrite(req)["lines"][0]
                times.append(time.time() - t + read_t[i])
                img = draw_output(line["strokes"])
                got = judge(render_strokes([np.asarray(p) for p in line["strokes"]])) if line["strokes"] else ""
                errs_leg.append(cer(got, line["text"]))
                errs_all.append(cer(got, c["intended"]))
                mine.append(line.get("style_used") == "mine")
                if s == 0:
                    img.save(os.path.join(args.out, f"{cfg.replace(',', '_')}_{i:02d}.png"))
                    rows.append((render_strokes([np.asarray(p, float) for p in c["strokes"]]), img,
                                 f"{i:02d}  wrote {line['text']!r}  judge read {got!r}  "
                                 f"style {line.get('style_used')}  cer {errs_all[-1]:.2f}"))
        contact_sheet(rows, os.path.join(args.out, f"sheet_{cfg.replace(',', '_')}.png"))
        res = dict(cfg=cfg, legible=np.mean(errs_leg), overall=np.mean(errs_all),
                   perfect=np.mean(np.array(errs_all) == 0), mine=np.mean(mine), time=np.mean(times))
        summary.append(res)
        print(f"{cfg:>14}: legible CER {res['legible']:.3f}  overall CER {res['overall']:.3f}  "
              f"perfect {res['perfect']:.0%}  mine {res['mine']:.0%}  {res['time']:.1f}s/line", flush=True)
    synth.write, reader.read = orig_write, orig_read
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
