"""End-to-end test: draw handwriting with a real (emulated) mouse in the browser.

Starts from ink in the eval set (or any JSON with "strokes"), moves the mouse
along it, waits for the magic, and saves screenshots along the way.

usage: python tests/e2e_playwright.py <out_dir> <case.json> [<case.json> ...]
          [--mode magic|tidy] [--url http://localhost:8765] [--chromium PATH]
"""
import argparse
import json
import math
import os
import time

from playwright.sync_api import sync_playwright


def draw_case(page, strokes, x0, y0, scale, spacing=5.0):
    all_pts = [p for s in strokes for p in s]
    minx = min(p[0] for p in all_pts)
    miny = min(p[1] for p in all_pts)
    for s in strokes:
        pts = [((p[0] - minx) * scale + x0, (p[1] - miny) * scale + y0) for p in s]
        # resample to roughly mouse-event spacing
        out = [pts[0]]
        for p in pts[1:]:
            if math.dist(p, out[-1]) >= spacing:
                out.append(p)
        if pts[-1] != out[-1]:
            out.append(pts[-1])
        page.mouse.move(*out[0])
        page.mouse.down()
        for p in out[1:]:
            page.mouse.move(*p)
        page.mouse.up()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--mode", default="magic")
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--chromium", default=os.environ.get("CHROMIUM", "/opt/pw-browsers/chromium"))
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--together", action="store_true", help="write all lines, then pause once")
    ap.add_argument("--gif", help="record the whole session to this animated GIF")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chromium if os.path.exists(args.chromium) else None)
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)
        logs = []
        page.on("console", lambda m: logs.append(m.text))
        page.on("pageerror", lambda e: logs.append(f"PAGE ERROR: {e}"))
        page.goto(args.url)
        if args.mode == "magic":
            t = time.time()
            page.wait_for_function("document.getElementById('status').classList.contains('ready')", timeout=900_000)
            print(f"models ready after {time.time() - t:.0f}s")
        page.click(f"#mode button[data-mode={args.mode}]")
        page.evaluate("window.__hw.settings.autoDelay = 600")

        frames = []

        def snap(path=None):
            png = page.screenshot(path=path)
            if args.gif:
                frames.append(png)

        def wait_done(label):
            t = time.time()
            time.sleep(1.0)
            snap(f"{args.out}/{label}_b_working.png")
            while page.evaluate("window.__hw.busy"):
                if time.time() - t > 600:
                    raise TimeoutError(label)
                snap() if args.gif else time.sleep(0.1)
            print(f"{label}: done in {time.time() - t:.1f}s")
            snap(f"{args.out}/{label}_c_done.png")

        y = 150 - 55
        if args.together:
            page.evaluate("window.__hw.settings.autoDelay = 60000")
        for i, path in enumerate(args.cases):
            case = json.load(open(path))
            draw_case(page, case["strokes"], 110, y, args.scale)
            snap(f"{args.out}/{i:02d}_a_drawn.png")
            if not args.together:
                print(f"{i:02d}: {case.get('written')!r}")
                wait_done(f"{i:02d}")
            y += 144
        if args.together:
            page.keyboard.press("Enter")
            wait_done("all")
        page.keyboard.down("Space")
        page.screenshot(path=f"{args.out}/zz_compare_original.png")
        page.keyboard.up("Space")
        page.screenshot(path=f"{args.out}/zz_final.png")
        print("\n".join(logs[-20:]))
        if args.gif and frames:
            import io
            from PIL import Image
            imgs = [Image.open(io.BytesIO(f)).convert("RGB").resize((700, 450)) for f in frames]
            imgs[0].save(args.gif, save_all=True, append_images=imgs[1:] + [imgs[-1]] * 8, duration=120, loop=0)
            print("gif:", args.gif, len(imgs), "frames")
        browser.close()


if __name__ == "__main__":
    main()
