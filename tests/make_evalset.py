"""Build a small, deliberately messy evaluation set of ink lines.

Uses the synthesiser at LOW bias (sloppy) plus low-frequency wobble and
jitter to imitate mouse/trackpad writing, with dyslexia-style misspellings.
Writes <out>/NN.json (strokes, written, intended) + NN.png.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from synth import HandwritingSynth
from ink import render_strokes

CASES = [
    ("I recieve my freind at the park", "I receive my friend at the park"),
    ("the wether was beutiful today", "the weather was beautiful today"),
    ("we went to the libary after scool", "we went to the library after school"),
    ("my favorit food is spagetti", "my favorite food is spaghetti"),
    ("I want to go their tomorow", "I want to go there tomorrow"),
    ("can you help me with my homwork", "can you help me with my homework"),
    ("The dog runs fast", "The dog runs fast"),
    ("thank you for comming to my party", "thank you for coming to my party"),
    ("Hello world", "Hello world"),
    ("I beleive it will rain", "I believe it will rain"),
    ("meet me at noon", "meet me at noon"),
    ("she was vary happy", "she was very happy"),
]

def wobble(strokes, rng, amp):
    out = []
    ph = rng.uniform(0, 6.28, 4)
    for s in strokes:
        s = s.copy()
        t = s[:, 0] / 40.0
        s[:, 1] += amp * np.sin(t * 0.7 + ph[0]) * 1.5    # wavy baseline
        s[:, 0] += amp * 0.4 * np.sin(s[:, 1] / 9 + ph[1])
        s += rng.normal(0, amp * 0.25, s.shape)            # hand tremor / mouse jitter
        out.append(s)
    return out

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.join(os.path.dirname(__file__), "..", "models")
    m = HandwritingSynth(os.path.join(root, "hand_synth.npz"), os.path.join(root, "styles"))
    rng = np.random.default_rng(7)
    written = [c[0] for c in CASES]
    styles = rng.integers(0, 13, len(CASES))
    coords = m.write(written, bias=0.15, primes=[m.style_prime(int(s)) for s in styles], seed=3)
    for i, ((w, intended), c) in enumerate(zip(CASES, coords)):
        c = c.copy(); c[:, 1] *= -1; c[:, :2] *= 3.0   # to canvas-ish px, y down
        strokes = [s[:, :2] for s in np.split(c, np.where(c[:, 2] == 1)[0] + 1) if len(s)]
        strokes = wobble(strokes, rng, amp=2.0)
        json.dump({"strokes": [s.round(2).tolist() for s in strokes], "written": w, "intended": intended},
                  open(f"{out_dir}/{i:02d}.json", "w"))
        render_strokes(strokes).save(f"{out_dir}/{i:02d}.png")
    print("wrote", len(CASES), "cases to", out_dir)

if __name__ == "__main__":
    main(sys.argv[1])
