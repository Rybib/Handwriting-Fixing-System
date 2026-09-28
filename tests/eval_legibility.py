"""Benchmark: is the rewritten handwriting legible AND correctly spelled?

For every case in the eval set, prime the synthesiser with the user's own
messy ink, write the INTENDED sentence, then have the VLM read the result
back literally. Reports character error rate (CER) per configuration.

usage: python tests/eval_legibility.py <evalset_dir> [bias,K ...]
"""
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from synth import HandwritingSynth, user_strokes_to_prime, sanitize
from ink import render_strokes
from recognize import VLMReader

LITERAL = "Transcribe this handwriting exactly, letter by letter. Reply with only the text."

def cer(a, b):
    a, b = a.lower().strip(" ."), b.lower().strip(" .")
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / max(1, len(b))

def main(ev, configs):
    root = os.path.join(os.path.dirname(__file__), "..", "models")
    m = HandwritingSynth(os.path.join(root, "hand_synth.npz"), os.path.join(root, "styles"))
    vlm = VLMReader()
    cases = [json.load(open(f)) for f in sorted(glob.glob(ev + "/*.json"))]
    for cfg in configs:
        bias, K = float(cfg.split(",")[0]), int(cfg.split(",")[1])
        errs, t_syn = [], 0
        for i, d in enumerate(cases):
            prime = user_strokes_to_prime([np.array(s) for s in d["strokes"]], sanitize(d["written"]))
            t = time.time()
            c = m.write([sanitize(d["intended"])], bias=bias, primes=[prime], seed=100 + i, n_candidates=K)[0]
            t_syn += time.time() - t
            c[:, 1] *= -1
            strokes = [s[:, :2] for s in np.split(c, np.where(c[:, 2] == 1)[0] + 1) if len(s)]
            img = render_strokes(strokes)
            img.save(f"{ev}/out_{cfg.replace(',', '_')}_{i:02d}.png")
            got = vlm._generate([{"type": "image", "image": img}, {"type": "text", "text": LITERAL}], 60).strip()
            errs.append(cer(got, d["intended"]))
            print(f"  {d['intended']!r:40} -> {got!r}  cer={errs[-1]:.2f}", flush=True)
        print(f"bias={bias} K={K}: mean CER={np.mean(errs):.3f}  perfect={np.mean(np.array(errs) == 0):.0%}  synth={t_syn / len(cases):.1f}s/line", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["0.75,1", "1.5,1", "1.5,4"])
