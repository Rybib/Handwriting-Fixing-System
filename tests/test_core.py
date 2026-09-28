"""Fast sanity tests (no reader model needed): python tests/test_core.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from recognize import cer, word_corrections  # noqa: E402
from synth import HandwritingSynth, offsets_to_coords, sanitize, user_strokes_to_prime  # noqa: E402
from weights import STYLES_DIR, ensure_weights  # noqa: E402


def test_sanitize():
    assert sanitize("Quiz  X-ray & Zoo’s") == "quiz x-ray and zoo's"


def test_corrections():
    assert word_corrections("I recieve my freind", "I receive my friend") == [["recieve", "receive"], ["freind", "friend"]]
    assert cer("hello world", "Hello, world!") == 0


def test_synthesis_and_priming():
    m = HandwritingSynth(ensure_weights(), STYLES_DIR)
    # unprimed + primed with a built-in style, batched, best-of-2
    out, info = m.write(["hello world", "neat writing"], bias=1.5, seed=0, n_candidates=2, return_info=True,
                        primes=[None, m.style_prime(3)])
    for c, inf in zip(out, info):
        assert c.shape[1] == 3 and len(c) > 50
        assert inf["score"] < 6, inf
        w = c[:, 0].max() - c[:, 0].min()
        assert 40 < w < 250, w          # ~10 units per character
    # priming with "user" ink: re-use a style sample as if it were canvas strokes
    offs, text = m.style_prime(7)
    coords = offsets_to_coords(offs)
    strokes = [s[:, :2] * [4, -4] for s in np.split(coords, np.where(coords[:, 2] == 1)[0] + 1) if len(s) > 1]
    prime = user_strokes_to_prime(strokes, text)
    _, info = m.write(["it still works"], bias=1.5, primes=[prime], seed=1, return_info=True)
    assert info[0]["prime_aligned"], info


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
