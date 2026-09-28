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


def test_trim_keeps_only_the_text():
    from synth import _trim
    # pen positions (x, y, pen lifts after this point) and the character attended to at each
    pts = [(0, 9, 0), (1, 9, 1),                 # still finishing the priming sample (chars < 5)
           (2, 0, 0), (6, 4, 0), (10, 0, 1),     # the text (chars 5-9)
           (7, 3, 0), (12, 3, 1),                # after the last letter, back over it: a t-bar
           (20, 0, 0), (24, 4, 1)]               # after the last letter, further right: junk
    att = [3, 4, 5, 7, 9, 10, 10, 11, 11]
    xy = np.array(pts, np.float32)
    off = np.concatenate([xy[:1], np.column_stack([np.diff(xy[:, :2], axis=0), xy[1:, 2]])])
    assert _trim(list(off), att, 5, 10)[:, 0].tolist() == [2, 6, 10, 7, 12]


def test_pick_prefers_the_version_that_reads_back():
    import app
    line = np.array([[0, 0, 0], [5, 3, 0], [10, 0, 1]], np.float32)
    cands = [{"coords": line, "score": s} for s in (0, 1, 2)]
    readings = iter(["hellp", "hello", "never read"])
    best = app.pick(cands, "hello", lambda img: next(readings), limit=3)
    assert best is cands[1] and best["cer"] == 0 and "cer" not in cands[2]
    # without a proofreader, a poor attention score counts as a misreading
    assert app.pick([{"coords": line, "score": 7}, {"coords": line, "score": 9}], "hi", None, 2)["score"] == 7


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
