"""Pure-numpy inference for the Graves (2013) handwriting synthesis network.

Architecture (matches sjvasquez/handwriting-synthesis exactly):
  3 stacked LSTMs (400 units) with a soft Gaussian attention window over the
  one-hot characters, and a 20-component bivariate Gaussian mixture output.

"Priming" feeds a real handwriting sample + its transcript through the network
first, so the text it then writes continues in that writer's style. We prime
with the user's own (cleaned-up) strokes, so the rewrite looks like *their*
handwriting, just neat. The `bias` knob sharpens the output distribution:
higher bias = neater, more regular writing.
"""

import os

import numpy as np
from scipy.signal import savgol_filter

ALPHABET = [
    "\x00", " ", "!", '"', "#", "'", "(", ")", ",", "-", ".",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";",
    "?", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
    "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
    "y", "z",
]
CHAR_TO_ID = {c: i for i, c in enumerate(ALPHABET)}
MAX_CHARS = 75
N_OUT = 20      # output mixture components
N_ATT = 10      # attention mixture components
H = 400         # LSTM size
V = len(ALPHABET)

_SUBSTITUTES = {"Q": "q", "X": "x", "Z": "z", "`": "'", "’": "'", "‘": "'",
                "“": '"', "”": '"', "—": "-", "–": "-", "&": "and"}


def sanitize(text):
    """Map text into the model's character set (IAM-OnDB lacks Q, X, Z capitals etc.)."""
    out = []
    for ch in text:
        ch = _SUBSTITUTES.get(ch, ch)
        out.append("".join(c for c in ch if c in CHAR_TO_ID and c != "\x00"))
    return " ".join("".join(out).split())


def encode(text):
    return np.array([CHAR_TO_ID[c] for c in text] + [0], dtype=np.int64)


def _sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def _softplus(x):
    return np.logaddexp(0.0, x)


# ----------------------------------------------------------------------------
# stroke utilities (coords are [x, y, eos] with y pointing UP, like IAM-OnDB)
# ----------------------------------------------------------------------------

def offsets_to_coords(offsets):
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def coords_to_offsets(coords):
    offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    return np.concatenate([np.array([[0, 0, 1]], dtype=coords.dtype), offsets], axis=0)


def split_strokes(coords):
    return [s for s in np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0) if len(s)]


def denoise(coords, window=7):
    out = []
    for s in split_strokes(coords):
        s = s.copy()
        if len(s) >= window:
            s[:, 0] = savgol_filter(s[:, 0], window, 3, mode="nearest")
            s[:, 1] = savgol_filter(s[:, 1], window, 3, mode="nearest")
        out.append(s)
    return np.vstack(out)


def align(coords):
    """Remove the global baseline tilt (least squares line through all points)."""
    coords = coords.copy()
    X = np.stack([np.ones(len(coords)), coords[:, 0]], axis=1)
    offset, slope = np.linalg.lstsq(X, coords[:, 1], rcond=None)[0]
    theta = np.arctan(slope)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords[:, :2] = coords[:, :2] @ R - offset
    return coords


class HandwritingSynth:
    def __init__(self, weights_path, styles_dir=None):
        w = np.load(weights_path)
        self.W = {k: w[k].astype(np.float32) for k in w.files}
        self.styles_dir = styles_dir
        # split each LSTM kernel into the input part and the recurrent part
        self.l1_Wx, self.l1_Wh = self.W["lstm1_W"][: V + 3], self.W["lstm1_W"][V + 3:]
        self.l2_Wx, self.l2_Wh = self.W["lstm2_W"][: 3 + H + V], self.W["lstm2_W"][3 + H + V:]
        self.l3_Wx, self.l3_Wh = self.W["lstm3_W"][: 3 + H + V], self.W["lstm3_W"][3 + H + V:]

    # -- core cell ---------------------------------------------------------
    @staticmethod
    def _lstm(x, h, c, Wx, Wh, b):
        z = x @ Wx + h @ Wh + b
        i, j, f, o = np.split(z, 4, axis=1)          # TF LSTMCell gate order
        c = _sigmoid(f + 1.0) * c + _sigmoid(i) * np.tanh(j)  # forget_bias = 1.0
        h = _sigmoid(o) * np.tanh(c)
        return h, c

    def _step(self, inp, st, onehot, mask, u):
        W = self.W
        h1, c1 = self._lstm(np.concatenate([st["w"], inp], 1), st["h1"], st["c1"],
                            self.l1_Wx, self.l1_Wh, W["lstm1_b"])
        att = _softplus(np.concatenate([st["w"], inp, h1], 1) @ W["attn_W"] + W["attn_b"])
        alpha, beta, kappa = np.split(att, 3, axis=1)
        kappa = st["kappa"] + kappa / 25.0
        beta = np.maximum(beta, 0.01)
        phi = np.sum(alpha[:, :, None] * np.exp(-np.square(kappa[:, :, None] - u) / beta[:, :, None]), axis=1)
        w = np.einsum("bu,buv->bv", phi * mask, onehot)
        h2, c2 = self._lstm(np.concatenate([inp, h1, w], 1), st["h2"], st["c2"],
                            self.l2_Wx, self.l2_Wh, W["lstm2_b"])
        h3, c3 = self._lstm(np.concatenate([inp, h2, w], 1), st["h3"], st["c3"],
                            self.l3_Wx, self.l3_Wh, W["lstm3_b"])
        return dict(h1=h1, c1=c1, h2=h2, c2=c2, h3=h3, c3=c3, kappa=kappa, w=w, phi=phi)

    def _sample_output(self, h3, bias, rng):
        """Sample the next pen offset. Returns (samples[B,3], eos_prob[B])."""
        z = h3 @ self.W["gmm_W"] + self.W["gmm_b"]
        pis, sig, rho, mu, e = np.split(z, np.cumsum([N_OUT, 2 * N_OUT, N_OUT, 2 * N_OUT]), axis=1)
        pis = pis * (1.0 + bias[:, None])
        sig = np.maximum(np.exp(sig - bias[:, None]), 1e-4)
        rho = np.clip(np.tanh(rho), -1 + 1e-8, 1 - 1e-8)
        e = np.clip(_sigmoid(e[:, 0]), 1e-8, 1 - 1e-8)
        e = np.where(e < 0.01, 0.0, e)
        pis = np.exp(pis - pis.max(1, keepdims=True))
        pis /= pis.sum(1, keepdims=True)
        pis = np.where(pis < 0.01, 0.0, pis)
        cdf = np.cumsum(pis, axis=1)
        cdf /= cdf[:, -1:]

        B = h3.shape[0]
        idx = np.minimum((cdf < rng.random((B, 1))).sum(1), N_OUT - 1)
        ar = np.arange(B)
        m1, m2 = mu[ar, idx], mu[ar, N_OUT + idx]
        s1, s2 = sig[ar, idx], sig[ar, N_OUT + idx]
        r = rho[ar, idx]
        z1, z2 = rng.standard_normal(B), rng.standard_normal(B)
        x1 = m1 + s1 * z1
        x2 = m2 + s2 * (r * z1 + np.sqrt(1 - r * r) * z2)
        eos = (rng.random(B) < e).astype(np.float32)
        return np.stack([x1, x2, eos], axis=1).astype(np.float32), e

    # -- public API ----------------------------------------------------------
    def style_prime(self, style_id):
        """Load one of the bundled IAM-OnDB writer samples as (offsets, text)."""
        strokes = np.load(os.path.join(self.styles_dir, f"style-{style_id}-strokes.npy"))
        chars = np.load(os.path.join(self.styles_dir, f"style-{style_id}-chars.npy")).tobytes().decode("utf-8")
        return strokes.astype(np.float32), chars

    def write(self, lines, bias=0.75, primes=None, seed=None, n_candidates=1, max_steps_per_char=40,
              return_info=False):
        """Synthesise handwriting for each line (batched).

        lines:        list of strings (already sanitised, <= 75 chars)
        bias:         float or list of floats (higher = neater)
        primes:       optional list (one per line) of (offsets[N,3], transcript) or None
        n_candidates: sample this many versions of every line in parallel and keep
                      the one whose attention visited every letter most evenly
                      (the RNN occasionally skips or garbles a letter; this is a
                      cheap way to catch that without a second recogniser pass).
        returns list of coords arrays [N,3] (x, y-up, eos), denoised + aligned
        (and, with return_info, a list of {"score", "prime_aligned"} per line:
        prime_aligned is False when the attention did NOT end the priming pass
        near the end of the prime transcript, i.e. the ink and its transcript
        disagree and the output is likely garbage)
        """
        rng = np.random.default_rng(seed)
        n_lines = len(lines)
        biases = np.broadcast_to(np.asarray(bias, dtype=np.float32), (n_lines,))
        primes = primes or [None] * n_lines
        K = max(1, int(n_candidates))
        lines_k = [t for t in lines for _ in range(K)]
        primes_k = [p for p in primes for _ in range(K)]
        bias = np.repeat(biases, K).astype(np.float32)
        B = len(lines_k)

        char_seqs, prime_seqs, text_start = [], [], []
        for text, p in zip(lines_k, primes_k):
            if p is not None:
                char_seqs.append(encode(p[1] + " " + text))
                prime_seqs.append(p[0])
                text_start.append(len(p[1]) + 1)
            else:
                char_seqs.append(encode(text))
                prime_seqs.append(np.zeros((0, 3), np.float32))
                text_start.append(0)

        U = max(len(c) for c in char_seqs)
        onehot = np.zeros((B, U, V), np.float32)
        mask = np.zeros((B, U), np.float32)
        c_len = np.array([len(c) for c in char_seqs])
        for b, cs in enumerate(char_seqs):
            onehot[b, np.arange(len(cs)), cs] = 1.0
            mask[b, : len(cs)] = 1.0
        u = np.arange(U, dtype=np.float32)[None, None, :]

        z = lambda n: np.zeros((B, n), np.float32)
        st = dict(h1=z(H), c1=z(H), h2=z(H), c2=z(H), h3=z(H), c3=z(H), kappa=z(N_ATT), w=z(V), phi=z(U))

        # 1) priming: teacher-force the style sample through the network
        P = max(len(p) for p in prime_seqs)
        for t in range(P):
            active = np.array([t < len(p) for p in prime_seqs])
            inp = np.stack([p[t] if t < len(p) else np.zeros(3, np.float32) for p in prime_seqs])
            new = self._step(inp, st, onehot, mask, u)
            for k in st:
                st[k] = np.where(active[:, None], new[k], st[k])

        prime_end = np.argmax(st["phi"], axis=1)
        prime_aligned = np.array([
            (not len(p)) or abs(int(prime_end[b]) - (text_start[b] - 1)) <= max(3, 0.25 * text_start[b])
            for b, p in enumerate(prime_seqs)])

        # 2) free-running generation
        has_prime = np.array([len(p) > 0 for p in prime_seqs])
        inp, _ = self._sample_output(st["h3"], bias, rng)
        inp[~has_prime] = np.array([0, 0, 1], np.float32)
        max_steps = max_steps_per_char * max(len(t) for t in lines) + 20
        outs = [[] for _ in range(B)]
        dwell = np.zeros((B, U + 1), np.int32)
        done = np.zeros(B, bool)
        natural = np.zeros(B, bool)
        for _ in range(max_steps):
            st_new = self._step(inp, st, onehot, mask, u)
            for k in st:
                st[k] = np.where(done[:, None], st[k], st_new[k])
            nxt, e = self._sample_output(st["h3"], bias, rng)
            char_idx = np.argmax(st["phi"], axis=1)
            for b in np.where(~done)[0]:
                outs[b].append(nxt[b])
                dwell[b, char_idx[b]] += 1
            # termination: attention has reached the end-of-text token and the pen lifts
            eos_probe = rng.random(B) < e
            finished = ((char_idx >= c_len - 1) & eos_probe) | (char_idx >= c_len)
            natural |= finished & ~done
            done |= finished
            if done.all():
                break
            inp = nxt

        results, infos = [], []
        for li in range(n_lines):
            best, best_score = None, None
            for b in range(li * K, li * K + K):
                text = lines_k[b]
                d = dwell[b, text_start[b]: text_start[b] + len(text)]
                letters = np.array([c != " " for c in text])
                skipped = int(np.sum((d < 2) & letters))
                score = skipped * 2 + (0 if natural[b] else 6) + int(np.sum(d > 5 * max(1, np.median(d[letters]) if letters.any() else 1)))
                if best_score is None or score < best_score:
                    best, best_score = b, score
            off = np.array(outs[best], np.float32) if outs[best] else np.zeros((1, 3), np.float32)
            off[-1, 2] = 1.0
            coords = offsets_to_coords(off)
            coords = denoise(coords)
            coords[:, :2] = align(coords[:, :2])
            results.append(coords)
            infos.append({"score": int(best_score), "prime_aligned": bool(prime_aligned[best])})
        return (results, infos) if return_info else results


# ----------------------------------------------------------------------------
# turning user ink into a priming sample
# ----------------------------------------------------------------------------

def resample_stroke(pts, spacing):
    """Uniform arc-length resampling of one stroke (pts: [N,2])."""
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = np.concatenate([[0], np.cumsum(seg)])
    total = d[-1]
    if total < 1e-6:
        return pts[:1].copy()
    n = max(2, int(np.ceil(total / spacing)) + 1)
    t = np.linspace(0, total, n)
    return np.stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], axis=1)


# Horizontal width of one character of IAM-OnDB ink in normalised offset units
# (the 13 bundled writer samples range 6.6-12.8, median ~10), and the network
# was trained on ink with ~1 unit between consecutive pen-down points.
IAM_UNITS_PER_CHAR = 10.0


def user_strokes_to_prime(strokes, transcript, units_per_char=IAM_UNITS_PER_CHAR):
    """Convert canvas strokes (list of [N,2] arrays, y DOWN) into IAM-like offsets.

    We rescale so the ink's width-per-character matches the training data, then
    resample to ~1 unit per point, which is roughly the pen speed the network
    learned. Returns (offsets[N,3], transcript) or None if unusable.
    """
    strokes = [np.asarray(s, np.float64) for s in strokes if len(s) >= 1]
    if not strokes or not transcript.strip():
        return None
    allpts = np.vstack(strokes)
    width = allpts[:, 0].max() - allpts[:, 0].min()
    n_chars = max(1, len(transcript))
    if width < 1:
        return None
    scale = units_per_char * n_chars / width
    coords = []
    for s in strokes:
        s = s.copy()
        s[:, 1] = -s[:, 1]                  # y up
        s = (s - allpts.min(0) * [1, -1]) * scale
        r = resample_stroke(s, 1.0)
        eos = np.zeros((len(r), 1))
        eos[-1] = 1
        coords.append(np.hstack([r, eos]))
    coords = np.vstack(coords)
    coords[:, :2] = align(coords[:, :2])
    offsets = coords_to_offsets(coords).astype(np.float32)
    offsets = offsets[:1200]
    return offsets, transcript
