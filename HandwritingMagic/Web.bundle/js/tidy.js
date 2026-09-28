// Geometry for handwriting: grouping ink into lines/words, measuring it, and
// "Tidy" - a style-preserving beautifier that keeps the writer's own letter
// shapes but straightens the baseline, evens out letter size, slant and word
// spacing, and smooths the strokes. Pure math, instant, no model needed.

const median = (a) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const percentile = (a, q) => {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const i = (s.length - 1) * q;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
};
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export function bbox(pts) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
  }
  return { minX, minY, maxX, maxY, w: maxX - minX, h: maxY - minY, cx: (minX + maxX) / 2, cy: (minY + maxY) / 2 };
}

// Upper/lower contour of a blob of ink, sampled in vertical slices.
function contours(pts, binW) {
  const bins = new Map();
  for (const p of pts) {
    const k = Math.floor(p.x / binW);
    const b = bins.get(k);
    if (!b) bins.set(k, { top: p.y, bot: p.y, xt: p.x, xb: p.x });
    else {
      if (p.y < b.top) { b.top = p.y; b.xt = p.x; }
      if (p.y > b.bot) { b.bot = p.y; b.xb = p.x; }
    }
  }
  return [...bins.values()];
}

// Robust baseline + x-height ("core") of some ink, from the horizontal ink
// density profile: the band between baseline and x-height holds most of the
// ink, while ascenders (l, h, t) and descenders (g, y, p) are sparse.
export function coreMetrics(pts, hint) {
  const bb = bbox(pts);
  const Hh = Math.max(bb.h, 1);
  const nb = 48, binH = Hh / nb;
  const hist = new Float64Array(nb);
  const steps = [];
  for (let i = 1; i < pts.length; i++) steps.push(Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y));
  const cap = 3 * (median(steps) || 1);
  for (let i = 0; i < pts.length; i++) {
    const w = Math.min(i + 1 < pts.length ? steps[i] : cap, cap) + 1e-3;   // arc-length weight, ignore pen-up jumps
    hist[Math.min(nb - 1, Math.floor((pts[i].y - bb.minY) / binH))] += w;
  }
  const sm = hist.map((_, i) => [-2, -1, 0, 1, 2].reduce((a, d) => a + (hist[i + d] || 0) * (3 - Math.abs(d)), 0));
  let peak = 0;
  sm.forEach((v, i) => { if (v > sm[peak]) peak = i; });
  const thr = sm[peak] * 0.4;
  let lo = peak, hi = peak;
  while (lo > 0 && sm[lo - 1] >= thr) lo--;
  while (hi < nb - 1 && sm[hi + 1] >= thr) hi++;
  const top = bb.minY + lo * binH, base = bb.minY + (hi + 1) * binH;
  return { base, top, core: Math.max(base - top, Hh * 0.15, 1) };
}

// Fit the baseline direction from the lower contour, ignoring descenders.
function baselineFit(pts, core) {
  const c = contours(pts, Math.max(2, core / 2)).map((b) => ({ x: b.xb, y: b.bot }));
  if (c.length < 3) return { slope: 0, icpt: median(c.map((p) => p.y)) };
  let fit = lsq(c);
  for (let it = 0; it < 2; it++) {
    const keep = c.filter((p) => p.y - (fit.icpt + fit.slope * p.x) < core * 0.35);
    if (keep.length >= 3) fit = lsq(keep);
  }
  return fit;
}
function lsq(pts) {
  const n = pts.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const p of pts) { sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y; }
  const den = n * sxx - sx * sx;
  const slope = Math.abs(den) < 1e-9 ? 0 : (n * sxy - sx * sy) / den;
  return { slope, icpt: (sy - slope * sx) / n };
}

// ---------------------------------------------------------------------------
// Group strokes into lines (in writing order) and lines into words.
// strokes: [{pts:[{x,y}], ...}] -> [[strokeIndex, ...], ...] top to bottom
export function groupLines(strokes) {
  const boxes = strokes.map((s) => bbox(s.pts));
  const hs = boxes.map((b) => b.h).filter((h) => h > 4);
  const h = clamp(median(hs) || 30, 10, 250);
  // Dots, i-dots, crossbars and connectors ("marks") sit above or between the
  // letters. Grouping them along with the letters let an i-dot start a line of
  // its own that then pulled in the next tall letters (h, k, l), splitting one
  // line in two. So letters are grouped first, and marks join afterwards.
  const isMark = (b) => b.h < h * 0.35 || Math.max(b.w, b.h) < h * 0.6;
  const lines = [];
  const newLine = (i) => {
    const b = boxes[i];
    lines.push({ idx: [i], cys: [b.cy], cy: b.cy, minY: b.minY, maxY: b.maxY, minX: b.minX, maxX: b.maxX });
  };
  const add = (L, i) => {
    const b = boxes[i];
    L.idx.push(i);
    if (!isMark(b)) { L.cys.push(b.cy); L.cy = median(L.cys); }
    L.minY = Math.min(L.minY, b.minY); L.maxY = Math.max(L.maxY, b.maxY);
    L.minX = Math.min(L.minX, b.minX); L.maxX = Math.max(L.maxX, b.maxX);
  };
  boxes.forEach((b, i) => {
    if (isMark(b)) return;
    let best = null, bestD = Infinity;
    for (const L of lines) {
      const d = Math.abs(b.cy - L.cy);
      const overlap = Math.min(b.maxY, L.maxY) - Math.max(b.minY, L.minY);
      if ((d < h * 0.9 || overlap > Math.min(b.h, L.maxY - L.minY) * 0.5) && d < bestD) { best = L; bestD = d; }
    }
    if (best) add(best, i); else newLine(i);
  });
  // Small leftovers (a crossbar, or a lone descender or ascender stroke like
  // the tail of a y, whose centre is off the line) fold into the nearest real
  // line when they are small or reach its core band.
  const big = lines.filter((L) => L.idx.length > 2 || (L.maxY - L.minY > h * 0.6 && L.maxX - L.minX > h * 2.5));
  for (const L of lines) {
    if (!big.length || big.includes(L)) continue;
    let best = big[0];
    for (const B of big) if (Math.abs(B.cy - L.cy) < Math.abs(best.cy - L.cy)) best = B;
    const reachesCore = L.minY < best.cy + h * 0.6 && L.maxY > best.cy - h * 0.6
      && L.maxX > best.minX - h * 2 && L.minX < best.maxX + h * 2;
    if (reachesCore || (L.maxY - L.minY <= h * 0.6 && Math.abs(best.cy - L.cy) < h * 2.2)) {
      L.idx.forEach((i) => add(best, i));
      L.idx = [];
    }
  }
  // Marks join the line whose ink is right beside them (an i-dot goes with
  // the stem under it), else the line whose centre is close.
  boxes.forEach((b, i) => {
    if (!isMark(b)) return;
    let best = null, bestD = h * 2.2;
    for (const L of lines) {
      if (!L.idx.length) continue;
      let d = Math.abs(b.cy - L.cy) + h;
      for (const j of L.idx) {
        const o = boxes[j];
        if (o.maxX > b.minX - h && o.minX < b.maxX + h) d = Math.min(d, Math.max(0, o.minY - b.cy, b.cy - o.maxY));
      }
      if (d < bestD) { best = L; bestD = d; }
    }
    if (best) add(best, i); else newLine(i);
  });
  return lines.filter((L) => L.idx.length).sort((a, b) => a.cy - b.cy).map((L) => L.idx.sort((a, b) => a - b));
}

// Lines written together - each one under the last, overlapping it side to
// side - are one piece of writing, and Magic reads them together so every line
// is read knowing the others. Writing somewhere else (a list off to the side,
// a note further down after a gap) is a block of its own. A line from
// groupLines is first split where it has a wide horizontal gap, since two
// things written side by side share a line but aren't one sentence.
// Returns blocks in reading order; each block is its lines (stroke indices), top to bottom.
export function groupBlocks(strokes, lines, lineGap = 72) {
  const boxOf = (idx) => bbox(idx.flatMap((i) => strokes[i].pts));
  const hs = lines.map((idx) => boxOf(idx).h).filter((h) => h > 4);
  const lineH = clamp(median(hs) || 40, 12, 400);
  // pieces of a line more than a few words' worth of blank apart
  const segs = [];
  for (const idx of lines) {
    const items = idx.map((i) => ({ i, b: bbox(strokes[i].pts) })).sort((a, b) => a.b.minX - b.b.minX);
    let cur = null;
    for (const it of items) {
      if (cur && it.b.minX - cur.maxX > lineH * 4) { segs.push(cur); cur = null; }
      if (!cur) cur = { idx: [], maxX: -Infinity };
      cur.idx.push(it.i); cur.maxX = Math.max(cur.maxX, it.b.maxX);
    }
    if (cur) segs.push(cur);
  }
  const S = segs.map((sg) => ({ idx: sg.idx.sort((a, b) => a - b), ...boxOf(sg.idx) })).sort((a, b) => a.cy - b.cy || a.minX - b.minX);
  const maxDy = Math.max(1.8 * lineH, 1.45 * lineGap);
  const blocks = [];
  for (const sg of S) {
    // the block whose last line is just above this one and overlaps it most
    let best = null, bestOverlap = 0;
    for (const B of blocks) {
      const last = B.at(-1);
      const dy = sg.cy - last.cy;
      if (dy < lineH * 0.3 || dy > maxDy) continue;
      const overlap = Math.min(sg.maxX, last.maxX) - Math.max(sg.minX, last.minX);
      const aligned = Math.abs(sg.minX - last.minX) < lineH * 3;
      const score = Math.max(overlap, aligned ? 1 : 0);
      if (score > bestOverlap) { best = B; bestOverlap = score; }
    }
    if (best) best.push(sg); else blocks.push([sg]);
  }
  return blocks.map((B) => B.map((sg) => sg.idx));
}

export function groupWords(strokePts, gapT) {
  const items = strokePts.map((pts, i) => ({ i, b: bbox(pts) })).sort((a, b) => a.b.minX - b.b.minX);
  const words = [];
  for (const it of items) {
    const w = words[words.length - 1];
    if (w && it.b.minX <= w.maxX + gapT) { w.idx.push(it.i); w.maxX = Math.max(w.maxX, it.b.maxX); }
    else words.push({ idx: [it.i], minX: it.b.minX, maxX: it.b.maxX });
  }
  return words;
}

// Word gaps are the "large" mode of the horizontal gaps between pieces of ink.
// Otsu's method finds the split between letter gaps and word gaps; we keep it
// within sane bounds relative to the x-height.
export function adaptiveWordGap(strokePts, core) {
  const blobs = groupWords(strokePts, core * 0.12);
  const gaps = blobs.slice(1).map((b, k) => b.minX - blobs[k].maxX).filter((g) => g > 0);
  const lo = core * 0.42, hi = core * 1.1, fallback = core * 0.6;
  if (gaps.length < 3) return fallback;
  const s = [...gaps].sort((a, b) => a - b);
  let best = fallback, bestVar = -1;
  for (let k = 1; k < s.length; k++) {
    const a = s.slice(0, k), b = s.slice(k);
    const ma = a.reduce((x, y) => x + y, 0) / a.length, mb = b.reduce((x, y) => x + y, 0) / b.length;
    const v = (a.length * b.length) * (ma - mb) ** 2;
    if (v > bestVar && mb > ma * 1.6) { bestVar = v; best = (s[k - 1] + s[k]) / 2; }
  }
  return clamp(best, lo, hi);
}

// Dominant slant of some strokes (0 = upright, >0 = leaning right).
function slantOf(strokePts) {
  const vals = [], wts = [];
  for (const pts of strokePts) {
    for (let i = 2; i < pts.length; i += 2) {
      const dx = pts[i].x - pts[i - 2].x, dy = pts[i].y - pts[i - 2].y;
      const len = Math.hypot(dx, dy);
      if (len < 1.5 || Math.abs(dy) < 1.7 * Math.abs(dx)) continue;
      vals.push(-dx / dy); wts.push(len);
    }
  }
  if (!vals.length) return 0;
  // weighted median
  const order = vals.map((v, i) => i).sort((a, b) => vals[a] - vals[b]);
  const tot = wts.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (const i of order) { acc += wts[i]; if (acc >= tot / 2) return vals[i]; }
  return 0;
}

function smooth(pts, passes = 2) {
  let cur = pts.map((p) => ({ ...p }));
  for (let k = 0; k < passes; k++) {
    const nxt = cur.map((p) => ({ ...p }));
    for (let i = 1; i < cur.length - 1; i++) {
      nxt[i].x = (cur[i - 1].x + 2 * cur[i].x + cur[i + 1].x) / 4;
      nxt[i].y = (cur[i - 1].y + 2 * cur[i].y + cur[i + 1].y) / 4;
    }
    cur = nxt;
  }
  return cur;
}

// ---------------------------------------------------------------------------
// Tidy one line. strokePts: [[{x,y,...}], ...]. Returns new point arrays with
// exactly the same number of points (so we can morph), plus line metrics.
export function tidyLine(strokePts, opts = {}) {
  const strength = opts.strength ?? 1;
  const all = strokePts.flat();
  const bb = bbox(all);
  const m0 = coreMetrics(all);
  const core0 = m0.core;

  // 1. straighten: rotate so the baseline is horizontal
  const fit = baselineFit(all, core0);
  let angle = Math.atan(fit.slope);
  if (bb.w < core0 * 4) angle *= 0.5;                   // short ink: fit is unreliable
  angle = clamp(angle, -0.35, 0.35) * strength;
  const px = bb.minX, py = fit.icpt + fit.slope * bb.minX;
  const ca = Math.cos(-angle), sa = Math.sin(-angle);
  let S = strokePts.map((pts) => pts.map((p) => ({ ...p,
    x: px + (p.x - px) * ca - (p.y - py) * sa,
    y: py + (p.x - px) * sa + (p.y - py) * ca })));

  // 2. words, each with its own baseline / core height / slant
  const mA = coreMetrics(S.flat());
  const words = groupWords(S, adaptiveWordGap(S, mA.core)).map((w) => {
    const pts = w.idx.flatMap((i) => S[i]);
    const m = coreMetrics(pts, mA.core);
    const narrow = w.maxX - w.minX < mA.core * 1.6;     // "I", "a", a lone letter: its core is unreliable
    return { ...w, base: m.base, core: m.core, narrow, slant: slantOf(w.idx.map((i) => S[i])), n: pts.length };
  });
  const origGaps = words.slice(1).map((w, k) => w.minX - words[k].maxX);
  const sized = words.filter((w) => !w.narrow);
  const targetCore = median((sized.length ? sized : words).flatMap((w) => Array(Math.max(1, Math.round(w.n / 20))).fill(w.core)));
  const lineBase = median(words.map((w) => w.base));
  const targetSlant = clamp(median(words.map((w) => w.slant)), -0.15, 0.4);

  // 3. per-word: scale to the common x-height, sit on the common baseline, even slant
  for (const w of words) {
    const s = w.narrow ? 1 : 1 + (clamp(targetCore / w.core, 0.75, 1.33) - 1) * strength * 0.85;
    const dSlant = (w.slant - targetSlant) * strength * 0.85;
    const ox = w.minX;
    for (const i of w.idx) {
      S[i] = S[i].map((p) => {
        let y = (p.y - w.base) * s;
        let x = (p.x - ox) * s - y * dSlant;
        return { ...p, x: x + ox, y: y + w.base + (lineBase - w.base) * strength };
      });
    }
  }

  // 4. even word spacing: pull every gap towards the typical word gap
  const boxes = words.map((w) => bbox(w.idx.flatMap((i) => S[i])));
  const gaps = origGaps;   // as the writer spaced them (before words were resized)
  const gapT = clamp(median(gaps) || targetCore, targetCore * 0.8, targetCore * 2.5);
  let cursor = boxes.length ? boxes[0].maxX : 0;
  words.forEach((w, k) => {
    if (k === 0) return;
    const g = gaps[k - 1] + (gapT - gaps[k - 1]) * strength * 0.8;
    const dx = cursor + Math.max(g, gapT * 0.7) - boxes[k].minX;
    for (const i of w.idx) S[i] = S[i].map((p) => ({ ...p, x: p.x + dx }));
    cursor = boxes[k].maxX + dx;
  });

  // 5. gentle smoothing
  S = S.map((pts) => smooth(pts, Math.round(2 * strength)));

  const fb = bbox(S.flat());
  return {
    strokes: S,
    metrics: { x0: fb.minX, right: fb.maxX, baseline: lineBase, core: targetCore, slant: targetSlant, height: fb.h },
  };
}

// Split normalised synthesised strokes into word clusters by horizontal gaps.
export function splitWords(strokes, gap = 0.55) {
  return groupWords(strokes, gap).map((w) => w.idx);
}
