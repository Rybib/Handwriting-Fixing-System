import { StrokeModeler } from "./inkmodeler.js";
import { bbox, clamp, coreMetrics, groupBlocks, groupLines, splitWords, tidyLine } from "./tidy.js";

// ---------------------------------------------------------------------------
// setup
const canvas = document.getElementById("paper");
const ctx = canvas.getContext("2d");
const $ = (id) => document.getElementById(id);
let W = 0, H = 0, DPR = 1;

const INK = "#1c2230";
const IMP = ["#4dc7ff", "#007aff", "#5759f5"];
const BASE_W = 2.7;
const RULE_GAP = 72;
let ruleTop = 150;      // first ruled line: below the toolbar, which wraps onto two rows on a phone

const settings = {
  neatness: 90,         // measured: cleaner fallback style, no worse copies of yours (tests/eval_pipeline.py)
  style: "mine",
  spelling: true,
};

// Which parts of the reading Magic uses (the ⚙ settings). All on by default:
// each covers for the others' mistakes. Saved on this device.
const READING_ALL_ON = { passages: true, mlkit: true, gemma: true, gemma_sees_ink: true, proofread: true };
const reading = { ...READING_ALL_ON };
try { Object.assign(reading, JSON.parse(localStorage.getItem("hw.reading") || "{}")); } catch {}

// The 13 writers the synthesiser learned from (IAM-OnDB), named for how their
// writing looks. The number is the server's style id; 9 is the clean fallback.
const STYLE_GROUPS = [
  ["Print", [[1, "Clean print"], [5, "Big clear print"], [9, "Tall narrow print"], [0, "Casual print"],
    [3, "Wide round print"], [6, "Airy spaced print"], [8, "Bold rounded print"], [4, "Small italic print"],
    [12, "Loopy upright print"]]],
  ["Joined-up", [[7, "Print with a hint of script"], [10, "Bouncy joined script"], [11, "Quick slanted script"],
    [2, "Flowing cursive"]]],
];
const STYLE_NAME = Object.fromEntries(STYLE_GROUPS.flatMap(([, styles]) => styles));

let paths = [];         // everything drawn: {id, kind:'user'|'synth', pts, orig, alpha, reveal, live, state, hidden}
let history = [];
let particles = [];
let tweens = [];
let comparing = false;
let nextId = 1;
let fixing = 0;         // lines sent to be rewritten and not back yet
let serverInfo = { loading: true, reader: null, synth: false, error: null };
let penSeen = false;

// The pen: saved on this device, so it's the same next time.
const INKS = [["Black", INK], ["Blue", "#1f56c9"], ["Red", "#c8322a"], ["Green", "#1d7f4a"]];
const pen = { tool: "pen", color: INK, size: 1, pressure: true };
try { Object.assign(pen, JSON.parse(localStorage.getItem("hw.pen") || "{}")); } catch {}
pen.tool = "pen";
const savePen = () => { try { localStorage.setItem("hw.pen", JSON.stringify({ ...pen, tool: undefined })); } catch {} };

// Settled ink is painted once onto an offscreen layer. While you write, a frame
// only copies that layer and draws the stroke under the pen, so a full page is
// as quick to write on as an empty one.
const layer = document.createElement("canvas");
const lctx = layer.getContext("2d");
let layerDirty = true;

function resize() {
  DPR = window.devicePixelRatio || 1;
  W = window.innerWidth; H = window.innerHeight;
  for (const [c, x] of [[canvas, ctx], [layer, lctx]]) {
    c.width = Math.round(W * DPR); c.height = Math.round(H * DPR);
    x.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  ruleTop = Math.max(150, Math.round($("topbar").getBoundingClientRect().bottom + 56));
  requestRender();
}
window.addEventListener("resize", resize);

// ---------------------------------------------------------------------------
// rendering
let renderQueued = false;
// layerChanged = false: only the stroke under the pen moved
function requestRender(layerChanged = true) {
  if (layerChanged) layerDirty = true;
  if (!renderQueued) { renderQueued = true; requestAnimationFrame(frame); }
}

function frame(now) {
  renderQueued = false;
  tweens = tweens.filter((tw) => {
    const k = Math.min(1, (now - tw.start) / tw.dur);
    if (k < 0) return true;
    tw.update(tw.ease ? tw.ease(k) : k, now);
    if (k >= 1) { tw.done && tw.done(); return false; }
    return true;
  });
  particles = particles.filter((p) => {
    const dt = 1 / 60;
    p.x += p.vx * dt; p.y += p.vy * dt; p.vy -= 18 * dt; p.vx *= 0.985;
    p.life -= dt;
    return p.life > 0;
  });
  draw(now);
  if (tweens.length || particles.length) requestRender(false);
}

function drawPaper(c) {
  c.fillStyle = "#fbfaf6";
  c.fillRect(0, 0, W, H);
  c.strokeStyle = "rgba(60,110,200,0.10)";
  c.lineWidth = 1;
  c.beginPath();
  for (let y = ruleTop; y < H; y += RULE_GAP) { c.moveTo(0, y + 0.5); c.lineTo(W, y + 0.5); }
  c.stroke();
}

// Variable-width polyline; `upto` = arc length to reveal (for the write-on effect).
function strokePts(c, pts, upto = Infinity, cum = null) {
  if (!pts.length) return null;
  if (pts.length === 1 || (cum && cum[cum.length - 1] < 0.5)) {
    const p = pts[0];
    if (upto <= 0) return null;
    c.beginPath(); c.arc(p.x, p.y, (p.w || BASE_W) / 2, 0, Math.PI * 2); c.fill();
    return p;
  }
  let curW = -1, head = pts[0];
  c.beginPath();
  for (let i = 1; i < pts.length; i++) {
    let a = pts[i - 1], b = pts[i];
    if (cum && cum[i] > upto) {
      const seg = cum[i] - cum[i - 1];
      const t = seg > 0 ? (upto - cum[i - 1]) / seg : 0;
      if (t <= 0) break;
      b = { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, w: b.w };
    }
    const w = Math.round((((a.w || BASE_W) + (b.w || BASE_W)) / 2) * 4) / 4;
    if (w !== curW) {
      if (curW > 0) c.stroke();
      c.beginPath(); c.lineWidth = w; curW = w; c.moveTo(a.x, a.y);
    }
    c.lineTo(b.x, b.y);
    head = b;
    if (cum && cum[i] > upto) break;
  }
  c.stroke();
  return head;
}

function draw(now) {
  // Only what is changing is drawn each frame (the stroke under the pen, ink
  // dissolving or writing itself in); everything else is copied from the layer.
  const live = paths.filter((p) => p.live);
  if (active && active.path) live.push(active.path);
  if (layerDirty) { drawPage(lctx, now, new Set(live)); layerDirty = false; }
  ctx.globalAlpha = 1; ctx.shadowBlur = 0;
  ctx.drawImage(layer, 0, 0, W, H);
  for (const p of live) drawPath(ctx, p, now);
  drawParticles(ctx);
  if (active && active.eraser && active.at) drawEraser(ctx, active.at);
}

function drawPage(c, now, skip) {
  drawPaper(c);
  for (const p of paths) if (!skip.has(p)) drawPath(c, p, now);
  c.globalAlpha = 1;
}

function drawPath(c, p, now) {
  c.lineCap = "round"; c.lineJoin = "round";
  if (p.kind === "synth") {
    if (comparing || p.hidden) return;
    c.strokeStyle = c.fillStyle = p.color || INK;
    c.globalAlpha = p.alpha;
    const head = strokePts(c, p.pts, p.reveal, p.cum);
    if (head && p.reveal < p.len && p.reveal > 0) drawNib(c, head);
    c.globalAlpha = 1;
    return;
  }
  const pts = comparing && p.orig ? p.orig : p.pts;
  const alpha = comparing ? 1 : p.alpha;
  if ((p.hidden && !comparing) || alpha <= 0.01) return;
  c.globalAlpha = alpha;
  c.strokeStyle = c.fillStyle = (!comparing && p.tint) || p.color || INK;
  c.shadowBlur = 0;
  strokePts(c, pts);
  c.globalAlpha = 1;
}

function drawParticles(c) {
  for (const q of particles) {
    c.globalAlpha = Math.max(0, q.life / q.max);
    c.fillStyle = q.color;
    c.beginPath(); c.arc(q.x, q.y, q.r * (0.5 + q.life / q.max), 0, Math.PI * 2); c.fill();
  }
  c.globalAlpha = 1;
}

function drawNib(c, p) {
  const g = c.createRadialGradient(p.x, p.y, 0, p.x, p.y, 14);
  g.addColorStop(0, "rgba(77,199,255,0.95)");
  g.addColorStop(0.35, "rgba(0,122,255,0.45)");
  g.addColorStop(1, "rgba(87,89,245,0)");
  c.save(); c.globalAlpha = 1; c.fillStyle = g;
  c.beginPath(); c.arc(p.x, p.y, 14, 0, Math.PI * 2); c.fill(); c.restore();
}

function drawEraser(c, p) {
  c.save(); c.globalAlpha = 1; c.lineWidth = 1.5; c.strokeStyle = "rgba(28,34,48,0.45)"; c.fillStyle = "rgba(255,255,255,0.5)";
  c.beginPath(); c.arc(p.x, p.y, ERASER_R, 0, Math.PI * 2); c.fill(); c.stroke(); c.restore();
}

const easeInOut = (k) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);
const easeOut = (k) => 1 - Math.pow(1 - k, 3);
function tween(dur, update, { delay = 0, ease = null, done = null } = {}) {
  tweens.push({ start: performance.now() + delay, dur, update, ease, done });
  requestRender();
}

function arcLengths(pts) {
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y));
  return cum;
}

function sparkle(pts, count = 1, spread = 1) {
  for (let i = 0; i < count; i++) {
    const p = pts[Math.floor(Math.random() * pts.length)];
    particles.push({
      x: p.x, y: p.y,
      vx: (Math.random() - 0.5) * 70 * spread, vy: -(20 + Math.random() * 70) * spread,
      r: 1 + Math.random() * 2.2, color: IMP[Math.floor(Math.random() * 3)],
      life: 0.5 + Math.random() * 0.6, max: 1.1,
    });
  }
}

// ---------------------------------------------------------------------------
// input: raw pointer events -> Ink Stroke Modeler -> variable-width ink
let active = null;          // the stroke (or eraser swipe) under the pen
const ERASER_R = 12;

function widthFor(pt, prev, type) {
  const base = BASE_W * pen.size;
  if (type === "pen" && pen.pressure) return base * (0.35 + 1.25 * (pt.p || 0.5));
  // mouse / touch: a little thinner when moving fast, like a real pen
  if (!prev) return base;
  const v = Math.hypot(pt.x - prev.x, pt.y - prev.y) / Math.max(1e-3, pt.t - prev.t);
  const target = base * Math.min(1.2, Math.max(0.72, 1.2 - v / 2600));
  return prev.w + (target - prev.w) * 0.25;
}

function pushModelled(out) {
  for (const m of out) {
    const prev = active.path.pts[active.path.pts.length - 1];
    const pt = { x: m.x, y: m.y, p: m.p, t: m.t };
    pt.w = widthFor(pt, prev, active.type);
    active.path.pts.push(pt);
  }
}

// iPad: without this, WebKit reads a quick lift-and-touch of the Pencil as a
// tap, double-tap or Scribble gesture and swallows or cancels the next stroke.
// touch-action: none alone doesn't stop those; cancelling the touches does
// (pointer events still arrive).
const noGesture = (e) => { if (e.cancelable) e.preventDefault(); };
for (const ev of ["touchstart", "touchmove", "touchend"]) canvas.addEventListener(ev, noGesture, { passive: false });
for (const ev of ["gesturestart", "gesturechange", "dblclick", "contextmenu"]) canvas.addEventListener(ev, noGesture);

canvas.addEventListener("pointerdown", (e) => {
  if (e.pointerType === "mouse" && e.button !== 0) return;
  if (e.pointerType === "pen") penSeen = true;
  if (penSeen && e.pointerType === "touch") return;   // palm rejection
  if (active) {
    if (e.pointerType === "touch" && active.type === "touch") return;   // a second finger
    // the pen after a palm that landed first: the palm's mark goes.
    // Otherwise the last stroke's pointerup never came: finish it now.
    if (e.pointerType === "pen" && active.type === "touch") dropStroke();
    else finishStroke();
  }
  // throws if the pointer is already up (a quick tap); the stroke still counts
  try { canvas.setPointerCapture(e.pointerId); } catch {}
  $("hint").classList.add("gone");
  if (pen.tool === "eraser") {
    active = { id: e.pointerId, type: e.pointerType, eraser: true, removed: [], at: null };
    eraseAt(e.offsetX, e.offsetY);
    return;
  }
  const path = { id: nextId++, kind: "user", pts: [], alpha: 1, state: "pending", color: pen.color, size: pen.size };
  active = { id: e.pointerId, type: e.pointerType, path, modeler: new StrokeModeler() };
  const pressure = e.pointerType === "pen" ? e.pressure : 0.5;
  pushModelled(active.modeler.begin(e.offsetX, e.offsetY, e.timeStamp / 1000, pressure));
  paths.push(path);
  requestRender(false);
});

canvas.addEventListener("pointermove", (e) => {
  if (!active || e.pointerId !== active.id) return;
  const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  for (const ev of evs.length ? evs : [e]) {
    if (active.eraser) { eraseAt(ev.offsetX, ev.offsetY); continue; }
    const pressure = active.type === "pen" ? ev.pressure : 0.5;
    pushModelled(active.modeler.move(ev.offsetX, ev.offsetY, ev.timeStamp / 1000, pressure));
  }
  requestRender(false);
});

function finishStroke() {
  if (active.eraser) {
    if (active.removed.length) history.push({ type: "erase", removed: active.removed });
    active = null;
    requestRender(false);
    return;
  }
  pushModelled(active.modeler.end());
  const path = active.path;
  active = null;
  path.pts = path.pts.filter((p, i, arr) => i === 0 || p.x !== arr[i - 1].x || p.y !== arr[i - 1].y);
  // gentle taper at the stroke ends
  const n = path.pts.length;
  for (let i = 0; i < Math.min(4, n); i++) {
    path.pts[i].w *= 0.7 + 0.075 * i;
    path.pts[n - 1 - i].w *= 0.7 + 0.075 * i;
  }
  history.push({ type: "stroke", path });
  if (!layerDirty) drawPath(lctx, path, performance.now());   // add it to the layer; no full repaint
  requestRender(false);
}

function dropStroke() {
  paths = paths.filter((p) => p !== active.path);
  active = null;
  requestRender();
}

const endStroke = (e) => { if (active && e.pointerId === active.id) finishStroke(); };
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);

// The eraser takes whole strokes: anything it touches, except ink that's
// being rewritten right now.
function eraseAt(x, y) {
  // sweep the whole way from the last sample, so a quick swipe misses nothing
  const from = active.at || { x, y };
  const steps = Math.max(1, Math.ceil(Math.hypot(x - from.x, y - from.y) / (ERASER_R / 2)));
  const spots = Array.from({ length: steps }, (_, s) => ({ x: from.x + ((x - from.x) * (s + 1)) / steps, y: from.y + ((y - from.y) * (s + 1)) / steps }));
  active.at = { x, y };
  let hit = false;
  for (let i = paths.length - 1; i >= 0; i--) {
    const p = paths[i];
    if (p.hidden || p.state === "busy" || (p.kind === "synth" && p.reveal < p.len)) continue;
    const r = ERASER_R + (p.pts[0]?.w || BASE_W) / 2;
    if (!spots.some((q) => touches(p.pts, q.x, q.y, r))) continue;
    active.removed.push({ path: p, index: i });
    paths.splice(i, 1);
    hit = true;
  }
  if (hit) requestRender();
}

function touches(pts, x, y, r) {
  const r2 = r * r;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[i + 1] || a;
    const dx = b.x - a.x, dy = b.y - a.y;
    const len2 = dx * dx + dy * dy;
    const t = len2 ? clamp(((x - a.x) * dx + (y - a.y) * dy) / len2, 0, 1) : 0;
    const ex = a.x + t * dx - x, ey = a.y + t * dy - y;
    if (ex * ex + ey * ey <= r2) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// the magic: only when asked (the ✨ Magic button or Enter), never while
// writing. A timer that fired on every pause used to grab half a sentence
// when you stopped to think, and rewrite it under your pen.
async function runMagic() {
  if (active) return;       // mid-stroke
  const pending = paths.filter((p) => p.kind === "user" && p.state === "pending");
  if (!pending.length) {
    if (!fixing) card("Write something first, then press ✨ Magic.", 3500);
    return;
  }
  if (!(serverInfo.synth && !serverInfo.loading && serverInfo.reader)) {
    card(serverInfo.loading ? `Magic isn't ready yet (${esc(serverInfo.phase || "loading")}). Try again when the dot turns green.`
      : `Magic is unavailable (${esc(serverInfo.error || "no handwriting reader")}).`, 5000, "err");
    return;
  }
  // Each block of lines written together is read as one piece of writing
  // (so a word can be worked out from the lines around it), and rewritten
  // line by line in place. Writing elsewhere on the page is its own block.
  const lines = groupLines(pending);
  const blocks = reading.passages ? groupBlocks(pending, lines, RULE_GAP) : lines.map((idx) => [idx]);
  const groups = blocks.map((block) => {
    const group = { id: nextId++, paths: block.flat().map((i) => pending[i]) };
    group.paths.forEach((p) => { p.state = "busy"; p.orig = p.pts.map((q) => ({ ...q })); p.group = group; });
    group.lines = block.map((idx) => {
      const paths = idx.map((i) => pending[i]);
      // the rewrite copies the style of a tidied version of your ink
      return { paths, tidy: tidyLine(paths.map((p) => p.orig), { strength: 0.45 + 0.55 * (settings.neatness / 100) }) };
    });
    return group;
  });
  fixing += groups.length;
  updateMagicButton();
  const results = await Promise.all(groups.map(applyMagic));
  fixing -= groups.length;
  updateMagicButton();
  // a stray mark reads as nothing; only say so when nothing on the page had words
  if (results.every((r) => r === "empty")) card("I couldn't find any words in that, so I left it as you wrote it.", 4000);
}

function updateMagicButton() {
  $("magic").classList.toggle("busy", fixing > 0);
  $("magic").textContent = fixing > 0 ? "✨ Fixing…" : "✨ Magic";
}

async function applyMagic(group) {
  group.mode = "magic";
  startShimmer(group);
  const xy = (pts) => pts.map((q) => [Math.round(q.x * 10) / 10, Math.round(q.y * 10) / 10]);
  const body = {
    lines: group.lines.map((L) => {
      // when each point was drawn (s): a stroke recogniser (ML Kit, on the phone) reads the pen's movement
      const t0 = Math.min(...L.paths.map((p) => p.orig[0]?.t ?? Infinity));
      const times = L.paths.map((p) => p.orig.map((q) => Math.round(((q.t ?? t0) - t0) * 1000) / 1000));
      return { strokes: L.paths.map((p) => xy(p.orig)), times, prime: L.tidy.strokes.map(xy) };
    }),
    style: settings.style, bias: 0.3 + (settings.neatness / 100) * 2.2,
    fix_spelling: settings.spelling, candidates: 8,   // each one proofread; 8 costs ~0.6 s on an M5
    reading, verify: reading.proofread,               // verify: the Mac server's name for proofreading
  };
  let res;
  try {
    const r = await fetch("/api/rewrite", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    res = await r.json();
    if (!r.ok) throw new Error(res.error || r.statusText);
  } catch (err) {
    // left as written, and still pending: press Magic to try again
    stopShimmer(group);
    group.paths.forEach((p) => (p.state = "pending"));
    card(`Couldn't rewrite that (${esc(err.message)}). Press ✨ Magic to try again.`, 5000, "err");
    return "error";
  }
  stopShimmer(group);
  if (group.paths.some((p) => p.state === "reverted")) return "undone";   // undone while we waited
  // line by line, top to bottom; a line whose rewrite wrapped pushes the ones under it down
  let placed = [], rewritten = [], minBase = -Infinity;
  group.lines.forEach((L, k) => {
    const line = res.lines[k];
    if (!line || !line.strokes.length) {           // a doodle: left as it is
      L.paths.forEach((p) => (p.state = "done"));
      return;
    }
    const out = placeSynth(line.strokes, L, line, minBase);
    placed = placed.concat(out.strokes);
    rewritten = rewritten.concat(L.paths);
    minBase = out.lastBase + out.lineH * 0.8;
  });
  if (!placed.length) return "empty";
  revealMagic(group, rewritten, placed);
  showResultCard(res.lines.filter((l) => l.strokes.length));
  history.push({ type: "fix", group });
  return "fixed";
}

// Scale/position the normalised synthesised ink onto the page, wrapping words
// onto new lines when they would run off the right edge. `written`: the line
// it replaces ({paths, tidy}). `minBase`: the
// highest baseline allowed (below the line before, if that one wrapped).
function placeSynth(strokes, written, line, minBase = -Infinity) {
  const metrics = written.tidy.metrics;
  const S = strokes.map((s) => s.map(([x, y]) => ({ x, y })));
  const m = coreMetrics(S.flat(), 1);
  // Size: give the rewrite the x-height of what was written. Measured directly,
  // the x-height of messy ink is often off by a quarter (a wavy baseline smears
  // it), so it is averaged with an estimate from the width per letter: in
  // handwriting a letter plus its share of the spaces is ~1.25 x-heights wide.
  // (Matching the width alone would blow up narrow styles like Style 10.)
  const ink = written.paths.flatMap((p) => p.orig);
  const inkCore = coreMetrics(ink).core;
  const perLetter = bbox(ink).w / Math.max(1, (line.written || line.text || "").length);
  const xh = clamp(Math.sqrt(inkCore * perLetter / 1.25), inkCore * 0.7, inkCore * 1.4);
  const k = xh / m.core;
  const scaled = S.map((s) => s.map((p) => ({ x: p.x * k, y: (p.y - m.base) * k })));
  const words = splitWords(scaled, 0.5 * xh);
  const right = W - 40;
  const lineH = Math.max(RULE_GAP, xh * 3.4);
  let cursorX = metrics.x0, y0 = Math.max(metrics.baseline, minBase), prevMax = null, out = [];
  const minX0 = Math.min(...scaled.flat().map((p) => p.x));
  words.forEach((idx, wi) => {
    const pts = idx.flatMap((i) => scaled[i]);
    const b = bbox(pts);
    let gap = prevMax === null ? 0 : b.minX - prevMax;
    let dx = (wi === 0 ? metrics.x0 - minX0 : cursorX + gap - b.minX);
    if (wi > 0 && b.maxX + dx > right) { y0 += lineH; dx = metrics.x0 - b.minX; }
    for (const i of idx) out.push({ order: b.minX, pts: scaled[i].map((p) => ({ x: p.x + dx, y: p.y + y0 })) });
    cursorX = b.maxX + dx; prevMax = b.maxX;
  });
  // keep the pen order (left to right, roughly as written)
  const widthScale = clamp(xh / 22, 0.8, 1.5) * (written.paths.reduce((sum, p) => sum + (p.size || 1), 0) / written.paths.length);
  const placed = out.map((o) => {
    const n = o.pts.length;
    o.pts.forEach((p, i) => {
      const edge = Math.min(i, n - 1 - i);
      p.w = BASE_W * widthScale * (edge < 3 ? 0.72 + 0.09 * edge : 1);
    });
    return o.pts;
  });
  return { strokes: placed, lastBase: y0, lineH };
}

function revealMagic(group, rewritten, placed) {
  // 1) the scrawl dissolves into sparkles
  rewritten.forEach((p) => {
    const from = p.orig;
    const count = Math.min(40, Math.ceil(arcLengths(from).at(-1) / 14));
    sparkle(from, count, 1);
    p.live = true;
    tween(520, (k) => { p.alpha = 1 - k; p.tint = k > 0.05 ? IMP[1] : null; }, {
      ease: easeOut, done: () => { p.hidden = true; p.state = "done"; p.live = false; requestRender(); },
    });
  });
  // 2) the neat version writes itself in, like an invisible pen
  const synth = placed.map((pts) => {
    const cum = arcLengths(pts);
    return { id: nextId++, kind: "synth", pts, cum, len: cum.at(-1), alpha: 1, reveal: 0, live: true, group, color: group.paths[0].color };
  });
  let t = 0;
  const starts = synth.map((s) => { const st = t; t += s.len + 25; return st; });
  const total = t;
  const speed = Math.max(900, total / 1.5);
  synth.forEach((s) => paths.push(s));
  group.synth = synth;
  tween((total / speed) * 1000, (k) => {
    const head = k * total;
    synth.forEach((s, i) => (s.reveal = Math.max(0, head - starts[i])));
  }, { delay: 180, done: () => { synth.forEach((s) => { s.reveal = Infinity; s.live = false; }); requestRender(); } });
}

// While Magic works, a band of colour sweeps over the ink. It is a CSS
// animation of a layer masked to the ink, so the system compositor runs it on
// its own: the page draws nothing per frame, and it stays smooth while the
// models keep the processor and the graphics chip busy.
function startShimmer(group) {
  const bb = bbox(group.paths.flatMap((p) => p.pts));
  const pad = 16;
  const x = bb.minX - pad, y = bb.minY - pad, w = bb.w + 2 * pad, h = bb.h + 2 * pad;
  const mask = document.createElement("canvas");
  mask.width = Math.ceil(w * DPR); mask.height = Math.ceil(h * DPR);
  const g = mask.getContext("2d");
  g.setTransform(DPR, 0, 0, DPR, -x * DPR, -y * DPR);
  g.lineCap = "round"; g.lineJoin = "round";
  g.strokeStyle = g.fillStyle = "#000";
  // a soft glow around the ink, then the ink itself (drawn once, here)
  g.shadowColor = "rgba(0,0,0,0.5)"; g.shadowBlur = 12 * DPR;
  group.paths.forEach((p) => strokePts(g, p.pts));
  g.shadowBlur = 0;
  group.paths.forEach((p) => strokePts(g, p.pts));
  const el = document.createElement("div");
  el.className = "shimmer";
  Object.assign(el.style, { left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px` });
  el.appendChild(document.createElement("i"));
  group.shimmer = el;
  mask.toBlob(async (blob) => {
    if (group.shimmer !== el || !blob) return;       // already done
    el.maskURL = URL.createObjectURL(blob);
    const img = new Image();
    img.src = el.maskURL;
    try { await img.decode(); } catch { return; }
    if (group.shimmer !== el) return;
    el.style.webkitMaskImage = el.style.maskImage = `url(${el.maskURL})`;
    document.body.insertBefore(el, $("topbar"));
    requestAnimationFrame(() => el.classList.add("on"));
  });
}

function stopShimmer(group) {
  const el = group.shimmer;
  if (!el) return;
  group.shimmer = null;
  el.classList.remove("on");
  setTimeout(() => { el.remove(); if (el.maskURL) URL.revokeObjectURL(el.maskURL); }, 300);
}

// ---------------------------------------------------------------------------
// UI
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function card(html, ttl = 6000, cls = "") {
  const el = document.createElement("div");
  el.className = `card ${cls}`;
  el.innerHTML = html;
  const box = $("transcript");
  box.appendChild(el);
  while (box.children.length > 3) box.firstChild.remove();
  setTimeout(() => { el.classList.add("fade"); setTimeout(() => el.remove(), 650); }, ttl);
}

function showResultCard(lines) {
  const fixes = lines.flatMap((l) => l.corrections || []).filter(([a, b]) => a.toLowerCase() !== b.toLowerCase());
  // the server only keeps "my handwriting" when the copy reads back as well as a clean style
  const other = lines.find((l) => l.style_used !== "mine");
  const note = settings.style === "mine" && other
    ? `<div class="note">Your handwriting was hard to copy neatly ${lines.length > 1 ? "in places" : "here"}, so I used ${STYLE_NAME[other.style_used] ? `the “${STYLE_NAME[other.style_used]}” style` : "a clean style"}${lines.length > 1 ? " there" : ""}.</div>` : "";
  if (fixes.length) {
    const chips = fixes.map(([a, b]) => `<del>${esc(a || "∅")}</del> → <ins>${esc(b || "∅")}</ins>`).join("&nbsp;&nbsp; ");
    card(`<span class="label">Fixed</span>${chips}${note}`, 7000);
  } else {
    card(`<span class="label">Read</span>${esc(lines.map((l) => l.text).join(" "))} &nbsp;<span class="ok">✓</span>${note}`, 5000);
  }
}

function undo() {
  const h = history.pop();
  if (!h) return;
  if (h.type === "stroke") {
    paths = paths.filter((p) => p !== h.path);
  } else if (h.type === "erase") {
    for (const { path, index } of h.removed.slice().reverse()) paths.splice(Math.min(index, paths.length), 0, path);
  } else {
    const g = h.group;
    if (g.synth) paths = paths.filter((p) => !g.synth.includes(p));
    g.paths.forEach((p) => {
      p.pts = p.orig; p.alpha = 1; p.hidden = false; p.tint = null; p.state = "reverted";
    });
  }
  requestRender();
}

function clearAll() {
  if (active) finishStroke();
  for (const p of paths) if (p.group) stopShimmer(p.group);
  paths = []; history = []; particles = []; tweens = [];
  $("transcript").innerHTML = "";
  requestRender();
}

$("magic").addEventListener("click", runMagic);
$("neatness").addEventListener("input", (e) => (settings.neatness = +e.target.value));
$("spelling").addEventListener("change", (e) => (settings.spelling = e.target.checked));
$("style").addEventListener("change", (e) => (settings.style = e.target.value === "mine" ? "mine" : +e.target.value));
for (const [group, styles] of STYLE_GROUPS) {
  const options = styles.map(([id, name]) => `<option value="${id}">${name}</option>`).join("");
  $("style").insertAdjacentHTML("beforeend", `<optgroup label="${group}">${options}</optgroup>`);
}

// pen toolbar
$("colors").innerHTML = INKS.map(([name, c]) =>
  `<button class="swatch" data-color="${c}" title="${name} ink" aria-label="${name} ink" style="--c:${c}"></button>`).join("");
function updatePenBar() {
  $("toolPen").classList.toggle("active", pen.tool === "pen");
  $("toolEraser").classList.toggle("active", pen.tool === "eraser");
  document.querySelectorAll(".swatch").forEach((b) => b.classList.toggle("active", pen.tool === "pen" && b.dataset.color === pen.color));
  $("size").value = pen.size;
  $("pressure").checked = pen.pressure;
  const d = Math.max(3, BASE_W * pen.size * 1.5);
  Object.assign($("sizeDot").style, { width: `${d}px`, height: `${d}px`, background: pen.color });
  canvas.classList.toggle("erasing", pen.tool === "eraser");
}
const setPen = (change) => { Object.assign(pen, change); savePen(); updatePenBar(); };
$("toolPen").addEventListener("click", () => setPen({ tool: "pen" }));
$("toolEraser").addEventListener("click", () => setPen({ tool: pen.tool === "eraser" ? "pen" : "eraser" }));
$("colors").addEventListener("click", (e) => { const c = e.target.dataset?.color; if (c) setPen({ tool: "pen", color: c }); });
$("size").addEventListener("input", (e) => setPen({ tool: "pen", size: +e.target.value }));
$("pressure").addEventListener("change", (e) => setPen({ pressure: e.target.checked }));
updatePenBar();

// settings sheet: a switch per part, with whether it works on this device
const READING_OPTS = [
  ["passages", "Read lines written together as one passage",
    "Lines under each other are read together, so each word is read knowing the lines around it. Off: every line on its own."],
  ["mlkit", "Google ML Kit reads your pen strokes",
    "It follows where the pen went, in what order: the best reader of messy writing. Off: Apple Vision reads a picture of the ink instead."],
  ["gemma", "Gemma works out what you meant",
    "The AI model fixes misspellings and mix-ups like their/there, using the whole passage. Off: ML Kit's reading is used as it is, with the system spellchecker when Fix spelling is on."],
  ["gemma_sees_ink", "Gemma looks at your ink too",
    "It checks each word against a picture of your writing, which catches letters ML Kit misread. Off: it only sees ML Kit's reading (faster).", "gemma"],
  ["proofread", "Proofread the rewrite",
    "Apple Vision reads back each neat version and keeps the clearest. Off: faster, but the odd rewrite can come out hard to read."],
];

async function openSettings() {
  $("settings").hidden = false;
  renderSettings();
  try { serverInfo = await (await fetch("/api/status")).json(); } catch {}
  renderSettings();
}

function renderSettings() {
  const parts = serverInfo.parts;
  $("readingOpts").innerHTML = READING_OPTS.map(([key, name, desc, parent]) => {
    const part = key === "passages" ? { ok: true, note: "" } : parts?.[key];
    const avail = key === "passages" || (part ? part.ok : false);
    const parentOff = parent && !reading[parent];
    const state = key === "passages" ? ""
      : serverInfo.loading ? `<div class="state">Loading…</div>`
      : !part ? `<div class="state no">Only in the iPhone and iPad app</div>`
      : `<div class="state ${part.ok ? "ok" : "no"}">${esc(part.note)}</div>`;
    return `<label class="opt${parent ? " sub" : ""}${parentOff ? " off" : ""}">
      <span class="txt"><div class="name">${name}</div><div class="desc">${desc}</div>${state}</span>
      <input type="checkbox" class="switch" data-key="${key}" ${reading[key] && (avail || serverInfo.loading) ? "checked" : ""} ${parentOff || (!avail && !serverInfo.loading && key !== "passages") ? "disabled" : ""}>
    </label>`;
  }).join("");
}

const saveReading = () => { try { localStorage.setItem("hw.reading", JSON.stringify(reading)); } catch {} };
$("readingOpts").addEventListener("change", (e) => {
  const key = e.target.dataset?.key;
  if (!key) return;
  reading[key] = e.target.checked;
  saveReading();
  renderSettings();
  showStatus();
});
$("allOn").addEventListener("click", () => { Object.assign(reading, READING_ALL_ON); saveReading(); renderSettings(); showStatus(); });
$("settingsBtn").addEventListener("click", openSettings);
$("settingsClose").addEventListener("click", () => ($("settings").hidden = true));
$("settings").addEventListener("click", (e) => { if (e.target === $("settings")) $("settings").hidden = true; });

const setCompare = (on) => { comparing = on; $("compare").classList.toggle("active", on); requestRender(); };
$("compare").addEventListener("pointerdown", () => setCompare(true));
["pointerup", "pointerleave", "pointercancel"].forEach((ev) => $("compare").addEventListener(ev, () => setCompare(false)));
$("undo").addEventListener("click", undo);
$("clear").addEventListener("click", clearAll);
window.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
  if (e.target === document.body && !e.metaKey && !e.ctrlKey) {
    if (e.key === "p") setPen({ tool: "pen" });
    if (e.key === "e") setPen({ tool: "eraser" });
  }
  if (e.code === "Space" && !e.repeat && e.target === document.body) { e.preventDefault(); setCompare(true); }
  if (e.key === "Escape") $("settings").hidden = true;
  if (e.key === "Enter" && $("settings").hidden) runMagic();
});
window.addEventListener("keyup", (e) => { if (e.code === "Space") setCompare(false); });

function showStatus() {
  const st = $("status");
  st.classList.toggle("ready", !serverInfo.loading && !!serverInfo.reader);
  st.classList.toggle("error", !serverInfo.loading && !serverInfo.reader);
  const pct = serverInfo.progress != null ? ` (${Math.round(serverInfo.progress * 100)}%)` : "";
  $("statusText").textContent = serverInfo.loading ? `${serverInfo.phase || "Loading models"}…${pct}`
    : serverInfo.reader ? `Ready · ${serverInfo.reader}${switchedOff()}`
    : serverInfo.error ? `Magic unavailable: ${serverInfo.error}` : "Reader off (--reader none)";
}

// the parts that work here but are switched off in the settings
function switchedOff() {
  const off = READING_OPTS.filter(([k]) => k !== "passages" && !reading[k] && serverInfo.parts?.[k]?.ok).map(([k]) => ({
    mlkit: "ML Kit", gemma: "Gemma", gemma_sees_ink: "Gemma's vision", proofread: "proofreading" }[k]));
  return off.length ? ` (off in settings: ${off.join(", ")})` : "";
}

async function pollStatus() {
  let reachable = true;
  try {
    const r = await fetch("/api/status");
    serverInfo = await r.json();
  } catch {
    reachable = false;
    serverInfo = { loading: false, error: "the demo server isn't running (start it again with the launcher)" };
  }
  showStatus();
  if (serverInfo.loading || !reachable) setTimeout(pollStatus, reachable ? 1500 : 3000);
}

// hooks for automated tests
window.__hw = {
  settings, get paths() { return paths; }, runMagic,
  get busy() { return paths.some((p) => p.state === "busy") || tweens.length > 0; },
};

if (matchMedia("(pointer: coarse)").matches) {
  $("hint").textContent = "Write something with your finger or an Apple Pencil, then tap ✨ Magic.";
}
resize();
pollStatus();
