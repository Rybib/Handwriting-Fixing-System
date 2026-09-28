import { StrokeModeler } from "./inkmodeler.js";
import { bbox, clamp, coreMetrics, groupLines, splitWords, tidyLine } from "./tidy.js";

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
  mode: "magic",        // magic | tidy | off
  neatness: 90,         // measured: cleaner fallback style, no worse copies of yours (tests/eval_pipeline.py)
  style: "mine",
  spelling: true,
  autoDelay: 1100,      // ms of stillness before the magic happens
};

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

let paths = [];         // everything drawn: {id, kind:'user'|'synth', pts, orig, alpha, reveal, shimmer, state, hidden}
let history = [];
let particles = [];
let tweens = [];
let comparing = false;
let nextId = 1;
let idleTimer = null;
let serverInfo = { loading: true, reader: null, synth: false, error: null };
let penSeen = false;

function resize() {
  DPR = window.devicePixelRatio || 1;
  W = window.innerWidth; H = window.innerHeight;
  canvas.width = Math.round(W * DPR); canvas.height = Math.round(H * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ruleTop = Math.max(150, Math.round($("toolbar").getBoundingClientRect().bottom + 56));
  requestRender();
}
window.addEventListener("resize", resize);

// ---------------------------------------------------------------------------
// rendering
let renderQueued = false;
function requestRender() {
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
  if (tweens.length || particles.length || paths.some((p) => p.shimmer)) requestRender();
}

function drawPaper() {
  ctx.fillStyle = "#fbfaf6";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "rgba(60,110,200,0.10)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let y = ruleTop; y < H; y += RULE_GAP) { ctx.moveTo(0, y + 0.5); ctx.lineTo(W, y + 0.5); }
  ctx.stroke();
}

// Variable-width polyline; `upto` = arc length to reveal (for the write-on effect).
function strokePts(pts, upto = Infinity, cum = null) {
  if (!pts.length) return null;
  if (pts.length === 1 || (cum && cum[cum.length - 1] < 0.5)) {
    const p = pts[0];
    if (upto <= 0) return null;
    ctx.beginPath(); ctx.arc(p.x, p.y, (p.w || BASE_W) / 2, 0, Math.PI * 2); ctx.fill();
    return p;
  }
  let curW = -1, head = pts[0];
  ctx.beginPath();
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
      if (curW > 0) ctx.stroke();
      ctx.beginPath(); ctx.lineWidth = w; curW = w; ctx.moveTo(a.x, a.y);
    }
    ctx.lineTo(b.x, b.y);
    head = b;
    if (cum && cum[i] > upto) break;
  }
  ctx.stroke();
  return head;
}

function draw(now) {
  drawPaper();
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  const shimmerT = (now / 900) % 1;
  for (const p of paths) {
    if (p.kind === "synth") {
      if (comparing || p.hidden) continue;
      ctx.strokeStyle = ctx.fillStyle = INK;
      ctx.globalAlpha = p.alpha;
      const head = strokePts(p.pts, p.reveal, p.cum);
      if (head && p.reveal < p.len && p.reveal > 0) drawNib(head);
      continue;
    }
    const pts = comparing && p.orig ? p.orig : p.pts;
    const alpha = comparing ? 1 : p.alpha;
    if ((p.hidden && !comparing) || alpha <= 0.01) continue;
    ctx.globalAlpha = alpha;
    if (p.shimmer) {
      const bb = p.bbox || (p.bbox = bbox(p.pts));
      const span = Math.max(240, bb.w * 2);
      const x0 = bb.minX - span + shimmerT * (bb.w + span * 2);
      const g = ctx.createLinearGradient(x0, 0, x0 + span, 0);
      g.addColorStop(0, INK); g.addColorStop(0.4, IMP[0]); g.addColorStop(0.55, IMP[1]); g.addColorStop(0.7, IMP[2]); g.addColorStop(1, INK);
      ctx.strokeStyle = ctx.fillStyle = g;
      ctx.shadowColor = "rgba(0,122,255,0.35)"; ctx.shadowBlur = 10;
    } else {
      ctx.strokeStyle = ctx.fillStyle = (!comparing && p.tint) || INK;
      ctx.shadowBlur = 0;
    }
    strokePts(pts);
    ctx.shadowBlur = 0;
  }
  ctx.globalAlpha = 1;
  for (const q of particles) {
    ctx.globalAlpha = Math.max(0, q.life / q.max);
    ctx.fillStyle = q.color;
    ctx.beginPath(); ctx.arc(q.x, q.y, q.r * (0.5 + q.life / q.max), 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawNib(p) {
  const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 14);
  g.addColorStop(0, "rgba(77,199,255,0.95)");
  g.addColorStop(0.35, "rgba(0,122,255,0.45)");
  g.addColorStop(1, "rgba(87,89,245,0)");
  ctx.save(); ctx.globalAlpha = 1; ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(p.x, p.y, 14, 0, Math.PI * 2); ctx.fill(); ctx.restore();
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
let active = null;

function widthFor(pt, prev, pointerType) {
  if (pointerType === "pen") return BASE_W * (0.35 + 1.25 * (pt.p || 0.5));
  // mouse / touch: a little thinner when moving fast, like a real pen
  if (!prev) return BASE_W;
  const v = Math.hypot(pt.x - prev.x, pt.y - prev.y) / Math.max(1e-3, pt.t - prev.t);
  const target = BASE_W * Math.min(1.2, Math.max(0.72, 1.2 - v / 2600));
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

canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 && e.pointerType === "mouse") return;
  if (e.pointerType === "pen") penSeen = true;
  if (penSeen && e.pointerType === "touch") return;   // palm rejection
  canvas.setPointerCapture(e.pointerId);
  clearTimeout(idleTimer);
  $("hint").classList.add("gone");
  const path = { id: nextId++, kind: "user", pts: [], alpha: 1, state: "pending" };
  active = { id: e.pointerId, type: e.pointerType, path, modeler: new StrokeModeler() };
  const pressure = e.pointerType === "pen" ? e.pressure : 0.5;
  pushModelled(active.modeler.begin(e.offsetX, e.offsetY, e.timeStamp / 1000, pressure));
  paths.push(path);
  requestRender();
});

canvas.addEventListener("pointermove", (e) => {
  if (!active || e.pointerId !== active.id) return;
  const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  for (const ev of evs.length ? evs : [e]) {
    const pressure = active.type === "pen" ? ev.pressure : 0.5;
    pushModelled(active.modeler.move(ev.offsetX, ev.offsetY, ev.timeStamp / 1000, pressure));
  }
  requestRender();
});

function endStroke(e) {
  if (!active || e.pointerId !== active.id) return;
  pushModelled(active.modeler.end());
  const path = active.path;
  path.pts = path.pts.filter((p, i, a) => i === 0 || p.x !== a[i - 1].x || p.y !== a[i - 1].y);
  // gentle taper at the stroke ends
  const n = path.pts.length;
  for (let i = 0; i < Math.min(4, n); i++) {
    path.pts[i].w *= 0.7 + 0.075 * i;
    path.pts[n - 1 - i].w *= 0.7 + 0.075 * i;
  }
  history.push({ type: "stroke", path });
  active = null;
  requestRender();
  scheduleMagic();
}
canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);

function scheduleMagic() {
  clearTimeout(idleTimer);
  if (settings.mode === "off") return;
  idleTimer = setTimeout(processPending, settings.autoDelay);
}

// ---------------------------------------------------------------------------
// the magic
function processPending() {
  if (active) return scheduleMagic();
  const pending = paths.filter((p) => p.kind === "user" && p.state === "pending");
  if (!pending.length) return;
  const lines = groupLines(pending);
  for (const idx of lines) {
    const group = { id: nextId++, paths: idx.map((i) => pending[i]) };
    group.paths.forEach((p) => { p.state = "busy"; p.orig = p.pts.map((q) => ({ ...q })); p.group = group; });
    const strength = 0.45 + 0.55 * (settings.neatness / 100);
    const tidy = tidyLine(group.paths.map((p) => p.orig), { strength });
    group.tidy = tidy;
    const canMagic = settings.mode === "magic" && serverInfo.synth && !serverInfo.loading && serverInfo.reader;
    if (!canMagic) {
      if (settings.mode === "magic") {
        card(serverInfo.loading ? `Magic isn't ready yet (${esc(serverInfo.phase || "loading")}), so I just tidied this. It will kick in once the status turns green.`
          : `Handwriting reader unavailable (${serverInfo.error || "no reader"}), so I tidied instead.`, 5000, "err");
      }
      applyTidy(group);
    } else {
      applyMagic(group);
    }
  }
}

function applyTidy(group) {
  group.mode = "tidy";
  const targets = group.tidy.strokes;
  const words = group.paths.map((p) => bbox(p.orig).minX);
  const minX = Math.min(...words), span = Math.max(1, Math.max(...words) - minX);
  group.paths.forEach((p, i) => {
    const from = p.orig, to = targets[i];
    const delay = ((words[i] - minX) / span) * 260;
    tween(700, (k) => {
      p.pts = from.map((a, j) => ({ ...a, x: a.x + (to[j].x - a.x) * k, y: a.y + (to[j].y - a.y) * k }));
      p.bbox = null;
    }, { delay, ease: easeInOut, done: () => { p.state = "done"; sparkle(p.pts, 4, 0.6); } });
  });
  history.push({ type: "fix", group });
}

async function applyMagic(group) {
  group.mode = "magic";
  group.paths.forEach((p) => { p.shimmer = true; p.bbox = null; });
  requestRender();
  const xy = (pts) => pts.map((q) => [Math.round(q.x * 10) / 10, Math.round(q.y * 10) / 10]);
  const body = {
    lines: [{ strokes: group.paths.map((p) => xy(p.orig)), prime: group.tidy.strokes.map(xy) }],
    style: settings.style, bias: 0.3 + (settings.neatness / 100) * 2.2,
    fix_spelling: settings.spelling, candidates: 8,   // each one proofread; 8 costs ~0.6 s on an M5
  };
  let res;
  try {
    const r = await fetch("/api/rewrite", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    res = await r.json();
    if (!r.ok) throw new Error(res.error || r.statusText);
  } catch (err) {
    group.paths.forEach((p) => (p.shimmer = false));
    card(`Couldn't rewrite that (${err.message}). Tidied it instead.`, 5000, "err");
    applyTidy(group);
    return;
  }
  group.paths.forEach((p) => (p.shimmer = false));
  const line = res.lines[0];
  if (group.paths.some((p) => p.state === "reverted")) return;   // undone while we waited
  if (!line.strokes.length) {
    card("I couldn't find any words in that, so I left it as you drew it.", 4000);
    group.paths.forEach((p) => (p.state = "done"));
    return;
  }
  const placed = placeSynth(line.strokes, group, line);
  revealMagic(group, placed);
  showResultCard(line);
  history.push({ type: "fix", group });
}

// Scale/position the normalised synthesised ink onto the page, wrapping words
// onto new lines when they would run off the right edge.
function placeSynth(strokes, group, line) {
  const metrics = group.tidy.metrics;
  const S = strokes.map((s) => s.map(([x, y]) => ({ x, y })));
  const m = coreMetrics(S.flat(), 1);
  // Size: give the rewrite the x-height of what was written. Measured directly,
  // the x-height of messy ink is often off by a quarter (a wavy baseline smears
  // it), so it is averaged with an estimate from the width per letter: in
  // handwriting a letter plus its share of the spaces is ~1.25 x-heights wide.
  // (Matching the width alone would blow up narrow styles like Style 10.)
  const ink = group.paths.flatMap((p) => p.orig);
  const inkCore = coreMetrics(ink).core;
  const perLetter = bbox(ink).w / Math.max(1, (line.written || line.text || "").length);
  const xh = clamp(Math.sqrt(inkCore * perLetter / 1.25), inkCore * 0.7, inkCore * 1.4);
  const k = xh / m.core;
  const scaled = S.map((s) => s.map((p) => ({ x: p.x * k, y: (p.y - m.base) * k })));
  const words = splitWords(scaled, 0.5 * xh);
  const right = W - 40;
  const lineH = Math.max(RULE_GAP, xh * 3.4);
  let cursorX = metrics.x0, y0 = metrics.baseline, prevMax = null, out = [];
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
  const widthScale = clamp(xh / 22, 0.8, 1.5);
  return out.map((o) => {
    const n = o.pts.length;
    o.pts.forEach((p, i) => {
      const edge = Math.min(i, n - 1 - i);
      p.w = BASE_W * widthScale * (edge < 3 ? 0.72 + 0.09 * edge : 1);
    });
    return o.pts;
  });
}

function revealMagic(group, placed) {
  // 1) the scrawl dissolves into sparkles
  group.paths.forEach((p) => {
    const from = p.orig;
    const count = Math.min(40, Math.ceil(arcLengths(from).at(-1) / 14));
    sparkle(from, count, 1);
    tween(520, (k) => { p.alpha = 1 - k; p.tint = k > 0.05 ? IMP[1] : null; }, {
      ease: easeOut, done: () => { p.hidden = true; p.state = "done"; },
    });
  });
  // 2) the neat version writes itself in, like an invisible pen
  const synth = placed.map((pts) => {
    const cum = arcLengths(pts);
    return { id: nextId++, kind: "synth", pts, cum, len: cum.at(-1), alpha: 1, reveal: 0, group };
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
  }, { delay: 180, done: () => synth.forEach((s) => (s.reveal = Infinity)) });
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

function showResultCard(line) {
  const fixes = (line.corrections || []).filter(([a, b]) => a.toLowerCase() !== b.toLowerCase());
  // the server only keeps "my handwriting" when the copy reads back as well as a clean style
  const note = settings.style === "mine" && line.style_used !== "mine"
    ? `<div class="note">Your handwriting was hard to copy neatly here, so I used ${STYLE_NAME[line.style_used] ? `the “${STYLE_NAME[line.style_used]}” style` : "a clean style"}.</div>` : "";
  if (fixes.length) {
    const chips = fixes.map(([a, b]) => `<del>${esc(a || "∅")}</del> → <ins>${esc(b || "∅")}</ins>`).join("&nbsp;&nbsp; ");
    card(`<span class="label">Fixed</span>${chips}${note}`, 7000);
  } else {
    card(`<span class="label">Read</span>${esc(line.text)} &nbsp;<span class="ok">✓</span>${note}`, 5000);
  }
}

function undo() {
  const h = history.pop();
  if (!h) return;
  if (h.type === "stroke") {
    paths = paths.filter((p) => p !== h.path);
  } else {
    const g = h.group;
    if (g.synth) paths = paths.filter((p) => !g.synth.includes(p));
    g.paths.forEach((p) => {
      p.pts = p.orig; p.alpha = 1; p.hidden = false; p.tint = null; p.bbox = null; p.state = "reverted";
    });
  }
  requestRender();
}

function clearAll() {
  paths = []; history = []; particles = []; tweens = [];
  $("transcript").innerHTML = "";
  requestRender();
}

document.querySelectorAll("#mode button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#mode button").forEach((x) => x.classList.toggle("on", x === b));
  settings.mode = b.dataset.mode;
  $("styleCtl").classList.toggle("disabled", settings.mode !== "magic");
  $("spellCtl").classList.toggle("disabled", settings.mode !== "magic");
}));
$("neatness").addEventListener("input", (e) => (settings.neatness = +e.target.value));
$("spelling").addEventListener("change", (e) => (settings.spelling = e.target.checked));
$("style").addEventListener("change", (e) => (settings.style = e.target.value === "mine" ? "mine" : +e.target.value));
for (const [group, styles] of STYLE_GROUPS) {
  const options = styles.map(([id, name]) => `<option value="${id}">${name}</option>`).join("");
  $("style").insertAdjacentHTML("beforeend", `<optgroup label="${group}">${options}</optgroup>`);
}

const setCompare = (on) => { comparing = on; $("compare").classList.toggle("active", on); requestRender(); };
$("compare").addEventListener("pointerdown", () => setCompare(true));
["pointerup", "pointerleave", "pointercancel"].forEach((ev) => $("compare").addEventListener(ev, () => setCompare(false)));
$("undo").addEventListener("click", undo);
$("clear").addEventListener("click", clearAll);
window.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
  if (e.code === "Space" && !e.repeat && e.target === document.body) { e.preventDefault(); setCompare(true); }
  if (e.key === "Enter") { clearTimeout(idleTimer); processPending(); }
});
window.addEventListener("keyup", (e) => { if (e.code === "Space") setCompare(false); });

async function pollStatus() {
  let reachable = true;
  try {
    const r = await fetch("/api/status");
    serverInfo = await r.json();
  } catch {
    reachable = false;
    serverInfo = { loading: false, error: "the demo server isn't running (start it again with the launcher)" };
  }
  const st = $("status");
  st.classList.toggle("ready", !serverInfo.loading && !!serverInfo.reader);
  st.classList.toggle("error", !serverInfo.loading && !serverInfo.reader);
  const pct = serverInfo.progress != null ? ` (${Math.round(serverInfo.progress * 100)}%)` : "";
  $("statusText").textContent = serverInfo.loading ? `${serverInfo.phase || "Loading models"}…${pct} · Tidy works already`
    : serverInfo.reader ? `Ready · ${serverInfo.reader}`
    : serverInfo.error ? `Magic unavailable: ${serverInfo.error}` : "Reader off (--reader none) · Tidy only";
  if (serverInfo.loading || !reachable) setTimeout(pollStatus, reachable ? 1500 : 3000);
}

// hooks for automated tests
window.__hw = {
  settings, get paths() { return paths; }, processPending,
  get busy() { return paths.some((p) => p.state === "busy") || tweens.length > 0; },
};

if (matchMedia("(pointer: coarse)").matches) {
  $("hint").textContent = "Write something with your finger or an Apple Pencil. Pause, and watch.";
}
resize();
pollStatus();
