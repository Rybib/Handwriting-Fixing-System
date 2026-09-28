// JavaScript port of the core of Google's Ink Stroke Modeler
// https://github.com/google/ink-stroke-modeler (Apache 2.0)
//
// Two stages, exactly as in the C++ library:
//   1. WobbleSmoother - a speed-aware moving average that removes the
//      high-frequency "wobble" from digitizer / mouse noise when moving slowly.
//   2. PositionModeler - the pen tip is a mass pulled by a spring towards the
//      (smoothed) input with drag. This yields beautiful, low-latency curves.
// Plus upsampling between inputs and end-of-stroke catch-up modelling.
// Units: positions in CSS px, time in seconds.

export const DEFAULT_PARAMS = {
  wobble: { enabled: true, timeout: 0.04, speedFloor: 60, speedCeiling: 110 },
  position: { springMassConstant: 11 / 32400, dragConstant: 72 },
  sampling: { minOutputRate: 180, endOfStrokeStoppingDistance: 0.01, endOfStrokeMaxIterations: 20, maxOutputsPerCall: 2000 },
};

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const lerp = (a, b, t) => a + (b - a) * t;
const normalize01 = (lo, hi, v) => (hi <= lo ? (v >= hi ? 1 : 0) : Math.min(1, Math.max(0, (v - lo) / (hi - lo))));

class WobbleSmoother {
  constructor(p) { this.p = p; }
  reset(pos, t) {
    this.samples = [{ x: pos.x, y: pos.y, wx: 0, wy: 0, d: 0, dt: 0, t }];
    this.sum = { wx: 0, wy: 0, d: 0, dt: 0 };
  }
  update(pos, t) {
    if (!this.p.enabled) return pos;
    const last = this.samples[this.samples.length - 1];
    const dt = t - last.t;
    const s = { x: pos.x, y: pos.y, wx: pos.x * dt, wy: pos.y * dt, d: dist(pos, last), dt, t };
    this.samples.push(s);
    this.sum.wx += s.wx; this.sum.wy += s.wy; this.sum.d += s.d; this.sum.dt += s.dt;
    while (this.samples[0].t < t - this.p.timeout) {
      const f = this.samples.shift();
      this.sum.wx -= f.wx; this.sum.wy -= f.wy; this.sum.d -= f.d; this.sum.dt -= f.dt;
    }
    if (this.sum.dt <= 1e-9) return pos;
    const avg = { x: this.sum.wx / this.sum.dt, y: this.sum.wy / this.sum.dt };
    const speed = this.sum.d / this.sum.dt;
    const k = normalize01(this.p.speedFloor, this.p.speedCeiling, speed);
    return { x: lerp(avg.x, pos.x, k), y: lerp(avg.y, pos.y, k) };
  }
}

class PositionModeler {
  constructor(p) { this.p = p; }
  reset(pos, t) { this.s = { x: pos.x, y: pos.y, vx: 0, vy: 0, t }; }
  update(anchor, t) {
    const s = this.s, p = this.p;
    const dt = t - s.t;
    const ax = (anchor.x - s.x) / p.springMassConstant - p.dragConstant * s.vx;
    const ay = (anchor.y - s.y) / p.springMassConstant - p.dragConstant * s.vy;
    s.vx += dt * ax; s.vy += dt * ay;
    s.x += dt * s.vx; s.y += dt * s.vy;
    s.t = t;
    return { x: s.x, y: s.y, t };
  }
}

// Streaming modeler: feed raw pointer events, get smooth modelled points.
export class StrokeModeler {
  constructor(params = DEFAULT_PARAMS) {
    this.params = params;
    this.wobble = new WobbleSmoother(params.wobble);
    this.pos = new PositionModeler(params.position);
  }

  // returns array of modelled points {x, y, t, p}
  begin(x, y, t, pressure = 0.5) {
    const pt = { x, y };
    this.wobble.reset(pt, t);
    this.pos.reset(pt, t);
    this.lastAnchor = { x, y, t };
    this.lastPressure = pressure;
    this.raw = [{ x, y, t, p: pressure }];
    return [{ x, y, t, p: pressure }];
  }

  move(x, y, t, pressure = 0.5) {
    if (t <= this.lastAnchor.t) t = this.lastAnchor.t + 1e-4;
    this.raw.push({ x, y, t, p: pressure });
    const anchor = this.wobble.update({ x, y }, t);
    const out = this._upsample(this.lastAnchor, { ...anchor, t }, pressure);
    this.lastAnchor = { ...anchor, t };
    this.lastPressure = pressure;
    return out;
  }

  // Let the simulated pen tip catch up with the final input position.
  end(x, y, t, pressure) {
    const out = x === undefined ? [] : this.move(x, y, t, pressure ?? this.lastPressure);
    const sp = this.params.sampling;
    const target = this.raw[this.raw.length - 1];
    let dt = 1 / sp.minOutputRate;
    for (let i = 0; i < sp.endOfStrokeMaxIterations; i++) {
      const prev = { ...this.pos.s };
      const cand = this.pos.update(target, prev.t + dt);
      if (dist(prev, cand) < sp.endOfStrokeStoppingDistance) break;
      // overshoot check: nearest point on prev->cand segment to target
      const sx = cand.x - prev.x, sy = cand.y - prev.y;
      const tt = ((target.x - prev.x) * sx + (target.y - prev.y) * sy) / (sx * sx + sy * sy || 1);
      if (tt < 1) { dt *= 0.5; this.pos.s = prev; continue; }
      out.push({ x: cand.x, y: cand.y, t: cand.t, p: this.lastPressure });
      if (dist(cand, target) < sp.endOfStrokeStoppingDistance) break;
    }
    return out;
  }

  _upsample(a, b, pressure) {
    const sp = this.params.sampling;
    const n = Math.min(sp.maxOutputsPerCall, Math.max(1, Math.ceil((b.t - a.t) * sp.minOutputRate)));
    const out = [];
    for (let i = 1; i <= n; i++) {
      const k = i / n;
      const s = this.pos.update({ x: lerp(a.x, b.x, k), y: lerp(a.y, b.y, k) }, lerp(a.t, b.t, k));
      out.push({ x: s.x, y: s.y, t: s.t, p: lerp(this.lastPressure, pressure, k) });
    }
    return out;
  }
}

// Convenience: model a whole recorded stroke at once.
export function modelStroke(raw, params = DEFAULT_PARAMS) {
  if (!raw.length) return [];
  const m = new StrokeModeler(params);
  const out = m.begin(raw[0].x, raw[0].y, raw[0].t, raw[0].p);
  for (let i = 1; i < raw.length; i++) out.push(...m.move(raw[i].x, raw[i].y, raw[i].t, raw[i].p));
  out.push(...m.end());
  return out;
}
