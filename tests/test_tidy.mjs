// Fast geometry tests for the Tidy beautifier: node tests/test_tidy.mjs
import assert from "node:assert/strict";
import { groupLines, tidyLine, coreMetrics } from "../web/js/tidy.js";

// A fake handwritten line: 4 "words" of loopy letters on a tilted, wavy baseline,
// with one word written much bigger than the rest.
function fakeLine(y0, tilt) {
  const strokes = [];
  let x = 20;
  [4, 3, 5, 2].forEach((nLetters, w) => {
    const size = w === 2 ? 30 : 18;
    const pts = [];
    for (let i = 0; i <= nLetters * 20; i++) {
      const t = i / 20;
      const px = x + t * size * 0.8 + Math.cos(t * Math.PI * 2) * size * 0.25;
      const py = y0 + tilt * px - (Math.sin(t * Math.PI * 2) * 0.5 + 0.5) * size;
      pts.push({ x: px, y: py });
    }
    strokes.push(pts);
    x = pts.at(-1).x + 26;
  });
  return strokes;
}

const line = fakeLine(200, 0.12);
const r = tidyLine(line, { strength: 1 });
assert.equal(r.strokes.length, line.length);
r.strokes.forEach((s, i) => assert.equal(s.length, line[i].length, "same point count -> can morph"));

// baseline got straighter: compare bottom of first and last word
const bottom = (pts) => Math.max(...pts.map((p) => p.y));
const before = Math.abs(bottom(line[0]) - bottom(line[3]));
const after = Math.abs(bottom(r.strokes[0]) - bottom(r.strokes[3]));
assert.ok(after < before * 0.4, `baseline not straightened: ${before.toFixed(1)} -> ${after.toFixed(1)}`);

// the oversized word got closer to the others
const h = (pts) => coreMetrics(pts).core;
assert.ok(Math.abs(h(r.strokes[2]) - h(r.strokes[0])) < Math.abs(h(line[2]) - h(line[0])), "sizes not evened out");

// two separate lines + a stray dot are grouped as two lines
const two = [...fakeLine(200, 0), ...fakeLine(300, 0), [{ x: 60, y: 160 }, { x: 61, y: 161 }]];
const lines = groupLines(two.map((pts) => ({ pts })));
assert.equal(lines.length, 2, "dot should join a line, not become its own");
console.log("ok tidy tests");
