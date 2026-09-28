// Fast geometry tests for the Tidy beautifier: node tests/test_tidy.mjs
import assert from "node:assert/strict";
import { groupBlocks, groupLines, tidyLine, coreMetrics } from "../web/js/tidy.js";

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

// Stroke boxes of "can you help me with my homwork" as drawn in the browser. The
// i-dot of "with" (x 506) used to start a line of its own that then took the
// tall h/k strokes after it, so the rewrite of half the line landed on top of
// the other half. Plus a lone descender (the tail of a y), which also used to
// become a line. Each stroke is its bounding-box diagonal.
const boxes = [[112, 124, 542, 564], [129, 180, 545, 561], [206, 227, 549, 586], [233, 274, 556, 567],
  [294, 353, 535, 569], [388, 432, 547, 562], [463, 496, 551, 562], [504, 505, 550, 564], [506, 506, 534, 535],
  [513, 514, 532, 567], [507, 550, 542, 576], [576, 606, 556, 569], [606, 624, 555, 584], [661, 680, 529, 559],
  [689, 735, 556, 562], [740, 828, 530, 564], [840, 866, 560, 614]];
const boxLine = (dy) => boxes.map(([x0, x1, y0, y1]) => ({ pts: [{ x: x0, y: y0 + dy }, { x: x1, y: y1 + dy }] }));
assert.equal(groupLines(boxLine(0)).length, 1, "an i-dot or a y tail must not split a line");
const stacked = groupLines([...boxLine(0), ...boxLine(72)]);   // next ruled line
assert.deepEqual(stacked.map((L) => L.length), [boxes.length, boxes.length], "two ruled lines stay separate");

// Blocks: three lines on consecutive ruled lines are one piece of writing; a
// line after a blank ruled line, and a note far off to the right on the same
// row as the first line, are blocks of their own.
const para = [...boxLine(0), ...boxLine(72), ...boxLine(144)];
const note = boxes.slice(0, 5).map(([x0, x1, y0, y1]) => ({ pts: [{ x: x0 + 1200, y: y0 }, { x: x1 + 1200, y: y1 }] }));
const later = boxLine(360);
const page = [...para, ...note, ...later];
const blocks = groupBlocks(page, groupLines(page), 72);
const n = boxes.length;
const shape = (B) => B.map((L) => L.length).join(",");
assert.deepEqual(blocks.map(shape).sort(), [`${n}`, `${n},${n},${n}`, "5"], "paragraph, side note, later line");
assert.ok(blocks.find((B) => shape(B) === "5").flat().every((i) => i >= 3 * n && i < 3 * n + 5), "the side note is its own block");
console.log("ok tidy tests");
