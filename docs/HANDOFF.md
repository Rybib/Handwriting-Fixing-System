# Handoff: Rytability Handwriting Magic (local Mac session)

Built in a cloud Claude session, which had no Apple Silicon Mac to test on.
This is the first time it runs on real hardware. The job is to get it working
well there.

**Update, 28 Sep 2026:** a local Mac session did this. See "Results of the local
Mac session" at the end.

## What it is
A local web demo. You handwrite with a mouse, trackpad or Apple Pencil, pause,
and the ink is read, its spelling fixed, and it is rewritten neatly in the
writer's own style. It is a proof of concept for the iPad version of Rytability,
a dyslexia writing app. Read `README.md` first for the architecture and the
research behind it.

Pipeline: `web/js/inkmodeler.js` (live smoothing) → `web/js/tidy.js`
(segmentation plus the "Tidy" beautifier) → `POST /api/rewrite` in
`server/app.py`. That runs `server/recognize.py` (Qwen3-VL-2B reads WRITTEN and
MEANT), then `server/synth.py` (a numpy Graves RNN, primed with the user's ink,
best-of-3), then a self-check where the reader proofreads the output and the
server falls back to built-in style 9. The browser then plays the animation.

## Where things are
- Repo: `~/Desktop/HandwritingMagic`, branch `claude/rytability-handwriting-cleanup-7juu45`
  of `Rybib/Handwriting-Fixing-System`. Commit and push to that branch.
- Launch: `./run.sh` or double-click `Launch Handwriting Magic.command`. Logs go
  to `logs/last-run.log`. The Python env is `.venv`, managed by uv.
- Models: `models/hand_synth.npz` + `models/styles/` (synthesiser), and
  `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct` (reader, about 4.5 GB).
- **Do not modify `Rybib/Rytability`.** It is read-only reference.

## Current status (reported by the user)
1. The first run hit "Address already in use" on port 8765, from another
   program. Fixed: the server now tries the next free port.
2. The page's status pill stayed amber on **"Loading models…"**. (Since then
   the pill shows the real phase: download progress in GB and %, "Loading…
   into memory", "Warming up", and a warning if the download stalls for 90 s.) Magic mode
   silently falls back to Tidy while it is loading, which is why the output was
   "not as clean" as expected. Find out why the reader isn't becoming ready:
   - Is the download still going or stalled? Check the Terminal output and
     `du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct`.
   - Did loading or warm-up on MPS hang or fail? Check
     `curl -s http://127.0.0.1:<port>/api/status` and the server log. The
     reader tries bfloat16 on MPS and retries in float32 on error, but a *hang*
     would not be caught.
   - Quick isolated check:
     `.venv/bin/python -c "import sys; sys.path.insert(0,'server'); import recognize as R; v=R.VLMReader(); print(v.device, v.read_literal(__import__('PIL.Image',fromlist=['x']).new('RGB',(64,32),'white')))"`

## Please verify, then improve
1. Fast tests: `node tests/test_tidy.mjs` and `.venv/bin/python tests/test_core.py`.
2. Get the status to **Ready**. Then time one Magic rewrite; the server prints
   `[rewrite] read Xs synth Ys`. On an M-series GPU the target is under about
   5 s in total. If numpy synthesis is slow, look at the per-step Python loops
   in `HandwritingSynth.write`.
3. End to end with the real UI (needs `pip install playwright` in the venv, and
   Chrome, or `playwright install chromium`):
   `python tests/make_evalset.py /tmp/ev && python tests/e2e_playwright.py /tmp/e2e /tmp/ev/00.json /tmp/ev/01.json --url http://127.0.0.1:<port>`
   Then look at the screenshots.
4. Output quality. The user wants the result **cleaner**. Levers, most useful
   first:
   - The Neatness slider maps to the sampling bias `0.3 + n/100*2.2`. Try a
     higher default (for example 75, which gives bias ≈ 2.0) and `candidates` 4.
   - Style "My handwriting" copies the user's style, including its mess. The
     preset styles (Style 1-13) are much cleaner. Consider making the fallback
     threshold stricter, or offering a "clean" default.
   - `HWFIX_VLM=Qwen/Qwen3-VL-4B-Instruct ./run.sh` gives better reading and
     spelling fixes, if RAM is 16 GB or more.
   - Proofreading is lenient because the VLM reads *through* small glitches.
     Consider a stricter CER threshold in `server/app.py` (currently 0.05 for
     "mine").
   - Show download or loading progress in the UI, so "Loading models…" isn't a
     mystery. Also make it obvious when Magic fell back to Tidy.
5. Try `./run.sh --reader apple`, the macOS Vision recogniser. It is untested
   because it was written on Linux.

Report back what you found, what you changed, and before/after screenshots.

## Results of the local Mac session (28 Sep 2026, M5 MacBook Pro, 32 GB)

**Why the status stayed on "Loading models".** Nothing hung. The first run
downloaded the 4.3 GB reader for about 6 minutes, and nothing showed it:
huggingface_hub hides its byte progress bars when stderr is not a terminal, and
`run.sh` pipes everything through `tee`. `print()` was also block-buffered on
that pipe, so `logs/last-run.log` stopped at "Loading weights" even after the
reader was ready. A cold load from the cache takes 5-10 s on MPS in bfloat16;
there's no GPU problem. Fixed: `snapshot_download` gets a tqdm class that
counts bytes, which feeds the page and a Terminal line every 5 s; stdout is
line-buffered. Also, huggingface_hub 1.x keeps finished blobs in a shared
`hub/blobs/` store, so `du` on the model folder shows only ~11 MB.

**What changed for cleaner output** (each measured with `tests/eval_pipeline.py`):
- Stray marks: after priming, the RNN often finished the *prime* first (dotting
  its last i). Strokes drawn before the attention reaches the text are dropped.
- "fasl" for "fast": generation stopped before the last t was crossed. The text
  now gets a trailing space; strokes after it are kept only if they go back
  over the words (`_trim` in `synth.py`).
- Candidates: 8 per style (0.6 s on the M5). macOS Vision proofreads them
  (40 ms each, strict; the VLM read *through* glitches). The first candidate
  that reads back perfectly in the user's style wins, else the clean style's
  best (`pick` in `app.py`).
- Reader: Qwen3-VL-4B by default with 16 GB+ of RAM (11/12 vs 7/12 on the eval
  ink). macOS Vision's literal reading goes into the prompt as a second opinion;
  it lifts 2B from 14 to 18 of 24 lines and is neutral for 4B.
- Neatness default 90 (was 60). Higher bias makes the clean styles more legible
  (Style 10: 4.0% -> 2.8% CER); it makes no difference to copies of the user's
  style, whose limit is the ink being copied.
- A bug that split one line in two: an i-dot started its own line and pulled in
  the next tall letters, so half the rewrite was drawn on top of the other half.
  Fixed in `groupLines`.
- Size: the rewrite's x-height came from a per-word estimate that was often off
  by a third. It is now the geometric mean of the line's x-height and its width
  per letter / 1.25.
- `--reader apple` works: 0.7 s a line, 15-17/24, below 4B's 19/24.

**Later the same day:** the reader went back to Qwen3-VL-2B (lighter, and the
model the iPhone/iPad app runs). Moving macOS Vision's hint BEFORE the
instructions stopped the model echoing it (reads went from 2.3 s to 1 s), and
2B now matches 4B: 10/12 and 9/12 intended sentences, 1.8 s a line. The 13
styles have names ("Clean print" ... "Flowing cursive") instead of numbers.
The iOS app is a separate project, `~/Desktop/HandwritingMagic-iOS`: this
page in a web view, with the pipeline ported to Swift (MLX for the reader).
The page now fits a phone (the toolbar wraps, safe areas) and says "finger
or Apple Pencil" on touchscreens; `scripts/sync_web.sh` there copies web/ in.

**Still open**
- Reading is the weak link on very messy words ("hanwak" -> "hamster").
  Candidates: Gemma 3 4B for the MEANT step (as in the iPad app), or asking the
  reader for several MEANT options and letting the user tap one.
- Proofreading uses macOS Vision, so on Linux it falls back to the VLM (slow,
  lenient, top 2 only).
- The user's own style copies their letter shapes, including loopy ones, as
  long as they read correctly.

**Magic is a button now (after trying the iOS app).** The rewrite used to run
after a 1.1 s pause, which on a touchscreen fired mid-sentence: half a line
went off to be read, came back as "couldn't find any words", and the rest was
fixed a few seconds later, under your pen. Now nothing changes until you press
✨ Magic (or Enter), which rewrites everything written since the last press.
Tidy and Off are gone as modes (tidying still prepares the style sample). "No
words" is only shown when nothing on the page had words, and if the reader's
reply is empty its OCR hint is used instead.
