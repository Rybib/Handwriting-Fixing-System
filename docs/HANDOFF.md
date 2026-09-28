# Handoff: Rytability Handwriting Magic (local Mac session)

Built in a cloud Claude session, which had no Apple Silicon Mac to test on.
This is the first time it runs on real hardware. The job is to get it working
well there.

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
2. The page's status pill stays amber on **"Loading models…"**. Magic mode
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
