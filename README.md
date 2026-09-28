# Rytability Handwriting Magic (proof of concept)

Write with a mouse, a trackpad or an Apple Pencil. When you pause, your scrawl
shimmers, dissolves into sparkles, and the same sentence writes itself back in.
It comes back **neat, spelled correctly, and still in your handwriting**.

![before and after](docs/before-after.png)

![the magic, animated](docs/demo.gif)

This is a standalone prototype for the iPad version of Rytability. Everything
runs locally on your Mac. The only network traffic is the one-time model download.

## Try it on your Mac

```bash
git clone https://github.com/Rybib/Handwriting-Fixing-System
cd Handwriting-Fixing-System
./run.sh
```

The first run sets up a Python environment, installs PyTorch and Transformers
(about 1 GB), then downloads the models (about 4.5 GB for the handwriting reader
and 43 MB for the handwriting synthesiser). After that, `./run.sh` opens
`http://localhost:8765` in a few seconds. You need Python 3.10 or newer. If
`run.sh` can't find it, run `brew install python@3.12`.

**Test with a real Apple Pencil:** run `./run.sh --lan`, then open the printed
`http://<your-mac-ip>:8765` address in Safari on an iPad on the same Wi-Fi.
Pencil pressure is used for line width, and palm rejection is on.

### Controls

| | |
|---|---|
| **✨ Magic** | Reads what you wrote, works out what you meant, fixes the spelling, and rewrites it neatly in your handwriting |
| **Tidy** | Keeps your exact letters. It straightens the baseline and evens out letter size, slant and word spacing. This is instant and needs no AI model |
| **Off** | Plain ink, smoothed live by the Ink Stroke Modeler |
| **Neatness** | How strongly the ink is regularised (the synthesiser's sampling bias, or how hard Tidy corrects) |
| **Style** | *My handwriting* copies your style. *Style 1-13* are other writers from the training data |
| **Fix spelling** | Off: Magic rewrites exactly what you wrote, just neater |
| **👁 / hold Space** | Shows what you originally wrote |
| **↶ / ⌘Z** | Undoes the last stroke or the last fix |
| **Enter** | Runs the magic now, without waiting for the pause |

## How it works

```
 pointer / Pencil events
        │
        ▼
 ① Ink Stroke Modeler (JS port)         live: spring-mass pen model + wobble smoothing
        │                                → smooth, low-latency, variable-width ink
        ▼  (pause ~1 s)
 ② Line + word segmentation, Tidy        pure geometry, instant
        │                                (Tidy mode stops here and morphs the ink)
        ▼
 ③ Reader: Qwen3-VL-2B (local)           one pass returns
        │                                  WRITTEN: "I recieve my freind at the park"
        │                                  MEANT:   "I receive my friend at the park"
        ▼
 ④ Handwriting synthesis (Graves RNN)    primed with YOUR tidied ink + its transcript,
        │                                writes MEANT in your style; 3 candidates, best kept
        ▼
 ⑤ Self-check                            attention alignment + the reader proofreads the
        │                                result; if your ink was too messy to copy,
        │                                falls back to a clean built-in style
        ▼
 ⑥ Magic animation                       scrawl dissolves, neat ink writes itself in
```

* **① [Ink Stroke Modeler](https://github.com/google/ink-stroke-modeler)**, one of
  the two projects you suggested. It's Google's real-time stroke smoother. I ported
  its wobble smoother, spring-mass position modeler, upsampling and end-of-stroke
  catch-up to JavaScript (`web/js/inkmodeler.js`). It removes mouse jitter while
  you draw, without lag.
* **③ Reader.** A small on-device vision-language model is prompted to give both a
  literal reading and the intended sentence. Because it sees the pixels and knows
  English, it can resolve "wether" → "weather" and "their tomorow" →
  "there tomorrow".
* **④ Synthesis.** [Graves (2013)](https://arxiv.org/abs/1308.0850) handwriting
  synthesis, using the pretrained weights from
  [sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis),
  re-implemented in pure numpy (`server/synth.py`, no TensorFlow). Its "priming"
  mechanism is what makes the output look like *you*. It first feeds your own
  strokes and their transcript through the network, then continues writing in
  that style.
* **⑤ Self-check.** Priming copies whatever it is shown, and that includes
  illegibility. The server measures whether the network's attention lined up with
  your ink, and has the reader proofread the result. If the result reads worse
  than a clean built-in style, the built-in style is used. Both versions are
  generated in the same batch, so the fallback adds almost no time.

## What I researched, and why this stack

| Candidate | Verdict |
|---|---|
| **Ink Stroke Modeler** (your pick) | ✅ Used for live smoothing. It's excellent at what it does, but it only smooths. It doesn't change letter shapes, so on its own it can't fix messy writing |
| **InkSight** (your pick) | ❌ Not used. It converts *photos* of handwriting into digital ink (derendering), and its output traces the original, mess included. We already have the digital ink. Its recognition is a side feature, and it needs TensorFlow + tensorflow-text, which are awkward on Apple Silicon |
| **Apple Smart Script** (iPadOS 18+) | The closest product to this idea, but it has no public API |
| **PencilKit `PKStrokeRecognizer`** (iPadOS/macOS 27, WWDC26) | ⭐ The production answer for step ③ on iPad: on-device stroke recognition in 29 languages, per-stroke IDs, no model to ship. `--reader apple` approximates it on the Mac with the Vision framework |
| **Graves RNN synthesis** | ✅ Used. It's tiny (3.6M parameters, 14 MB) and fast on CPU, supports style priming, and runs easily in Core ML |
| **DiffInk** (ICLR 2026) and other diffusion ink models | Better style fidelity in the papers, but trained on Chinese only. Worth watching |
| **TrOCR** | Older OCR model. It has tokenizer breakage on current Transformers, and a VLM beats it on messy lines |
| **Qwen3-VL-2B / 4B** | ✅ 2B is the default. Set `HWFIX_VLM=Qwen/Qwen3-VL-4B-Instruct` to try 4B if your Mac has 16 GB+ RAM |

### Measured results (12 deliberately messy test lines: sloppy writing + mouse wobble + dyslexic misspellings)

* **Reader:** got the intended sentence exactly right on **8/12**. Two of the four
  misses were deliberately near-illegible scribbles.
* **Rewrite legibility:** each rewritten line was read back by the VLM. **9/12**
  came back perfectly, when primed with the user's own messy ink. The failures
  were the illegible inputs, which the self-check now catches, plus one dropped
  letter.
* **Speed** on the 4-core cloud CPU I built this on: about 9 s to read, about 5 s
  to synthesise, and about 4 s to verify, per line. Your Mac's GPU should be a
  good deal faster, but I couldn't measure that here. Tidy mode is instant.

Benchmarks and test tooling are in `tests/`: `make_evalset.py`,
`eval_legibility.py`, and `e2e_playwright.py`, which drives the real UI with an
emulated mouse. The quick checks run in a few seconds:

```bash
node tests/test_tidy.mjs
.venv/bin/python tests/test_core.py
```

## Honest limitations

* **English only.** The synthesiser's alphabet has no capital Q, X or Z, so
  those are written lowercase.
* The synthesiser still sometimes wobbles on a letter. The best-of-3 selection
  and the self-check reduce this, but don't remove it.
* The 2B reader occasionally "fixes" a word into the wrong word. The 4B model is
  better at this, and in the real app, Gemma 3 4B (already bundled) could do this
  step.
* Drawings and diagrams aren't detected. If the reader can't find any letters,
  the ink is left alone.

## Path into Rytability (iPad)

1. **Input:** `PKCanvasView`, which already gives you Apple's own stroke smoothing.
   The JS Ink Stroke Modeler only matters for the web demo.
2. **Read:** `PKStrokeRecognizer` (iPadOS 27) for WRITTEN, then the app's existing
   Gemma 3 / Apple Foundation Models text pipeline for MEANT. This is the same
   job as Improve, with a prompt aimed at handwriting.
3. **Rewrite:** convert `server/synth.py` to Core ML or MLX. It's only an LSTM
   with 3.6M parameters, and priming uses the `PKStroke` points. Tidy mode is
   about 200 lines of geometry, which ports easily to Swift.
4. **Licensing (important before shipping):** the pretrained synthesis weights come
   from a repo with **no licence**, and were trained on **IAM-OnDB**, which is
   licensed for non-commercial research only. That's fine for a proof of concept.
   For the App Store, retrain the same small network on data you can use
   commercially, for example handwriting collected from consenting users or
   licensed from a vendor. Ink Stroke Modeler and Qwen3-VL are Apache-2.0.

## Repo layout

```
run.sh                 one-command launcher (macOS / Linux)
server/app.py          local web server + /api/rewrite pipeline
server/synth.py        Graves handwriting synthesis in numpy (priming, bias, best-of-N)
server/weights.py      fetches + converts the pretrained checkpoint (no TensorFlow)
server/recognize.py    readers: Qwen3-VL (default), Apple Vision (macOS)
server/ink.py          rasterising + measuring ink
web/js/inkmodeler.js   Ink Stroke Modeler port
web/js/tidy.js         line/word segmentation, x-height/baseline/slant, Tidy beautifier
web/js/app.js          canvas, input, magic animations, UI
tests/                 eval set generator, legibility benchmark, Playwright end-to-end
```
