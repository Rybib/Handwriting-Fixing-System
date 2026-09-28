# Handwriting Magic for iPhone and iPad

The Mac demo (`~/Desktop/HandwritingMagic`), running on a touchscreen. Write with
your finger or an Apple Pencil, then tap **✨ Magic**: everything you wrote
dissolves into sparkles and comes back neat, spelled correctly, and in your handwriting.
Everything runs on the device, and it works in Airplane Mode from the first launch.

This is a playground for trying the idea on a real screen before any of it
goes into Rytability.

## Getting the code

This app lives on the `ios` branch of
[Rybib/Handwriting-Fixing-System](https://github.com/Rybib/Handwriting-Fixing-System);
the Mac demo is on the default branch. Clone them side by side on the Desktop:

```bash
cd ~/Desktop
git clone https://github.com/Rybib/Handwriting-Fixing-System HandwritingMagic
git clone -b ios https://github.com/Rybib/Handwriting-Fixing-System HandwritingMagic-iOS
cd HandwritingMagic && ./run.sh --reader none      # first run: sets up Python, fetches the synthesiser (Ctrl-C when it's up)
cd ../HandwritingMagic-iOS
../HandwritingMagic/.venv/bin/python scripts/export_synth.py   # -> HandwritingMagic/Resources/hand_synth.bin/json
scripts/get_model.sh                                          # -> Gemma 3 4B + its vision tower (~3 GB), GemmaMLXModel.bundle
scripts/get_mlkit_model.sh                                    # -> ML Kit's 13 MB English model, MLKitEnglishModel.bundle
```

Three things are deliberately not in git: the AI model (~3 GB, too big), ML
Kit's English model (Google's files, fetched from dl.google.com), and the
synthesiser's weights (they come from a repo with no licence and were trained
on IAM-OnDB, which is for non-commercial research only, so each machine
fetches its own copy, as the Mac demo does). The Xcode project also expects
the vendored packages at `~/Desktop/Files/App Work/dependencies`: Rytability's
(mlx-swift-lm, swift-transformers and theirs) plus the frozen ML Kit and its
five Google packages.

## Run it on your iPhone or iPad

1. Open `HandwritingMagic.xcodeproj` in Xcode.
2. Pick the **HandwritingMagic** scheme and your device at the top, then press ▶︎.
   Signing uses the Rytability team (85P7KNQ5P3) and the bundle id
   `com.RytechLabsLLC.HandwritingMagic`; Xcode registers it on the first run.
3. The first install copies the ~3 GB model to the device, so it takes a
   minute. At launch the status (bottom right) says *Loading the handwriting
   reader…* for a few seconds, then turns green: *Ready · ML Kit + Gemma 3 4B (sees the ink) on this iPad*.
   Gemma needs a device with 6 GB of memory or more; with less, the app
   says so and fixes spelling with the system spellchecker instead.
   You can write while it loads.

The Simulator runs the app too, but MLX needs a real GPU, so there it reads
with ML Kit and fixes spelling with the system spellchecker only (so "their"
stays "their"), and it's slower. Use it for layout; use a device for the real thing.

The page and its controls are the Mac demo's (✨ Magic, Neatness, the 13 named
styles, Fix spelling, 👁 hold to see what you wrote, ↶ undo, ✕ clear). Nothing
is rewritten until you tap Magic, so it never grabs a sentence mid-thought.
After the first Pencil stroke, fingers stop drawing (palm rejection).

## How it's put together

```
HandwritingMagic/            the app
  App/                       SwiftUI shell: a web view showing the page, and
                             PageServer, which answers the page's /api calls
  Web.bundle/                the page, copied from the Mac demo's web/
  Resources/hand_synth.*     the handwriting synthesiser's weights + 13 styles
  GemmaMLXModel.bundle/      Gemma 3 4B QAT, 4-bit MLX, Rytability's: what was meant (not in git)
  MLKitEnglishModel.bundle/  ML Kit's English handwriting model (FileData not in git)
Engine/                      the pipeline, in Swift (shared with MagicCheck)
  MagicEngine.swift          read -> write in your style + a clean style -> proofread
  InkReader.swift            Google ML Kit Digital Ink: reads the pen strokes literally
  Readers.swift              Gemma 3 4B via MLX (what was meant), Apple Vision
  Gemma3MemoryPatched.swift  Rytability's Gemma 3, whose vision pass doesn't spike memory
  SpellFixer.swift           the system spellchecker, for when there's no AI model
  HandwritingSynth.swift     Graves handwriting synthesis on Accelerate
  Ink.swift, TextTools.swift ink geometry, spelling diff, text helpers
MagicCheck/                  a Mac command-line tool: runs Engine/ on test ink
scripts/                     get_model.sh, sync_web.sh, export_synth.py
```

Magic reads a **passage** at a time: the page groups lines written one under
the next (overlapping side to side) into a block, and sends the block as one
request, so every line is read knowing the lines around it. Writing somewhere
else on the page, like a list off to the side or a note after a blank line,
is a block of its own. The rewrite still goes line by line, in place.

Reading a passage takes two steps. **ML Kit Digital Ink Recognition** reads
each line's pen strokes (where the pen went, in what order), with the end of
the line before as context, and returns what is literally written,
misspellings and all, in ~20 ms a line. **Gemma 3 4B** (Rytability's model,
with its vision tower) then gets all of those readings and a picture of the
whole passage, and works out what each line meant ("their tomorow" -> "there
tomorrow"). Gemma squeezes every picture to 896 x 896, so long lines are
wrapped at their widest gaps and the picture is padded to a square instead of
stretched (`Ink.renderPassage`). If Gemma's reading of a line has little to
do with what ML Kit saw there (it merged or split lines), that line is read
again on its own, with the rest of the passage as context. Without ML Kit,
Apple Vision reads a picture of each line instead.

The page talks to `magic://app/api/rewrite` exactly as it talks to the Python
server on the Mac, so both run the same `web/`. The Swift engine is a port of
`server/app.py`, `recognize.py` and `synth.py`: same "your style
or the clean *Tall narrow print* style, whichever reads back better" rule, 8
candidates of each, proofread by Vision.

ML Kit is **frozen**: a local copy in `~/Desktop/Files/App Work/dependencies/google-mlkit-swiftpm`,
trimmed to Digital Ink, made from [d-date/google-mlkit-swiftpm](https://github.com/d-date/google-mlkit-swiftpm)
9.0.2 (a community Swift Package of Google's ML Kit binaries; Google only
publishes it for CocoaPods). Its five Google packages (promises, GoogleDataTransport,
GoogleUtilities, gtm-session-fetcher, nanopb) are local copies next to it, so
building fetches nothing and nothing changes unless those folders are replaced
by hand; `VENDORED.md` there says exactly what's in it. It needs `-ObjC -all_load` in Other Linker Flags
and `HandwritingMagic/MLKitDigitalInkRecognition_resource.bundle` (the model
download manifest; without it the download silently never starts).

ML Kit only offers its handwriting models as a download (it has 300+
languages). So the app carries the English one, the exact files ML Kit would
download (`MLKitEnglishModel.bundle`), and on first launch copies them to where
ML Kit keeps its downloads, plus its one small bookkeeping file, named after
the app's bundle id. ML Kit then sees the model as already downloaded (checked
on a fresh install in the Simulator). This isn't an official ML Kit feature: if
an update stores downloads differently, the copy does nothing and ML Kit just
downloads the model once instead. Before shipping it in Rytability, check that
Google's ML Kit terms allow carrying the model in the app.

The MLX packages come from Rytability's vendored copies in
`~/Desktop/Files/App Work/dependencies` (`mlx-swift-lm` and `swift-transformers`,
local references), so the code builds against exactly what Rytability ships.
The tokenizer loader is written out by hand, so no Swift macro has to be
trusted.

## Settings (⚙)

The ⚙ button opens a sheet with a switch for each part of the reading, and
whether it works on this device (from `/api/status`'s `parts`): reading lines
together as one passage, ML Kit, Gemma, Gemma looking at the ink, and
proofreading the rewrite with Apple Vision. **Turn everything on** puts them all
back; that's the default, and the best: each part covers for the others'
mistakes. The status line says when something is switched off. The page sends
the switches with each Magic (`"reading"`), and `MagicEngine.Reading` honours
them. With Gemma off you see exactly what ML Kit read.

On the M5 Mac (Apple Vision standing in for ML Kit), the 12 eval lines:
everything on 10/12 at 2.75 s a line; Gemma not looking at the ink 10/12 at
1.6 s; Gemma off 3/12 at 0.5 s (the spellchecker can't fix misreadings). In
the Simulator, with ML Kit off, Apple Vision read "my faworil food is
spagetto" where ML Kit read "my favorit food is spagetti".

## The Magic effect

While the models work, a band of colour sweeps over your ink. It's a CSS
animation of a layer masked to the ink (`startShimmer` in `app.js`), so the
system compositor runs it by itself: the page draws nothing while it waits,
and the effect stays smooth even with Gemma keeping the GPU busy. (It used to
be redrawn on the canvas every frame, with a blur on every stroke, which
stuttered on the iPad once Gemma arrived.) The models run off the main
thread (`PageServer` hands each request to a detached task).

## Keeping it in step with the Mac demo

- Changed the page? `scripts/sync_web.sh` copies `../HandwritingMagic/web` in.
- Fresh checkout, or the model is missing? `scripts/get_model.sh` puts it
  together from Rytability's copy (or your Hugging Face cache, or a download).
- Changed the synthesiser's weights or styles? `python scripts/export_synth.py`.

## Checking the pipeline on the Mac

MLX can't run in the Simulator, so `MagicCheck` runs the app's `Engine/`
code on the Mac demo's messy test lines, one at a time and then in passages of three:

```bash
cd ~/Desktop/HandwritingMagic && .venv/bin/python tests/make_evalset.py /tmp/evalset
cd ~/Desktop/HandwritingMagic-iOS
xcodebuild -scheme MagicCheck -configuration Release -destination platform=macOS \
  -derivedDataPath /tmp/hm-dd build
/tmp/hm-dd/Build/Products/Release/MagicCheck /tmp/evalset /tmp/magiccheck
```

With Gemma it gets the intended sentence on 10 of the 12 lines, alone or in
passages (see below). The synthesiser writes 16 candidates in about 0.3 s on
the Mac. The iOS Simulator has no GPU for MLX, so there it reads with ML Kit
and the spellchecker only.

In a Debug build, `MAGIC_SELFTEST=<request.json>` makes the app rewrite that
request once it has loaded and print the result, which checks ML Kit in the
Simulator:

```bash
SIMCTL_CHILD_MAGIC_SELFTEST=/path/to/request.json xcrun simctl launch --console-pty booted com.RytechLabsLLC.HandwritingMagic
```

## The model: Gemma 3 4B, as in Rytability (28 Sep 2026)

The app used Qwen3-VL-2B; it now uses the Gemma 3 4B that Rytability carries
(mlx-community/gemma-3-4b-it-qat-4bit), vision tower included.
`scripts/get_model.sh` builds `GemmaMLXModel.bundle` from Rytability's
`GemmaMLXModel.bundle` (the same `model-text.safetensors`, 2.56 GB) plus the
vision tower (434 MB), which Rytability ships as a separate download: it is
split out of the full model (Rytability's backup in `~/Desktop/Files/App Work`,
the Hugging Face cache, or a download) by `scripts/split_vision.py`. MLX loads
every `*.safetensors` in the folder, so the two files are the whole model.
`Gemma3MemoryPatched.swift` is Rytability's copy of MLX's Gemma 3 with its
vision-attention memory fix; on A14 / M1-class GPUs the model runs in float32,
as in Rytability.

MagicCheck on the M5 MacBook Pro, where Apple Vision gives the literal reading
(ML Kit is iOS only), on the 12 eval lines:

| | Right | Time |
|---|---|---|
| Gemma 3 4B with the ink, each line alone | 10/12 | 2.75 s a line |
| Gemma 3 4B with the ink, in passages of 3 lines | 10/12 | 5.9 s a passage |
| Qwen3-VL-2B with the ink, each line alone (before) | 9/12 | 1.1 s a line |

(Times include writing and proofreading the rewrite, about 0.5 s a line.)
The two misses are the same hard ones as before: a stray scribble in the eval
ink, and a near-illegible "thank you for comming to my party". Gemma is
slower than Qwen: expect a few seconds a line on an iPad.

## Which model reads best? (earlier comparison, 28 Sep 2026)

`MagicCheck --readers <model folders>` runs any MLX vision model through the
app's reader code on the 24 eval lines (12 eval ink, 12 drawn with a mouse in
the browser). On the M5 Mac, with Vision's reading as the hint:

| Model | Right | Time a line | Size |
|---|---|---|---|
| **Qwen3-VL-2B 4-bit (the app's)** | 19/24 | 0.45 s | 1.8 GB |
| Qwen3-VL-2B 8-bit | 19/24 | 0.63 s | 2.7 GB |
| Qwen3.5-2B 4-bit | 20/24 | 0.84 s | 1.75 GB |
| Qwen3-VL-4B 4-bit | 19/24 | 0.78 s | 3.1 GB |
| Gemma 3 4B QAT (Rytability's) | 20/24 | 2.2 s | 3.0 GB |
| LFM2.5-VL-1.6B | 13/24 | 0.89 s | 1.5 GB |
| Qwen3.5-0.8B | 12/24 | 0.59 s | 0.65 GB |

(Gemma 4 E2B's MLX build doesn't load in this mlx-swift-lm.) No model of the
same size reads clearly better. What made the difference was the hint: given
the *correct* literal reading, Qwen3-VL-2B gets **23/24** and Qwen3.5-2B 24/24.
So the weak step was the literal reading, which is why ML Kit now does it.

On the 12 lines whose strokes we have, ML Kit read 8 exactly (Vision's
readings were like "frecieue my freind"), in 10-30 ms. What was meant, from
ML Kit's reading:

| | Right | Time |
|---|---|---|
| Qwen3-VL-2B, looking at the ink too (the app) | 10/12 | 0.40 s |
| Qwen3-VL-2B, text only | 9/12 | 0.14 s |
| Gemma 3 4B (Rytability's), text only | 9/12 | 0.39 s |
| Qwen3.5-0.8B, text only | 6/12 | 0.17 s |
| System spellchecker, no AI | ~4/12 | 0.02 s |

The two it can't get are the same for every setup: a stray scribble in the
eval ink, and a near-illegible "thank you for comming to my party". This ink
is generated, so real handwriting (which ML Kit is trained on) should favour
ML Kit more.

## Before merging into Rytability

- **No second model needed:** ML Kit (13 MB) reads, and Rytability's own Gemma
  works out what was meant, as this app now does.

- Measure it on the devices: time per passage and memory. Gemma needs about
  3 GB, and nearer 4 GB while it looks at a picture; the app asks for the
  increased memory limit, as Rytability does.
- The page would become native SwiftUI + PencilKit. The ink smoothing and Tidy
  are in `Web.bundle/js`; Engine/ can be used as it is.
