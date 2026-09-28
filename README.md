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
scripts/get_model.sh                                          # -> the 1.7 GB reader, Qwen3VLModel.bundle
scripts/get_mlkit_model.sh                                    # -> ML Kit's 13 MB English model, MLKitEnglishModel.bundle
```

Three things are deliberately not in git: the reader (1.7 GB, too big), ML
Kit's English model (Google's files, fetched from dl.google.com), and the
synthesiser's weights (they come from a repo with no licence and were trained
on IAM-OnDB, which is for non-commercial research only, so each machine
fetches its own copy, as the Mac demo does). The Xcode project also expects
Rytability's vendored packages at `~/Desktop/Files/App Work/dependencies`.

## Run it on your iPhone or iPad

1. Open `HandwritingMagic.xcodeproj` in Xcode.
2. Pick the **HandwritingMagic** scheme and your device at the top, then press ▶︎.
   Signing uses the Rytability team (85P7KNQ5P3) and the bundle id
   `com.RytechLabsLLC.HandwritingMagic`; Xcode registers it on the first run.
3. The first install copies the 1.7 GB reader to the device, so it takes a
   minute. At launch the status (bottom right) says *Loading the handwriting
   reader…* for a few seconds, then turns green: *Ready · Qwen3-VL-2B on this iPad*.
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
  Qwen3VLModel.bundle/       Qwen3-VL-2B-Instruct, 4-bit MLX: what was meant (not in git)
  MLKitEnglishModel.bundle/  ML Kit's English handwriting model (FileData not in git)
Engine/                      the pipeline, in Swift (shared with MagicCheck)
  MagicEngine.swift          read -> write in your style + a clean style -> proofread
  InkReader.swift            Google ML Kit Digital Ink: reads the pen strokes literally
  Readers.swift              Qwen3-VL-2B via MLX (what was meant), Apple Vision
  SpellFixer.swift           the system spellchecker, for when there's no AI model
  HandwritingSynth.swift     Graves handwriting synthesis on Accelerate
  Ink.swift, TextTools.swift ink geometry, spelling diff, text helpers
MagicCheck/                  a Mac command-line tool: runs Engine/ on test ink
scripts/                     get_model.sh, sync_web.sh, export_synth.py
```

Reading a line takes two steps. **ML Kit Digital Ink Recognition** reads the
pen strokes (where the pen went, in what order) and returns what is literally
written, misspellings and all, in ~20 ms. **Qwen3-VL-2B** then looks at the
ink with that reading as a hint and works out what was meant ("their
tomorow" -> "there tomorrow"). Without ML Kit, Apple Vision reads a picture of
the ink instead.

The page talks to `magic://app/api/rewrite` exactly as it talks to the Python
server on the Mac, so both run the same `web/`. The Swift engine is a port of
`server/app.py`, `recognize.py` and `synth.py`: same prompt, same "your style
or the clean *Tall narrow print* style, whichever reads back better" rule, 8
candidates of each, proofread by Vision.

ML Kit comes from [d-date/google-mlkit-swiftpm](https://github.com/d-date/google-mlkit-swiftpm)
9.0.2, a community Swift Package of Google's ML Kit binaries (Google only
publishes it for CocoaPods). It needs `-ObjC -all_load` in Other Linker Flags
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

## Keeping it in step with the Mac demo

- Changed the page? `scripts/sync_web.sh` copies `../HandwritingMagic/web` in.
- Fresh checkout, or the model is missing? `scripts/get_model.sh` copies it from
  your Hugging Face cache, or downloads it.
- Changed the synthesiser's weights or styles? `python scripts/export_synth.py`.

## Checking the pipeline on the Mac

MLX can't run in the Simulator, so `MagicCheck` runs the app's `Engine/`
code on the Mac demo's messy test lines:

```bash
cd ~/Desktop/HandwritingMagic && .venv/bin/python tests/make_evalset.py /tmp/evalset
cd ~/Desktop/HandwritingMagic-iOS
xcodebuild -scheme MagicCheck -configuration Release -destination platform=macOS \
  -derivedDataPath /tmp/hm-dd build
/tmp/hm-dd/Build/Products/Release/MagicCheck /tmp/evalset /tmp/magiccheck
```

On the M5 MacBook Pro (28 Sep 2026) it got the intended sentence on 9 of the
12 lines at 1.1 s a line; the Python demo gets 10/12 at 1.8 s. The misses are
the same hard ones: "homwork" drawn as "hanwak", and a near-illegible
"thank you for comming to my party". The synthesiser writes 16 candidates in
about 0.3 s on the Mac. The iOS Simulator is ~3 s a line; it has no GPU for
MLX and runs Vision on the CPU, so real devices should be quicker, but that
still needs measuring.

## Which model reads best? (28 Sep 2026)

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

- **No second model needed:** ML Kit (13 MB) reads, and Rytability's Gemma,
  text only, works out what was meant: 9/12 above, one behind Qwen with the ink.

- Measure it on the devices: time per line and memory. The reader needs about
  2 GB; the app asks for the increased memory limit, as Rytability does.
- The page would become native SwiftUI + PencilKit. The ink smoothing and Tidy
  are in `Web.bundle/js`; Engine/ can be used as it is.
