# Handwriting Magic for iPhone and iPad

The Mac demo (`~/Desktop/HandwritingMagic`), running on a touchscreen. Write with
your finger or an Apple Pencil. When you pause, your scrawl dissolves into
sparkles and comes back neat, spelled correctly, and in your handwriting.
Everything runs on the device; it works in Airplane Mode.

This is a playground for trying the idea on a real screen before any of it
goes into Rytability.

## Run it on your iPhone or iPad

1. Open `HandwritingMagic.xcodeproj` in Xcode.
2. Pick the **HandwritingMagic** scheme and your device at the top, then press ▶︎.
   Signing uses the Rytability team (85P7KNQ5P3) and the bundle id
   `com.RytechLabsLLC.HandwritingMagic`; Xcode registers it on the first run.
3. The first install copies the 1.7 GB reader to the device, so it takes a
   minute. At launch the status (bottom right) says *Loading the handwriting
   reader…* for a few seconds, then turns green: *Ready · Qwen3-VL-2B on this iPad*.
   Tidy works straight away.

The Simulator runs the app too, but MLX needs a real GPU, so there it reads
with Apple Vision only (no spelling fixes) and is slower. Use it for layout;
use a device for the real thing.

The page and its controls are the Mac demo's (Magic / Tidy / Off, Neatness,
the 13 named styles, Fix spelling, 👁 hold to see what you wrote, ↶ undo, ✕ clear).
After the first Pencil stroke, fingers stop drawing (palm rejection).

## How it's put together

```
HandwritingMagic/            the app
  App/                       SwiftUI shell: a web view showing the page, and
                             PageServer, which answers the page's /api calls
  Web.bundle/                the page, copied from the Mac demo's web/
  Resources/hand_synth.*     the handwriting synthesiser's weights + 13 styles
  Qwen3VLModel.bundle/       the reader: Qwen3-VL-2B-Instruct, 4-bit MLX (not in git)
Engine/                      the pipeline, in Swift (shared with MagicCheck)
  MagicEngine.swift          read -> write in your style + a clean style -> proofread
  Readers.swift              Qwen3-VL-2B via MLX, and Apple Vision
  HandwritingSynth.swift     Graves handwriting synthesis on Accelerate
  Ink.swift, TextTools.swift ink geometry, spelling diff, text helpers
MagicCheck/                  a Mac command-line tool: runs Engine/ on test ink
scripts/                     get_model.sh, sync_web.sh, export_synth.py
```

The page talks to `magic://app/api/rewrite` exactly as it talks to the Python
server on the Mac, so both run the same `web/`. The Swift engine is a port of
`server/app.py`, `recognize.py` and `synth.py`: same prompt, same "your style
or the clean *Tall narrow print* style, whichever reads back better" rule, 8
candidates of each, proofread by Vision.

Packages come from Rytability's vendored copies in
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

## Before merging into Rytability

- Measure it on the devices: time per line and memory. The reader needs about
  2 GB; the app asks for the increased memory limit, as Rytability does.
- Rytability's model is Gemma, bundled the same way (`GemmaMLXModel.bundle`).
  Two models in one app is 4+ GB, so decide whether Gemma can do the reading,
  or whether the reader is a downloadable add-on.
- The page would become native SwiftUI + PencilKit. The ink smoothing and Tidy
  are in `Web.bundle/js`; Engine/ can be used as it is.
