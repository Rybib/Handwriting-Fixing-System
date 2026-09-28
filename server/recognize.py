"""Reading the handwriting, and working out what the writer meant.

Readers (pick with --reader):
  vlm    Qwen3-VL (default 2B; HWFIX_VLM overrides). One pass reads the ink
         literally AND infers the intended, correctly spelled sentence, using
         both the pixels and language context. Runs on-device (Apple Silicon
         GPU via MPS, or CPU).
  apple  macOS Vision framework (VNRecognizeTextRequest) via pyobjc. Same
         on-device recogniser family as PencilKit's PKStrokeRecognizer on
         iPadOS, which is what the real iPad app would use. The "meant" text
         then comes from the VLM in text-only mode if loaded, else a
         dictionary spellchecker.
"""

import difflib
import os
import platform
import re
import threading

PROMPT = (
    "This image is one line of handwriting by a person with dyslexia; it may be messy and misspelled. "
    "Reply in exactly this format:\n"
    "WRITTEN: <exactly what is written, letter by letter, keeping spelling mistakes>\n"
    "MEANT: <the same words spelled correctly. Fix misspellings and wrong homophones (their/there, to/too, "
    "your/you're) from context. Keep the same words in the same order; do not add or remove words>"
)

# A literal OCR reading (macOS Vision) as a second opinion, put BEFORE the
# instructions: after them, the model echoed it back and took 2.3 s instead of
# 1 s. It took Qwen3-VL-2B from 14 to 19 of the 24 eval lines (raw and browser
# ink from tests/make_evalset.py); 4B was unchanged at 19.
OCR_HINT = "An OCR engine read this line as \"{ocr}\", but it is often wrong about single letters, so trust the image.\n"

LITERAL_PROMPT = "Transcribe this handwriting exactly, letter by letter. Reply with only the text."

TEXT_FIX_PROMPT = (
    "A person with dyslexia handwrote the text below and it was read by OCR, so it may contain "
    "misspellings, OCR errors or missing spaces. Rewrite it as the sentence they meant, with correct "
    "spelling and capitalization, changing as little as possible. Reply with only the corrected text.\n\n"
    "Text: {text}"
)


def _parse(reply):
    written = meant = None
    for line in reply.splitlines():
        m = re.match(r"\s*(WRITTEN|MEANT)\s*:\s*(.*)", line, re.I)
        if m:
            if m.group(1).upper() == "WRITTEN":
                written = m.group(2).strip()
            else:
                meant = m.group(2).strip()
    if written is None:
        written = reply.strip().splitlines()[0] if reply.strip() else ""
    if meant is None:
        meant = written
    # the small model sometimes letter-spaces its literal reading ("h e l l o")
    if re.fullmatch(r"(\S )+\S", written or ""):
        written = meant
    return written.strip(' "'), meant.strip(' "')


def default_vlm():
    """Qwen3-VL-2B: the same model the iPhone/iPad app runs. With macOS Vision's
    second opinion it reads as well as 4B (19 of the 24 eval lines each) at
    half the size and time. HWFIX_VLM=Qwen/Qwen3-VL-4B-Instruct for 4B."""
    return os.environ.get("HWFIX_VLM") or "Qwen/Qwen3-VL-2B-Instruct"


class VLMReader:
    name = "vlm"

    def __init__(self, model_id=None):
        import torch
        from transformers import AutoProcessor

        self.model_id = model_id or default_vlm()
        if torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.bfloat16
        elif torch.cuda.is_available():
            self.device, dtype = "cuda", torch.bfloat16
        else:
            self.device, dtype = "cpu", torch.float32
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.torch = torch
        self.ocr = None         # optional literal OCR function, see OCR_HINT
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.lock = threading.Lock()
        self._load(dtype)
        try:
            self._generate([{"type": "text", "text": "Say OK."}], 4)     # smoke test / warm-up
        except Exception as e:  # noqa: BLE001  (older macOS: no bfloat16 on the GPU)
            if dtype == torch.float32:
                raise
            print(f"[reader] {dtype} failed on {self.device} ({e}); retrying in float32")
            self._load(torch.float32)

    def _load(self, dtype):
        from transformers import AutoModelForImageTextToText
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, dtype=dtype).to(self.device).eval()

    def _generate(self, content, max_new_tokens=96):
        msgs = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        with self.lock, self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def read(self, image):
        ocr = self.ocr(image) if self.ocr else ""
        prompt = (OCR_HINT.format(ocr=ocr) if ocr else "") + PROMPT
        return _parse(self._generate([{"type": "image", "image": image}, {"type": "text", "text": prompt}]))

    def read_literal(self, image):
        return self._generate([{"type": "image", "image": image}, {"type": "text", "text": LITERAL_PROMPT}], 60).strip()

    def fix_text(self, text):
        reply = self._generate([{"type": "text", "text": TEXT_FIX_PROMPT.format(text=text)}], 80)
        return reply.strip().splitlines()[0].strip(' "') if reply.strip() else text


class AppleVisionReader:
    """macOS only. Uses the system handwriting recogniser (no download)."""

    name = "apple"

    def __init__(self, fixer=None):
        if platform.system() != "Darwin":
            raise RuntimeError("Apple Vision is only available on macOS")
        import Vision  # noqa: F401  (pyobjc-framework-Vision)
        from Foundation import NSData  # noqa: F401
        self.fixer = fixer

    def recognize(self, image):
        import io

        import Vision
        from Foundation import NSData

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)   # we want what was WRITTEN
        request.setRecognitionLanguages_(["en-US"])
        ok, err = handler.performRequests_error_([request], None)
        if not ok:
            raise RuntimeError(f"Vision failed: {err}")
        pieces = []
        for obs in request.results() or []:
            cands = obs.topCandidates_(1)
            if cands and len(cands):
                box = obs.boundingBox()
                pieces.append((box.origin.x, str(cands[0].string())))
        return " ".join(t for _, t in sorted(pieces))

    def read_literal(self, image):
        return self.recognize(image)

    def read(self, image):
        written = self.recognize(image)
        meant = self.fixer(written) if (self.fixer and written) else written
        return written, meant


class SpellFixer:
    """Dictionary fallback when no language model is loaded."""

    def __init__(self):
        from spellchecker import SpellChecker
        self.sc = SpellChecker()

    def __call__(self, text):
        out = []
        for tok in re.findall(r"\w+|\W+", text):
            if tok.isalpha() and tok.lower() not in self.sc:
                fix = self.sc.correction(tok.lower()) or tok
                if tok[0].isupper():
                    fix = fix.capitalize()
                out.append(fix)
            else:
                out.append(tok)
        return "".join(out)


def cer(a, b):
    """Character error rate of reading `a` against reference `b` (letters/digits only)."""
    a = re.sub(r"[^a-z0-9]", "", a.lower())
    b = re.sub(r"[^a-z0-9]", "", b.lower())
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(b))


def word_corrections(written, meant):
    """[(from, to), ...] word-level differences, for the UI."""
    a, b = written.split(), meant.split()
    sm = difflib.SequenceMatcher(a=[w.lower().strip(".,!?;:") for w in a],
                                 b=[w.lower().strip(".,!?;:") for w in b], autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            out.append([" ".join(a[i1:i2]), " ".join(b[j1:j2])])
    return out


def build_proofreader(reader):
    """A strict, literal reader for checking the rewritten ink: (read(image), fast).

    macOS Vision reads exactly what is on the page in ~40 ms and is independent
    of the VLM, which tends to read *through* small glitches in the output.
    Elsewhere the VLM reads literally (slower, so fewer candidates are checked).
    """
    if platform.system() == "Darwin":
        try:
            return AppleVisionReader().recognize, True
        except Exception as e:  # noqa: BLE001
            print(f"[reader] macOS Vision unavailable for proofreading ({e})")
    if reader is not None:
        return reader.read_literal, False
    return None, False


def build_reader(kind="vlm"):
    """Returns (reader, description). Falls back gracefully."""
    if kind == "apple":
        fixer = None
        try:
            fixer = VLMReader().fix_text
        except Exception as e:  # noqa: BLE001
            print(f"[reader] VLM text fixer unavailable ({e}); using dictionary spellchecker")
            try:
                fixer = SpellFixer()
            except Exception:  # noqa: BLE001
                fixer = None
        return AppleVisionReader(fixer=fixer), "Apple Vision (macOS) + " + ("LLM fix" if fixer else "no fix")
    reader = VLMReader()
    return reader, f"{reader.model_id} on {reader.device}"
