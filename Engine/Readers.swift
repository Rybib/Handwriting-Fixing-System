import CoreGraphics
import CoreImage
import Foundation
import Metal
import MLX
import MLXLMCommon
import MLXVLM
import Vision

/// Works out what a passage of handwriting says: what is literally written on
/// each line, and what the writer meant. The lines are read together, as one
/// piece of writing, so a word can be worked out from the lines around it.
/// `literals` are a stroke recogniser's (ML Kit) or OCR's reading of each line;
/// `passage` shows the whole passage, `lines` each line on its own.
protocol HandwritingReader: Sendable {
    /// Shown in the page's status line.
    var description: String { get }
    /// useInk: false = don't look at the pictures, only the literal readings (faster)
    func read(passage: CGImage, lines: [CGImage?], literals: [String?], useInk: Bool) async throws -> [(written: String, meant: String)]
}

/// Apple's on-device text recogniser (Vision). It reads exactly what is on
/// the page in a few tens of milliseconds, and it is independent of the
/// reader, which tends to read *through* small glitches. So it proofreads the
/// rewrites and gives the reader a second opinion.
enum VisionOCR {
    static func read(_ image: CGImage) -> String? {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false      // we want what was WRITTEN
        request.recognitionLanguages = ["en-US"]
        do {
            try VNImageRequestHandler(cgImage: image).perform([request])
        } catch {
            return nil
        }
        let pieces = (request.results ?? []).compactMap { obs in
            obs.topCandidates(1).first.map { (obs.boundingBox.minX, $0.string) }
        }
        return pieces.sorted { $0.0 < $1.0 }.map(\.1).joined(separator: " ")
    }

    /// Whether Vision works here (it can't create its model in some Simulators).
    static func works() -> Bool {
        let hello: [Stroke] = [[SIMD2(0, 0), SIMD2(0, 40)], [SIMD2(0, 20), SIMD2(20, 20)], [SIMD2(20, 0), SIMD2(20, 40)],
                               [SIMD2(34, 0), SIMD2(34, 40)]]
        guard let image = Ink.render(hello) else { return false }
        return read(image) != nil
    }
}

/// No AI model: the literal reading, with the system spellchecker for what
/// was meant. Used where Gemma can't run (the Simulator, too little memory, or
/// the model wasn't copied into the app). It fixes "freind", but not "their"
/// for "there".
struct SpellcheckReader: HandwritingReader {
    let description: String

    func read(passage: CGImage, lines: [CGImage?], literals: [String?], useInk: Bool) async throws -> [(written: String, meant: String)] {
        literals.map { ($0 ?? "", SpellFixer.fix($0 ?? "")) }
    }
}

/// Gemma 3 4B (4-bit QAT, MLX): the same model Rytability carries, with its
/// vision tower. It gets the recogniser's reading of every line AND a picture
/// of the ink, and returns each line as written and as meant. Knowing English
/// and seeing the whole passage, it can tell "their" from "there" and a
/// misread letter from a misspelling.
final class GemmaReader: HandwritingReader, @unchecked Sendable {
    var description: String
    /// Whether the vision tower loaded (model-vision.safetensors is in the model folder).
    let seesInk: Bool
    private let container: ModelContainer

    #if os(iOS)
    /// GPUs without native bfloat16 (A14 / M1 and older) make NaN garbage of
    /// Gemma's bf16 maths, so there it runs in float32, as in Rytability.
    static let needsFloat32Compute: Bool = {
        guard let device = MTLCreateSystemDefaultDevice() else { return true }
        return !device.supportsFamily(.apple8)
    }()
    #endif

    init(modelDirectory: URL, description: String) async throws {
        let fm = FileManager.default
        // Rytability's split: text weights + the vision tower in a file of its own
        // (or one model.safetensors with both)
        seesInk = fm.fileExists(atPath: modelDirectory.appendingPathComponent("model-vision.safetensors").path)
            || fm.fileExists(atPath: modelDirectory.appendingPathComponent("model.safetensors").path)
        // Rytability's Gemma 3, whose vision attention stays on the fused Metal
        // kernel (the stock one spikes ~2.5 GB on every picture; see Gemma3MemoryPatched.swift)
        await Gemma3MemoryPatch.register()
        Gemma3MemoryPatch.includeVisionTower = seesInk
        container = try await VLMModelFactory.shared.loadContainer(from: modelDirectory, using: TransformersTokenizerLoader())
        #if os(iOS)
        if Self.needsFloat32Compute {
            await container.update { context in
                context.model.apply { array in
                    switch array.dtype {
                    case .bfloat16, .float16: return array.asType(.float32)
                    default: return array
                    }
                }
                eval(context.model.parameters())
            }
        }
        #endif
        self.description = description
    }

    /// The recogniser's readings go BEFORE the instructions (after them, small
    /// models echo them back and take twice as long).
    static func prompt(_ literals: [String], seesInk: Bool) -> String {
        let n = literals.count
        let lines = n == 1 ? "one line" : "\(n) lines"
        var p = seesInk
            ? "The image shows handwriting by a person with dyslexia: \(lines) of one piece of writing, read top to bottom"
                + (n > 1 ? " (a long line may be wrapped in the image)" : "") + ". It may be messy and misspelled.\n"
            : "Here is handwriting by a person with dyslexia: \(lines) of one piece of writing. It may be messy and misspelled.\n"
        p += "A handwriting recognizer read \(n == 1 ? "it" : "the lines") as:\n"
        for (i, l) in literals.enumerated() { p += "\(i + 1): \"\(l)\"\n" }
        if seesInk { p += "The recognizer is often wrong about single letters, so check every word against the image.\n" }
        p += n == 1 ? "\nReply with exactly two lines, in this format:\n" : "\nReply with exactly two lines for each of the \(n) lines, in this format:\n"
        p += "1 WRITTEN: <exactly what is written on line 1, letter by letter, keeping spelling mistakes>\n"
        p += "1 MEANT: <line 1 spelled correctly>\n"
        if n > 1 { p += "2 WRITTEN: ...\n2 MEANT: ...\nand so on.\n" }
        p += "In MEANT, fix misspellings and wrong homophones (their/there, to/too, your/you're) using the whole passage "
            + "as context"
            + (n > 1 ? ", since a sentence can carry on from one line to the next" : "")
            + ". Keep the same words in the same order and on the same line; do not add, remove or move words. "
            + "Reply with nothing else."
        return p
    }

    func read(passage: CGImage, lines: [CGImage?], literals: [String?], useInk: Bool) async throws -> [(written: String, meant: String)] {
        let hints = literals.map { $0 ?? "" }
        let seesInk = seesInk && useInk
        let reply = try await generate(Self.prompt(hints, seesInk: seesInk), image: seesInk ? passage : nil,
                                       maxTokens: Self.budget(hints))
        var parsed = Self.parse(reply, lines: hints.count)
        // A line whose reading has little to do with what the recogniser saw
        // there was probably shifted onto a neighbour's line (it merged or split
        // lines). Read it again on its own, with the passage around it as context.
        if hints.count > 1 {
            for i in hints.indices where Self.drifted(parsed[i], hint: hints[i]) {
                let context = hints.indices.filter { $0 != i }.map { parsed[$0].meant ?? hints[$0] }
                let image = seesInk ? lines[i] : nil
                let again = try await generate(Self.linePrompt(hints[i], context: context, seesInk: image != nil),
                                               image: image, maxTokens: Self.budget([hints[i]]))
                parsed[i] = Self.parse(again, lines: 1)[0]
            }
        }
        let hasWords = { (s: String) in s.rangeOfCharacter(from: .alphanumerics) != nil }
        return zip(hints, parsed).map { hint, got in
            // a line it skipped, or left wordless: the recogniser's reading, spellchecked, is better than nothing
            var written = got.written.flatMap { hasWords($0) ? $0 : nil } ?? hint
            let meant = got.meant.flatMap { hasWords($0) ? $0 : nil } ?? SpellFixer.fix(written)
            // it sometimes letter-spaces its literal reading ("h e l l o")
            if written.range(of: #"^(\S )+\S$"#, options: .regularExpression) != nil { written = hint.isEmpty ? meant : hint }
            return (written, meant)
        }
    }

    /// Two readings a line, a few tokens a word.
    static func budget(_ hints: [String]) -> Int { min(1200, hints.reduce(24) { $0 + 2 * ($1.count / 2 + 12) }) }

    static func drifted(_ got: (written: String?, meant: String?), hint: String) -> Bool {
        guard let written = got.written, got.meant != nil else { return true }
        return hint.filter(\.isLetter).count >= 4 && TextTools.cer(written, hint) > 0.5
    }

    /// One line, knowing what the rest of the passage says.
    static func linePrompt(_ hint: String, context: [String], seesInk: Bool) -> String {
        var p = seesInk ? "The image shows one line of handwriting by a person with dyslexia. " : "Here is one line of handwriting by a person with dyslexia. "
        p += "It may be messy and misspelled. It comes from a longer piece of writing; the other lines say: "
            + context.map { "\"\($0)\"" }.joined(separator: " ") + "\n"
        if !hint.isEmpty {
            p += "A handwriting recognizer read this line as \"\(hint)\"" + (seesInk ? ", but it is often wrong about single letters, so trust the image" : "") + ".\n"
        }
        return p + "\nReply in exactly this format:\n"
            + "WRITTEN: <exactly what is written on this line, letter by letter, keeping spelling mistakes>\n"
            + "MEANT: <the same words spelled correctly. Fix misspellings and wrong homophones (their/there, to/too, "
            + "your/you're) from context. Keep the same words in the same order; do not add or remove words>"
    }

    func generate(_ prompt: String, image: CGImage?, maxTokens: Int = 96) async throws -> String {
        // greedy: the same passage should always read the same way
        let session = ChatSession(container, generateParameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
                                  processing: UserInput.Processing(resize: nil))
        let reply = try await session.respond(to: prompt, images: image.map { [.ciImage(CIImage(cgImage: $0))] } ?? [],
                                              videos: [])
        await session.clear()
        return reply
    }

    /// "1 WRITTEN: ... / 1 MEANT: ... / 2 WRITTEN: ..." -> each line's (written, meant),
    /// nil where the reply skipped it. Tolerant of "Line 1", "1.", "**WRITTEN**:"
    /// and, for a single line, no numbers at all.
    static func parse(_ reply: String, lines n: Int) -> [(written: String?, meant: String?)] {
        var out = [(written: String?, meant: String?)](repeating: (nil, nil), count: n)
        let reply = reply.replacingOccurrences(of: #"(?s)<think>.*?</think>"#, with: "", options: .regularExpression)
        let label = try! NSRegularExpression(
            pattern: #"^[\s>*#-]*(?:line\s*)?(\d+)?\s*[.):]?\s*\**\s*(WRITTEN|MEANT)\s*\**\s*[:：]\**\s*(.*)$"#,
            options: [.caseInsensitive, .anchorsMatchLines])
        let ns = reply as NSString
        var lastLine = 0
        for m in label.matches(in: reply, range: NSRange(location: 0, length: ns.length)) {
            let kind = ns.substring(with: m.range(at: 2)).uppercased()
            var i: Int
            if m.range(at: 1).location != NSNotFound, let k = Int(ns.substring(with: m.range(at: 1))) {
                i = k - 1
            } else {
                // unnumbered: WRITTEN starts the next line
                i = kind == "WRITTEN" ? (out[min(lastLine, n - 1)].written == nil ? lastLine : lastLine + 1) : lastLine
            }
            guard i >= 0, i < n else { continue }
            lastLine = i
            let value = ns.substring(with: m.range(at: 3)).trimmingCharacters(in: CharacterSet(charactersIn: " \"*"))
            if kind == "WRITTEN" { out[i].written = value } else { out[i].meant = value }
        }
        return out
    }
}
