import CoreGraphics
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Vision

/// Reads one line of handwriting: what is literally written, and what the
/// writer meant. Ported from the Mac demo (`server/recognize.py`).
protocol HandwritingReader: Sendable {
    /// Shown in the page's status line.
    var description: String { get }
    func read(_ image: CGImage) async throws -> (written: String, meant: String)
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

/// Vision on its own: reads, but can't fix spelling. Used when the Qwen model
/// isn't available (the Simulator, or the model wasn't copied into the app).
struct VisionReader: HandwritingReader {
    let description: String

    func read(_ image: CGImage) async throws -> (written: String, meant: String) {
        let text = VisionOCR.read(image) ?? ""
        return (text, text)
    }
}

/// Qwen3-VL-2B-Instruct (4-bit MLX), the same model and prompt as the Mac demo.
/// One pass returns both a literal reading and the intended sentence, because
/// it sees the pixels and knows English ("wether" -> "weather").
final class QwenReader: HandwritingReader, @unchecked Sendable {
    static let prompt =
        "This image is one line of handwriting by a person with dyslexia; it may be messy and misspelled. "
        + "Reply in exactly this format:\n"
        + "WRITTEN: <exactly what is written, letter by letter, keeping spelling mistakes>\n"
        + "MEANT: <the same words spelled correctly. Fix misspellings and wrong homophones (their/there, to/too, "
        + "your/you're) from context. Keep the same words in the same order; do not add or remove words>"

    /// Vision's literal reading, put BEFORE the instructions (after them, the
    /// model echoed it back and took twice as long). It took 2B from 14 to 19
    /// of the 24 eval lines on the Mac.
    static func ocrHint(_ ocr: String) -> String {
        "An OCR engine read this line as \"\(ocr)\", but it is often wrong about single letters, so trust the image.\n"
    }

    let description: String
    private let container: ModelContainer
    /// Whether to ask Vision for a second opinion first.
    var useOCRHint = true

    init(modelDirectory: URL, description: String) async throws {
        container = try await VLMModelFactory.shared.loadContainer(from: modelDirectory, using: TransformersTokenizerLoader())
        self.description = description
    }

    func read(_ image: CGImage) async throws -> (written: String, meant: String) {
        let ocr = useOCRHint ? (VisionOCR.read(image) ?? "") : ""
        let prompt = (ocr.isEmpty ? "" : Self.ocrHint(ocr)) + Self.prompt
        return Self.parse(try await generate(prompt, image: image))
    }

    func generate(_ prompt: String, image: CGImage?, maxTokens: Int = 96) async throws -> String {
        // greedy, like the Mac demo; no resize: Qwen's processor keeps the line's aspect ratio
        let session = ChatSession(container, generateParameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
                                  processing: UserInput.Processing(resize: nil))
        let reply = try await session.respond(to: prompt, images: image.map { [.ciImage(CIImage(cgImage: $0))] } ?? [],
                                              videos: [])
        await session.clear()
        return reply
    }

    /// "WRITTEN: ... / MEANT: ..." -> (written, meant), tolerant of a sloppy reply.
    static func parse(_ reply: String) -> (written: String, meant: String) {
        var written: String?, meant: String?
        for line in reply.split(separator: "\n", omittingEmptySubsequences: false) {
            let s = line.trimmingCharacters(in: .whitespaces)
            guard let colon = s.firstIndex(of: ":") else { continue }
            let key = s[..<colon].trimmingCharacters(in: .whitespaces).uppercased()
            let value = s[s.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            if key == "WRITTEN" { written = value } else if key == "MEANT" { meant = value }
        }
        let trimmed = reply.trimmingCharacters(in: .whitespacesAndNewlines)
        var w = written ?? trimmed.split(separator: "\n").first.map(String.init) ?? ""
        let m = meant ?? w
        // the small model sometimes letter-spaces its literal reading ("h e l l o")
        if w.range(of: #"^(\S )+\S$"#, options: .regularExpression) != nil { w = m }
        let strip = CharacterSet(charactersIn: " \"")
        return (w.trimmingCharacters(in: strip), m.trimmingCharacters(in: strip))
    }
}
