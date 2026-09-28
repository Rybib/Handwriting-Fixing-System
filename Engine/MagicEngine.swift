import CoreGraphics
import Foundation
import MLX
import os
#if os(iOS)
import UIKit
#endif

/// The whole Magic pipeline, the Swift twin of the Mac demo's server
/// (`server/app.py`): it answers the page's `/api/status` and `/api/rewrite`.
///
///   read the line: ML Kit reads the pen strokes literally, then Qwen3-VL-2B
///   looks at the ink too and works out what was meant (spelling, their/there)
///   -> write what was meant, in the user's style AND a clean built-in style
///   -> Vision proofreads the candidates; the user's style wins if it reads
///      back perfectly, else whichever reads best
final class MagicEngine: @unchecked Sendable {
    static let shared = MagicEngine()

    struct Status: Sendable {
        var loading = true
        var phase = "Starting"
        var reader: String?
        var synth = false
        var error: String?

        var json: [String: Any] {
            ["loading": loading, "phase": phase, "progress": NSNull(), "synth": synth,
             "reader": reader ?? NSNull(), "error": error ?? NSNull()]
        }
    }

    private let state = OSAllocatedUnfairLock(initialState: Status())
    var status: Status { state.withLock { $0 } }

    private var synth: HandwritingSynth?
    private var reader: HandwritingReader?
    #if canImport(MLKitDigitalInkRecognition)
    private var inkReader: InkReader?
    #endif
    private var literalSource = "Apple Vision"
    private var proofreads = false
    /// One rewrite at a time: they share the CPU and GPU.
    private let queue = AsyncQueue()

    private func update(_ change: (inout Status) -> Void) { state.withLock { change(&$0) } }

    // MARK: - loading

    /// Loads everything in the background; you can write meanwhile.
    /// nil folders: the app's own (hand_synth.bin/json, Qwen3VLModel.bundle).
    func load(synthDirectory: URL? = nil, modelDirectory: URL? = nil) async {
        update { $0.phase = "Getting the handwriting synthesiser" }
        do {
            synth = try synthDirectory.map {
                try HandwritingSynth(weightsURL: $0.appendingPathComponent("hand_synth.bin"),
                                     indexURL: $0.appendingPathComponent("hand_synth.json"))
            } ?? HandwritingSynth()
            update { $0.synth = true }
        } catch {
            update { $0.error = "synthesis model failed: \(error.localizedDescription)" }
        }
        proofreads = VisionOCR.works()
        #if canImport(MLKitDigitalInkRecognition)
        update { $0.phase = "Getting Google's handwriting recognizer ready" }
        do {
            inkReader = try await InkReader.load()
            literalSource = "ML Kit"
        } catch {
            print("[models] ML Kit unavailable, reading with Vision: \(error)")
        }
        #endif
        reader = await loadReader(modelDirectory ?? Bundle.main.url(forResource: "Qwen3VLModel", withExtension: "bundle"))
        update {
            $0.reader = self.reader?.description
            $0.loading = false
            $0.phase = "Ready"
        }
    }

    private func loadReader(_ modelDirectory: URL?) async -> HandwritingReader? {
        let canRead = proofreads || literalSource != "Apple Vision"
        #if targetEnvironment(simulator)
        // MLX needs a real GPU: the Simulator has no AI model
        return canRead ? SpellcheckReader(description: "\(literalSource) + spellchecker (Simulator: no AI fixes)") : nil
        #else
        let fallback = canRead
            ? SpellcheckReader(description: "\(literalSource) + spellchecker, no AI fixes (Qwen3-VL-2B isn't in the app: run scripts/get_model.sh)")
            : nil
        guard let dir = modelDirectory, FileManager.default.fileExists(atPath: dir.appendingPathComponent("model.safetensors").path) else {
            if fallback == nil { update { $0.error = "Qwen3-VL-2B isn't in the app (run scripts/get_model.sh)" } }
            return fallback
        }
        do {
            update { $0.phase = "Loading the handwriting reader into memory" }
            // MLX recycles freed GPU buffers up to this size; keep it small next to 1.7 GB of weights
            MLX.Memory.cacheLimit = 128 * 1024 * 1024
            let qwen = try await QwenReader(modelDirectory: dir, description: "\(literalSource) + Qwen3-VL-2B on this \(Self.deviceName)")
            update { $0.phase = "Warming up the handwriting reader" }
            let blank = Ink.render([[SIMD2(0, 0), SIMD2(40, 0)]])!
            _ = try await qwen.generate("Say OK.", image: blank, maxTokens: 4)
            return qwen
        } catch {
            update { $0.error = "the reader didn't load: \(error.localizedDescription)" }
            return fallback == nil ? nil : SpellcheckReader(description: "\(literalSource) + spellchecker, no AI fixes (Qwen3-VL-2B failed to load)")
        }
        #endif
    }

    static var deviceName: String {
        #if os(iOS)
        ProcessInfo.processInfo.isiOSAppOnMac ? "Mac" : (UIDevice.current.userInterfaceIdiom == .pad ? "iPad" : "iPhone")
        #else
        "Mac"
        #endif
    }

    // MARK: - rewriting

    struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    /// POST /api/rewrite, same request and reply as the Mac server:
    ///   {"lines": [{"strokes": [[[x, y], ...], ...], "times": [[s, ...], ...], "prime": [...]}], "style": "mine" | 0...12,
    ///    "bias": 2.28, "fix_spelling": true, "candidates": 8}
    ///   -> {"lines": [{"written", "meant", "text", "corrections", "strokes", "style_used", "timing"}]}
    /// Returned strokes: left edge x 0, baseline y 0, x-height 1, y down.
    func rewrite(_ body: Data) async throws -> Data {
        try await queue.run {
            guard let req = try JSONSerialization.jsonObject(with: body) as? [String: Any],
                  let lines = req["lines"] as? [[String: Any]] else { throw Failure("bad request") }
            return try JSONSerialization.data(withJSONObject: try await self.rewrite(req, lines: lines))
        }
    }

    private func rewrite(_ req: [String: Any], lines: [[String: Any]]) async throws -> [String: Any] {
        guard let synth else { throw Failure(status.error ?? "models still loading") }
        let style: Int? = (req["style"] as? String) == "mine" || req["style"] == nil ? nil : (req["style"] as? NSNumber)?.intValue
        let bias = Float((req["bias"] as? NSNumber)?.doubleValue ?? 1.5)
        let fix = (req["fix_spelling"] as? Bool) ?? true
        let nCand = (req["candidates"] as? NSNumber)?.intValue ?? 8

        var out: [[String: Any]] = []
        for line in lines {
            let strokes = Self.strokes(line["strokes"])
            let t0 = Date()
            let written: String, meant: String
            if let text = line["text"] as? String {         // caller already knows the text (tests)
                (written, meant) = (text, text)
            } else {
                guard let reader else { throw Failure(status.error ?? "handwriting reader still loading") }
                guard let image = Ink.render(strokes) else { throw Failure("no ink") }
                let literal = await literalReading(strokes, times: line["times"] as? [[Double]], image: image)
                (written, meant) = try await reader.read(image, literal: literal)
            }
            let tRead = Date().timeIntervalSince(t0)
            let target = TextTools.sanitize(fix ? meant : written)
            let writtenS = TextTools.sanitize(written)
            if target.range(of: "[A-Za-z0-9]", options: .regularExpression) == nil {   // a doodle: leave it alone
                out.append(["written": written, "meant": meant, "text": "", "corrections": [], "strokes": []])
                continue
            }

            let chunks = TextTools.wrap(target)
            let n = chunks.count
            let safePrime = synth.styles[style ?? HandwritingSynth.fallbackStyle]
            var minePrime: HandwritingSynth.Prime?
            if style == nil, !writtenS.isEmpty {
                let primeInk = Self.strokes(line["prime"] ?? line["strokes"])
                minePrime = Ink.userStrokesToPrime(primeInk, transcript: String(writtenS.prefix(HandwritingSynth.maxChars)))
            }

            let t1 = Date()
            var coords: [[PenPoint]] = []
            var styleUsed: Any = style.map { $0 as Any } ?? HandwritingSynth.fallbackStyle
            if let minePrime {
                // One batch: the user's own style AND a clean built-in style as a safety net.
                let res = synth.write(chunks + chunks, bias: bias, primes: Array(repeating: minePrime, count: n) + Array(repeating: safePrime, count: n),
                                      candidates: nCand)
                styleUsed = "mine"
                for (i, chunk) in chunks.enumerated() {
                    // Copying the user's style only works if their ink is legible and the network
                    // lined it up with its transcript; otherwise it faithfully copies the mess.
                    var best = res[i].primeAligned ? pick(res[i].candidates, chunk, limit: nCand) : nil
                    if best == nil || best!.cer > 0 {
                        let safe = pick(res[n + i].candidates, chunk, limit: nCand)
                        if best == nil || safe.cer < best!.cer { best = safe; styleUsed = HandwritingSynth.fallbackStyle }
                    }
                    coords.append(best!.coords)
                }
            } else {
                let res = synth.write(chunks, bias: bias, primes: Array(repeating: safePrime, count: n), candidates: nCand)
                coords = zip(res, chunks).map { pick($0.candidates, $1, limit: nCand).coords }
            }
            let tSynth = Date().timeIntervalSince(t1)

            // stitch the chunks into one long line; the page wraps it to its width
            var strokesOut: [[[Double]]] = []
            var xOff: Float = 0
            for c in coords {
                let norm = Ink.normaliseOutput(c)
                let w = norm.flatMap { $0 }.map(\.x).max() ?? 0
                strokesOut += norm.map { s in s.map { [Self.round4($0.x + xOff), Self.round4($0.y)] } }
                xOff += w + 0.9
            }
            out.append([
                "written": written, "meant": meant, "text": target,
                "corrections": fix ? TextTools.wordCorrections(written, target) : [],
                "strokes": strokesOut, "style_used": styleUsed,
                "timing": ["read": Self.round4(Float(tRead)), "synth": Self.round4(Float(tSynth))],
            ])
            print("[rewrite] read \(String(format: "%.1f", tRead))s synth \(String(format: "%.1f", tSynth))s  \(written) -> \(target)")
        }
        return ["lines": out]
    }

    /// What is literally written: ML Kit reads the pen strokes (and their
    /// timing); without it, Vision reads a picture of them.
    private func literalReading(_ strokes: [Stroke], times: [[Double]]?, image: CGImage) async -> String? {
        #if canImport(MLKitDigitalInkRecognition)
        if let inkReader, let best = await inkReader.read(strokes, times: times).first { return best }
        #endif
        return proofreads ? VisionOCR.read(image) : nil
    }

    /// The candidate that reads back best (ties: the better attention score).
    ///
    /// Candidates come sorted by attention score, so the first one Vision reads
    /// back perfectly wins. Without Vision, a poor attention score (a skipped
    /// letter, or the pen never lifting at the end) counts as a misreading.
    private func pick(_ cands: [HandwritingSynth.Candidate], _ text: String, limit: Int) -> (coords: [PenPoint], cer: Double) {
        var best: (coords: [PenPoint], cer: Double)?
        for c in cands.prefix(max(1, limit)) {
            let cer: Double
            if proofreads, let image = Ink.render(Ink.toStrokes(c.coords)) {
                cer = TextTools.cer(VisionOCR.read(image) ?? "", text)
            } else {
                cer = c.score >= 6 ? 1 : 0
            }
            if best == nil || cer < best!.cer { best = (c.coords, cer) }
            if cer == 0 { break }
        }
        return best!
    }

    private static func strokes(_ any: Any?) -> [Stroke] {
        ((any as? [[[NSNumber]]]) ?? []).map { s in s.compactMap { p in p.count >= 2 ? SIMD2(p[0].floatValue, p[1].floatValue) : nil } }
            .filter { !$0.isEmpty }
    }

    private static func round4(_ v: Float) -> Double { (Double(v) * 10_000).rounded() / 10_000 }
}

/// Runs async jobs one after another.
actor AsyncQueue {
    private var tail: Task<Void, Never>?

    func run<T: Sendable>(_ job: @escaping @Sendable () async throws -> T) async throws -> T {
        let previous = tail
        let task = Task<T, Error> {
            await previous?.value
            return try await job()
        }
        tail = Task { _ = try? await task.value }
        return try await task.value
    }
}

