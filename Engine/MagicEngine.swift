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
///   read the passage: ML Kit reads each line's pen strokes literally, then
///   Gemma 3 4B looks at all of it (those readings and a picture of the ink)
///   and works out what each line meant (spelling, their/there), in context
///   -> write what was meant, line by line, in the user's style AND a clean built-in style
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
        /// Each part of the reading, for the page's settings: whether it works here, and a word on it.
        var parts: [String: Part] = [:]

        struct Part: Sendable {
            var ok: Bool
            var note: String
        }

        var json: [String: Any] {
            ["loading": loading, "phase": phase, "progress": NSNull(), "synth": synth,
             "reader": reader ?? NSNull(), "error": error ?? NSNull(),
             "parts": parts.mapValues { ["ok": $0.ok, "note": $0.note] }]
        }
    }

    /// Which parts of the reading to use, from the page's settings (all on by default):
    ///   {"mlkit": true, "gemma": true, "gemma_sees_ink": true, "proofread": true}
    struct Reading {
        var mlkit = true, gemma = true, gemmaSeesInk = true, proofread = true

        init(_ any: Any?) {
            let d = any as? [String: Any] ?? [:]
            mlkit = d["mlkit"] as? Bool ?? true
            gemma = d["gemma"] as? Bool ?? true
            gemmaSeesInk = d["gemma_sees_ink"] as? Bool ?? true
            proofread = d["proofread"] as? Bool ?? true
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
    /// Why Gemma isn't reading, when it isn't.
    private var gemmaNote = ""
    private var mlkitNote = "Only in the iPhone and iPad app"
    /// One rewrite at a time: they share the CPU and GPU.
    private let queue = AsyncQueue()

    private func update(_ change: (inout Status) -> Void) { state.withLock { change(&$0) } }

    // MARK: - loading

    /// Loads everything in the background; you can write meanwhile.
    /// nil folders: the app's own (hand_synth.bin/json, GemmaMLXModel.bundle).
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
            mlkitNote = "Didn't start: \(error.localizedDescription)"
            print("[models] ML Kit unavailable, reading with Vision: \(error)")
        }
        #endif
        reader = await loadReader(modelDirectory ?? Bundle.main.url(forResource: "GemmaMLXModel", withExtension: "bundle"))
        let gemma = reader as? GemmaReader
        #if canImport(MLKitDigitalInkRecognition)
        let mlkit = inkReader != nil
        #else
        let mlkit = false
        #endif
        let parts: [String: Status.Part] = [
            "mlkit": .init(ok: mlkit, note: mlkit ? "Ready: its English model is in the app, so it works offline" : mlkitNote),
            "gemma": .init(ok: gemma != nil, note: gemma != nil ? "Ready: Gemma 3 4B on this \(Self.deviceName)" : gemmaNote),
            "gemma_sees_ink": .init(ok: gemma?.seesInk == true,
                                    note: gemma == nil ? "Needs Gemma" : gemma!.seesInk ? "Ready: its vision tower is loaded" : "The vision tower isn't in the app"),
            "proofread": .init(ok: proofreads, note: proofreads ? "Ready: Apple Vision" : "Apple Vision doesn't work here"),
        ]
        update {
            $0.reader = self.reader?.description
            $0.parts = parts
            $0.loading = false
            $0.phase = "Ready"
        }
    }

    private func loadReader(_ modelDirectory: URL?) async -> HandwritingReader? {
        let canRead = proofreads || literalSource != "Apple Vision"
        #if targetEnvironment(simulator)
        // MLX needs a real GPU: the Simulator has no AI model
        gemmaNote = "The Simulator can't run it (MLX needs a real GPU)"
        return canRead ? SpellcheckReader(description: "\(literalSource) + spellchecker (Simulator: no AI fixes)") : nil
        #else
        let fallback = canRead
            ? SpellcheckReader(description: "\(literalSource) + spellchecker, no AI fixes (Gemma isn't in the app: run scripts/get_model.sh)")
            : nil
        let fm = FileManager.default
        guard let dir = modelDirectory,
              let files = try? fm.contentsOfDirectory(atPath: dir.path), files.contains(where: { $0.hasSuffix(".safetensors") }) else {
            gemmaNote = "It isn't in the app (run scripts/get_model.sh)"
            if fallback == nil { update { $0.error = "Gemma isn't in the app (run scripts/get_model.sh)" } }
            return fallback
        }
        // Gemma 3 4B needs about 3 GB, 4 with a picture: more than a 4 GB iPad or iPhone can spare
        let gigabytes = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        if gigabytes < 5.5 {
            gemmaNote = "This device has too little memory for it (it needs 6 GB)"
            return canRead ? SpellcheckReader(description: "\(literalSource) + spellchecker, no AI fixes "
                                              + "(Gemma needs a device with 6 GB of memory or more)") : nil
        }
        do {
            update { $0.phase = "Loading Gemma into memory" }
            // MLX recycles freed GPU buffers up to this size (Rytability's sizes)
            MLX.Memory.cacheLimit = (gigabytes >= 7.5 ? 384 : 224) * 1024 * 1024
            let gemma = try await GemmaReader(modelDirectory: dir, description: "")
            update { $0.phase = gemma.seesInk ? "Warming up Gemma's vision" : "Warming up Gemma" }
            // runs a picture through the vision tower once, so a broken one shows up now, not at the first Magic
            let blank = Ink.render([[SIMD2(0, 0), SIMD2(40, 0)]])!
            _ = try await gemma.generate("Say OK.", image: gemma.seesInk ? blank : nil, maxTokens: 4)
            gemma.description = "\(literalSource) + Gemma 3 4B\(gemma.seesInk ? " (sees the ink)" : "") on this \(Self.deviceName)"
            return gemma
        } catch {
            gemmaNote = "It didn't load: \(error.localizedDescription)"
            update { $0.error = "Gemma didn't load: \(error.localizedDescription)" }
            return fallback == nil ? nil : SpellcheckReader(description: "\(literalSource) + spellchecker, no AI fixes (Gemma failed to load)")
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
    ///    "bias": 2.28, "fix_spelling": true, "candidates": 8, "reading": {...} (see Reading)}
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
        let reading = Reading(req["reading"])
        let proofread = reading.proofread

        // Read all the lines together: they're one piece of writing, so each
        // line is read knowing the others (the page sends a block of lines that
        // sit together; writing somewhere else on the page comes separately).
        let t0 = Date()
        let readings = try await read(lines, reading)
        let tRead = Date().timeIntervalSince(t0)

        var out: [[String: Any]] = []
        for (line, (written, meant)) in zip(lines, readings) {
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
                    var best = res[i].primeAligned ? pick(res[i].candidates, chunk, limit: nCand, proofread: proofread) : nil
                    if best == nil || best!.cer > 0 {
                        let safe = pick(res[n + i].candidates, chunk, limit: nCand, proofread: proofread)
                        if best == nil || safe.cer < best!.cer { best = safe; styleUsed = HandwritingSynth.fallbackStyle }
                    }
                    coords.append(best!.coords)
                }
            } else {
                let res = synth.write(chunks, bias: bias, primes: Array(repeating: safePrime, count: n), candidates: nCand)
                coords = zip(res, chunks).map { pick($0.candidates, $1, limit: nCand, proofread: proofread).coords }
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

    /// What each line says, as written and as meant, read as one passage.
    private func read(_ lines: [[String: Any]], _ reading: Reading) async throws -> [(written: String, meant: String)] {
        var readings: [(written: String, meant: String)?] = lines.map { l in (l["text"] as? String).map { ($0, $0) } }
        let toRead = lines.indices.filter { readings[$0] == nil }       // the caller knows the text of the others (tests)
        guard !toRead.isEmpty else { return readings.map { $0! } }
        guard var reader else { throw Failure(status.error ?? "handwriting reader still loading") }
        // Gemma switched off in the settings: the literal reading, spellchecked
        if !reading.gemma, reader is GemmaReader { reader = SpellcheckReader(description: "") }
        let inks = toRead.map { Self.strokes(lines[$0]["strokes"]) }
        guard let image = Ink.renderPassage(inks) else { throw Failure("no ink") }
        var literals: [String?] = []
        for (k, i) in toRead.enumerated() {
            // the line before tells ML Kit how this one starts ("... the" -> "park", not "pork")
            let before = literals.compactMap { $0 }.joined(separator: " ")
            literals.append(await literalReading(inks[k], times: lines[i]["times"] as? [[Double]],
                                                 preContext: String(before.suffix(20)), mlkit: reading.mlkit))
        }
        let got = try await reader.read(passage: image, lines: inks.map { Ink.render($0) }, literals: literals,
                                        useInk: reading.gemmaSeesInk)
        for (k, i) in toRead.enumerated() { readings[i] = k < got.count ? got[k] : (literals[k] ?? "", literals[k] ?? "") }
        print("[read] \(toRead.count) line(s)\n  literal: \(literals.map { $0 ?? "-" })\n  meant:   \(got.map(\.meant))")
        return readings.map { $0! }
    }

    /// What is literally written: ML Kit reads the pen strokes (and their
    /// timing); without it, Vision reads a picture of them.
    private func literalReading(_ strokes: [Stroke], times: [[Double]]?, preContext: String, mlkit: Bool) async -> String? {
        #if canImport(MLKitDigitalInkRecognition)
        // with a preContext, ML Kit starts its reading with the space after it
        if mlkit, let inkReader, let best = await inkReader.read(strokes, times: times, preContext: preContext).first {
            return best.trimmingCharacters(in: .whitespaces)
        }
        #endif
        return proofreads ? Ink.render(strokes).flatMap(VisionOCR.read) : nil
    }

    /// The candidate that reads back best (ties: the better attention score).
    ///
    /// Candidates come sorted by attention score, so the first one Vision reads
    /// back perfectly wins. Without Vision, a poor attention score (a skipped
    /// letter, or the pen never lifting at the end) counts as a misreading.
    private func pick(_ cands: [HandwritingSynth.Candidate], _ text: String, limit: Int, proofread: Bool) -> (coords: [PenPoint], cer: Double) {
        var best: (coords: [PenPoint], cer: Double)?
        for c in cands.prefix(max(1, limit)) {
            let cer: Double
            if proofread, proofreads, let image = Ink.render(Ink.toStrokes(c.coords)) {
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

