#if canImport(MLKitDigitalInkRecognition)
import CoreGraphics
import Foundation
import MLKitCommon
import MLKitDigitalInkRecognition

/// Google ML Kit Digital Ink Recognition: reads the pen strokes themselves
/// (where the pen went, and in what order), not a picture of them. That makes
/// it far better at messy handwriting than reading an image, and it's small:
/// the English model is 13 MB, carried in the app (see installBundledModel).
///
/// It returns what is literally WRITTEN. Working out what was MEANT (spelling,
/// their/there) is up to Qwen or the spellchecker, see MagicEngine.
final class InkReader: @unchecked Sendable {
    private let recognizer: DigitalInkRecognizer

    private init(_ recognizer: DigitalInkRecognizer) { self.recognizer = recognizer }

    /// Uses the model the app carries; downloads it only if that copy is missing.
    static func load(language: String = "en-US") async throws -> InkReader {
        let id = try DigitalInkRecognitionModelIdentifier.from(languageTag: language)
        let model = DigitalInkRecognitionModel(modelIdentifier: id)
        let manager = ModelManager.modelManager()
        if !manager.isModelDownloaded(model) { installBundledModel() }
        if !manager.isModelDownloaded(model) {
            try await withCheckedThrowingContinuation { (done: CheckedContinuation<Void, Error>) in
                let center = NotificationCenter.default
                var observers: [NSObjectProtocol] = []
                let finish = { (error: Error?) in
                    observers.forEach(center.removeObserver)
                    observers = []
                    if let error { done.resume(throwing: error) } else { done.resume() }
                }
                observers.append(center.addObserver(forName: .mlkitModelDownloadDidSucceed, object: nil, queue: .main) { _ in finish(nil) })
                observers.append(center.addObserver(forName: .mlkitModelDownloadDidFail, object: nil, queue: .main) { note in
                    finish(note.userInfo?[ModelDownloadUserInfoKey.error.rawValue] as? Error
                           ?? MagicEngine.Failure("the handwriting model didn't download (no internet?)"))
                })
                // needs MLKitDigitalInkRecognition_resource.bundle in the app (the download's manifest)
                _ = manager.download(model, conditions: ModelDownloadConditions(allowsCellularAccess: true, allowsBackgroundDownloading: true))
            }
        }
        let options = DigitalInkRecognizerOptions(model: model)
        options.maxResultCount = 5
        return InkReader(DigitalInkRecognizer.digitalInkRecognizer(options: options))
    }

    /// Works offline from the first launch: ML Kit only offers the model as a
    /// download, so the app carries the files it would download
    /// (MLKitEnglishModel.bundle, from scripts/get_mlkit_model.sh) and puts
    /// them where ML Kit keeps its downloads. Its bookkeeping is one small file
    /// named after the app. Not an official ML Kit feature: if a future ML Kit
    /// stores downloads differently, this does nothing and it downloads again.
    private static func installBundledModel() {
        let fm = FileManager.default
        guard let bundled = Bundle.main.url(forResource: "MLKitEnglishModel", withExtension: "bundle"),
              let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first,
              let files = try? fm.contentsOfDirectory(atPath: bundled.appendingPathComponent("FileData").path), files.count == 3
        else { return }
        let download = support.appendingPathComponent("com.google.mlkit.vision.digitalinkrecognition/Download")
        let name = "\(Bundle.main.bundleIdentifier ?? "app").com.google.mlkit.mdd.cellular."
            + "com.google.mlkit.vision.digitalinkrecognition.en-US.0001790626403812901"
        do {
            for dir in ["FileData", "FileGroupMetadata"] {
                try fm.createDirectory(at: download.appendingPathComponent(dir), withIntermediateDirectories: true)
            }
            for f in files where !fm.fileExists(atPath: download.appendingPathComponent("FileData/\(f)").path) {
                try fm.copyItem(at: bundled.appendingPathComponent("FileData/\(f)"), to: download.appendingPathComponent("FileData/\(f)"))
            }
            let metadata = download.appendingPathComponent("FileGroupMetadata/\(name)")
            if !fm.fileExists(atPath: metadata.path) {
                try fm.copyItem(at: bundled.appendingPathComponent("FileGroupMetadata.pb"), to: metadata)
            }
        } catch {
            print("[models] couldn't install the bundled handwriting model (\(error)); downloading it instead")
        }
    }

    /// Candidate readings, best first. `times`: seconds since the first point,
    /// one per point (the recogniser uses the pen's timing); nil = evenly spaced.
    func read(_ strokes: [Stroke], times: [[Double]]? = nil, preContext: String = "") async -> [String] {
        var clock = 0.0
        // ML Kit's Ink/Stroke, not the engine's own types of the same name
        let ink = MLKitDigitalInkRecognition.Ink(strokes: strokes.enumerated().map { i, s in
            let t = times.flatMap { i < $0.count && $0[i].count == s.count ? $0[i] : nil }
            if t == nil && i > 0 { clock += 0.25 }      // a pen lift
            return MLKitDigitalInkRecognition.Stroke(points: s.enumerated().map { j, p in
                let ms: Double
                if let t { ms = t[j] * 1000 } else { clock += 0.012; ms = clock * 1000 }
                return StrokePoint(x: p.x, y: p.y, t: Int(ms))
            })
        })
        // the size of the line helps it tell a small o from a big O
        let ys = strokes.flatMap { $0 }.map(\.y), xs = strokes.flatMap { $0 }.map(\.x)
        let area = WritingArea(width: Float((xs.max() ?? 1) - (xs.min() ?? 0)) + 40,
                               height: Float((ys.max() ?? 1) - (ys.min() ?? 0)) * 2)
        let context = DigitalInkRecognitionContext(preContext: preContext, writingArea: area)
        return await withCheckedContinuation { done in
            recognizer.recognize(ink: ink, context: context) { result, _ in
                done.resume(returning: result?.candidates.map(\.text) ?? [])
            }
        }
    }
}
#endif
