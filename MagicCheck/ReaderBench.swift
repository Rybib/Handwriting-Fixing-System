import CoreGraphics
import Foundation
import ImageIO
import MLX

/// `MagicCheck --readers <model dir> ...`: compares reading models the way the
/// app runs them (Engine/Readers.swift on MLX), on the 24 eval images from the
/// Mac demo (12 eval-set lines, 12 drawn through the browser), each with the
/// literal OCR reading as the hint. Any vision model mlx-swift-lm supports works.
///
/// items: [{"path": "...png", "intended": "...", "prompt": "<hint + the app's prompt>"}]
/// (MAGICCHECK_ITEMS, default /tmp/mlxeval/items_v2.json)
enum ReaderBench {
    static func run(_ modelDirs: [String]) async {
        let itemsURL = URL(fileURLWithPath: ProcessInfo.processInfo.environment["MAGICCHECK_ITEMS"] ?? "/tmp/mlxeval/items_v2.json")
        guard let data = try? Data(contentsOf: itemsURL),
              let items = try? JSONSerialization.jsonObject(with: data) as? [[String: String]] else {
            print("can't read \(itemsURL.path)")
            return
        }
        MLX.Memory.cacheLimit = 128 * 1024 * 1024
        for dir in modelDirs {
            let url = URL(fileURLWithPath: dir)
            do {
                var t = Date()
                let reader = try await QwenReader(modelDirectory: url, description: url.lastPathComponent)
                let load = Date().timeIntervalSince(t)
                MLX.GPU.resetPeakMemory()
                var right = ["ev": 0, "evb": 0], times: [Double] = [], wrong: [String] = []
                for (i, item) in items.enumerated() {
                    // no "path": a text-only prompt (the reading is in the prompt)
                    let path = item["path"] ?? ""
                    let image = path.isEmpty ? nil : loadImage(path)
                    if !path.isEmpty && image == nil { continue }
                    t = Date()
                    let (_, meant) = QwenReader.parse(try await reader.generate(item["prompt"] ?? QwenReader.prompt, image: image))
                    if i > 0 { times.append(Date().timeIntervalSince(t)) }      // the first includes warm-up
                    let set = URL(fileURLWithPath: path).lastPathComponent.hasPrefix("evb") ? "evb" : "ev"
                    if normal(meant) == normal(item["intended"] ?? "") { right[set, default: 0] += 1 } else { wrong.append("\(set): \(meant)") }
                }
                let size = (try? FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: [.fileSizeKey]))?
                    .reduce(0) { $0 + ((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) } ?? 0
                print(String(format: "%@  %.2f GB on disk | intended %d/\(items.count) (eval %d, browser %d) | %.2f s a line | peak %.2f GB | load %.1f s",
                             url.lastPathComponent, Double(size) / 1e9, right["ev"]! + right["evb"]!, right["ev"]!, right["evb"]!,
                             times.reduce(0, +) / Double(max(1, times.count)), Double(MLX.GPU.peakMemory) / 1e9, load))
                print("    misses: " + wrong.joined(separator: " | "))
            } catch {
                print("\(url.lastPathComponent): failed: \(error)")
            }
            MLX.Memory.clearCache()
        }
    }

    static func loadImage(_ path: String) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    static func normal(_ s: String) -> String {
        s.lowercased().filter { $0.isLetter || $0.isNumber || $0 == " " || $0 == "'" }.split(separator: " ").joined(separator: " ")
    }
}
