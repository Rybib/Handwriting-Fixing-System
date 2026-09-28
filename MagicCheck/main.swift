import CoreGraphics
import Foundation
import ImageIO

// Runs the app's Magic pipeline (Engine/) on this Mac, on the Mac demo's eval
// ink, to check the reader and the rewrites without an iPhone. MLX can't run
// in the Simulator, so this is the way to test Qwen3-VL-2B off the device.
//
// usage: MagicCheck <evalset dir> [out dir]
//   make the eval set in the Mac repo: .venv/bin/python tests/make_evalset.py /tmp/evalset
//   each case is NN.json: {"strokes": [[[x, y], ...], ...], "written": ..., "intended": ...}
//
// MagicCheck --readers <model dir> ...   compares reading models instead (ReaderBench.swift)

if CommandLine.arguments.dropFirst().first == "--readers" {
    await ReaderBench.run(Array(CommandLine.arguments.dropFirst(2)))
    exit(0)
}

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let args = CommandLine.arguments
let evalDir = URL(fileURLWithPath: args.count > 1 ? args[1] : "/tmp/evalset")
let outDir = URL(fileURLWithPath: args.count > 2 ? args[2] : "/tmp/magiccheck")
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

let engine = MagicEngine.shared
var t = Date()
await engine.load(synthDirectory: root.appendingPathComponent("HandwritingMagic/Resources"),
                  modelDirectory: root.appendingPathComponent("HandwritingMagic/Qwen3VLModel.bundle"))
let status = engine.status
print("loaded in \(String(format: "%.1f", Date().timeIntervalSince(t)))s: reader \(status.reader ?? "none"), error \(status.error ?? "none")")

func normal(_ s: String) -> String {
    s.lowercased().filter { $0.isLetter || $0.isNumber || $0 == " " || $0 == "'" }.split(separator: " ").joined(separator: " ")
}

func save(_ image: CGImage, _ url: URL) {
    let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

let cases = try FileManager.default.contentsOfDirectory(atPath: evalDir.path).filter { $0.hasSuffix(".json") }.sorted()
var meantOK = 0, mine = 0, times: [Double] = []
for name in cases {
    let item = try JSONSerialization.jsonObject(with: Data(contentsOf: evalDir.appendingPathComponent(name))) as! [String: Any]
    let intended = item["intended"] as! String
    // what the page sends: neatness 90 -> bias 0.3 + 0.9 * 2.2
    let req: [String: Any] = ["lines": [["strokes": item["strokes"]!]], "style": "mine", "bias": 2.28,
                              "fix_spelling": true, "candidates": 8]
    t = Date()
    let reply = try JSONSerialization.jsonObject(with: await engine.rewrite(JSONSerialization.data(withJSONObject: req))) as! [String: Any]
    times.append(Date().timeIntervalSince(t))
    let line = (reply["lines"] as! [[String: Any]])[0]
    let meant = line["meant"] as! String
    let ok = normal(meant) == normal(intended)
    meantOK += ok ? 1 : 0
    let styleUsed = "\(line["style_used"] ?? "-")"
    mine += styleUsed == "mine" ? 1 : 0
    let strokes = (line["strokes"] as! [[[Double]]]).map { $0.map { SIMD2(Float($0[0]), Float($0[1])) } }
    if let image = Ink.render(strokes) { save(image, outDir.appendingPathComponent(name.replacingOccurrences(of: ".json", with: ".png"))) }
    print("\(ok ? "✓" : "✗") \(name)  \(String(format: "%.1f", times.last!))s  style \(styleUsed)\n    written \"\(line["written"]!)\"\n    meant   \"\(meant)\"   (intended \"\(intended)\")")
}
print("\nintended sentence \(meantOK)/\(cases.count), kept own style \(mine)/\(cases.count), "
      + "\(String(format: "%.2f", times.reduce(0, +) / Double(max(1, times.count))))s a line. Rewrites in \(outDir.path)")
