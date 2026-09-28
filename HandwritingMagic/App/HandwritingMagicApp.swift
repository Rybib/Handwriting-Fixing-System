import SwiftUI

/// Handwriting Magic for iPhone and iPad: the Mac demo's page, running on a
/// touchscreen, with the whole pipeline on the device (see Engine/).
@main
struct HandwritingMagicApp: App {
    init() {
        // loads in the background; you can write meanwhile
        Task.detached(priority: .userInitiated) {
            await MagicEngine.shared.load()
            #if DEBUG
            // MAGIC_SELFTEST=<request.json>: rewrite it once loaded and print the reply (checks ML Kit in the Simulator)
            if let path = ProcessInfo.processInfo.environment["MAGIC_SELFTEST"], let body = FileManager.default.contents(atPath: path) {
                print("[selftest] reader: \(MagicEngine.shared.status.reader ?? "none"), error: \(MagicEngine.shared.status.error ?? "none")")
                do {
                    let reply = try await MagicEngine.shared.rewrite(body)
                    let lines = (try JSONSerialization.jsonObject(with: reply) as? [String: Any])?["lines"] as? [[String: Any]] ?? []
                    for l in lines { print("[selftest] written \"\(l["written"] ?? "")\" -> \"\(l["text"] ?? "")\"") }
                } catch {
                    print("[selftest] failed: \(error)")
                }
            }
            #endif
        }
    }

    var body: some Scene {
        WindowGroup {
            MagicPageView()
                .ignoresSafeArea()      // the paper fills the screen; the page keeps its controls in the safe area
                .background(Color(red: 0xFB / 255, green: 0xFA / 255, blue: 0xF6 / 255))
                .persistentSystemOverlays(.hidden)      // the home indicator fades while you write
        }
    }
}
