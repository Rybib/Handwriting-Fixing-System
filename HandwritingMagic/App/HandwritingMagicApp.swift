import SwiftUI

/// Handwriting Magic for iPhone and iPad: the Mac demo's page, running on a
/// touchscreen, with the whole pipeline on the device (see Engine/).
@main
struct HandwritingMagicApp: App {
    init() {
        // loads in the background; the page works in Tidy mode meanwhile
        Task.detached(priority: .userInitiated) { await MagicEngine.shared.load() }
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
