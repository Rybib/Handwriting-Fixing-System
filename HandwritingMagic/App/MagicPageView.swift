import SwiftUI
import WebKit

/// The page from the Mac demo (Web.bundle), in a web view. Its requests to
/// `/api/...` are answered on the device by `PageServer`, so the page runs
/// unchanged: Apple Pencil pressure and palm rejection come from its
/// pointer-event handling.
struct MagicPageView: UIViewRepresentable {
    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.setURLSchemeHandler(PageServer.shared, forURLScheme: PageServer.scheme)
        let web = WKWebView(frame: .zero, configuration: config)
        web.isOpaque = false
        web.backgroundColor = UIColor(red: 0xFB / 255, green: 0xFA / 255, blue: 0xF6 / 255, alpha: 1)
        // it's a sheet of paper, not a document: no scrolling, bouncing or zooming
        web.scrollView.isScrollEnabled = false
        web.scrollView.bounces = false
        web.scrollView.pinchGestureRecognizer?.isEnabled = false
        // hand every Pencil touch to the page at once, so a quick lift-and-touch
        // between letters isn't held back while the scroll view decides what it is
        web.scrollView.delaysContentTouches = false
        web.scrollView.contentInsetAdjustmentBehavior = .never
        web.allowsLinkPreview = false
        #if DEBUG
        web.isInspectable = true        // Safari > Develop > (device) to debug the page
        #endif
        web.load(URLRequest(url: URL(string: "\(PageServer.scheme)://app/index.html")!))
        return web
    }

    func updateUIView(_ web: WKWebView, context: Context) {}
}
