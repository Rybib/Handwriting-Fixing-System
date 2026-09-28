import Foundation
import UniformTypeIdentifiers
import WebKit

/// Serves the page to the web view and answers its API, standing in for the
/// Mac demo's Python server:
///
///   magic://app/<file>           a file from Web.bundle
///   GET  magic://app/api/status  loading progress, which reader is in use
///   POST magic://app/api/rewrite read + rewrite a line (MagicEngine)
@MainActor
final class PageServer: NSObject, WKURLSchemeHandler {
    static let scheme = "magic"
    static let shared = PageServer()

    private let web = Bundle.main.url(forResource: "Web", withExtension: "bundle")!
    /// Tasks the web view gave up on (page reloaded); answering them would crash.
    private var stopped = Set<ObjectIdentifier>()

    func webView(_ webView: WKWebView, start task: any WKURLSchemeTask) {
        let path = task.request.url?.path ?? "/"
        switch path {
        case "/api/status":
            reply(task, json: try? JSONSerialization.data(withJSONObject: MagicEngine.shared.status.json))
        case "/api/rewrite":
            guard let body = Self.body(of: task.request) else {
                return reply(task, status: 400, json: Self.error("no request body"))
            }
            // the models run away from the main thread, which keeps the web view (and its animations) going
            Task.detached(priority: .userInitiated) {
                do {
                    let result = try await MagicEngine.shared.rewrite(body)
                    await self.reply(task, json: result)
                } catch {
                    await self.reply(task, status: 500, json: Self.error(error.localizedDescription))
                }
            }
        default:
            let file = web.appendingPathComponent(path == "/" ? "index.html" : String(path.dropFirst())).standardizedFileURL
            guard file.path.hasPrefix(web.standardizedFileURL.path), let data = try? Data(contentsOf: file) else {
                return reply(task, status: 404, data: Data(), type: "text/plain")
            }
            let type = UTType(filenameExtension: file.pathExtension)?.preferredMIMEType ?? "application/octet-stream"
            reply(task, data: data, type: type == "application/javascript" ? "text/javascript" : type)
        }
    }

    func webView(_ webView: WKWebView, stop task: any WKURLSchemeTask) {
        stopped.insert(ObjectIdentifier(task))
    }

    private func reply(_ task: any WKURLSchemeTask, status: Int = 200, json: Data?) {
        reply(task, status: status, data: json ?? Data("{}".utf8), type: "application/json")
    }

    private func reply(_ task: any WKURLSchemeTask, status: Int = 200, data: Data, type: String) {
        guard stopped.remove(ObjectIdentifier(task)) == nil, let url = task.request.url else { return }
        let headers = ["Content-Type": type, "Content-Length": "\(data.count)", "Cache-Control": "no-store",
                       "Access-Control-Allow-Origin": "*"]
        task.didReceive(HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!)
        task.didReceive(data)
        task.didFinish()
    }

    private static func body(of request: URLRequest) -> Data? {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return nil }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while case let n = stream.read(&buffer, maxLength: buffer.count), n > 0 { data.append(buffer, count: n) }
        return data
    }

    private static func error(_ message: String) -> Data? {
        try? JSONSerialization.data(withJSONObject: ["error": message])
    }
}
