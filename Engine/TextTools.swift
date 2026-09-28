import Foundation

/// Text helpers shared by the reader and the synthesiser. Ported from the Mac
/// demo (`server/synth.py`, `server/recognize.py`, `server/app.py`).
enum TextTools {
    /// The synthesiser's character set. It was trained on IAM-OnDB, which has no
    /// capital Q, X or Z. Index 0 is the end-of-text marker.
    static let alphabet: [Character] = Array("\0 !\"#'(),-.0123456789:;?ABCDEFGHIJKLMNOPRSTUVWYabcdefghijklmnopqrstuvwxyz")
    static let charToID: [Character: Int] = Dictionary(uniqueKeysWithValues: alphabet.enumerated().map { ($1, $0) })

    private static let substitutes: [Character: String] = [
        "Q": "q", "X": "x", "Z": "z", "`": "'", "\u{2019}": "'", "\u{2018}": "'",
        "\u{201C}": "\"", "\u{201D}": "\"", "\u{2014}": "-", "\u{2013}": "-", "&": "and",
    ]

    /// Map text into the synthesiser's character set and collapse whitespace.
    static func sanitize(_ text: String) -> String {
        var out = ""
        for ch in text {
            let mapped = substitutes[ch] ?? String(ch)
            for c in mapped where c != "\0" && charToID[c] != nil { out.append(c) }
        }
        return out.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// Character ids of `text` plus the end-of-text marker.
    static func encode(_ text: String) -> [Int] {
        text.map { charToID[$0] ?? 1 } + [0]
    }

    /// Character error rate of reading `a` against reference `b` (letters and digits only).
    static func cer(_ a: String, _ b: String) -> Double {
        let clean = { (s: String) in Array(s.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) && $0.isASCII }) }
        let x = clean(a), y = clean(b)
        var prev = Array(0...y.count)
        for (i, ca) in x.enumerated() {
            var cur = [i + 1]
            for (j, cb) in y.enumerated() {
                cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (ca == cb ? 0 : 1)))
            }
            prev = cur
        }
        return Double(prev[y.count]) / Double(max(1, y.count))
    }

    /// Word-level differences between what was written and what is meant, as
    /// [from, to] pairs for the result card ("recieve" -> "receive").
    static func wordCorrections(_ written: String, _ meant: String) -> [[String]] {
        let a = written.split(separator: " ").map(String.init)
        let b = meant.split(separator: " ").map(String.init)
        let key = { (w: String) in w.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: ".,!?;:")) }
        let ka = a.map(key), kb = b.map(key)
        // longest common subsequence, then everything between matches is one change
        var lcs = Array(repeating: Array(repeating: 0, count: kb.count + 1), count: ka.count + 1)
        for i in stride(from: ka.count - 1, through: 0, by: -1) {
            for j in stride(from: kb.count - 1, through: 0, by: -1) {
                lcs[i][j] = ka[i] == kb[j] ? lcs[i + 1][j + 1] + 1 : max(lcs[i + 1][j], lcs[i][j + 1])
            }
        }
        var out: [[String]] = []
        var i = 0, j = 0, i0 = 0, j0 = 0
        func flush() {
            if i > i0 || j > j0 {
                out.append([a[i0..<i].joined(separator: " "), b[j0..<j].joined(separator: " ")])
            }
        }
        while i < ka.count || j < kb.count {
            if i < ka.count, j < kb.count, ka[i] == kb[j] {
                flush()
                i += 1; j += 1; i0 = i; j0 = j
            } else if j < kb.count, i == ka.count || lcs[i][j + 1] >= lcs[i + 1][j] {
                j += 1
            } else {
                i += 1
            }
        }
        flush()
        return out
    }

    /// The synthesiser writes at most 75 characters per sequence; split long text on spaces.
    static func wrap(_ text: String, limit: Int = HandwritingSynth.maxChars - 5) -> [String] {
        var chunks: [String] = [], cur = ""
        for w in text.split(separator: " ") {
            if !cur.isEmpty, cur.count + 1 + w.count > limit {
                chunks.append(cur)
                cur = String(w)
            } else {
                cur = cur.isEmpty ? String(w) : cur + " " + w
            }
        }
        if !cur.isEmpty { chunks.append(cur) }
        return chunks
    }
}
