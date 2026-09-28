import Accelerate
import Foundation

/// Graves (2013) handwriting synthesis, ported from the Mac demo's numpy version
/// (`server/synth.py`, weights from sjvasquez/handwriting-synthesis).
///
/// 3 stacked LSTMs (400 units) with a soft Gaussian attention window over the
/// characters, and a 20-component bivariate Gaussian mixture for the next pen
/// movement. "Priming" first feeds a handwriting sample and its transcript
/// through the network, so the text it then writes continues in that style.
/// `bias` sharpens the sampling: higher is neater.
final class HandwritingSynth {
    static let maxChars = 75
    /// "Tall narrow print": the clean style used when the user's own can't be copied legibly.
    static let fallbackStyle = 9

    private let V = TextTools.alphabet.count     // 73
    private let H = 400
    private let nOut = 20
    private let nAtt = 10

    struct Prime {
        var offsets: [PenPoint]
        var text: String
    }

    struct Candidate {
        var coords: [PenPoint]      // x, y up, pen lifts; denoised and straightened
        var score: Int              // attention check: lower is better
    }

    struct Line {
        var candidates: [Candidate]     // best attention score first
        var primeAligned: Bool          // false: the ink and its transcript disagree
    }

    private let l1Wx, l1Wh, l1b, l2Wx, l2Wh, l2b, l3Wx, l3Wh, l3b, attW, attb, gmmW, gmmb: [Float]
    /// The 13 built-in writers (see STYLE_GROUPS in Web.bundle/js/app.js for their names).
    let styles: [Prime]

    init(weightsURL: URL, indexURL: URL) throws {
        struct Index: Decodable {
            struct Tensor: Decodable { let offset: Int; let shape: [Int] }
            struct Style: Decodable { let text: String; let offsets: [Float] }
            let tensors: [String: Tensor]
            let styles: [Style]
        }
        let data = try Data(contentsOf: weightsURL)
        let all: [Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: indexURL))
        func tensor(_ name: String) -> [Float] {
            let t = index.tensors[name]!
            return Array(all[t.offset ..< t.offset + t.shape.reduce(1, *)])
        }
        let (V, H) = (TextTools.alphabet.count, 400)
        // each LSTM kernel is [inputs + units, 4 * units]: split off the recurrent part
        let l1 = tensor("lstm1_W"), l2 = tensor("lstm2_W"), l3 = tensor("lstm3_W")
        l1Wx = Array(l1[..<((V + 3) * 4 * H)]); l1Wh = Array(l1[((V + 3) * 4 * H)...])
        l2Wx = Array(l2[..<((3 + H + V) * 4 * H)]); l2Wh = Array(l2[((3 + H + V) * 4 * H)...])
        l3Wx = Array(l3[..<((3 + H + V) * 4 * H)]); l3Wh = Array(l3[((3 + H + V) * 4 * H)...])
        l1b = tensor("lstm1_b"); l2b = tensor("lstm2_b"); l3b = tensor("lstm3_b")
        attW = tensor("attn_W"); attb = tensor("attn_b")
        gmmW = tensor("gmm_W"); gmmb = tensor("gmm_b")
        styles = index.styles.map { s in
            Prime(offsets: stride(from: 0, to: s.offsets.count, by: 3).map {
                PenPoint(x: s.offsets[$0], y: s.offsets[$0 + 1], eos: s.offsets[$0 + 2])
            }, text: s.text)
        }
    }

    convenience init(bundle: Bundle = .main) throws {
        guard let bin = bundle.url(forResource: "hand_synth", withExtension: "bin"),
              let json = bundle.url(forResource: "hand_synth", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        try self.init(weightsURL: bin, indexURL: json)
    }

    // MARK: - network state (row-major, one row per sequence in the batch)

    private struct State {
        var h1, c1, h2, c2, h3, c3, kappa, w, phi: [Float]

        init(_ B: Int, H: Int, V: Int, nAtt: Int, U: Int) {
            h1 = .init(repeating: 0, count: B * H); c1 = h1; h2 = h1; c2 = h1; h3 = h1; c3 = h1
            kappa = .init(repeating: 0, count: B * nAtt)
            w = .init(repeating: 0, count: B * V)
            phi = .init(repeating: 0, count: B * U)
        }

        /// Copy `rows` of `other` into self (rows that are still active).
        mutating func take(_ other: State, rows: [Int], H: Int, V: Int, nAtt: Int, U: Int) {
            for b in rows {
                for (dst, src, n) in [(\State.h1, \State.h1, H), (\.c1, \.c1, H), (\.h2, \.h2, H), (\.c2, \.c2, H),
                                      (\.h3, \.h3, H), (\.c3, \.c3, H), (\.kappa, \.kappa, nAtt), (\.w, \.w, V),
                                      (\.phi, \.phi, U)] as [(WritableKeyPath<State, [Float]>, KeyPath<State, [Float]>, Int)] {
                    self[keyPath: dst].replaceSubrange(b * n ..< (b + 1) * n, with: other[keyPath: src][b * n ..< (b + 1) * n])
                }
            }
        }

        /// Repeat every row k times (priming is deterministic, so the k candidates share it).
        func repeated(_ k: Int, H: Int, V: Int, nAtt: Int, U: Int) -> State {
            func rep(_ a: [Float], _ n: Int) -> [Float] {
                stride(from: 0, to: a.count, by: n).flatMap { i in (0..<k).flatMap { _ in a[i ..< i + n] } }
            }
            var s = self
            s.h1 = rep(h1, H); s.c1 = rep(c1, H); s.h2 = rep(h2, H); s.c2 = rep(c2, H); s.h3 = rep(h3, H); s.c3 = rep(c3, H)
            s.kappa = rep(kappa, nAtt); s.w = rep(w, V); s.phi = rep(phi, U)
            return s
        }
    }

    /// C[m x n] (+)= A[m x k] @ B[k x n]
    private func gemm(_ A: [Float], _ B: [Float], _ C: inout [Float], m: Int, k: Int, n: Int, accumulate: Bool) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k),
                    1, A, Int32(k), B, Int32(n), accumulate ? 1 : 0, &C, Int32(n))
    }

    /// One TF LSTMCell step (gate order i, j, f, o; forget bias 1) for B rows.
    private func lstm(_ x: [Float], xDim: Int, h: inout [Float], c: inout [Float], Wx: [Float], Wh: [Float], b: [Float], B: Int) {
        let G = 4 * H
        var z = [Float](repeating: 0, count: B * G)
        gemm(x, Wx, &z, m: B, k: xDim, n: G, accumulate: false)
        gemm(h, Wh, &z, m: B, k: H, n: G, accumulate: true)
        // sigmoid(v) = 0.5 * (tanh(v / 2) + 1): prescale the sigmoid gates, then one vectorised tanh
        z.withUnsafeMutableBufferPointer { zp in
            for r in 0..<B {
                let row = zp.baseAddress! + r * G
                for j in 0..<G {
                    let gate = j / H
                    let v = row[j] + b[j]
                    row[j] = gate == 1 ? v : 0.5 * (gate == 2 ? v + 1 : v)
                }
            }
            var n = Int32(B * G)
            vvtanhf(zp.baseAddress!, zp.baseAddress!, &n)
            h.withUnsafeMutableBufferPointer { hp in
                c.withUnsafeMutableBufferPointer { cp in
                    for r in 0..<B {
                        let row = zp.baseAddress! + r * G
                        for j in 0..<H {
                            let i = 0.5 * (row[j] + 1), g = row[H + j], f = 0.5 * (row[2 * H + j] + 1)
                            cp[r * H + j] = f * cp[r * H + j] + i * g
                        }
                    }
                    var th = [Float](repeating: 0, count: B * H)
                    var m = Int32(B * H)
                    vvtanhf(&th, cp.baseAddress!, &m)
                    for r in 0..<B {
                        let row = zp.baseAddress! + r * G
                        for j in 0..<H { hp[r * H + j] = 0.5 * (row[3 * H + j] + 1) * th[r * H + j] }
                    }
                }
            }
        }
    }

    /// One step of the whole network for B rows.
    private func step(_ inp: [Float], _ st: State, chars: [[Int]], U: Int, B: Int) -> State {
        var s = st
        // layer 1 sees the previous attention window and the pen movement
        var x1 = [Float](repeating: 0, count: B * (V + 3))
        for b in 0..<B {
            for v in 0..<V { x1[b * (V + 3) + v] = st.w[b * V + v] }
            for d in 0..<3 { x1[b * (V + 3) + V + d] = inp[b * 3 + d] }
        }
        lstm(x1, xDim: V + 3, h: &s.h1, c: &s.c1, Wx: l1Wx, Wh: l1Wh, b: l1b, B: B)

        // attention window: where along the text the pen is
        let aDim = V + 3 + H
        var xa = [Float](repeating: 0, count: B * aDim)
        for b in 0..<B {
            for v in 0..<V { xa[b * aDim + v] = st.w[b * V + v] }
            for d in 0..<3 { xa[b * aDim + V + d] = inp[b * 3 + d] }
            for j in 0..<H { xa[b * aDim + V + 3 + j] = s.h1[b * H + j] }
        }
        var att = [Float](repeating: 0, count: B * 3 * nAtt)
        gemm(xa, attW, &att, m: B, k: aDim, n: 3 * nAtt, accumulate: false)
        var expArg = [Float](repeating: 0, count: B * nAtt * U)
        var alpha = [Float](repeating: 0, count: B * nAtt), beta = alpha
        for b in 0..<B {
            for k in 0..<nAtt {
                let sp = { (v: Float) in max(v, 0) + log1p(exp(-abs(v))) }     // softplus
                alpha[b * nAtt + k] = sp(att[b * 3 * nAtt + k] + attb[k])
                beta[b * nAtt + k] = max(sp(att[b * 3 * nAtt + nAtt + k] + attb[nAtt + k]), 0.01)
                s.kappa[b * nAtt + k] = st.kappa[b * nAtt + k] + sp(att[b * 3 * nAtt + 2 * nAtt + k] + attb[2 * nAtt + k]) / 25
                for u in 0..<U {
                    let d = s.kappa[b * nAtt + k] - Float(u)
                    expArg[(b * nAtt + k) * U + u] = -d * d / beta[b * nAtt + k]
                }
            }
        }
        var n = Int32(expArg.count)
        vvexpf(&expArg, expArg, &n)
        for b in 0..<B {
            for u in 0..<U {
                var p: Float = 0
                for k in 0..<nAtt { p += alpha[b * nAtt + k] * expArg[(b * nAtt + k) * U + u] }
                s.phi[b * U + u] = p
            }
            for v in 0..<V { s.w[b * V + v] = 0 }
            for (u, ch) in chars[b].enumerated() { s.w[b * V + ch] += s.phi[b * U + u] }
        }

        // layers 2 and 3 see the pen movement, the layer below, and the window
        let xDim = 3 + H + V
        var x2 = [Float](repeating: 0, count: B * xDim)
        for b in 0..<B {
            for d in 0..<3 { x2[b * xDim + d] = inp[b * 3 + d] }
            for j in 0..<H { x2[b * xDim + 3 + j] = s.h1[b * H + j] }
            for v in 0..<V { x2[b * xDim + 3 + H + v] = s.w[b * V + v] }
        }
        lstm(x2, xDim: xDim, h: &s.h2, c: &s.c2, Wx: l2Wx, Wh: l2Wh, b: l2b, B: B)
        for b in 0..<B { for j in 0..<H { x2[b * xDim + 3 + j] = s.h2[b * H + j] } }
        lstm(x2, xDim: xDim, h: &s.h3, c: &s.c3, Wx: l3Wx, Wh: l3Wh, b: l3b, B: B)
        return s
    }

    /// Sample the next pen movement for every row. Returns (movements [B*3], pen-lift probabilities).
    private func sample(_ h3: [Float], bias: Float, B: Int, rng: inout SplitMix64) -> ([Float], [Float]) {
        let D = 6 * nOut + 1
        var z = [Float](repeating: 0, count: B * D)
        gemm(h3, gmmW, &z, m: B, k: H, n: D, accumulate: false)
        var out = [Float](repeating: 0, count: B * 3), eosP = [Float](repeating: 0, count: B)
        for b in 0..<B {
            let r = b * D
            func p(_ i: Int) -> Float { z[r + i] + gmmb[i] }
            var pis = (0..<nOut).map { p($0) * (1 + bias) }
            let mx = pis.max()!
            pis = pis.map { exp($0 - mx) }
            let sum = pis.reduce(0, +)
            pis = pis.map { $0 / sum < 0.01 ? 0 : $0 / sum }
            let tot = pis.reduce(0, +)
            let u = Float.random(in: 0..<1, using: &rng)
            var idx = 0, acc: Float = 0
            for k in 0..<nOut { acc += pis[k] / tot; if acc > u || k == nOut - 1 { idx = k; break } }
            let s1 = max(exp(p(nOut + idx) - bias), 1e-4), s2 = max(exp(p(2 * nOut + idx) - bias), 1e-4)
            let rho = min(max(tanh(p(3 * nOut + idx)), -1 + 1e-7), 1 - 1e-7)
            let m1 = p(4 * nOut + idx), m2 = p(5 * nOut + idx)
            let (z1, z2) = rng.gaussianPair()
            var e = 1 / (1 + exp(-p(6 * nOut)))
            e = min(max(e, 1e-8), 1 - 1e-8)
            if e < 0.01 { e = 0 }
            out[b * 3] = m1 + s1 * z1
            out[b * 3 + 1] = m2 + s2 * (rho * z1 + (1 - rho * rho).squareRoot() * z2)
            out[b * 3 + 2] = Float.random(in: 0..<1, using: &rng) < e ? 1 : 0
            eosP[b] = e
        }
        return (out, eosP)
    }

    // MARK: - public API

    /// Synthesise every line (batched), `candidates` versions of each.
    ///
    /// lines: sanitised text, at most 75 characters each; primes: one per line
    /// (nil = no style). Each line's candidates come back sorted by how evenly
    /// the attention visited every letter; the caller picks with a recogniser.
    func write(_ lines: [String], bias: Float, primes: [Prime?], candidates: Int, seed: UInt64? = nil,
               maxStepsPerChar: Int = 40) -> [Line] {
        var rng = SplitMix64(seed: seed ?? UInt64.random(in: 0...UInt64.max))
        let L = lines.count, K = max(1, candidates)
        // The trailing space gives the pen its usual end-of-word moment to cross
        // the last t and dot the last i. Whatever it writes after that is trimmed.
        var chars: [[Int]] = [], textStart: [Int] = []
        for (text, p) in zip(lines, primes) {
            if let p {
                chars.append(TextTools.encode(p.text + " " + text + " "))
                textStart.append(p.text.count + 1)
            } else {
                chars.append(TextTools.encode(text + " "))
                textStart.append(0)
            }
        }
        let U = chars.map(\.count).max() ?? 1
        var st = State(L, H: H, V: V, nAtt: nAtt, U: U)

        // 1) priming: teacher-force each line's style sample through the network (once per line)
        let primeOffsets = primes.map { $0?.offsets ?? [] }
        for t in 0 ..< (primeOffsets.map(\.count).max() ?? 0) {
            let active = (0..<L).filter { t < primeOffsets[$0].count }
            var inp = [Float](repeating: 0, count: L * 3)
            for b in active {
                let o = primeOffsets[b][t]
                inp[b * 3] = o.x; inp[b * 3 + 1] = o.y; inp[b * 3 + 2] = o.eos
            }
            st.take(step(inp, st, chars: chars, U: U, B: L), rows: active, H: H, V: V, nAtt: nAtt, U: U)
        }
        let primeAligned = (0..<L).map { b -> Bool in
            guard !primeOffsets[b].isEmpty else { return true }
            let row = st.phi[b * U ..< (b + 1) * U]
            let end = row.indices.max { row[$0] < row[$1] }! - b * U
            return Double(abs(end - (textStart[b] - 1))) <= max(3, 0.25 * Double(textStart[b]))
        }

        // 2) free-running generation, K candidates per line
        let B = L * K
        st = st.repeated(K, H: H, V: V, nAtt: nAtt, U: U)
        let rowChars = chars.flatMap { c in Array(repeating: c, count: K) }
        let cLen = rowChars.map(\.count)
        let hasPrime = primeOffsets.flatMap { Array(repeating: !$0.isEmpty, count: K) }
        var (inp, _) = sample(st.h3, bias: bias, B: B, rng: &rng)
        for b in 0..<B where !hasPrime[b] { inp[b * 3] = 0; inp[b * 3 + 1] = 0; inp[b * 3 + 2] = 1 }
        let maxSteps = maxStepsPerChar * (lines.map(\.count).max() ?? 1) + 20
        var outs = [[PenPoint]](repeating: [], count: B), attended = [[Int]](repeating: [], count: B)
        var dwell = [[Int]](repeating: [Int](repeating: 0, count: U + 1), count: B)
        var done = [Bool](repeating: false, count: B), natural = done
        for _ in 0..<maxSteps {
            let next = step(inp, st, chars: rowChars, U: U, B: B)
            let active = (0..<B).filter { !done[$0] }
            st.take(next, rows: active, H: H, V: V, nAtt: nAtt, U: U)
            let (nxt, e) = sample(st.h3, bias: bias, B: B, rng: &rng)
            for b in 0..<B {
                let row = st.phi[b * U ..< (b + 1) * U]
                let charIdx = row.indices.max { row[$0] < row[$1] }! - b * U
                if !done[b] {
                    outs[b].append(PenPoint(x: nxt[b * 3], y: nxt[b * 3 + 1], eos: nxt[b * 3 + 2]))
                    attended[b].append(charIdx)
                    dwell[b][charIdx] += 1
                }
                // finished: the attention reached the end of the text and the pen lifts
                let probe = Float.random(in: 0..<1, using: &rng) < e[b]
                if !done[b] && ((charIdx >= cLen[b] - 1 && probe) || charIdx >= cLen[b]) {
                    done[b] = true; natural[b] = true
                }
            }
            if !done.contains(false) { break }
            inp = nxt
        }

        return (0..<L).map { li in
            var cands: [Candidate] = []
            for b in li * K ..< li * K + K {
                let text = Array(lines[li])
                let d = dwell[b][textStart[li] ..< textStart[li] + text.count]
                let letters = text.map { $0 != " " }
                let letterDwell = zip(d, letters).filter(\.1).map(\.0).sorted()
                let median = letterDwell.isEmpty ? 1 : (letterDwell.count % 2 == 1
                    ? Double(letterDwell[letterDwell.count / 2])
                    : Double(letterDwell[letterDwell.count / 2 - 1] + letterDwell[letterDwell.count / 2]) / 2)
                let skipped = zip(d, letters).filter { $0.1 && $0.0 < 2 }.count
                let overlong = d.filter { Double($0) > 5 * max(1, median) }.count
                let score = skipped * 2 + (natural[b] ? 0 : 6) + overlong
                var coords = Self.trim(outs[b], attended[b], start: textStart[li], end: textStart[li] + text.count)
                coords = Ink.align(Ink.denoise(coords))
                cands.append(Candidate(coords: coords, score: score))
            }
            let sorted = cands.enumerated().sorted { ($0.element.score, $0.offset) < ($1.element.score, $1.offset) }.map(\.element)
            return Line(candidates: sorted, primeAligned: primeAligned[li])
        }
    }

    /// One sample's movements -> coords of only the strokes that write the text.
    ///
    /// Right after priming the pen sometimes finishes the *sample* first (dots
    /// its last i): strokes drawn before the attention reaches the text are
    /// dropped. After the last letter, strokes that go back over the words (a
    /// crossbar, an i-dot) are kept; anything that starts further right is the
    /// pen running on.
    static func trim(_ outs: [PenPoint], _ att: [Int], start: Int, end: Int) -> [PenPoint] {
        var off = outs.isEmpty ? [PenPoint(x: 0, y: 0, eos: 1)] : outs
        off[off.count - 1].eos = 1
        let att = outs.isEmpty ? [start] : att
        let coords = Ink.offsetsToCoords(off)
        var strokes: [Range<Int>] = []
        var a = 0
        for (i, p) in off.enumerated() where p.eos == 1 { strokes.append(a ..< i + 1); a = i + 1 }
        while strokes.count > 1, att[strokes[0]].allSatisfy({ $0 < start }) { strokes.removeFirst() }
        let after = strokes.filter { r in att[r].allSatisfy { $0 >= end } }
        let written = strokes.filter { r in !after.contains(r) }
        if !written.isEmpty {
            let right = written.map { r in coords[r].map(\.x).max()! }.max()!
            strokes = (written + after.filter { r in coords[r].map(\.x).min()! < right }).sorted { $0.lowerBound < $1.lowerBound }
        }
        return strokes.flatMap { coords[$0] }
    }
}

/// Small, fast, seedable random numbers (the synthesiser draws thousands per line).
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Two independent standard normal samples (Box-Muller).
    mutating func gaussianPair() -> (Float, Float) {
        let u1 = max(Float.random(in: 0..<1, using: &self), 1e-12), u2 = Float.random(in: 0..<1, using: &self)
        let r = (-2 * log(u1)).squareRoot()
        return (r * cos(2 * .pi * u2), r * sin(2 * .pi * u2))
    }
}
