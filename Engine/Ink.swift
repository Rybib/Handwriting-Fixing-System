import CoreGraphics
import Foundation
import simd

/// A point of a stroke on the page, y pointing down (canvas coordinates).
typealias Stroke = [SIMD2<Float>]

/// One step of the synthesiser's pen: x, y (pointing UP, as in IAM-OnDB) and
/// whether the pen lifts after this point.
struct PenPoint {
    var x: Float
    var y: Float
    var eos: Float
}

/// Ink geometry, ported from the Mac demo (`server/ink.py`, `server/synth.py`).
enum Ink {
    // MARK: - rendering and measuring

    /// Rasterise strokes (y down) to a white image with black ink, scaled so the
    /// ink is about `targetHeight` px tall. This is what the readers look at.
    static func render(_ strokes: [Stroke], targetHeight: Float = 96, pad: Float = 24, lineWidth: Float? = nil) -> CGImage? {
        let pts = strokes.flatMap { $0 }
        guard let first = pts.first else { return nil }
        var lo = first, hi = first
        for p in pts { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let h = max(hi.y - lo.y, 1)
        let scale = targetHeight / h
        let width = max(32, Int((hi.x - lo.x) * scale + 2 * pad))
        let height = max(32, Int(h * scale + 2 * pad))
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        // CoreGraphics' origin is bottom-left: flip so y points down like the canvas
        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1, y: -1)
        let lw = CGFloat(lineWidth ?? max(2, (targetHeight / 22).rounded()))
        ctx.setStrokeColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.setLineWidth(lw)
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)
        for s in strokes where !s.isEmpty {
            let mapped = s.map { CGPoint(x: CGFloat(($0.x - lo.x) * scale + pad), y: CGFloat(($0.y - lo.y) * scale + pad)) }
            if mapped.count == 1 {
                ctx.fillEllipse(in: CGRect(x: mapped[0].x - lw / 2, y: mapped[0].y - lw / 2, width: lw, height: lw))
            } else {
                ctx.addLines(between: mapped)
                ctx.strokePath()
            }
        }
        return ctx.makeImage()
    }

    /// numpy.percentile with linear interpolation.
    static func percentile(_ values: [Float], _ q: Float) -> Float {
        let s = values.sorted()
        guard !s.isEmpty else { return 0 }
        let i = Float(s.count - 1) * q
        let lo = Int(i.rounded(.down)), hi = Int(i.rounded(.up))
        return s[lo] + (s[hi] - s[lo]) * (i - Float(lo))
    }

    /// Robust (baseline, core height) of some ink, y pointing down. The same
    /// percentiles are used for the user's ink and for synthesised ink, so they
    /// can be matched without caring about either's absolute scale.
    static func bodyMetrics(_ ys: [Float]) -> (base: Float, core: Float) {
        let lo = percentile(ys, 0.20), hi = percentile(ys, 0.85)
        return (hi, max(hi - lo, 1e-3))
    }

    // MARK: - pen-point sequences

    static func offsetsToCoords(_ off: [PenPoint]) -> [PenPoint] {
        var x: Float = 0, y: Float = 0
        return off.map { p in
            x += p.x; y += p.y
            return PenPoint(x: x, y: y, eos: p.eos)
        }
    }

    static func coordsToOffsets(_ coords: [PenPoint]) -> [PenPoint] {
        guard !coords.isEmpty else { return [] }
        var out = [PenPoint(x: 0, y: 0, eos: 1)]
        for i in 1..<coords.count {
            out.append(PenPoint(x: coords[i].x - coords[i - 1].x, y: coords[i].y - coords[i - 1].y, eos: coords[i].eos))
        }
        return out
    }

    /// Split at pen lifts.
    static func splitStrokes(_ coords: [PenPoint]) -> [[PenPoint]] {
        var out: [[PenPoint]] = [], cur: [PenPoint] = []
        for p in coords {
            cur.append(p)
            if p.eos == 1 { out.append(cur); cur = [] }
        }
        if !cur.isEmpty { out.append(cur) }
        return out
    }

    /// Savitzky-Golay smoothing (window 7, cubic) of every stroke long enough,
    /// like scipy's savgol_filter(..., 7, 3, mode="nearest").
    static func denoise(_ coords: [PenPoint]) -> [PenPoint] {
        let w: [Float] = [-2, 3, 6, 7, 6, 3, -2].map { $0 / 21 }
        var out: [PenPoint] = []
        out.reserveCapacity(coords.count)
        for s in splitStrokes(coords) {
            guard s.count >= 7 else { out += s; continue }
            for i in 0..<s.count {
                var x: Float = 0, y: Float = 0
                for k in -3...3 {
                    let p = s[min(max(i + k, 0), s.count - 1)]
                    x += w[k + 3] * p.x
                    y += w[k + 3] * p.y
                }
                out.append(PenPoint(x: x, y: y, eos: s[i].eos))
            }
        }
        return out
    }

    /// Remove the overall baseline tilt (least-squares line through all points).
    static func align(_ coords: [PenPoint]) -> [PenPoint] {
        guard coords.count > 1 else { return coords }
        let n = Float(coords.count)
        var sx: Float = 0, sy: Float = 0, sxx: Float = 0, sxy: Float = 0
        for p in coords { sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y }
        let den = n * sxx - sx * sx
        let slope = abs(den) < 1e-9 ? 0 : (n * sxy - sx * sy) / den
        let offset = (sy - slope * sx) / n
        let t = atan(slope), c = cos(t), s = sin(t)
        // [x y] @ [[c, -s], [s, c]] - offset, exactly as the numpy version
        return coords.map { p in PenPoint(x: p.x * c + p.y * s - offset, y: -p.x * s + p.y * c - offset, eos: p.eos) }
    }

    /// Uniform arc-length resampling of one stroke.
    static func resample(_ pts: Stroke, spacing: Float) -> Stroke {
        guard pts.count >= 2 else { return pts }
        var d: [Float] = [0]
        for i in 1..<pts.count { d.append(d[i - 1] + simd_distance(pts[i], pts[i - 1])) }
        let total = d.last!
        guard total > 1e-6 else { return [pts[0]] }
        let n = max(2, Int((total / spacing).rounded(.up)) + 1)
        var out: Stroke = []
        var j = 0
        for k in 0..<n {
            let t = total * Float(k) / Float(n - 1)
            while j < d.count - 2 && d[j + 1] < t { j += 1 }
            let seg = d[j + 1] - d[j]
            let f = seg > 0 ? min(max((t - d[j]) / seg, 0), 1) : 0
            out.append(pts[j] + (pts[j + 1] - pts[j]) * f)
        }
        return out
    }

    // MARK: - between the page and the synthesiser

    /// Width of one character of IAM-OnDB ink in the network's units (the 13
    /// built-in styles range 6.6-12.8, median ~10); it was trained on ink with
    /// ~1 unit between consecutive pen-down points.
    static let iamUnitsPerChar: Float = 10

    /// The user's (tidied) strokes, y down, as a priming sample: rescaled so
    /// the width per character matches the training data and resampled to about
    /// the pen speed the network learned. nil if unusable.
    static func userStrokesToPrime(_ strokes: [Stroke], transcript: String) -> HandwritingSynth.Prime? {
        let strokes = strokes.filter { !$0.isEmpty }
        let pts = strokes.flatMap { $0 }
        guard let first = pts.first, !transcript.trimmingCharacters(in: .whitespaces).isEmpty else { return nil }
        var lo = first, hi = first
        for p in pts { lo = simd_min(lo, p); hi = simd_max(hi, p) }
        let width = hi.x - lo.x
        guard width >= 1 else { return nil }
        let scale = iamUnitsPerChar * Float(max(1, transcript.count)) / width
        var coords: [PenPoint] = []
        for s in strokes {
            // y up, origin at the ink's top-left corner (as numpy: s - min * [1, -1])
            let flipped = s.map { SIMD2<Float>(($0.x - lo.x) * scale, (-$0.y + lo.y) * scale) }
            let r = resample(flipped, spacing: 1)
            for (i, p) in r.enumerated() { coords.append(PenPoint(x: p.x, y: p.y, eos: i == r.count - 1 ? 1 : 0)) }
        }
        let offsets = coordsToOffsets(align(coords))
        return HandwritingSynth.Prime(offsets: Array(offsets.prefix(1200)), text: transcript)
    }

    /// Synthesiser coords (y up) -> strokes with y down.
    static func toStrokes(_ coords: [PenPoint]) -> [Stroke] {
        splitStrokes(coords).map { $0.map { SIMD2<Float>($0.x, -$0.y) } }
    }

    /// Synthesiser coords -> strokes with y down, baseline 0, core height 1 and
    /// left edge 0: the format the page scales onto the canvas.
    static func normaliseOutput(_ coords: [PenPoint]) -> [Stroke] {
        let strokes = toStrokes(coords)
        let all = strokes.flatMap { $0 }
        guard !all.isEmpty else { return [] }
        let (base, core) = bodyMetrics(all.map(\.y))
        let x0 = all.map(\.x).min()!
        return strokes.map { $0.map { SIMD2<Float>(($0.x - x0) / core, ($0.y - base) / core) } }
    }
}
