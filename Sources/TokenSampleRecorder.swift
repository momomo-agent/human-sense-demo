import Foundation
import HumanSenseKit

@MainActor
final class TokenSampleRecorder: ObservableObject {
    struct Sample {
        let ts: Double
        let jaw: Float
        let vol: Float
        let gaze: Bool
        let headFwd: Bool
    }

    struct TokenRow: Identifiable {
        let id = UUID()
        let text: String
        let startTime: Double
        let endTime: Double
        let avgJaw: Float
        let maxJaw: Float
        let avgVol: Float
        let maxVol: Float
        let gazeRatio: Float   // fraction of samples where isLookingAtScreen
        let headFwdRatio: Float // fraction of samples where isFacingForward
        let sampleCount: Int
        let isFinal: Bool
    }

    private var samples: [Sample] = []
    private let maxSamples = 6000

    /// Finalized rows — permanent.
    private var finalizedTokens: [TokenRow] = []
    /// Current volatile rows — replaced entirely on each volatile update.
    private var volatileTokens: [TokenRow] = []

    /// Combined view for the UI.
    @Published var tokens: [TokenRow] = []

    var audioStreamStartTime: Date?

    func recordSample(ts: Double, jaw: Float, vol: Float, gaze: Bool, headFwd: Bool) {
        samples.append(Sample(ts: ts, jaw: jaw, vol: vol, gaze: gaze, headFwd: headFwd))
        if samples.count > maxSamples {
            samples.removeFirst(samples.count - maxSamples)
        }
    }

    func recordTokens(_ incoming: [SpeechToken], isFinal: Bool) {
        let base = audioStreamStartTime?.timeIntervalSince1970 ?? 0
        let now = Date().timeIntervalSince1970
        if let first = incoming.first, let last = incoming.last {
            let wStart = base + first.startTime
            let wEnd = base + last.endTime
            let sRange = samples.isEmpty ? "empty" : String(format: "%.3f-%.3f", samples.first!.ts, samples.last!.ts)
            print(String(format: "[TOK-DBG] base:%.3f now:%.3f tokWall:[%.3f-%.3f] samples:%d range:%@ isFinal:%d",
                         base, now, wStart, wEnd, samples.count, sRange, isFinal ? 1 : 0))
        }
        let rows = splitToCharRows(incoming, base: base, isFinal: isFinal)

        if isFinal {
            finalizedTokens.append(contentsOf: rows)
            volatileTokens.removeAll()
            if finalizedTokens.count > 500 {
                finalizedTokens.removeFirst(finalizedTokens.count - 500)
            }
        } else {
            volatileTokens = rows
        }
        tokens = finalizedTokens + volatileTokens
    }

    func clear() {
        finalizedTokens.removeAll()
        volatileTokens.removeAll()
        tokens.removeAll()
        samples.removeAll()
    }

    private func splitToCharRows(_ tokens: [SpeechToken], base: Double, isFinal: Bool) -> [TokenRow] {
        var rows: [TokenRow] = []
        for t in tokens {
            let wallStart = base + t.startTime
            let wallEnd = base + t.endTime
            let chars = Array(t.text)
            guard !chars.isEmpty else { continue }
            let charDur = (wallEnd - wallStart) / Double(chars.count)
            for (i, ch) in chars.enumerated() {
                let cs = wallStart + Double(i) * charDur
                let ce = cs + charDur
                rows.append(buildRow(text: String(ch), wallStart: cs, wallEnd: ce, isFinal: isFinal))
            }
        }
        return rows
    }

    private func buildRow(text: String, wallStart: Double, wallEnd: Double, isFinal: Bool) -> TokenRow {
        let matched = samples.filter { $0.ts >= wallStart && $0.ts <= wallEnd }
        let jaws = matched.map { $0.jaw }
        let vols = matched.map { $0.vol }
        let gazeCount = matched.filter { $0.gaze }.count
        let headCount = matched.filter { $0.headFwd }.count
        let n = matched.count
        return TokenRow(
            text: text,
            startTime: wallStart,
            endTime: wallEnd,
            avgJaw: jaws.isEmpty ? 0 : jaws.reduce(0, +) / Float(n),
            maxJaw: jaws.max() ?? 0,
            avgVol: vols.isEmpty ? 0 : vols.reduce(0, +) / Float(n),
            maxVol: vols.max() ?? 0,
            gazeRatio: n > 0 ? Float(gazeCount) / Float(n) : 0,
            headFwdRatio: n > 0 ? Float(headCount) / Float(n) : 0,
            sampleCount: n,
            isFinal: isFinal
        )
    }
}
