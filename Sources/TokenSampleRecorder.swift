import Foundation
import HumanSenseKit

/// Records per-frame (jaw, vol, gaze, headFwd) samples and aligns them to
/// STT tokens. For each token (or per-character), computes LOCAL statistics
/// from samples inside the token's audio time range:
///  - avgJaw / maxJaw / jawStd
///  - avgVol / maxVol / volStd
///  - localPearson (jaw-vs-vol correlation in this window)
///  - gazeRatio / headFwdRatio
///  - isUser (direct per-token verdict)
///  - filledBySentence (upgraded by sentence-level majority vote)
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
        let jawStd: Float
        let avgVol: Float
        let maxVol: Float
        let volStd: Float
        let localPearson: Float     // jaw-vs-vol correlation, computed LOCALLY from this window
        let fusedScore: Float       // combined confidence
        let gazeRatio: Float
        let headFwdRatio: Float
        let sampleCount: Int
        let effectiveWindow: Double // how wide we actually looked (may expand if too few samples)
        let isUser: Bool
        let filledBySentence: Bool
        let isFinal: Bool
    }

    struct Verdict {
        let userRatio: Float
        let peakScore: Float
        let longestSpanRatio: Float
        let isUserDominant: Bool
    }

    private var samples: [Sample] = []
    private let maxSamples = 6000

    private var finalizedTokens: [TokenRow] = []
    private var volatileTokens: [TokenRow] = []

    @Published var tokens: [TokenRow] = []
    @Published var lastVerdict = Verdict(userRatio: 0, peakScore: 0, longestSpanRatio: 0, isUserDominant: false)
    /// Reconstructed user sentence from the CURRENT volatile (live) + recent finalized tokens.
    @Published var userSentence: String = ""

    var audioStreamStartTime: Date?

    // MARK: - Thresholds
    // Primary decision is JAW ACTIVITY (you told me that was more accurate).
    // Pearson is just a tie-breaker / negative signal.
    let jawActivityThreshold: Float = 0.12      // maxJaw in window ≥ this → probably speaking
    let jawStdThreshold: Float = 0.008          // jaw std ≥ this → mouth moving
    let volActiveThreshold: Float = 0.008       // vol present
    let minSampleCount = 3                      // below this we expand the window
    let windowExpandMs: Double = 0.08           // ±80 ms expansion per try
    let maxWindowMs: Double = 0.30              // cap expansion

    func recordSample(ts: Double, jaw: Float, vol: Float, gaze: Bool, headFwd: Bool) {
        samples.append(Sample(ts: ts, jaw: jaw, vol: vol, gaze: gaze, headFwd: headFwd))
        if samples.count > maxSamples {
            samples.removeFirst(samples.count - maxSamples)
        }
    }

    func recordTokens(_ incoming: [SpeechToken], isFinal: Bool) {
        let base = audioStreamStartTime?.timeIntervalSince1970 ?? 0
        let rows = splitToCharRows(incoming, base: base, isFinal: isFinal)
        let votedRows = applySentenceVote(rows)

        if isFinal {
            finalizedTokens.append(contentsOf: votedRows)
            volatileTokens.removeAll()
            if finalizedTokens.count > 500 {
                finalizedTokens.removeFirst(finalizedTokens.count - 500)
            }
        } else {
            volatileTokens = votedRows
        }
        tokens = finalizedTokens + volatileTokens

        // Reconstruct user sentence from volatile + last utterance-worth of finalized
        userSentence = reconstructSentence(volatile: volatileTokens, finalized: finalizedTokens)
    }

    func clear() {
        finalizedTokens.removeAll()
        volatileTokens.removeAll()
        tokens.removeAll()
        samples.removeAll()
        userSentence = ""
    }

    // MARK: - Split to char rows

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
        // Expand window if sample count is too low (common for 50-100ms tokens)
        var expand: Double = 0
        var matched = samples.filter { $0.ts >= wallStart && $0.ts <= wallEnd }
        while matched.count < minSampleCount && expand < maxWindowMs {
            expand += windowExpandMs
            let s = wallStart - expand
            let e = wallEnd + expand
            matched = samples.filter { $0.ts >= s && $0.ts <= e }
        }

        let jaws = matched.map { $0.jaw }
        let vols = matched.map { $0.vol }
        let n = matched.count

        let jawAvg = n == 0 ? 0 : jaws.reduce(0, +) / Float(n)
        let volAvg = n == 0 ? 0 : vols.reduce(0, +) / Float(n)
        let jawStd = std(jaws, mean: jawAvg)
        let volStd = std(vols, mean: volAvg)

        // Local Pearson: jaw vs vol inside this (possibly expanded) window
        let pearson = localPearson(jaws: jaws, jawMean: jawAvg, vols: vols, volMean: volAvg, jawStd: jawStd, volStd: volStd)

        // Fused score: Pearson is primary, boost by jaw activity, penalize silent-mouth+active-voice
        let fused = fuse(pearson: pearson, jawStd: jawStd, volStd: volStd, volAvg: volAvg)

        let gazeCount = matched.filter { $0.gaze }.count
        let headCount = matched.filter { $0.headFwd }.count

        return TokenRow(
            text: text,
            startTime: wallStart,
            endTime: wallEnd,
            avgJaw: jawAvg,
            maxJaw: jaws.max() ?? 0,
            jawStd: jawStd,
            avgVol: volAvg,
            maxVol: vols.max() ?? 0,
            volStd: volStd,
            localPearson: pearson,
            fusedScore: fused,
            gazeRatio: n > 0 ? Float(gazeCount) / Float(n) : 0,
            headFwdRatio: n > 0 ? Float(headCount) / Float(n) : 0,
            sampleCount: n,
            effectiveWindow: (wallEnd - wallStart) + 2 * expand,
            isUser: decideIsUser(
                maxJaw: jaws.max() ?? 0, jawStd: jawStd, volStd: volStd,
                volAvg: volAvg, pearson: pearson,
                gazeRatio: n > 0 ? Float(gazeCount) / Float(n) : 0,
                headFwdRatio: n > 0 ? Float(headCount) / Float(n) : 0
            ),
            filledBySentence: false,
            isFinal: isFinal
        )
    }

    // MARK: - Fusion & correlation

    private func std(_ v: [Float], mean: Float) -> Float {
        guard v.count >= 2 else { return 0 }
        let ss = v.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
        return sqrt(ss / Float(v.count))
    }

    private func localPearson(jaws: [Float], jawMean: Float, vols: [Float], volMean: Float, jawStd: Float, volStd: Float) -> Float {
        guard jaws.count == vols.count, jaws.count >= 2 else { return 0 }
        // Guard against low variance: if either signal is flat, correlation is undefined
        guard jawStd > 0.0005 && volStd > 0.0005 else { return 0 }
        var sum: Float = 0
        for i in jaws.indices {
            sum += (jaws[i] - jawMean) * (vols[i] - volMean)
        }
        let cov = sum / Float(jaws.count)
        let r = cov / (jawStd * volStd)
        return max(-1, min(1, r))
    }

    /// Fused score mostly for debug display. Primary decision uses direct rules below.
    private func fuse(pearson: Float, jawStd: Float, volStd: Float, volAvg: Float) -> Float {
        var score: Float = 0
        if jawStd >= jawStdThreshold { score += 0.4 }
        if volStd > 0.005 { score += 0.2 }
        if pearson > 0.1 { score += 0.2 }
        if pearson > 0.3 { score += 0.2 }
        // Penalty: silent mouth but loud audio → probably someone else
        if jawStd < 0.004 && volAvg > 0.02 { score -= 0.5 }
        return max(-1, min(1, score))
    }

    /// Direct per-token verdict — MUST be looking at screen + head forward,
    /// AND have jaw activity or voice-synced mouth movement.
    private func decideIsUser(maxJaw: Float, jawStd: Float, volStd: Float, volAvg: Float, pearson: Float, gazeRatio: Float, headFwdRatio: Float) -> Bool {
        // GATE: must be looking at the screen AND facing forward for most of the token
        let lookingAtScreen = gazeRatio > 0.5 && headFwdRatio > 0.5
        if !lookingAtScreen { return false }

        // HARD NO: silent mouth + audio present (someone else speaking nearby)
        if jawStd < 0.004 && volAvg > 0.02 && pearson < 0.1 { return false }
        // STRONG YES: clear jaw activity
        if maxJaw >= jawActivityThreshold { return true }
        // MEDIUM YES: jaw moving + voice active together
        if jawStd >= jawStdThreshold && volStd > volActiveThreshold { return true }
        // TIE-BREAKER: Pearson actually agrees
        if pearson >= 0.25 && volStd > 0.003 { return true }
        return false
    }

    // MARK: - Sentence-level vote

    /// If the sentence is dominantly the user, upgrade low-conf tokens to isUser.
    private func applySentenceVote(_ rows: [TokenRow]) -> [TokenRow] {
        guard rows.count >= 2 else {
            lastVerdict = Verdict(userRatio: 0, peakScore: 0, longestSpanRatio: 0, isUserDominant: false)
            return rows
        }
        let userCount = rows.filter { $0.isUser }.count
        let ratio = Float(userCount) / Float(rows.count)
        let peak = rows.map { $0.fusedScore }.max() ?? 0

        var longest = 0, cur = 0
        for r in rows {
            if r.isUser { cur += 1; longest = max(longest, cur) } else { cur = 0 }
        }
        let spanRatio = Float(longest) / Float(rows.count)

        // Thresholds tuned to relax: any of these — ratio ≥ 40%, peak ≥ 0.4, span ≥ 30%
        let hits = [ratio >= 0.40, peak >= 0.40, spanRatio >= 0.30].filter { $0 }.count
        let dominant = hits >= 2

        lastVerdict = Verdict(userRatio: ratio, peakScore: peak, longestSpanRatio: spanRatio, isUserDominant: dominant)

        guard dominant else { return rows }

        // Upgrade all non-user rows unless they have very strong evidence against
        return rows.map { row in
            if row.isUser { return row }
            // Sentence vote can only fill in tokens where user was also looking at screen.
            // If they looked away, that's the sentence boundary — don't steal across.
            let lookingAtScreen = row.gazeRatio > 0.5 && row.headFwdRatio > 0.5
            if !lookingAtScreen { return row }
            // Keep as non-user if it's clearly not: completely silent mouth AND loud audio
            let clearlyNot = row.jawStd < 0.003 && row.avgVol > 0.03
            if clearlyNot { return row }
            return TokenRow(
                text: row.text, startTime: row.startTime, endTime: row.endTime,
                avgJaw: row.avgJaw, maxJaw: row.maxJaw, jawStd: row.jawStd,
                avgVol: row.avgVol, maxVol: row.maxVol, volStd: row.volStd,
                localPearson: row.localPearson, fusedScore: row.fusedScore,
                gazeRatio: row.gazeRatio, headFwdRatio: row.headFwdRatio,
                sampleCount: row.sampleCount, effectiveWindow: row.effectiveWindow,
                isUser: true, filledBySentence: true, isFinal: row.isFinal
            )
        }
    }

    // MARK: - Sentence reconstruction

    /// Reconstruct the user's live/current sentence: take volatile rows if present,
    /// otherwise take the tail of finalized rows until we hit a non-user gap or sentence boundary.
    private func reconstructSentence(volatile: [TokenRow], finalized: [TokenRow]) -> String {
        if !volatile.isEmpty {
            return volatile.filter { $0.isUser }.map(\.text).joined()
        }
        // No volatile — show the last finalized run of user tokens
        // (most recent contiguous isUser stretch)
        var tail: [TokenRow] = []
        for row in finalized.reversed() {
            if row.isUser {
                tail.insert(row, at: 0)
            } else if !tail.isEmpty {
                break
            }
        }
        return tail.map(\.text).joined()
    }
}
