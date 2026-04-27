import SwiftUI
import HumanSenseKit

struct TokenTableView: View {
    @ObservedObject var recorder: UserSentenceReconstructor
    var engine: HumanStateEngine?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            sentenceBar
            Divider()
            verdictBar
            Divider()
            // Live signals
            if let engine {
                HStack(spacing: 16) {
                    liveSignal("👄", engine.humanState.face.jawOpen > 0.15,
                               detail: String(format: "%.2f", engine.humanState.face.jawOpen))
                    liveSignal("👁", engine.humanState.face.isLookingAtScreen)
                    liveSignal("🧭", engine.humanState.face.headOrientation.isFacingForward)
                    Spacer()
                    Button("Clear") { recorder.clear() }.buttonStyle(.bordered)
                }
                .padding(.horizontal)
                .padding(.vertical, 6)
                Divider()
            }

            header.padding(.horizontal).padding(.vertical, 4)
            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(recorder.tokens) { row in
                            TokenRowView(row: row).id(row.id)
                            Divider().background(Color.white.opacity(0.06))
                        }
                    }
                }
                .onChange(of: recorder.tokens.count) {
                    if let last = recorder.tokens.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }
        }
        .background(Color.black)
    }

    // MARK: - Top: reconstructed user sentence

    private var sentenceBar: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("你对屏幕说的话").font(.caption.bold()).foregroundStyle(.secondary)
                Spacer()
                Text("\(recorder.tokens.count) tokens")
                    .font(.caption2.monospaced()).foregroundStyle(.secondary)
            }
            if recorder.userSentence.isEmpty {
                Text("（等待语音…）").font(.title3).italic()
                    .foregroundStyle(.secondary.opacity(0.5))
            } else {
                Text(recorder.userSentence)
                    .font(.title2.bold())
                    .foregroundStyle(.green)
                    .lineLimit(3)
                    .animation(.easeInOut(duration: 0.15), value: recorder.userSentence)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color.white.opacity(0.03))
    }

    // MARK: - Verdict bar

    private var verdictBar: some View {
        let v = recorder.lastVerdict
        return HStack(spacing: 10) {
            verdictPill("ratio", String(format: "%.0f%%", v.userRatio * 100), hit: v.userRatio >= 0.5)
            verdictPill("peak",  String(format: "%.2f", v.peakScore),     hit: v.peakScore >= 0.45)
            verdictPill("span",  String(format: "%.0f%%", v.longestSpanRatio * 100), hit: v.longestSpanRatio >= 0.35)
            Spacer()
            Text(v.isUserDominant ? "✓ 整句算你的" : "· 分字判断")
                .font(.caption.bold())
                .foregroundStyle(v.isUserDominant ? .green : .orange)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(Color.white.opacity(0.02))
    }

    private func verdictPill(_ label: String, _ value: String, hit: Bool) -> some View {
        HStack(spacing: 4) {
            Circle().fill(hit ? Color.green : Color.gray).frame(width: 6, height: 6)
            Text("\(label):").font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
            Text(value).font(.system(size: 10, design: .monospaced).bold())
                .foregroundStyle(hit ? .green : .gray)
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 2) {
            Text("词").frame(minWidth: 44, alignment: .leading)
            Text("conf").frame(width: 34, alignment: .trailing).bold()
            Text("G").frame(width: 20, alignment: .trailing)
            Text("M").frame(width: 20, alignment: .trailing)
            Text("S").frame(width: 20, alignment: .trailing)
            Text("jaw").frame(width: 28, alignment: .trailing)
            Text("vol").frame(width: 28, alignment: .trailing)
            Text("r").frame(width: 28, alignment: .trailing)
            Text("👄").frame(width: 18, alignment: .center)
            Text("👁").frame(width: 18, alignment: .center)
            Text("🧭").frame(width: 18, alignment: .center)
            Text("u").frame(width: 16, alignment: .center)
            Text("u+").frame(width: 20, alignment: .center).bold()
            Spacer(minLength: 0)
        }
        .font(.system(size: 9, design: .monospaced))
        .foregroundStyle(.secondary)
    }

    private func liveSignal(_ emoji: String, _ on: Bool, detail: String? = nil) -> some View {
        HStack(spacing: 4) {
            Text(emoji)
            Circle().fill(on ? Color.green : Color.red.opacity(0.4)).frame(width: 8, height: 8)
            if let detail {
                Text(detail).font(.caption2.monospaced()).foregroundStyle(.secondary)
            }
        }
    }
}

private struct TokenRowView: View {
    let row: UserSentenceReconstructor.TokenRow

    var body: some View {
        HStack(spacing: 2) {
            Text(row.text)
                .font(.system(size: 13, design: .monospaced))
                .frame(width: 44, alignment: .leading)
                .foregroundStyle(textColor)
                .bold(row.isUser)
                // Force CJK-friendly wrapping inside the 44pt cell so a
                // 4-char word breaks to two lines instead of pushing the
                // numeric columns off screen. fixedSize lets the row grow
                // vertically to accommodate.
                .lineLimit(nil)
                .fixedSize(horizontal: false, vertical: true)

            // Composed confidence — main "is this really you" answer.
            Text(String(format: "%.2f", row.userConfidence))
                .font(.system(size: 11, design: .monospaced).bold())
                .frame(width: 34, alignment: .trailing)
                .foregroundStyle(confColor(row.userConfidence))

            // Sub-scores as two-digit ints (0-100) to save width.
            subScore(row.gateScore)
            subScore(row.mouthScore)
            subScore(row.syncScore)

            Text(String(format: "%.2f", row.maxJaw))
                .font(.system(size: 9, design: .monospaced))
                .frame(width: 28, alignment: .trailing)
                .foregroundStyle(row.maxJaw > 0.15 ? .green : Color.secondary)

            Text(String(format: "%.2f", row.maxVol))
                .font(.system(size: 9, design: .monospaced))
                .frame(width: 28, alignment: .trailing)
                .foregroundStyle(row.maxVol > 0.01 ? .green : Color.secondary)

            Text(String(format: "%+.1f", row.localPearson))
                .font(.system(size: 9, design: .monospaced))
                .frame(width: 28, alignment: .trailing)
                .foregroundStyle(scoreColor(row.localPearson))

            Text(row.maxJaw > 0.15 ? "✓" : "·")
                .font(.system(size: 11))
                .frame(width: 18, alignment: .center)
                .foregroundStyle(row.maxJaw > 0.15 ? .green : Color.secondary)
            Text(row.gazeRatio >= UserSentenceReconstructor.gazeRatioThreshold ? "✓" : "·")
                .font(.system(size: 11))
                .frame(width: 18, alignment: .center)
                .foregroundStyle(row.gazeRatio >= UserSentenceReconstructor.gazeRatioThreshold ? .green : Color.secondary)
            Text(row.headFwdRatio >= UserSentenceReconstructor.headFwdRatioThreshold ? "✓" : "·")
                .font(.system(size: 11))
                .frame(width: 18, alignment: .center)
                .foregroundStyle(row.headFwdRatio >= UserSentenceReconstructor.headFwdRatioThreshold ? .green : Color.secondary)

            Text(userGlyph)
                .font(.system(size: 11).bold())
                .frame(width: 16, alignment: .center)
                .foregroundStyle(userColor)
            // Span-aware verdict: '✓' when this token sits inside a user
            // span identified by the Schmitt-trigger on smoothed conf.
            // Compare against the plain 'u' column to see span extraction
            // rescuing mid-utterance dips or rejecting foreign-speech blips.
            Text(row.isUserWithConfidence ? "✓" : "·")
                .font(.system(size: 12).bold())
                .frame(width: 20, alignment: .center)
                .foregroundStyle(row.isUserWithConfidence ? Color.blue : Color.secondary)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(row.filledBySentence ? Color.yellow.opacity(0.08) : Color.clear)
    }

    /// Two-digit integer cell for a 0-1 sub-score (e.g. 0.87 → "87",
    /// 1.0 → "100"). Saves a lot of width vs "0.87".
    private func subScore(_ v: Float) -> some View {
        let pct = Int((v * 100).rounded())
        return Text("\(pct)")
            .font(.system(size: 10, design: .monospaced))
            .frame(width: 20, alignment: .trailing)
            .foregroundStyle(subColor(v))
    }

    private func subColor(_ v: Float) -> Color {
        if v >= 0.7 { return .green }
        if v >= 0.4 { return .yellow }
        if v >= 0.2 { return .orange }
        return Color.secondary
    }

    /// Composed confidence color — stricter bands than sub-scores.
    private func confColor(_ v: Float) -> Color {
        if v >= 0.6 { return .green }
        if v >= 0.35 { return .yellow }
        if v >= 0.15 { return .orange }
        return .red
    }

    private var userGlyph: String {
        if !row.isUser { return "·" }
        return row.filledBySentence ? "↑" : "✓"
    }

    private var userColor: Color {
        if !row.isUser { return .gray }
        return row.filledBySentence ? .yellow : .green
    }

    private var textColor: Color {
        if !row.isUser { return .gray }
        if row.filledBySentence { return .yellow }
        if row.gazeRatio >= UserSentenceReconstructor.gazeRatioThreshold { return .green }
        return .cyan   // user speaking but not looking at screen
    }

    private var jawStdColor: Color {
        if row.jawStd > 0.02 { return .green }
        if row.jawStd > 0.005 { return .yellow }
        return Color.secondary
    }

    private func scoreColor(_ v: Float) -> Color {
        if v >= 0.5 { return .green }
        if v >= 0.3 { return .yellow }
        if v >= 0.0 { return .orange }
        return .red
    }
}
