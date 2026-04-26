import SwiftUI
import HumanSenseKit

struct TokenTableView: View {
    @ObservedObject var recorder: TokenSampleRecorder
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
        HStack(spacing: 4) {
            Text("字").frame(width: 26, alignment: .leading)
            Text("jStd").frame(width: 38, alignment: .trailing)
            Text("mxJ").frame(width: 36, alignment: .trailing)
            Text("vol").frame(width: 40, alignment: .trailing)
            Text("r").frame(width: 38, alignment: .trailing)       // local Pearson
            Text("fus").frame(width: 40, alignment: .trailing)
            Text("👁").frame(width: 22, alignment: .center)
            Text("🧭").frame(width: 22, alignment: .center)
            Text("n").frame(width: 22, alignment: .trailing)
            Text("w").frame(width: 30, alignment: .trailing)       // effective window ms
            Text("user").frame(width: 34, alignment: .center)
            Spacer()
        }
        .font(.caption2.monospaced())
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
    let row: TokenSampleRecorder.TokenRow

    var body: some View {
        HStack(spacing: 4) {
            Text(row.text)
                .font(.body.monospaced())
                .frame(width: 26, alignment: .leading)
                .foregroundStyle(textColor)
                .bold(row.isUser)

            Text(String(format: "%.3f", row.jawStd))
                .font(.caption.monospaced())
                .frame(width: 38, alignment: .trailing)
                .foregroundStyle(jawStdColor)

            Text(String(format: "%.2f", row.maxJaw))
                .font(.caption.monospaced())
                .frame(width: 36, alignment: .trailing)
                .foregroundStyle(row.maxJaw > 0.15 ? .green : Color.secondary)

            Text(String(format: "%.3f", row.maxVol))
                .font(.caption.monospaced())
                .frame(width: 40, alignment: .trailing)
                .foregroundStyle(row.maxVol > 0.01 ? .green : Color.secondary)

            Text(String(format: "%+.2f", row.localPearson))
                .font(.caption.monospaced())
                .frame(width: 38, alignment: .trailing)
                .foregroundStyle(scoreColor(row.localPearson))

            Text(String(format: "%+.2f", row.fusedScore))
                .font(.caption.monospaced().bold())
                .frame(width: 40, alignment: .trailing)
                .foregroundStyle(scoreColor(row.fusedScore))

            Text(row.gazeRatio > 0.5 ? "✓" : "·")
                .frame(width: 22, alignment: .center)
                .foregroundStyle(row.gazeRatio > 0.5 ? .green : Color.secondary)
            Text(row.headFwdRatio > 0.5 ? "✓" : "·")
                .frame(width: 22, alignment: .center)
                .foregroundStyle(row.headFwdRatio > 0.5 ? .green : Color.secondary)
            Text("\(row.sampleCount)")
                .font(.system(size: 9, design: .monospaced))
                .frame(width: 22, alignment: .trailing)
                .foregroundStyle(row.sampleCount < 3 ? Color.orange : Color.secondary)
            Text(String(format: "%.0f", row.effectiveWindow * 1000))
                .font(.system(size: 9, design: .monospaced))
                .frame(width: 30, alignment: .trailing)
                .foregroundStyle(.secondary)

            Text(userGlyph)
                .frame(width: 34, alignment: .center)
                .foregroundStyle(userColor)
                .bold()
            Spacer()
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 2)
        .background(row.filledBySentence ? Color.yellow.opacity(0.08) : Color.clear)
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
        if row.gazeRatio > 0.5 { return .green }
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
