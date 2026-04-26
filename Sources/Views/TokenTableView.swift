import SwiftUI
import HumanSenseKit

/// Token-level attribution debug view.
/// Top: reconstructed user sentence + sentence verdict.
/// Below: per-token table with raw Pearson, fused signals, smoothed conf,
/// jaw features, face visibility, and isUser flag (+ whether it was
/// upgraded by the sentence-level vote).
struct TokenTableView: View {
    @ObservedObject var sttManager: STTManager
    var engine: HumanStateEngine

    @State private var refreshTick: Int = 0
    private let timer = Timer.publish(every: 0.15, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            reconstructedSentenceBar
            Divider()
            verdictBar
            Divider()
            tokenList
        }
        .background(Color.black)
        .onReceive(timer) { _ in refreshTick += 1 }
    }

    // MARK: - Top: reconstructed sentence

    private var reconstructedSentenceBar: some View {
        let attributor = sttManager.tokenAttributor
        let sentence = attributor.userSentence
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("你对屏幕说的话")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(attributor.currentTokens.count) tokens")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }

            if sentence.isEmpty {
                Text("（等待语音…）")
                    .font(.title3)
                    .foregroundStyle(.secondary.opacity(0.5))
                    .italic()
            } else {
                Text(sentence)
                    .font(.title2.bold())
                    .foregroundStyle(.green)
                    .lineLimit(3)
                    .animation(.easeInOut(duration: 0.15), value: sentence)
            }

            if !attributor.currentRuns.isEmpty {
                runsInlineView(attributor.currentRuns)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color.white.opacity(0.03))
    }

    private func runsInlineView(_ runs: [TokenAttributor.AttributedRun]) -> some View {
        HStack(spacing: 2) {
            ForEach(runs) { run in
                Text(run.text)
                    .font(.callout)
                    .foregroundStyle(run.isUser ? .green : .gray)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 2)
                    .background(run.isUser ? Color.green.opacity(0.1) : Color.gray.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }
            Spacer(minLength: 0)
        }
    }

    // MARK: - Sentence verdict bar

    private var verdictBar: some View {
        let v = sttManager.tokenAttributor.lastVerdict
        return HStack(spacing: 10) {
            verdictPill("ratio", String(format: "%.0f%%", v.userTokenRatio * 100), hit: v.userTokenRatio >= 0.5)
            verdictPill("peak",  String(format: "%.2f", v.peakConfidence), hit: v.peakConfidence >= 0.45)
            verdictPill("span",  String(format: "%.0f%%", v.longestUserSpan * 100), hit: v.longestUserSpan >= 0.35)
            Spacer()
            Text(v.isUserDominant ? "✓ 整句算你的" : "· 分字判断")
                .font(.caption.bold())
                .foregroundStyle(v.isUserDominant ? .green : .orange)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.white.opacity(0.02))
    }

    private func verdictPill(_ label: String, _ value: String, hit: Bool) -> some View {
        HStack(spacing: 4) {
            Circle().fill(hit ? Color.green : Color.gray).frame(width: 6, height: 6)
            Text("\(label):")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 10, design: .monospaced).bold())
                .foregroundStyle(hit ? .green : .gray)
        }
    }

    // MARK: - Token Table

    private var tokenList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 2) {
                headerRow
                ForEach(Array(sttManager.tokenAttributor.currentTokens.enumerated()), id: \.offset) { idx, token in
                    tokenRow(idx: idx, token: token)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
        }
    }

    private var headerRow: some View {
        HStack(spacing: 6) {
            Text("#").frame(width: 22, alignment: .trailing)
            Text("字").frame(width: 40, alignment: .leading)
            Text("raw").frame(width: 42, alignment: .trailing)
            Text("fus").frame(width: 42, alignment: .trailing)
            Text("sm").frame(width: 42, alignment: .trailing)
            Text("jaw").frame(width: 38, alignment: .trailing)
            Text("rate").frame(width: 38, alignment: .trailing)
            Text("face").frame(width: 36, alignment: .trailing)
            Text("user").frame(width: 40, alignment: .center)
            Spacer()
        }
        .font(.system(size: 9).monospaced())
        .foregroundStyle(.secondary)
        .padding(.bottom, 4)
    }

    private func tokenRow(idx: Int, token: TokenAttributor.AttributedToken) -> some View {
        HStack(spacing: 6) {
            Text("\(idx)")
                .frame(width: 22, alignment: .trailing)
                .foregroundStyle(.secondary)

            Text(token.text)
                .frame(width: 40, alignment: .leading)
                .foregroundStyle(token.isUser ? (token.filledBySentence ? .yellow : .green) : .gray)
                .bold(token.isUser)

            Text(String(format: "%+.2f", token.rawConfidence))
                .frame(width: 42, alignment: .trailing)
                .foregroundStyle(confColor(token.rawConfidence))
            Text(String(format: "%+.2f", token.fusedConfidence))
                .frame(width: 42, alignment: .trailing)
                .foregroundStyle(confColor(token.fusedConfidence))
            Text(String(format: "%+.2f", token.smoothedConfidence))
                .frame(width: 42, alignment: .trailing)
                .foregroundStyle(confColor(token.smoothedConfidence))
                .bold()

            Text(String(format: "%.3f", token.features.jawStd))
                .frame(width: 38, alignment: .trailing)
                .foregroundStyle(.secondary)
            Text(String(format: "%.1f", token.features.jawPeakRate))
                .frame(width: 38, alignment: .trailing)
                .foregroundStyle(peakRateColor(token.features.jawPeakRate))
            Text(String(format: "%.0f%%", token.features.faceVisibleRatio * 100))
                .frame(width: 36, alignment: .trailing)
                .foregroundStyle(token.features.faceVisibleRatio > 0.5 ? Color.secondary : Color.red)

            Text(token.isUser ? (token.filledBySentence ? "↑" : "✓") : "·")
                .frame(width: 40, alignment: .center)
                .foregroundStyle(token.isUser ? (token.filledBySentence ? .yellow : .green) : .gray)
                .bold()
            Spacer()
        }
        .font(.system(size: 11, design: .monospaced))
        .padding(.vertical, 1)
        .background(token.filledBySentence ? Color.yellow.opacity(0.08) : Color.clear)
    }

    private func confColor(_ v: Float) -> Color {
        if v >= 0.5 { return .green }
        if v >= 0.3 { return .yellow }
        if v >= 0.0 { return .orange }
        return .red
    }

    private func peakRateColor(_ v: Float) -> Color {
        if v >= 3 && v <= 8 { return .green }   // speech band
        if v > 8 { return .orange }              // too fast (laugh/chew)
        return Color.gray                        // too slow
    }
}
