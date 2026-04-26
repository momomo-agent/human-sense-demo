import SwiftUI
import HumanSenseKit

struct TokenTableView: View {
    @ObservedObject var recorder: TokenSampleRecorder
    var engine: HumanStateEngine?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Token × Signals").font(.title3.bold())
                Spacer()
                Button("Clear") { recorder.clear() }
                    .buttonStyle(.bordered)
            }
            .padding(.horizontal)

            // Real-time signal indicators
            if let engine {
                HStack(spacing: 16) {
                    liveSignal("👄", engine.humanState.face.jawOpen > 0.15,
                               detail: String(format: "%.2f", engine.humanState.face.jawOpen))
                    liveSignal("👁", engine.humanState.face.isLookingAtScreen)
                    liveSignal("🧭", engine.humanState.face.headOrientation.isFacingForward)
                }
                .padding(.horizontal)
            }

            header.padding(.horizontal)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(recorder.tokens) { row in
                            TokenRowView(row: row).id(row.id)
                            Divider().background(Color.white.opacity(0.08))
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
        .padding(.vertical)
        .background(Color.black)
    }

    private var header: some View {
        HStack(spacing: 6) {
            Text("时间").frame(width: 55, alignment: .leading)
            Text("字").frame(width: 24, alignment: .leading)
            Text("avgJ").frame(width: 36, alignment: .trailing)
            Text("maxJ").frame(width: 36, alignment: .trailing)
            Text("vol").frame(width: 44, alignment: .trailing)
            Text("👄").frame(width: 24, alignment: .center)
            Text("👁").frame(width: 24, alignment: .center)
            Text("🧭").frame(width: 24, alignment: .center)
            Text("n").frame(width: 20, alignment: .trailing)
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
        HStack(spacing: 6) {
            Text(timeString)
                .font(.system(size: 9, design: .monospaced))
                .frame(width: 55, alignment: .leading)
                .foregroundStyle(.secondary)
            Text(row.text)
                .font(.body.monospaced())
                .frame(width: 24, alignment: .leading)
                .foregroundStyle(textColor)
            Text(String(format: "%.2f", row.avgJaw))
                .font(.caption.monospaced())
                .frame(width: 36, alignment: .trailing)
                .foregroundStyle(jawColor)
            Text(String(format: "%.2f", row.maxJaw))
                .font(.caption.monospaced())
                .frame(width: 36, alignment: .trailing)
                .foregroundStyle(jawColor)
            Text(String(format: "%.3f", row.avgVol))
                .font(.caption.monospaced())
                .frame(width: 44, alignment: .trailing)
                .foregroundStyle(volColor)
            Text(row.maxJaw > 0.15 ? "✅" : "❌")
                .frame(width: 24, alignment: .center)
            Text(row.gazeRatio > 0.5 ? "✅" : "❌")
                .frame(width: 24, alignment: .center)
            Text(row.headFwdRatio > 0.5 ? "✅" : "❌")
                .frame(width: 24, alignment: .center)
            Text("\(row.sampleCount)")
                .font(.system(size: 9, design: .monospaced))
                .frame(width: 20, alignment: .trailing)
                .foregroundStyle(.secondary)
            Spacer()
            Text(row.isFinal ? "✅" : "…")
                .font(.caption2)
        }
        .padding(.horizontal)
        .padding(.vertical, 3)
    }

    private var timeString: String {
        let date = Date(timeIntervalSince1970: row.startTime)
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f.string(from: date)
    }

    private var textColor: Color {
        let mouthMoving = row.maxJaw > 0.15
        let gazeOn = row.gazeRatio > 0.5
        let headFwd = row.headFwdRatio > 0.5
        if mouthMoving && gazeOn && headFwd { return .yellow }   // 对着屏幕说
        if mouthMoving { return .cyan }                           // 我说的但没对着屏幕
        return .gray                                              // 不是我说的
    }

    private var jawColor: Color {
        row.avgJaw > 0.15 ? .green : .secondary
    }

    private var volColor: Color {
        row.maxVol > 0.01 ? .green : .secondary
    }
}
