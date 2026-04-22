import SwiftUI
import HumanSenseKit

/// Dedicated STT test view showing volatile vs final results clearly.
struct STTTestView: View {
    @ObservedObject var sttManager: STTManager
    var engine: HumanStateEngine

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            legend
            segmentsList
            if let error = sttManager.lastError {
                errorView(error)
            }
            controls
        }
        .padding()
        .background(Color.black)
    }

    private var header: some View {
        HStack {
            Text("STT Test")
                .font(.title2.bold())
            Spacer()
            Circle()
                .fill(sttManager.isListening ? Color.green : Color.red)
                .frame(width: 10, height: 10)
            Text(sttManager.isListening ? "Listening" : "Off")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var legend: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 16) {
                legendItem(color: .white, label: "Final")
                legendItem(color: .purple.opacity(0.6), label: "Volatile")
                legendItem(color: .gray, label: "Ambient")
            }
            .foregroundStyle(.secondary)

            // Real-time signals
            HStack(spacing: 12) {
                signalPill("👄 Mouth", engine.humanState.face.jawOpen > 0.2 || abs(engine.humanState.face.jawOpen - previousJaw) > 0.04)
                signalPill("🔗 Corr", engine.lipAudioCorrelator.isCorrelated,
                           detail: String(format: "%.2f", engine.lipAudioCorrelator.correlation))
                signalPill("👁 Gaze", engine.humanState.face.isLookingAtScreen)
                signalPill("🧭 Head", engine.humanState.face.headOrientation.isFacingForward)
                signalPill("🔊 Audio", engine.humanState.audio.isSpeaking)
            }

            HStack(spacing: 12) {
                Text("isSpeaking: \(sttManager.isSpeaking ? "✅" : "❌")")
                    .font(.caption2.monospaced())
                    .foregroundStyle(sttManager.isSpeaking ? .green : .red)
                Text("activity: \(engine.humanState.activity.rawValue)")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var previousJaw: Float { 0 } // approximation for display

    private func signalPill(_ label: String, _ on: Bool, detail: String? = nil) -> some View {
        HStack(spacing: 3) {
            Circle().fill(on ? Color.green : Color.red.opacity(0.5)).frame(width: 6, height: 6)
            Text(label).font(.system(size: 9))
            if let detail {
                Text(detail).font(.system(size: 9, design: .monospaced)).foregroundStyle(.secondary)
            }
        }
        .foregroundStyle(on ? .white : .gray)
    }

    private func legendItem(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label).font(.caption)
        }
    }

    private var segmentsList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(sttManager.segments) { segment in
                        SegmentRow(segment: segment)
                            .id(segment.id)
                    }
                }
                .padding(.horizontal, 4)
            }
            .onChange(of: sttManager.segments.count) {
                if let last = sttManager.segments.last {
                    withAnimation {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private func errorView(_ error: String) -> some View {
        Text("Error: \(error)")
            .font(.caption)
            .foregroundStyle(.red)
            .padding(8)
            .background(Color.red.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var controls: some View {
        HStack {
            Button("Clear") {
                sttManager.clearSegments()
            }
            .buttonStyle(.bordered)
        }
    }
}

private struct SegmentRow: View {
    let segment: SpeechSegment

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Text(statusEmoji)
                .font(.caption)

            VStack(alignment: .leading, spacing: 2) {
                Text(segment.text)
                    .font(segment.isFinal ? .body.bold() : .body.italic())
                    .foregroundStyle(textColor)

                tags
            }
        }
        .padding(8)
        .background(segment.isFinal ? Color.white.opacity(0.05) : Color.purple.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var tags: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                if segment.isFinal {
                    Text("FINAL")
                        .font(.caption2.bold())
                        .foregroundStyle(.green)
                } else {
                    Text("VOLATILE")
                        .font(.caption2)
                        .foregroundStyle(.purple)
                }
                if segment.isFromUser {
                    Text("user")
                        .font(.caption2)
                        .foregroundStyle(.blue)
                }
                if segment.sentenceStartedLookingAtScreen {
                    Text("👀 screen")
                        .font(.caption2)
                }
            }
            // Signal snapshot at time of this segment
            HStack(spacing: 6) {
                sigDot("👄", segment.signals.mouthMoving)
                sigDot("🔊", segment.signals.audioActive)
                sigDot("👁", segment.signals.gazeOnScreen)
                sigDot("🧭", segment.signals.headForward)
                sigDot("🔗", segment.signals.lipCorrelated)
                Text(String(format: "r=%.2f", segment.signals.lipCorrelation))
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.secondary)
                Text(segment.signals.activity)
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func sigDot(_ emoji: String, _ on: Bool) -> some View {
        HStack(spacing: 1) {
            Text(emoji).font(.system(size: 8))
            Circle().fill(on ? Color.green : Color.red.opacity(0.4)).frame(width: 5, height: 5)
        }
    }

    private var textColor: Color {
        if !segment.isFromUser { return .gray }
        if segment.isFinal { return .white }
        return .purple.opacity(0.8)
    }

    private var statusEmoji: String {
        if !segment.isFromUser { return "🔇" }
        if segment.isFinal { return "✅" }
        return "💬"
    }
}
