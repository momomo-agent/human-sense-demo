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
            correlationWaveform
            segmentsList
            if let error = sttManager.lastError {
                errorView(error)
            }
            controls
        }
        .padding()
        .background(Color.black)
    }

    // MARK: - Correlation Waveform

    private var correlationWaveform: some View {
        let points = engine.lipAudioCorrelator.samplePoints
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Lip vs Audio")
                    .font(.caption2.bold())
                Spacer()
                Text(String(format: "%.0f%%", engine.lipAudioCorrelator.correlation * 100))
                    .font(.caption2.monospaced())
                    .foregroundStyle(engine.lipAudioCorrelator.isCorrelated ? .green : .red)
            }
            .foregroundStyle(.secondary)

            // Waveform chart
            Canvas { context, size in
                guard points.count > 1 else { return }
                let w = size.width
                let h = size.height

                // Draw lip activity (orange)
                var lipPath = Path()
                for (i, p) in points.enumerated() {
                    let x = w * CGFloat(p.timeOffset) / 1.0  // 1s window
                    let y = h * (1 - CGFloat(p.lipActivity))
                    if i == 0 { lipPath.move(to: CGPoint(x: x, y: y)) }
                    else { lipPath.addLine(to: CGPoint(x: x, y: y)) }
                }
                context.stroke(lipPath, with: .color(.orange), lineWidth: 1.5)

                // Draw audio RMS (cyan)
                var audioPath = Path()
                for (i, p) in points.enumerated() {
                    let x = w * CGFloat(p.timeOffset) / 1.0
                    let y = h * (1 - CGFloat(p.audioRMS))
                    if i == 0 { audioPath.move(to: CGPoint(x: x, y: y)) }
                    else { audioPath.addLine(to: CGPoint(x: x, y: y)) }
                }
                context.stroke(audioPath, with: .color(.cyan), lineWidth: 1.5)
            }
            .frame(height: 60)
            .background(Color.white.opacity(0.05))
            .clipShape(RoundedRectangle(cornerRadius: 6))

            // Legend
            HStack(spacing: 12) {
                HStack(spacing: 4) {
                    Circle().fill(Color.orange).frame(width: 6, height: 6)
                    Text("Lip").font(.system(size: 9))
                }
                HStack(spacing: 4) {
                    Circle().fill(Color.cyan).frame(width: 6, height: 6)
                    Text("Audio").font(.system(size: 9))
                }
            }
            .foregroundStyle(.secondary)
        }
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
                signalPill("👄 Mouth", engine.humanState.face.jawOpen > 0.15 || engine.lipAudioCorrelator.lipActivity > 0.5,
                           detail: String(format: "lip:%.1f", engine.lipAudioCorrelator.lipActivity))
                signalPill("🔗 CoOcc", engine.lipAudioCorrelator.isCorrelated,
                           detail: String(format: "%.0f%%", engine.lipAudioCorrelator.correlation * 100))
                signalPill("👁 Gaze", engine.humanState.face.isLookingAtScreen)
                signalPill("🧭 Head", engine.humanState.face.headOrientation.isFacingForward)
                signalPill("🔊 Audio", engine.humanState.audio.isSpeaking)
            }

            // Co-occurrence frame counts
            HStack(spacing: 8) {
                Text("✅both:\(engine.lipAudioCorrelator.bothCount)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.green)
                Text("👄lip:\(engine.lipAudioCorrelator.lipOnlyCount)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.orange)
                Text("🔊audio:\(engine.lipAudioCorrelator.audioOnlyCount)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.cyan)
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
                        if segment.text.trimmingCharacters(in: .whitespaces).isEmpty {
                            // Sentence separator
                            Divider()
                                .padding(.vertical, 4)
                        } else {
                            SegmentRow(segment: segment)
                                .id(segment.id)
                        }
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
                Text(String(format: "%.0f%%", segment.signals.lipCorrelation * 100))
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.secondary)
                Text(segment.signals.activity)
                    .font(.system(size: 8, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            // Mini waveform snapshot
            if !segment.signals.waveform.isEmpty {
                miniWaveform
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

    private var miniWaveform: some View {
        Canvas { context, size in
            let points = segment.signals.waveform
            guard points.count > 1 else { return }
            let w = size.width
            let h = size.height

            var lipPath = Path()
            var audioPath = Path()
            for (i, p) in points.enumerated() {
                let x = w * CGFloat(p.timeOffset) / 1.0
                let lipY = h * (1 - CGFloat(p.lipActivity))
                let audioY = h * (1 - CGFloat(p.audioRMS))
                if i == 0 {
                    lipPath.move(to: CGPoint(x: x, y: lipY))
                    audioPath.move(to: CGPoint(x: x, y: audioY))
                } else {
                    lipPath.addLine(to: CGPoint(x: x, y: lipY))
                    audioPath.addLine(to: CGPoint(x: x, y: audioY))
                }
            }
            context.stroke(lipPath, with: .color(.orange), lineWidth: 1)
            context.stroke(audioPath, with: .color(.cyan), lineWidth: 1)
        }
        .frame(height: 30)
        .background(Color.white.opacity(0.03))
        .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    private var statusEmoji: String {
        if !segment.isFromUser { return "🔇" }
        if segment.isFinal { return "✅" }
        return "💬"
    }
}
