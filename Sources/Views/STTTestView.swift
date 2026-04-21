import SwiftUI
import HumanSenseKit

/// Dedicated STT test view showing volatile vs final results clearly.
struct STTTestView: View {
    @ObservedObject var sttManager: STTManager

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
        HStack(spacing: 16) {
            legendItem(color: .white, label: "Final")
            legendItem(color: .purple.opacity(0.6), label: "Volatile")
            legendItem(color: .gray, label: "Ambient")
        }
        .foregroundStyle(.secondary)
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
