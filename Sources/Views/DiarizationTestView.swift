import SwiftUI
import AVFoundation
import FluidAudio

private final class DiarizerBox: @unchecked Sendable {
    let value: DiarizerManager
    init(_ value: DiarizerManager) { self.value = value }
}

@MainActor
struct DiarizationTestView: View {
    @State private var status = "Tap to download models"
    @State private var segments: [(speaker: String, start: Double, end: Double)] = []
    @State private var isRecording = false
    @State private var diarizerBox: DiarizerBox?
    @State private var audioEngine = AVAudioEngine()
    @State private var audioBuffer: [Float] = []

    var body: some View {
        VStack(spacing: 16) {
            Text("Speaker Diarization Test").font(.headline)
            Text(status).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
            Button(diarizerBox == nil ? "Download Models" : (isRecording ? "Stop" : "Start Recording")) {
                if diarizerBox == nil { Task { await downloadModels() } }
                else if isRecording { stopRecording() }
                else { startRecording() }
            }.buttonStyle(.borderedProminent)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(segments.indices, id: \.self) { i in
                        let seg = segments[i]
                        HStack {
                            Text(seg.speaker).font(.caption.bold())
                                .foregroundStyle(speakerColor(seg.speaker)).frame(width: 80)
                            Text(String(format: "%.1f–%.1fs", seg.start, seg.end)).font(.caption2).foregroundStyle(.secondary)
                        }.padding(.horizontal)
                    }
                }
            }
        }.padding()
    }

    private func downloadModels() async {
        status = "Downloading models..."
        do {
            let models = try await DiarizerModels.download { p in
                Task { @MainActor in status = "Downloading \(Int(p.fractionCompleted * 100))%" }
            }
            let d = DiarizerManager()
            d.initialize(models: models)
            diarizerBox = DiarizerBox(d)
            status = "Ready — tap to record"
        } catch { status = "Download failed: \(error.localizedDescription)" }
    }

    private func startRecording() {
        audioBuffer = []
        isRecording = true
        status = "Recording..."
        let input = audioEngine.inputNode
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        input.installTap(onBus: 0, bufferSize: 4096, format: fmt) { buf, _ in
            guard let data = buf.floatChannelData?[0] else { return }
            let s = Array(UnsafeBufferPointer(start: data, count: Int(buf.frameLength)))
            Task { @MainActor in self.audioBuffer.append(contentsOf: s) }
        }
        try? audioEngine.start()
    }

    private func stopRecording() {
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        isRecording = false
        status = "Processing..."
        Task { await runDiarization() }
    }

    private func runDiarization() async {
        guard let box = diarizerBox, !audioBuffer.isEmpty else { return }
        let buf = audioBuffer
        do {
            let result = try await box.value.performCompleteDiarization(buf, sampleRate: 16000)
            segments = result.segments.map { ($0.speakerId, Double($0.startTimeSeconds), Double($0.endTimeSeconds)) }
            status = "Done — \(result.segments.count) segments, \(Set(result.segments.map(\.speakerId)).count) speakers"
        } catch { status = "Error: \(error.localizedDescription)" }
    }

    private func speakerColor(_ s: String) -> Color {
        let colors: [Color] = [.blue, .green, .orange, .purple, .red]
        return colors[(s.unicodeScalars.reduce(0) { $0 + Int($1.value) }) % colors.count]
    }
}
