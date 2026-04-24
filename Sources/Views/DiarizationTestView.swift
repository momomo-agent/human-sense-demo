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
    @State private var isEnrolling = false
    @State private var diarizerBox: DiarizerBox?
    @State private var audioEngine = AVAudioEngine()
    @State private var audioBuffer: [Float] = []
    @State private var enrolledName: String? = nil

    var body: some View {
        VStack(spacing: 16) {
            Text("Speaker Diarization Test").font(.headline)
            Text(status).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)

            if let name = enrolledName {
                Label("Enrolled: \(name)", systemImage: "person.fill.checkmark")
                    .font(.caption).foregroundStyle(.green)
            }

            HStack(spacing: 12) {
                if diarizerBox == nil {
                    Button("Download Models") { Task { await downloadModels() } }
                        .buttonStyle(.borderedProminent)
                } else {
                    Button(isEnrolling ? "Stop Enroll" : "Enroll Voice") {
                        if isEnrolling { stopEnroll() } else { startEnroll() }
                    }
                    .buttonStyle(.bordered)
                    .tint(.orange)

                    Button(isRecording ? "Stop" : "Start") {
                        if isRecording { stopRecording() } else { startRecording() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isEnrolling)
                }
            }

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(segments.indices, id: \.self) { i in
                        let seg = segments[i]
                        HStack {
                            Text(seg.speaker).font(.caption.bold())
                                .foregroundStyle(speakerColor(seg.speaker)).frame(width: 80)
                            Text(String(format: "%.1f–%.1fs", seg.start, seg.end))
                                .font(.caption2).foregroundStyle(.secondary)
                        }.padding(.horizontal)
                    }
                }
            }
        }.padding()
    }

    // MARK: - Model Download

    private func downloadModels() async {
        status = "Downloading models..."
        do {
            let models = try await DiarizerModels.download { p in
                Task { @MainActor in status = "Downloading \(Int(p.fractionCompleted * 100))%" }
            }
            let d = DiarizerManager()
            d.initialize(models: models)
            diarizerBox = DiarizerBox(d)
            status = "Ready — enroll your voice or start recording"
        } catch { status = "Download failed: \(error.localizedDescription)" }
    }

    // MARK: - Voice Enrollment

    private func startEnroll() {
        audioBuffer = []
        isEnrolling = true
        status = "Recording your voice for enrollment (5s)..."
        startAudioCapture()
        // Auto-stop after 5 seconds
        Task {
            try? await Task.sleep(for: .seconds(5))
            if isEnrolling { stopEnroll() }
        }
    }

    private func stopEnroll() {
        stopAudioCapture()
        isEnrolling = false
        status = "Processing enrollment..."
        Task { await runEnrollment() }
    }

    private func runEnrollment() async {
        guard let box = diarizerBox, !audioBuffer.isEmpty else { return }
        let buf = audioBuffer
        do {
            let embedding = try box.value.extractSpeakerEmbedding(from: buf)
            let speaker = Speaker(id: "kenefe", name: "kenefe", currentEmbedding: embedding, isPermanent: true)
            await box.value.initializeKnownSpeakers([speaker])
            enrolledName = "kenefe"
            status = "Enrolled! Now start recording to test diarization"
        } catch { status = "Enrollment failed: \(error.localizedDescription)" }
    }

    // MARK: - Diarization Recording

    private func startRecording() {
        audioBuffer = []
        isRecording = true
        status = "Recording..."
        startAudioCapture()
    }

    private func stopRecording() {
        stopAudioCapture()
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
            let speakerCount = Set(result.segments.map(\.speakerId)).count
            status = "Done — \(result.segments.count) segments, \(speakerCount) speakers"
        } catch { status = "Error: \(error.localizedDescription)" }
    }

    // MARK: - Audio Capture

    private func startAudioCapture() {
        let input = audioEngine.inputNode
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        input.installTap(onBus: 0, bufferSize: 4096, format: fmt) { buf, _ in
            guard let data = buf.floatChannelData?[0] else { return }
            let s = Array(UnsafeBufferPointer(start: data, count: Int(buf.frameLength)))
            Task { @MainActor in self.audioBuffer.append(contentsOf: s) }
        }
        try? audioEngine.start()
    }

    private func stopAudioCapture() {
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
    }

    // MARK: - Helpers

    private func speakerColor(_ s: String) -> Color {
        if s == "kenefe" { return .blue }
        let colors: [Color] = [.green, .orange, .purple, .red]
        return colors[(s.unicodeScalars.reduce(0) { $0 + Int($1.value) }) % colors.count]
    }
}
