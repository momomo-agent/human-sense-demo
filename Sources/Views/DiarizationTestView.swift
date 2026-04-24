import SwiftUI
import AVFoundation
import FluidAudio

@MainActor
private final class DiarizationModel: ObservableObject {
    @Published var status = "Tap to download models"
    @Published var segments: [(speaker: String, start: Double, end: Double)] = []
    @Published var isRecording = false
    @Published var isEnrolling = false
    @Published var enrolledName: String? = nil

    var isReady: Bool { diarizer != nil }
    private var diarizer: DiarizerManager?
    private let audioEngine = AVAudioEngine()
    private var audioBuffer: [Float] = []
    private var actualSampleRate: Int = 44100

    func downloadModels() async {
        guard diarizer == nil else { return }
        status = "Downloading models..."
        do {
            let models = try await DiarizerModels.download { p in
                Task { @MainActor in self.status = "Downloading \(Int(p.fractionCompleted * 100))%" }
            }
            let d = DiarizerManager()
            d.initialize(models: models)
            diarizer = d
            status = "Ready — enroll your voice or start recording"
        } catch { status = "Download failed: \(error.localizedDescription)" }
    }

    func startEnroll() {
        audioBuffer = []
        isEnrolling = true
        status = "Recording your voice (5s)..."
        startCapture()
        Task {
            try? await Task.sleep(for: .seconds(5))
            if isEnrolling { await stopEnroll() }
        }
    }

    func stopEnroll() async {
        stopCapture()
        isEnrolling = false
        status = "Processing enrollment..."
        guard let d = diarizer, !audioBuffer.isEmpty else { return }
        let buf = audioBuffer
        do {
            let embedding = try d.extractSpeakerEmbedding(from: buf)
            let speaker = Speaker(id: "kenefe", name: "kenefe", currentEmbedding: embedding, isPermanent: true)
            await d.initializeKnownSpeakers([speaker])
            enrolledName = "kenefe"
            status = "Enrolled! Start recording to test diarization"
        } catch { status = "Enrollment failed: \(error.localizedDescription)" }
    }

    func startRecording() {
        audioBuffer = []
        isRecording = true
        status = "Recording..."
        startCapture()
    }

    func stopRecording() async {
        stopCapture()
        isRecording = false
        status = "Processing..."
        guard let d = diarizer, !audioBuffer.isEmpty else { return }
        let buf = audioBuffer
        let sr = actualSampleRate
        nonisolated(unsafe) let d2 = d
        do {
            let result = try await Task.detached(priority: .userInitiated) {
                try await d2.performCompleteDiarization(buf, sampleRate: sr)
            }.value
            segments = result.segments.map { ($0.speakerId, Double($0.startTimeSeconds), Double($0.endTimeSeconds)) }
            let n = Set(result.segments.map(\.speakerId)).count
            status = "Done — \(result.segments.count) segments, \(n) speakers"
        } catch { status = "Error: \(error.localizedDescription)" }
    }

    private func startCapture() {
        let input = audioEngine.inputNode
        let fmt = input.inputFormat(forBus: 0)
        actualSampleRate = Int(fmt.sampleRate)
        input.installTap(onBus: 0, bufferSize: 4096, format: fmt) { [weak self] buf, _ in
            guard let data = buf.floatChannelData?[0] else { return }
            let s = Array(UnsafeBufferPointer(start: data, count: Int(buf.frameLength)))
            DispatchQueue.main.async { self?.audioBuffer.append(contentsOf: s) }
        }
        try? audioEngine.start()
    }

    private func stopCapture() {
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
    }
}

@MainActor
struct DiarizationTestView: View {
    @StateObject private var model = DiarizationModel()

    var body: some View {
        VStack(spacing: 16) {
            Text("Speaker Diarization Test").font(.headline)
            Text(model.status).font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)

            if let name = model.enrolledName {
                Label("Enrolled: \(name)", systemImage: "person.fill.checkmark")
                    .font(.caption).foregroundStyle(.green)
            }

            HStack(spacing: 12) {
                if !model.isReady {
                    Button("Download Models") { Task { await model.downloadModels() } }
                        .buttonStyle(.borderedProminent)
                } else {
                    Button(model.isEnrolling ? "Stop Enroll" : "Enroll Voice") {
                        if model.isEnrolling { Task { await model.stopEnroll() } }
                        else { model.startEnroll() }
                    }
                    .buttonStyle(.bordered).tint(.orange)

                    Button(model.isRecording ? "Stop" : "Start") {
                        if model.isRecording { Task { await model.stopRecording() } }
                        else { model.startRecording() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.isEnrolling)
                }
            }

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(model.segments.indices, id: \.self) { i in
                        let seg = model.segments[i]
                        HStack {
                            Text(seg.speaker).font(.caption.bold())
                                .foregroundStyle(color(seg.speaker)).frame(width: 80)
                            Text(String(format: "%.1f–%.1fs", seg.start, seg.end))
                                .font(.caption2).foregroundStyle(.secondary)
                        }.padding(.horizontal)
                    }
                }
            }
        }.padding()
    }

    private func color(_ s: String) -> Color {
        if s == "kenefe" { return .blue }
        let c: [Color] = [.green, .orange, .purple, .red]
        return c[(s.unicodeScalars.reduce(0) { $0 + Int($1.value) }) % c.count]
    }
}
