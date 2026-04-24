import SwiftUI
import AVFoundation
import FluidAudio

@MainActor
private final class DiarizationModel: ObservableObject {
    @Published var status = "Tap to download models"
    @Published var segments: [(speaker: String, start: Double, end: Double)] = []
    @Published var isRecording = false
    @Published var enrolledName: String? = nil

    var isReady: Bool { diarizer != nil }
    private var diarizer: DiarizerManager?
    private let audioEngine = AVAudioEngine()
    private var audioStream: AudioStream?

    func downloadModels() async {
        guard diarizer == nil else { return }
        status = "Downloading models..."
        do {
            let models = try await DiarizerModels.downloadIfNeeded()
            let d = DiarizerManager()
            d.initialize(models: models)
            diarizer = d
            status = "Ready"
        } catch { status = "Download failed: \(error.localizedDescription)" }
    }

    func startRecording() {
        guard let d = diarizer else { return }
        segments = []
        isRecording = true
        status = "Recording..."
        do {
        let hwSampleRate = audioEngine.inputNode.inputFormat(forBus: 0).sampleRate
        var stream = try AudioStream(chunkDuration: 5.0, chunkSkip: 2.0, sampleRate: hwSampleRate)
        nonisolated(unsafe) let dRef = d
        stream.bind { (chunk: [Float], time: TimeInterval) async in
            do {
                let result = try dRef.performCompleteDiarization(chunk, sampleRate: Int(hwSampleRate), atTime: time)
                DispatchQueue.main.async {
                    for seg in result.segments {
                        self.segments.append((seg.speakerId, Double(seg.startTimeSeconds), Double(seg.endTimeSeconds)))
                    }
                }
            } catch {
                DispatchQueue.main.async { self.status = "Error: \(error.localizedDescription)" }
            }
        }
        audioStream = stream

        let input = audioEngine.inputNode
        let fmt = input.inputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 4096, format: fmt) { [weak self] buf, _ in
            try? self?.audioStream?.write(from: buf)
        }
        try? audioEngine.start()
        } catch {
            status = "Failed to start: \(error.localizedDescription)"
            isRecording = false
        }
    }

    func stopRecording() {
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        audioStream = nil
        isRecording = false
        status = segments.isEmpty ? "No segments detected" : "Done — \(Set(segments.map(\.speaker)).count) speakers"
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

            if !model.isReady {
                Button("Download Models") { Task { await model.downloadModels() } }
                    .buttonStyle(.borderedProminent)
            } else {
                Button(model.isRecording ? "Stop" : "Start Recording") {
                    if model.isRecording { model.stopRecording() } else { model.startRecording() }
                }.buttonStyle(.borderedProminent)
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
        let c: [Color] = [.blue, .green, .orange, .purple, .red]
        return c[(s.unicodeScalars.reduce(0) { $0 + Int($1.value) }) % c.count]
    }
}
