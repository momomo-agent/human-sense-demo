import SwiftUI
import HumanSenseKit

@main
struct HumanSenseDemoApp: App {
    @State private var engine = HumanStateEngine()
    
    var body: some Scene {
        WindowGroup {
            TabView {
                ContentView(engine: engine)
                    .tabItem { Label("Sense", systemImage: "eye") }
                STTTestView(sttManager: engine.sttManager, engine: engine)
                    .tabItem { Label("STT Test", systemImage: "waveform") }
                DiarizationTestView()
                    .tabItem { Label("Diarization", systemImage: "person.2") }
            }
            .onAppear {
                engine.start()
            }
            .onReceive(Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()) { _ in
                let ts = Date().timeIntervalSince1970
                let jaw = engine.humanState.face.jawOpen
                let vol = engine.humanState.audio.volume
                let speaking = engine.sttManager.isSpeaking
                if let seg = engine.sttManager.segments.last(where: { !$0.isFinal }) {
                    let base = engine.sttManager.audioStreamStartTime?.timeIntervalSince1970 ?? ts
                    let start = seg.audioStartTime.map { String(format: "%.3f", base + $0) } ?? "?"
                    let end = seg.audioEndTime.map { String(format: "%.3f", base + $0) } ?? "?"
                    print(String(format: "[HSK] %.3f jaw:%.3f vol:%.3f speaking:\(speaking ? 1 : 0) audio:[\(start)-\(end)] text:\"\(seg.text)\"", ts, jaw, vol))
                } else {
                    print(String(format: "[HSK] %.3f jaw:%.3f vol:%.3f speaking:\(speaking ? 1 : 0)", ts, jaw, vol))
                }
            }
        }
    }
}
