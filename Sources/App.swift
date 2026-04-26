import SwiftUI
import HumanSenseKit

@main
struct HumanSenseDemoApp: App {
    @State private var engine = HumanStateEngine()
    @StateObject private var tokenRecorder = TokenSampleRecorder()

    var body: some Scene {
        WindowGroup {
            TabView {
                ContentView(engine: engine)
                    .tabItem { Label("Sense", systemImage: "eye") }
                STTTestView(sttManager: engine.sttManager, engine: engine)
                    .tabItem { Label("STT Test", systemImage: "waveform") }
                TokenTableView(recorder: tokenRecorder, engine: engine)
                    .tabItem { Label("Tokens", systemImage: "tablecells") }
                DiarizationTestView()
                    .tabItem { Label("Diarization", systemImage: "person.2") }
            }
            .onAppear {
                engine.start()
                engine.sttManager.onTokens = { tokens, isFinal in
                    tokenRecorder.audioStreamStartTime = engine.sttManager.audioStreamStartTime
                    tokenRecorder.recordTokens(tokens, isFinal: isFinal)
                }
            }
            .onReceive(Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()) { _ in
                let ts = Date().timeIntervalSince1970
                let jaw = engine.humanState.face.jawOpen
                let vol = engine.humanState.audio.volume
                tokenRecorder.recordSample(
                    ts: ts, jaw: jaw, vol: vol,
                    gaze: engine.humanState.face.isLookingAtScreen,
                    headFwd: engine.humanState.face.headOrientation.isFacingForward
                )
            }
        }
    }
}
