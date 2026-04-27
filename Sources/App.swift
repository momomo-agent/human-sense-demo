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
                TokenTableView(
                    recorder: engine.sttManager.userSentenceReconstructor,
                    engine: engine
                )
                .tabItem { Label("Tokens", systemImage: "tablecells") }
                DiarizationTestView(humanEngine: engine)
                    .tabItem { Label("Gaze+Speaker", systemImage: "person.2") }
            }
            .onAppear {
                engine.start()
                // STT token stream and per-frame sensor samples are now
                // plumbed through HumanStateEngine → STTManager →
                // UserSentenceReconstructor. No manual wiring needed.
            }
        }
    }
}
