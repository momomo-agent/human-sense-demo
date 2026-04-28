import SwiftUI
import HumanSenseKit

@main
struct HumanSenseDemoApp: App {
    @State private var engine = HumanStateEngine()
    @State private var selectedTab = DemoTab.initialSelection

    var body: some Scene {
        WindowGroup {
            TabView(selection: $selectedTab) {
                ContentView(engine: engine)
                    .tabItem { Label("Sense", systemImage: "eye") }
                    .tag(DemoTab.sense)
                STTTestView(sttManager: engine.sttManager, engine: engine)
                    .tabItem { Label("STT Test", systemImage: "waveform") }
                    .tag(DemoTab.sttTest)
                TokenTableView(
                    recorder: engine.sttManager.userSentenceReconstructor,
                    engine: engine
                )
                .tabItem { Label("Tokens", systemImage: "tablecells") }
                .tag(DemoTab.tokens)
                DiarizationTestView(humanEngine: engine)
                    .tabItem { Label("Gaze+Speaker", systemImage: "person.2") }
                    .tag(DemoTab.diarization)
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

private enum DemoTab {
    case sense
    case sttTest
    case tokens
    case diarization

    static var initialSelection: DemoTab {
        #if DEBUG
        if ProcessInfo.processInfo.arguments.contains("--open-diarization-tab") {
            return .diarization
        }
        #endif

        return .sense
    }
}
