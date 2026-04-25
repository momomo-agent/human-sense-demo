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
        }
    }
}
