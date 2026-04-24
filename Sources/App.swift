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
            .onChange(of: engine.humanState.face.headYaw) { _, yaw in
                // Mute STT when user is not facing the screen (yaw > 45°)
                engine.sttManager.isMuted = abs(yaw) > 0.785
            }
        }
    }
}
