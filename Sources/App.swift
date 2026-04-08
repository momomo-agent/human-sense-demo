import SwiftUI

@main
struct HumanSenseDemoApp: App {
    @StateObject private var sttManager = STTManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(sttManager)
                .onAppear {
                    print("App: onAppear called, starting STT...")
                    sttManager.start()
                }
        }
    }
}
