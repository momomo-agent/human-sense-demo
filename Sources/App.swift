import SwiftUI

@main
struct HumanSenseDemoApp: App {
    @State private var engine = HumanStateEngine()
    
    var body: some Scene {
        WindowGroup {
            ContentView(engine: engine)
                .onAppear {
                    engine.start()
                }
        }
    }
}
