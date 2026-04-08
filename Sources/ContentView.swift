import SwiftUI
import ARKit

struct ContentView: View {
    @State private var engine: HumanStateEngine
    
    init() {
        let faceManager = FaceTrackingManager()
        let audioManager = AudioDetectionManager()
        _engine = State(initialValue: HumanStateEngine(faceManager: faceManager, audioManager: audioManager))
    }
    
    var state: HumanState { engine.humanState }
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            VStack(spacing: 12) {
                // State card
                StateCard(state: state)
                    .padding(.horizontal)
                
                // Gaze point overlay area
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.white.opacity(0.05))
                        .frame(height: 120)
                    
                    if state.face.faceDetected {
                        GazeOverlay(
                            gazePoint: normalizedGazePoint(state.face.gazePoint),
                            isLooking: state.face.isLookingAtScreen
                        )
                        .frame(height: 120)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    } else {
                        Text("未检测到人脸")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal)
                
                // Data panels
                VStack(spacing: 10) {
                    HeadOrientationView(
                        yaw: state.face.headYaw,
                        pitch: state.face.headPitch,
                        roll: state.face.headRoll
                    )
                    
                    BlendShapePanel(face: state.face)
                    
                    AudioVisualizerView(audio: state.audio)
                }
                .padding()
                .background(Color.white.opacity(0.05))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)
                
                Spacer()
            }
            .padding(.top)
        }
        .preferredColorScheme(.dark)
        .onAppear { engine.start() }
        .onDisappear { engine.stop() }
    }
    
    // Map gaze point to the overlay area (120pt height)
    private func normalizedGazePoint(_ point: CGPoint) -> CGPoint {
        let screenSize = UIScreen.main.bounds.size
        let overlayHeight: CGFloat = 120
        let x = (point.x / screenSize.width) * (screenSize.width - 32)
        let y = (point.y / screenSize.height) * overlayHeight
        return CGPoint(x: x + 16, y: y)
    }
}
