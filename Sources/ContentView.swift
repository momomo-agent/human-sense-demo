import SwiftUI
import ARKit

struct ContentView: View {
    @State private var faceManager = FaceTrackingManager()
    @State private var audioManager = AudioDetectionManager()
    @State private var handManager = HandGestureManager()
    @State private var sttManager = STTManager()
    @State private var engine: HumanStateEngine?

    var state: HumanState { engine?.humanState ?? HumanState() }
    var history: [(date: Date, activity: HumanActivity)] { engine?.stateHistory ?? [] }

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                StateCard(state: state).padding(.horizontal)
                
                // Speech text display
                if !sttManager.recognizedText.isEmpty {
                    Text(sttManager.recognizedText)
                        .font(.body)
                        .foregroundStyle(state.face.isLookingAtScreen ? .blue : .orange)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.white.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .padding(.horizontal)
                }

                // Face mesh + gaze overlay
                ZStack {
                    RoundedRectangle(cornerRadius: 12).fill(Color.white.opacity(0.05))
                    if state.face.faceDetected {
                        FaceMeshView(faceAnchor: faceManager.currentAnchor)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        GazeTrailView(trail: mappedTrail)
                        GazeOverlay(gazePoint: mappedGaze(state.face.gazePoint), isLooking: state.face.isLookingAtScreen)
                    } else {
                        Text("未检测到人脸").font(.caption).foregroundStyle(.secondary)
                    }
                }
                .frame(height: 200)
                .padding(.horizontal)

                // Eyes + mouth + distance
                HStack(spacing: 12) {
                    EyeVisualizerView(face: state.face)
                    MouthVisualizerView(jawOpen: state.face.jawOpen, mouthClose: state.face.mouthClose)
                    DistanceIndicatorView(distance: state.face.distanceFromCamera, label: state.face.distanceLabel)
                }
                .padding(.horizontal)

                // Hand gesture
                HandGestureView(hand: state.hand)
                    .padding()
                    .background(Color.white.opacity(0.05))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)

                // Data panels
                VStack(spacing: 10) {
                    StateTimelineView(history: history)
                    HeadOrientationView(yaw: state.face.headYaw, pitch: state.face.headPitch, roll: state.face.headRoll)
                    BlendShapePanel(face: state.face)
                    AudioVisualizerView(audio: state.audio)
                }
                .padding()
                .background(Color.white.opacity(0.05))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)
            }
            .padding(.top)
        }
        .background(Color.black.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .onAppear {
            faceManager.handManager = handManager
            let e = HumanStateEngine(faceManager: faceManager, audioManager: audioManager, handManager: handManager)
            engine = e
            e.start()
            sttManager.start()
        }
        .onDisappear { 
            engine?.stop()
            sttManager.stop()
        }
    }

    private var mappedTrail: [CGPoint] { faceManager.gazeTrail.map { mappedGaze($0) } }

    private func mappedGaze(_ point: CGPoint) -> CGPoint {
        let screen = UIScreen.main.bounds.size
        let x = (point.x / screen.width) * (screen.width - 32)
        let y = (point.y / screen.height) * 200
        return CGPoint(x: x + 16, y: y)
    }
}
