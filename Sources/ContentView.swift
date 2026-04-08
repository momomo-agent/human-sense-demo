import SwiftUI
import ARKit

struct ContentView: View {
    @State private var faceManager = FaceTrackingManager()
    @State private var audioManager = AudioDetectionManager()
    @State private var handManager = HandGestureManager()
    @State private var sttManager = STTManager()
    @State private var deviceMotion = DeviceMotionManager()
    @State private var engine: HumanStateEngine?

    var state: HumanState { engine?.humanState ?? HumanState() }
    var history: [(date: Date, activity: HumanActivity)] { engine?.stateHistory ?? [] }

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                StateCard(state: state).padding(.horizontal)
                
                // Device motion state
                HStack(spacing: 12) {
                    Text("📱 \(deviceMotion.motionState.posture.rawValue)")
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.white.opacity(0.1))
                        .clipShape(Capsule())
                    
                    Text("🔄 \(deviceMotion.motionState.orientation.rawValue)")
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.white.opacity(0.1))
                        .clipShape(Capsule())
                    
                    Text(deviceMotion.motionState.isHolding ? "✋ 持握" : "📍 放置")
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(deviceMotion.motionState.isHolding ? Color.green.opacity(0.2) : Color.gray.opacity(0.2))
                        .clipShape(Capsule())
                    
                    if deviceMotion.motionState.isWalking {
                        Text("🚶 行走中")
                            .font(.caption)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color.green.opacity(0.2))
                            .clipShape(Capsule())
                    }
                }
                .padding(.horizontal)
                
                // Debug info
                VStack(alignment: .leading, spacing: 4) {
                    Text("DEBUG INFO").font(.caption2.bold()).foregroundStyle(.yellow)
                    Text("Head Roll: \(String(format: "%.2f", state.face.headRoll))").font(.caption2.monospacedDigit())
                    Text("Device Pitch: \(String(format: "%.2f", deviceMotion.debugPitch))").font(.caption2.monospacedDigit())
                    Text("Accel Variance: \(String(format: "%.6f", deviceMotion.debugVariance))").font(.caption2.monospacedDigit())
                    Text("Audio Volume: \(String(format: "%.4f", state.audio.volume))").font(.caption2.monospacedDigit())
                }
                .padding()
                .background(Color.yellow.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal)
                
                // Speech text display with segmented colors
                if !sttManager.segments.isEmpty {
                    ScrollViewReader { proxy in
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 0) {
                                ForEach(sttManager.segments) { segment in
                                    Text(segment.text)
                                        .foregroundStyle(segment.isToScreen ? .blue : .orange)
                                }
                                .id("end")
                            }
                            .font(.body)
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .background(Color.white.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .padding(.horizontal)
                        .onChange(of: sttManager.segments.count) { _ in
                            withAnimation {
                                proxy.scrollTo("end", anchor: .trailing)
                            }
                        }
                    }
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

                // Hand gesture (temporarily hidden - needs tuning)
                /*
                HandGestureView(hand: state.hand)
                    .padding()
                    .background(Color.white.opacity(0.05))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)
                */

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
            deviceMotion.start()
            
            // Temporarily disable hand gesture processing
            // handManager.start()
            
            // Sync isLookingAtScreen to STTManager
            Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
                sttManager.isLookingAtScreen = state.face.isLookingAtScreen
            }
        }
        .onDisappear { 
            engine?.stop()
            sttManager.stop()
            deviceMotion.stop()
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
