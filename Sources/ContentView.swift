import SwiftUI
import ARKit
import HumanSenseKit

struct ContentView: View {
    var engine: HumanStateEngine
    @ObservedObject private var sttManager: STTManager

    init(engine: HumanStateEngine) {
        self.engine = engine
        self._sttManager = ObservedObject(wrappedValue: engine.sttManager)
    }

    private var state: HumanState { engine.humanState }
    private var history: [(date: Date, activity: HumanActivity)] { engine.stateHistory }

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                StateCard(state: state).padding(.horizontal)
                
                // Debug: all raw signals
                if state.face.faceDetected {
                    DebugSignalsView(
                        state: state,
                        sttIsSpeaking: sttManager.isSpeaking,
                        sttIsListening: sttManager.isListening,
                        lipAudioCorrelation: engine.lipAudioCorrelator.correlation,
                        lipAudioCorrelated: engine.lipAudioCorrelator.isCorrelated,
                        faceAnchor: engine.currentFaceAnchor
                    )
                    .padding(.horizontal)
                }
                
                // Device state
                HStack(spacing: 12) {
                    Text("📱 \(state.device.posture.rawValue)")
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.white.opacity(0.1))
                        .clipShape(Capsule())
                    
                    Text(state.device.isHolding ? "✋ 持握" : "📍 放置")
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(state.device.isHolding ? Color.green.opacity(0.2) : Color.gray.opacity(0.2))
                        .clipShape(Capsule())
                    
                    if state.device.isWalking {
                        Text("🚶 行走中")
                            .font(.caption)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color.green.opacity(0.2))
                            .clipShape(Capsule())
                    }
                }
                .padding(.horizontal)
                
                // Speech text display — right-aligned, latest text anchored to trailing edge
                if !sttManager.segments.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 0) {
                            ForEach(sttManager.segments) { segment in
                                Text(segment.text)
                                    .foregroundStyle(segmentColor(segment))
                            }
                        }
                        .font(sttFontSize)
                        .padding()
                        .frame(minWidth: UIScreen.main.bounds.width - 32, alignment: .trailing)
                    }
                    .defaultScrollAnchor(.trailing)
                    .background(Color.white.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)
                }

                // Face mesh + gaze overlay
                ZStack {
                    RoundedRectangle(cornerRadius: 12).fill(Color.white.opacity(0.05))
                    if state.face.faceDetected {
                        FaceMeshView(faceAnchor: engine.currentFaceAnchor)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        GazeTrailView(trail: mappedTrail)
                    } else {
                        Text("未检测到人脸").foregroundStyle(.secondary)
                    }
                }
                .frame(height: 300)
                .padding(.horizontal)

                // Panels
                HStack(spacing: 12) {
                    EyeVisualizerView(face: state.face)
                    MouthVisualizerView(jawOpen: state.face.jawOpen, mouthClose: state.face.mouthClose)
                }
                .padding(.horizontal)

                HStack(spacing: 12) {
                    HeadOrientationView(yaw: state.face.headYaw, pitch: state.face.headPitch, roll: state.face.headRoll)
                    DistanceIndicatorView(distance: state.face.distanceFromCamera, label: state.face.distanceLabel)
                }
                .padding(.horizontal)

                AudioVisualizerView(audio: state.audio)
                    .padding(.horizontal)

                // Hand gesture (disabled: needs accuracy/perf work)
                // HandGestureView(hand: state.hand)
                //     .padding(.horizontal)

                StateTimelineView(history: history)
                    .padding(.horizontal)
            }
            .padding(.vertical)
        }
        .overlay {
            if state.face.faceDetected {
                GazeOverlay(gazePoint: state.face.gazePoint, gazePointEye: state.face.gazePointEye, isLooking: state.face.isLookingAtScreen, distance: state.face.distanceFromCamera, pitch: state.face.headPitch, yaw: state.face.headYaw)
                    .allowsHitTesting(false)
            }
        }
        .background(Color.black)
        .preferredColorScheme(.dark)
    }

    // MARK: - Helpers

    private func segmentColor(_ segment: SpeechSegment) -> Color {
        if !segment.isFromUser { return .blue }
        let gazeOn = segment.isFinal ? (segment.speakingToAIScore >= 0.5) : engine.humanState.face.isLookingAtScreen
        return gazeOn ? .yellow : .orange
    }
    
    private func mappedGaze(_ point: CGPoint) -> CGPoint {
        let screenSize = UIScreen.main.bounds.size
        return CGPoint(
            x: point.x / screenSize.width * 300,
            y: point.y / screenSize.height * 300
        )
    }
    
    private var mappedTrail: [CGPoint] { engine.gazeTrail.map { mappedGaze($0) } }
    
    private var sttFontSize: Font {
        let distance = state.face.distanceFromCamera
        let baseFontSize: CGFloat = 17
        let dist = CGFloat(distance)
        let scaleFactor: CGFloat
        
        switch dist {
        case 0..<0.3:
            scaleFactor = 1.0
        case 0.3..<0.8:
            let ratio = (dist - 0.3) / 0.5
            scaleFactor = 1.0 + ratio * 0.3
        case 0.8..<1.2:
            let ratio = (dist - 0.8) / 0.4
            scaleFactor = 1.3 + ratio * 0.35
        case 1.2..<1.5:
            let ratio = (dist - 1.2) / 0.3
            scaleFactor = 1.65 + ratio * 0.35
        default:
            scaleFactor = 2.0
        }
        
        return .system(size: baseFontSize * scaleFactor)
    }
}
