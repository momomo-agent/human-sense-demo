import SwiftUI
import ARKit
import HumanSenseKit

/// Shows all raw signals that feed into activity/STT decisions.
struct DebugSignalsView: View {
    let state: HumanState
    let sttIsSpeaking: Bool
    let sttIsListening: Bool
    let lipAudioCorrelation: Float
    let lipAudioCorrelated: Bool
    let faceAnchor: ARFaceAnchor?
    
    private var face: FaceState { state.face }
    private var audio: AudioState { state.audio }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("DEBUG SIGNALS")
                .font(.caption2.bold())
                .foregroundStyle(.gray)
            
            HStack(spacing: 12) {
                signal("👁 Gaze", face.isLookingAtScreen)
                signal("🧭 Head", face.headOrientation.isFacingForward)
                signal("👄 Mouth", mouthMoving)
                signal("🔊 Audio", audio.isSpeaking)
                signal("🔗 Corr", lipAudioCorrelated)
            }
            
            HStack(spacing: 12) {
                signal("🗣 isSpeaking", sttIsSpeaking)
                signal("📝 STT Active", sttIsListening)
            }
            
            // Raw values
            HStack(spacing: 8) {
                rawValue("jaw", String(format: "%.3f", face.jawOpen))
                rawValue("vol", String(format: "%.3f", audio.volume))
                rawValue("yaw", String(format: "%.0f°", face.headYaw * 180 / .pi))
                rawValue("pitch", String(format: "%.0f°", face.headPitch * 180 / .pi))
                rawValue("corr", String(format: "%.2f", lipAudioCorrelation))
            }

            // Gaze compensation
            HStack(spacing: 8) {
                let pitch = face.headPitch
                let tiltThreshold: Float = 0.35
                let maxPitch: Float = 1.05
                let tiltRatio = abs(pitch) > tiltThreshold
                    ? min((abs(pitch) - tiltThreshold) / (maxPitch - tiltThreshold), 1.0)
                    : 0
                let compensation = tiltRatio * 0.35 * 100
                rawValue("gaze-comp", String(format: "%.0f%%", compensation))
                rawValue("pitch-raw", String(format: "%.3f", pitch))
            }

            // Eye openness
            HStack(spacing: 8) {
                rawValue("eyeL", String(format: "%.2f", 1 - face.eyeBlinkLeft))
                rawValue("eyeR", String(format: "%.2f", 1 - face.eyeBlinkRight))
                rawValue("gazeV", String(format: "%.2f", face.gazeV))
                rawValue("gazeH", String(format: "%.2f", face.gazeH))
            }

            // Eye geometry (from face mesh vertices)
            if let anchor = faceAnchor {
                let verts = anchor.geometry.vertices
                if verts.count > 468 {
                    // Left eye: top=159, bottom=145, left=33, right=133
                    let lTop = verts[159]; let lBot = verts[145]
                    let lLeft = verts[33]; let lRight = verts[133]
                    let lH = abs(lTop.y - lBot.y)
                    let lW = abs(lRight.x - lLeft.x)
                    // Right eye: top=386, bottom=374, left=362, right=263
                    let rTop = verts[386]; let rBot = verts[374]
                    let rLeft = verts[362]; let rRight = verts[263]
                    let rH = abs(rTop.y - rBot.y)
                    let rW = abs(rRight.x - rLeft.x)
                    HStack(spacing: 8) {
                        rawValue("lEye h/w", String(format: "%.3f/%.3f=%.2f", lH, lW, lW > 0 ? lH/lW : 0))
                        rawValue("rEye h/w", String(format: "%.3f/%.3f=%.2f", rH, rW, rW > 0 ? rH/rW : 0))
                    }
                }
            }

            // Gaze point coordinates
            HStack(spacing: 8) {
                let scaleX = face.distanceFromCamera > 0 ? 0.5 / face.distanceFromCamera : 1.0
                let scaleY = scaleX * (1.0 + abs(face.headPitch) * 0.5)
                rawValue("dist", String(format: "%.2fm", face.distanceFromCamera))
                rawValue("scX", String(format: "%.2f", scaleX))
                rawValue("scY", String(format: "%.2f", scaleY))
                rawValue("face-xy", String(format: "%.0f,%.0f", face.gazePoint.x, face.gazePoint.y))
                rawValue("eye-xy", String(format: "%.0f,%.0f", face.gazePointEye.x, face.gazePointEye.y))
            }
            
            // Activity result
            Text("→ \(state.activity.rawValue)")
                .font(.caption.bold())
                .foregroundStyle(activityColor)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
    
    private var mouthMoving: Bool {
        face.jawOpen > 0.2
    }
    
    private var activityColor: Color {
        switch state.activity {
        case .speakingToScreen: return .yellow
        case .speakingToOther: return .blue
        case .listening: return .green
        case .distracted: return .orange
        case .eyesClosed: return .purple
        case .absent: return .gray
        }
    }
    
    private func signal(_ label: String, _ on: Bool) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(on ? Color.green : Color.red.opacity(0.5))
                .frame(width: 8, height: 8)
            Text(label)
                .font(.caption2)
                .foregroundStyle(on ? .white : .gray)
        }
    }
    
    private func rawValue(_ label: String, _ value: String) -> some View {
        Text("\(label):\(value)")
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(.gray)
    }
}
