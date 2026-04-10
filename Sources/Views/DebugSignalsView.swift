import SwiftUI

/// Shows all raw signals that feed into activity/STT decisions.
struct DebugSignalsView: View {
    let state: HumanState
    @ObservedObject var sttManager: STTManager
    
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
            }
            
            HStack(spacing: 12) {
                signal("🗣 isSpeaking", sttManager.isSpeaking)
                signal("📝 STT Active", sttManager.isListening)
            }
            
            // Raw values
            HStack(spacing: 8) {
                rawValue("jaw", String(format: "%.3f", face.jawOpen))
                rawValue("vol", String(format: "%.3f", audio.volume))
                rawValue("yaw", String(format: "%.2f", face.headYaw))
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
        face.jawOpen > 0.2 // simplified — actual uses jawDelta too
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
