import SwiftUI

struct StateCard: View {
    let state: HumanState

    var body: some View {
        VStack(spacing: 8) {
            if state.face.faceDetected {
                // Multiple simultaneous indicators
                HStack(spacing: 16) {
                    // Gaze indicator
                    StatusPill(
                        emoji: state.face.isLookingAtScreen ? "👁" : "👀",
                        label: state.face.isLookingAtScreen ? "看屏幕" : "看别处",
                        color: state.face.isLookingAtScreen ? .green : .orange,
                        active: true
                    )

                    // Speech indicator with color distinction
                    StatusPill(
                        emoji: state.activity.isSpeaking ? "🗣" : "🤫",
                        label: state.activity.isSpeaking ? "在说话" : "安静",
                        color: speechColor,
                        active: true
                    )

                    // Face orientation indicator
                    StatusPill(
                        emoji: faceOrientationEmoji,
                        label: faceOrientationLabel,
                        color: .blue,
                        active: true
                    )
                    
                    // Eye state indicator
                    StatusPill(
                        emoji: state.face.eyesClosed ? "😴" : "👀",
                        label: state.face.eyesClosed ? "闭眼" : "睁眼",
                        color: state.face.eyesClosed ? .purple : .green,
                        active: true
                    )
                }
            } else {
                // No face detected
                Text("未检测到人脸").font(.caption).foregroundStyle(.secondary)
            }
            
            // Emotion indicator
            if state.face.faceDetected {
                Text(state.face.emotion.rawValue)
                    .font(.title2)
                    .padding(.vertical, 4)
            }

            // Head gesture indicator (fixed height to prevent layout jumping)
            ZStack {
                if state.face.headGesture != .none {
                    Text(state.face.headGesture.rawValue)
                        .font(.caption.bold())
                        .foregroundStyle(.cyan)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 4)
                        .background(Color.cyan.opacity(0.2))
                        .clipShape(Capsule())
                }
            }
            .frame(height: 24)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(Color.white.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    var speechColor: Color {
        if !state.activity.isSpeaking { return .gray }
        return state.face.isLookingAtScreen ? .yellow : .orange
    }
    
    var faceOrientationEmoji: String {
        if !state.face.faceDetected { return "👻" }
        let yaw = state.face.headYaw
        if yaw > 0.3 { return "👈" }  // Looking left
        if yaw < -0.3 { return "👉" }  // Looking right
        return "😊"  // Facing forward
    }
    
    var faceOrientationLabel: String {
        if !state.face.faceDetected { return "不在" }
        let yaw = state.face.headYaw
        if yaw > 0.3 { return "朝左" }
        if yaw < -0.3 { return "朝右" }
        return "朝前"
    }
}

struct StatusPill: View {
    let emoji: String
    let label: String
    let color: Color
    let active: Bool

    var body: some View {
        VStack(spacing: 4) {
            Text(emoji).font(.title2)
            Text(label)
                .font(.caption2.bold())
                .foregroundStyle(active ? color : .secondary)
        }
        .frame(maxWidth: .infinity)
    }
}
