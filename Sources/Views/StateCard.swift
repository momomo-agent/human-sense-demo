import SwiftUI

struct StateCard: View {
    let state: HumanState

    var body: some View {
        VStack(spacing: 8) {
            // Status indicators (fixed height to prevent jumping)
            ZStack {
                if state.face.faceDetected {
                    HStack(spacing: 16) {
                        // Combined presence/gaze indicator
                        StatusPill(
                            emoji: presenceEmoji,
                            label: presenceLabel,
                            color: presenceColor,
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
                    }
                } else {
                    // No face detected
                    StatusPill(
                        emoji: "🚫",
                        label: "无人",
                        color: .secondary,
                        active: true
                    )
                }
            }
            .frame(height: 60)
            
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
    
    // Combined presence/gaze state: 看屏幕 / 看别处 / 闭眼
    var presenceEmoji: String {
        if state.face.eyesClosed { return "😴" }
        return state.face.isLookingAtScreen ? "👁" : "👀"
    }
    
    var presenceLabel: String {
        if state.face.eyesClosed { return "闭眼" }
        return state.face.isLookingAtScreen ? "看屏幕" : "看别处"
    }
    
    var presenceColor: Color {
        if state.face.eyesClosed { return .purple }
        return state.face.isLookingAtScreen ? .green : .orange
    }
    
    var faceOrientationEmoji: String {
        switch state.face.headOrientation {
        case .left: return "👈"
        case .right: return "👉"
        case .forward: return "😊"
        }
    }
    
    var faceOrientationLabel: String {
        switch state.face.headOrientation {
        case .left: return "朝左"
        case .right: return "朝右"
        case .forward: return "朝前"
        }
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
