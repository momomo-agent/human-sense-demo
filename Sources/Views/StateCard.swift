import SwiftUI

struct StateCard: View {
    let state: HumanState

    var body: some View {
        VStack(spacing: 8) {
            // Multiple simultaneous indicators
            HStack(spacing: 16) {
                // Gaze indicator
                StatusPill(
                    emoji: state.face.isLookingAtScreen ? "👁" : "👀",
                    label: state.face.isLookingAtScreen ? "看屏幕" : "看别处",
                    color: state.face.isLookingAtScreen ? .green : .orange,
                    active: state.face.faceDetected
                )

                // Speech indicator with color distinction
                StatusPill(
                    emoji: state.audio.isSpeaking ? "🗣" : "🤫",
                    label: state.audio.isSpeaking ? "在说话" : "安静",
                    color: speechColor,
                    active: true
                )

                // Presence indicator
                StatusPill(
                    emoji: !state.face.faceDetected ? "👻" : (state.face.eyesClosed ? "😴" : "😊"),
                    label: !state.face.faceDetected ? "不在" : (state.face.eyesClosed ? "闭眼" : "检测中"),
                    color: !state.face.faceDetected ? .gray : (state.face.eyesClosed ? .purple : .green),
                    active: true
                )
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

            // Combined state description
            Text(state.activity.rawValue)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(Color.white.opacity(0.05))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    var speechColor: Color {
        if !state.audio.isSpeaking { return .gray }
        // Speaking to screen (looking + speaking) = blue
        // Speaking elsewhere (not looking + speaking) = orange
        return state.face.isLookingAtScreen ? .blue : .orange
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
