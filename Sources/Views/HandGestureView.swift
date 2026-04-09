import SwiftUI

struct HandGestureView: View {
    let hand: HandState
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("手势识别").font(.caption).foregroundStyle(.secondary)
            
            HStack(spacing: 12) {
                Circle()
                    .fill(hand.detected ? Color.green : Color.gray.opacity(0.3))
                    .frame(width: 10, height: 10)
                
                Text(hand.handLabel)
                    .font(.caption2)
                    .foregroundStyle(hand.detected ? .primary : .secondary)
            }
            
            // Finger curl indicators
            if hand.detected && !hand.fingerData.isEmpty {
                HStack(spacing: 8) {
                    ForEach(Finger.allCases, id: \.rawValue) { finger in
                        if let data = hand.fingerData[finger] {
                            VStack(spacing: 2) {
                                Text(fingerEmoji(finger))
                                    .font(.caption2)
                                curlIndicator(data.curl)
                            }
                        }
                    }
                }
            }
            
            // Fixed height canvas to prevent layout jumping
            Canvas { context, size in
                if hand.detected && !hand.handPoints.isEmpty {
                    for point in hand.handPoints {
                        let x = point.x * size.width
                        let y = point.y * size.height
                        context.fill(
                            Circle().path(in: CGRect(x: x - 3, y: y - 3, width: 6, height: 6)),
                            with: .color(.cyan)
                        )
                    }
                }
            }
            .frame(height: 60)
            .background(Color.white.opacity(0.05))
            .clipShape(RoundedRectangle(cornerRadius: 4))
        }
    }
    
    private func fingerEmoji(_ finger: Finger) -> String {
        switch finger {
        case .thumb: return "👆"
        case .index: return "☝️"
        case .middle: return "🖕"
        case .ring: return "💍"
        case .pinky: return "🤙"
        }
    }
    
    @ViewBuilder
    private func curlIndicator(_ curl: FingerCurl) -> some View {
        let (color, height): (Color, CGFloat) = switch curl {
        case .noCurl: (.green, 16)
        case .halfCurl: (.yellow, 10)
        case .fullCurl: (.red, 4)
        }
        RoundedRectangle(cornerRadius: 1)
            .fill(color)
            .frame(width: 6, height: height)
    }
}
