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
            
            if hand.detected && !hand.handPoints.isEmpty {
                Canvas { context, size in
                    for point in hand.handPoints {
                        let x = point.x * size.width
                        let y = point.y * size.height
                        context.fill(
                            Circle().path(in: CGRect(x: x - 3, y: y - 3, width: 6, height: 6)),
                            with: .color(.cyan)
                        )
                    }
                }
                .frame(height: 60)
                .background(Color.white.opacity(0.05))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            }
        }
    }
}
