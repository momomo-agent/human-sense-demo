import SwiftUI

/// Full-screen overlay showing gaze point and valid gaze region.
/// Uses raw screen coordinates — matches FaceTrackingManager's isLookingAtScreen logic exactly.
struct GazeOverlay: View {
    let gazePoint: CGPoint
    let gazePointEye: CGPoint
    let isLooking: Bool
    let distance: Float
    let pitch: Float
    let yaw: Float
    
    var body: some View {
        let screen = UIScreen.main.bounds.size
        let marginX = screen.width * 0.15
        
        ZStack {
            // Valid gaze region (exactly matches FaceTrackingManager logic)
            Rectangle()
                .strokeBorder(isLooking ? Color.green.opacity(0.4) : Color.red.opacity(0.4), lineWidth: 2)
                .background(
                    Rectangle()
                        .fill(isLooking ? Color.green.opacity(0.05) : Color.red.opacity(0.05))
                )
                .frame(width: screen.width - marginX * 2, height: screen.height)
                .position(x: screen.width / 2, y: screen.height / 2)
            
            // Yellow point: center + (blue - red) * distance scale
            let center = CGPoint(x: screen.width / 2, y: screen.height / 2)
            let diffX = gazePoint.x - gazePointEye.x
            let diffY = gazePoint.y - gazePointEye.y
            // scale: reference distance 0.5m → scale=1.0, closer=larger, farther=smaller
            let scaleX = CGFloat(max(distance, 0.1) > 0 ? 0.5 / Double(max(distance, 0.1)) : 1.0)
            let scaleY = scaleX * CGFloat(1.0 + abs(Double(pitch)) * 0.5)
            let yellowPoint = CGPoint(x: center.x + diffX * scaleX, y: center.y + diffY * scaleY)
            Circle()
                .fill(Color.yellow.opacity(0.8))
                .frame(width: 16, height: 16)
                .shadow(color: .yellow, radius: 4)
                .position(yellowPoint)

            // Orange point: yaw/pitch mapped to screen
            let yawScale = screen.width / 2
            let pitchScale = screen.height / 2
            let orangePoint = CGPoint(
                x: center.x - CGFloat(yaw) * yawScale,
                y: center.y - CGFloat(pitch) * pitchScale
            )
            Circle()
                .fill(Color.orange.opacity(0.8))
                .frame(width: 16, height: 16)
                .shadow(color: .orange, radius: 4)
                .position(orangePoint)

            // Eye gaze point (red)
            Circle()
                .fill(Color.red.opacity(0.8))
                .frame(width: 16, height: 16)
                .shadow(color: .red, radius: 4)
                .position(gazePointEye)

            // Face-ray gaze point (blue)
            Circle()
                .fill(isLooking ? Color.blue.opacity(0.8) : Color.red.opacity(0.6))
                .frame(width: 16, height: 16)
                .shadow(color: isLooking ? .blue : .red, radius: 4)
                .position(gazePoint)
        }
        .frame(width: screen.width, height: screen.height)
    }
}
