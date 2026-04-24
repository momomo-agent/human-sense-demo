import SwiftUI

struct GazeOverlay: View {
    let gazePoint: CGPoint
    let isLooking: Bool
    
    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let marginX = w * 0.1
            
            // Valid gaze region (matches FaceTrackingManager logic)
            Rectangle()
                .strokeBorder(isLooking ? Color.green.opacity(0.3) : Color.red.opacity(0.3), lineWidth: 2)
                .background(
                    Rectangle()
                        .fill(isLooking ? Color.green.opacity(0.05) : Color.red.opacity(0.05))
                )
                .frame(width: w - marginX * 2, height: h)
                .position(x: w / 2, y: h / 2)
            
            // Gaze point
            Circle()
                .fill(isLooking ? Color.blue.opacity(0.7) : Color.red.opacity(0.5))
                .frame(width: 20, height: 20)
                .position(gazePoint)
        }
    }
}
