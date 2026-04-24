import SwiftUI

/// Full-screen overlay showing gaze point and valid gaze region.
/// Uses raw screen coordinates — matches FaceTrackingManager's isLookingAtScreen logic exactly.
struct GazeOverlay: View {
    let gazePoint: CGPoint   // screen coordinates
    let isLooking: Bool
    
    var body: some View {
        let screen = UIScreen.main.bounds.size
        let marginX = screen.width * 0.1
        
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
            
            // Gaze point
            Circle()
                .fill(isLooking ? Color.blue.opacity(0.8) : Color.red.opacity(0.6))
                .frame(width: 16, height: 16)
                .shadow(color: isLooking ? .blue : .red, radius: 4)
                .position(gazePoint)
        }
        .frame(width: screen.width, height: screen.height)
    }
}
