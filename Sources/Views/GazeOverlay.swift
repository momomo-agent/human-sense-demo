import SwiftUI

struct GazeOverlay: View {
    let gazePoint: CGPoint
    let isLooking: Bool
    
    var body: some View {
        GeometryReader { geo in
            Circle()
                .fill(isLooking ? Color.blue.opacity(0.7) : Color.red.opacity(0.5))
                .frame(width: 20, height: 20)
                .position(gazePoint)
        }
    }
}
