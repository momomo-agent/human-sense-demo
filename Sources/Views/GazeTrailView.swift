import SwiftUI

struct GazeTrailView: View {
    let trail: [CGPoint]
    
    var body: some View {
        Canvas { context, size in
            guard trail.count > 1 else { return }
            
            var path = Path()
            path.move(to: trail[0])
            for point in trail.dropFirst() {
                path.addLine(to: point)
            }
            
            // Draw trail with gradient opacity (older = more transparent)
            for (index, point) in trail.enumerated() {
                let opacity = Double(index) / Double(trail.count)
                context.fill(
                    Circle().path(in: CGRect(x: point.x - 2, y: point.y - 2, width: 4, height: 4)),
                    with: .color(.blue.opacity(opacity * 0.5))
                )
            }
            
            context.stroke(
                path,
                with: .color(.blue.opacity(0.3)),
                lineWidth: 1
            )
        }
    }
}
