import SwiftUI

struct MouthVisualizerView: View {
    let jawOpen: Float
    let mouthClose: Float
    
    var body: some View {
        VStack(spacing: 4) {
            Text("嘴巴").font(.caption2).foregroundStyle(.secondary)
            
            ZStack {
                // Upper lip
                Capsule()
                    .fill(Color.red.opacity(0.8))
                    .frame(width: 80, height: 8)
                    .offset(y: -CGFloat(jawOpen) * 15)
                
                // Lower lip
                Capsule()
                    .fill(Color.red.opacity(0.8))
                    .frame(width: 80, height: 8)
                    .offset(y: CGFloat(jawOpen) * 15)
            }
            .frame(height: 40)
            
            Text(String(format: "张开: %.2f", jawOpen))
                .font(.caption2.monospacedDigit())
        }
    }
}
