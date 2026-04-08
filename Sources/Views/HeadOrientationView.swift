import SwiftUI

struct HeadOrientationView: View {
    let yaw: Float
    let pitch: Float
    let roll: Float
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("头部朝向").font(.caption).foregroundStyle(.secondary)
            OrientationBar(label: "左右", value: yaw, range: 0.8)
            OrientationBar(label: "上下", value: pitch, range: 0.6)
            OrientationBar(label: "歪头", value: roll, range: 1.0)  // Increased from 0.5 to 1.0
        }
    }
}

struct OrientationBar: View {
    let label: String
    let value: Float
    let range: Float
    
    var normalized: CGFloat {
        // Clamp to [-range, range] then map to [0, 1]
        let clamped = max(-range, min(range, value))
        return CGFloat((clamped / range + 1) / 2)
    }
    
    var body: some View {
        HStack(spacing: 8) {
            Text(label).font(.caption2).frame(width: 28, alignment: .leading)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.secondary.opacity(0.2)).frame(height: 8)
                    Capsule().fill(Color.accentColor).frame(width: 2, height: 16)
                        .offset(x: normalized * geo.size.width - 1)
                }
            }
            .frame(height: 16)
            Text(String(format: "%.2f", value)).font(.caption2.monospacedDigit()).frame(width: 40)
        }
    }
}
