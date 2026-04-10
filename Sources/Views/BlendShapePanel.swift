import SwiftUI
import HumanSenseKit

struct BlendShapePanel: View {
    let face: FaceState
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("面部状态").font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 12) {
                BlendBar(label: "张嘴", value: face.jawOpen, color: .orange)
                BlendBar(label: "左眼", value: face.eyeBlinkLeft, color: .purple)
                BlendBar(label: "右眼", value: face.eyeBlinkRight, color: .purple)
            }
            HStack(spacing: 12) {
                BlendBar(label: "眼H", value: abs(face.gazeH), color: .blue)
                BlendBar(label: "眼V", value: abs(face.gazeV), color: .blue)
            }
        }
    }
}

struct BlendBar: View {
    let label: String
    let value: Float
    let color: Color
    
    var body: some View {
        VStack(spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            GeometryReader { geo in
                ZStack(alignment: .bottom) {
                    RoundedRectangle(cornerRadius: 4).fill(color.opacity(0.15))
                    RoundedRectangle(cornerRadius: 4).fill(color)
                        .frame(height: geo.size.height * CGFloat(value))
                }
            }
            .frame(height: 40)
            Text(String(format: "%.2f", value)).font(.caption2.monospacedDigit())
        }
        .frame(maxWidth: .infinity)
    }
}
