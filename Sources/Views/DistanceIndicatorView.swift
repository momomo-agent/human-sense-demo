import SwiftUI

struct DistanceIndicatorView: View {
    let distance: Float
    let label: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("距离").font(.caption).foregroundStyle(.secondary)
            
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .stroke(Color.secondary.opacity(0.3), lineWidth: 2)
                        .frame(width: 40, height: 40)
                    
                    Circle()
                        .fill(distanceColor)
                        .frame(width: CGFloat(max(10, 40 - distance * 30)), height: CGFloat(max(10, 40 - distance * 30)))
                }
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(label).font(.caption2.bold())
                    Text(String(format: "%.2f 米", distance))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
    
    var distanceColor: Color {
        switch distance {
        case 0: return .gray
        case ..<0.3: return .red
        case ..<0.5: return .orange
        case ..<0.8: return .green
        case ..<1.2: return .blue
        default: return .purple
        }
    }
}
