import SwiftUI
import HumanSenseKit

struct StateTimelineView: View {
    let history: [(date: Date, activity: HumanActivity)]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("状态时间线 (10秒)").font(.caption).foregroundStyle(.secondary)
            
            // Legend
            HStack(spacing: 12) {
                LegendItem(color: .gray, label: "不在")
                LegendItem(color: .purple, label: "闭眼")
                LegendItem(color: Color(red: 0, green: 0.5, blue: 0), label: "分心")
                LegendItem(color: .green, label: "倾听")
                LegendItem(color: .yellow, label: "对屏幕说")
                LegendItem(color: .orange, label: "对别处说")
            }
            .font(.caption2)
            
            GeometryReader { geo in
                Canvas { context, size in
                    guard !history.isEmpty else { return }
                    
                    let now = Date()
                    let startTime = now.addingTimeInterval(-10)
                    let width = size.width
                    
                    for (_, entry) in history.enumerated() {
                        let elapsed = entry.date.timeIntervalSince(startTime)
                        let x = CGFloat(elapsed / 10.0) * width
                        
                        let color: Color = {
                            switch entry.activity {
                            case .absent: return .gray
                            case .eyesClosed: return .purple
                            case .distracted: return Color(red: 0, green: 0.5, blue: 0)
                            case .listening: return .green
                            case .speakingToScreen: return .yellow
                            case .speakingToOther: return .orange
                            }
                        }()
                        
                        let rect = CGRect(x: x, y: 0, width: max(2, width / CGFloat(history.count)), height: size.height)
                        context.fill(Path(rect), with: .color(color))
                    }
                }
            }
            .frame(height: 20)
            .background(Color.secondary.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 4))
        }
    }
}

struct LegendItem: View {
    let color: Color
    let label: String
    
    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(label)
        }
    }
}
