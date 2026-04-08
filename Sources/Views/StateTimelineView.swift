import SwiftUI

struct StateTimelineView: View {
    let history: [(date: Date, activity: HumanActivity)]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("状态时间线 (10秒)").font(.caption).foregroundStyle(.secondary)
            
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
                            case .distracted: return .orange
                            case .listening: return .green
                            case .speaking: return .blue
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
