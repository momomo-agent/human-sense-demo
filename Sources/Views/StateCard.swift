import SwiftUI

struct StateCard: View {
    let state: HumanState
    
    var activity: HumanActivity { state.activity }
    
    var cardColor: Color {
        switch activity {
        case .absent:     return .gray
        case .eyesClosed: return .purple
        case .distracted: return .orange
        case .listening:  return .green
        case .speaking:   return .blue
        }
    }
    
    var body: some View {
        VStack(spacing: 8) {
            Text(activity.emoji)
                .font(.system(size: 48))
            Text(activity.rawValue)
                .font(.title2.bold())
                .foregroundStyle(.white)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 20)
        .background(cardColor.gradient)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}
