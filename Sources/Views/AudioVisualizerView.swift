import SwiftUI

struct AudioVisualizerView: View {
    let audio: AudioState
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("音频").font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 12) {
                // Volume bar
                VStack(spacing: 2) {
                    Text("音量").font(.caption2).foregroundStyle(.secondary)
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color.secondary.opacity(0.2)).frame(height: 8)
                            Capsule()
                                .fill(audio.isSpeaking ? Color.green : Color.gray)
                                .frame(width: geo.size.width * CGFloat(min(audio.volume * 20, 1.0)), height: 8)
                        }
                    }
                    .frame(height: 8)
                }
                
                // Speaking indicator
                HStack(spacing: 6) {
                    Circle()
                        .fill(audio.isSpeaking ? Color.green : Color.gray.opacity(0.3))
                        .frame(width: 10, height: 10)
                        .animation(.easeInOut(duration: 0.1), value: audio.isSpeaking)
                    Text(audio.isSpeaking ? "有声音" : "静音")
                        .font(.caption2)
                        .foregroundStyle(audio.isSpeaking ? .primary : .secondary)
                }
            }
        }
    }
}
