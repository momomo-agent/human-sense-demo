import Foundation

struct SpeechSegment: Identifiable {
    let id = UUID()
    let text: String
    let isToScreen: Bool
    let sentenceStartedLookingAtScreen: Bool
}

struct SpeechState {
    var segments: [SpeechSegment] = []
    var isListening: Bool = false
}
