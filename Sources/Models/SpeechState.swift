import Foundation

struct SpeechSegment: Identifiable {
    let id: UUID
    let text: String
    let isToScreen: Bool
    let sentenceStartedLookingAtScreen: Bool
    
    init(id: UUID = UUID(), text: String, isToScreen: Bool, sentenceStartedLookingAtScreen: Bool) {
        self.id = id
        self.text = text
        self.isToScreen = isToScreen
        self.sentenceStartedLookingAtScreen = sentenceStartedLookingAtScreen
    }
}

struct SpeechState {
    var segments: [SpeechSegment] = []
    var isListening: Bool = false
}
