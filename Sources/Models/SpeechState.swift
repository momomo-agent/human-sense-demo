import Foundation

struct SpeechSegment: Identifiable {
    let id: UUID
    let text: String
    let isToScreen: Bool
    let sentenceStartedLookingAtScreen: Bool
    /// true = user's own speech (mouth moving + audio), false = ambient/other people
    let isFromUser: Bool
    
    init(id: UUID = UUID(), text: String, isToScreen: Bool, sentenceStartedLookingAtScreen: Bool, isFromUser: Bool = true) {
        self.id = id
        self.text = text
        self.isToScreen = isToScreen
        self.sentenceStartedLookingAtScreen = sentenceStartedLookingAtScreen
        self.isFromUser = isFromUser
    }
}

struct SpeechState {
    var segments: [SpeechSegment] = []
    var isListening: Bool = false
}
