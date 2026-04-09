import Foundation

enum HumanActivity: String, CaseIterable {
    case absent           = "不在画面中"
    case eyesClosed       = "闭着眼睛"
    case distracted       = "看向别处"
    case listening        = "在看屏幕"
    case speakingToScreen = "对屏幕说话"
    case speakingToOther  = "对别处说话"

    var emoji: String {
        switch self {
        case .absent:           return "👻"
        case .eyesClosed:       return "😴"
        case .distracted:       return "👀"
        case .listening:        return "👁"
        case .speakingToScreen: return "🗣"
        case .speakingToOther:  return "🗣"
        }
    }

    var isSpeaking: Bool {
        self == .speakingToScreen || self == .speakingToOther
    }
}

struct HumanState {
    var activity: HumanActivity = .absent
    var face: FaceState = FaceState()
    var audio: AudioState = AudioState()
    var hand: HandState = HandState()
    var device: DeviceState = DeviceState()
    var speech: SpeechState = SpeechState()
}
