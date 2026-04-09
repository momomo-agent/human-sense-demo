import Foundation

enum DevicePosture: String {
    case uprightStand = "竖放"
    case landscapeStand = "横放"
    case faceUp = "平躺"
    case faceDown = "盖着"
    case holdingWalking = "持握行走"
}

enum DeviceOrientation: String {
    case portrait = "竖屏"
    case landscape = "横屏"
    case unknown = "未知"
}

struct DeviceState {
    var posture: DevicePosture = .uprightStand
    var orientation: DeviceOrientation = .portrait
    var isWalking: Bool = false
    var isHolding: Bool = false
}
