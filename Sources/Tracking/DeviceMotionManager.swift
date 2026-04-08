import Foundation
import CoreMotion
import UIKit
import Combine

enum DevicePosture: String {
    case flat = "平放"
    case upright = "竖立"
    case tilted = "倾斜"
    case upsideDown = "倒置"
}

enum DeviceOrientation: String {
    case portrait = "竖屏"
    case landscape = "横屏"
    case unknown = "未知"
}

struct DeviceMotionState {
    var posture: DevicePosture = .upright
    var orientation: DeviceOrientation = .portrait
    var isWalking: Bool = false
}

@MainActor
class DeviceMotionManager: ObservableObject {
    @Published var motionState = DeviceMotionState()
    
    private let motionManager = CMMotionManager()
    private let activityManager = CMMotionActivityManager()
    
    func start() {
        // Start device motion updates
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 0.1
            motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
                guard let motion = motion else { return }
                self?.updatePosture(from: motion)
            }
        }
        
        // Start activity updates
        if CMMotionActivityManager.isActivityAvailable() {
            activityManager.startActivityUpdates(to: .main) { [weak self] activity in
                guard let activity = activity else { return }
                self?.motionState.isWalking = activity.walking
            }
        }
        
        // Monitor orientation changes
        NotificationCenter.default.addObserver(
            forName: UIDevice.orientationDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.updateOrientation()
        }
        
        updateOrientation()
    }
    
    func stop() {
        motionManager.stopDeviceMotionUpdates()
        activityManager.stopActivityUpdates()
        NotificationCenter.default.removeObserver(self)
    }
    
    private func updatePosture(from motion: CMDeviceMotion) {
        let gravity = motion.gravity
        
        // Calculate pitch angle (forward/backward tilt)
        let pitch = atan2(gravity.y, sqrt(gravity.x * gravity.x + gravity.z * gravity.z))
        
        // Determine posture based on pitch angle
        // pitch > 0: device tilted back (screen facing up)
        // pitch < 0: device tilted forward (screen facing down)
        if abs(pitch) < 0.5 {
            // Nearly horizontal
            motionState.posture = .flat
        } else if pitch > 0.8 && pitch < 2.0 {
            // Normal upright position (45° to 90°)
            motionState.posture = .upright
        } else if pitch < -0.8 && pitch > -2.0 {
            // Upside down
            motionState.posture = .upsideDown
        } else {
            // In between
            motionState.posture = .tilted
        }
    }
    
    private func updateOrientation() {
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene else {
            motionState.orientation = .unknown
            return
        }
        
        let orientation = windowScene.interfaceOrientation
        switch orientation {
        case .portrait, .portraitUpsideDown:
            motionState.orientation = .portrait
        case .landscapeLeft, .landscapeRight:
            motionState.orientation = .landscape
        default:
            motionState.orientation = .unknown
        }
    }
}
