import Foundation
import CoreMotion
import UIKit
import Combine

enum DevicePosture: String {
    case uprightStand = "竖放"      // Standing upright on stand
    case landscapeStand = "横放"    // Landscape on stand
    case faceUp = "平躺"            // Flat face up
    case faceDown = "盖着"          // Flat face down
    case holdingPortrait = "持握竖"  // Holding portrait
    case holdingLandscape = "持握横" // Holding landscape
    case holdingWalking = "持握行走"  // Holding while walking
}

enum DeviceOrientation: String {
    case portrait = "竖屏"
    case landscape = "横屏"
    case unknown = "未知"
}

struct DeviceMotionState {
    var posture: DevicePosture = .uprightStand
    var orientation: DeviceOrientation = .portrait
    var isWalking: Bool = false
    var isHolding: Bool = false  // New: holding vs placed
}

@MainActor
class DeviceMotionManager: ObservableObject {
    @Published var motionState = DeviceMotionState()
    @Published var debugPitch: Float = 0
    @Published var debugVariance: Double = 0
    @Published var debugGravityY: Double = 0
    @Published var debugGravityZ: Double = 0
    
    private let motionManager = CMMotionManager()
    private let activityManager = CMMotionActivityManager()
    private var accelerationHistory: [Double] = []
    private let historySize = 10
    
    func start() {
        // Start device motion updates
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 0.1
            motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
                guard let motion = motion else { return }
                self?.updatePosture(from: motion)
                self?.updateHoldingState(from: motion)
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
        
        debugGravityY = gravity.y
        debugGravityZ = gravity.z
        
        let absY = abs(gravity.y)
        let absZ = abs(gravity.z)
        
        // Determine base posture from gravity
        if absZ > 0.8 {
            // Flat (screen up or down)
            if gravity.z > 0 {
                motionState.posture = .faceUp
            } else {
                motionState.posture = .faceDown
            }
        } else if absY > 0.7 {
            // Vertical orientation
            if motionState.isWalking {
                motionState.posture = .holdingWalking
            } else if motionState.isHolding {
                if motionState.orientation == .landscape {
                    motionState.posture = .holdingLandscape
                } else {
                    motionState.posture = .holdingPortrait
                }
            } else {
                // Standing on surface
                if gravity.z < -0.3 {
                    // Tilted back (on stand)
                    motionState.posture = .uprightStand
                } else {
                    motionState.posture = .uprightStand
                }
            }
        } else {
            // Horizontal/landscape
            if motionState.isHolding {
                motionState.posture = .holdingLandscape
            } else {
                motionState.posture = .landscapeStand
            }
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
    
    private func updateHoldingState(from motion: CMDeviceMotion) {
        // Calculate total acceleration magnitude (excluding gravity)
        let userAccel = motion.userAcceleration
        let magnitude = sqrt(userAccel.x * userAccel.x + 
                           userAccel.y * userAccel.y + 
                           userAccel.z * userAccel.z)
        
        // Add to history
        accelerationHistory.append(magnitude)
        if accelerationHistory.count > historySize {
            accelerationHistory.removeFirst()
        }
        
        // Calculate variance over history
        guard accelerationHistory.count == historySize else { return }
        let mean = accelerationHistory.reduce(0, +) / Double(historySize)
        let variance = accelerationHistory.map { pow($0 - mean, 2) }.reduce(0, +) / Double(historySize)
        
        debugVariance = variance
        
        // If variance is above threshold, device is being held (micro-movements)
        // If variance is near zero, device is placed on a surface
        motionState.isHolding = variance > 0.0001
    }
}
