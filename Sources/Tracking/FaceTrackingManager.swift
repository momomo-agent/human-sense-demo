import Foundation
import ARKit
import Combine

@MainActor
class FaceTrackingManager: NSObject, ObservableObject {
    @Published var faceState = FaceState()
    
    private let arSession = ARSession()
    private let processingQueue = DispatchQueue(label: "com.momomo.facetracking", qos: .userInitiated)
    
    private var gazeFilterX: LowPassFilter?
    private var gazeFilterY: LowPassFilter?
    private var previousJawOpen: Float = 0
    
    override init() {
        super.init()
        arSession.delegate = self
    }
    
    func start() {
        guard ARFaceTrackingConfiguration.isSupported else {
            print("Face tracking not supported")
            return
        }
        
        let config = ARFaceTrackingConfiguration()
        config.worldAlignment = .camera
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
    }
    
    func stop() {
        arSession.pause()
    }
    
    nonisolated private func extractHeadOrientation(from transform: simd_float4x4) -> (yaw: Float, pitch: Float, roll: Float) {
        let yaw = atan2(transform.columns.0.z, transform.columns.2.z)
        let pitch = asin(-transform.columns.1.z)
        let roll = atan2(transform.columns.1.x, transform.columns.1.y)
        return (yaw, pitch, roll)
    }
}

extension FaceTrackingManager: ARSessionDelegate {
    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let anchor = frame.anchors.first as? ARFaceAnchor else {
            Task { @MainActor in
                self.faceState.faceDetected = false
            }
            return
        }
        
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Extract head orientation
            let (yaw, pitch, roll) = self.extractHeadOrientation(from: anchor.transform)
            
            // Gaze mapping using ARCamera.projectPoint
            let lookAtVector = anchor.transform * SIMD4<Float>(anchor.lookAtPoint, 1)
            
            guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                  let orientation = scene.windows.first?.windowScene?.interfaceOrientation else { return }
            
            let lookPoint = frame.camera.projectPoint(
                SIMD3<Float>(x: lookAtVector.x, y: lookAtVector.y, z: lookAtVector.z),
                orientation: orientation,
                viewportSize: UIScreen.main.bounds.size
            )
            
            let size = UIScreen.main.bounds.size
            let adjusted = (x: size.width - lookPoint.x, y: size.height - lookPoint.y)
            
            // Extract blend shapes
            let bs = anchor.blendShapes
            let jawOpen = bs[.jawOpen]?.floatValue ?? 0
            let mouthClose = bs[.mouthClose]?.floatValue ?? 0
            let eyeBlinkL = bs[.eyeBlinkLeft]?.floatValue ?? 0
            let eyeBlinkR = bs[.eyeBlinkRight]?.floatValue ?? 0
            
            let eyeLookInL = bs[.eyeLookInLeft]?.floatValue ?? 0
            let eyeLookOutL = bs[.eyeLookOutLeft]?.floatValue ?? 0
            let eyeLookUpL = bs[.eyeLookUpLeft]?.floatValue ?? 0
            let eyeLookDownL = bs[.eyeLookDownLeft]?.floatValue ?? 0
            
            let eyeLookInR = bs[.eyeLookInRight]?.floatValue ?? 0
            let eyeLookOutR = bs[.eyeLookOutRight]?.floatValue ?? 0
            let eyeLookUpR = bs[.eyeLookUpRight]?.floatValue ?? 0
            let eyeLookDownR = bs[.eyeLookDownRight]?.floatValue ?? 0
            
            // Update on main thread
            Task { @MainActor in
                // Initialize filters if needed
                if self.gazeFilterX == nil {
                    self.gazeFilterX = LowPassFilter(value: adjusted.x)
                    self.gazeFilterY = LowPassFilter(value: adjusted.y)
                } else {
                    self.gazeFilterX?.update(with: adjusted.x)
                    self.gazeFilterY?.update(with: adjusted.y)
                }
                
                var newState = FaceState()
                newState.faceDetected = true
                newState.gazePoint = CGPoint(x: self.gazeFilterX?.value ?? adjusted.x,
                                            y: self.gazeFilterY?.value ?? adjusted.y)
                
                newState.headYaw = yaw
                newState.headPitch = pitch
                newState.headRoll = roll
                
                // Looking at screen if head is roughly facing forward
                newState.isLookingAtScreen = abs(yaw) < 0.4 && abs(pitch) < 0.35
                
                newState.jawOpen = jawOpen
                newState.mouthClose = mouthClose
                
                newState.eyeBlinkLeft = eyeBlinkL
                newState.eyeBlinkRight = eyeBlinkR
                
                newState.eyeLookInLeft = eyeLookInL
                newState.eyeLookOutLeft = eyeLookOutL
                newState.eyeLookUpLeft = eyeLookUpL
                newState.eyeLookDownLeft = eyeLookDownL
                
                newState.eyeLookInRight = eyeLookInR
                newState.eyeLookOutRight = eyeLookOutR
                newState.eyeLookUpRight = eyeLookUpR
                newState.eyeLookDownRight = eyeLookDownR
                
                self.faceState = newState
                self.previousJawOpen = jawOpen
            }
        }
    }
}
