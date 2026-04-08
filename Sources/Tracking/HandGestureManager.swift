import Foundation
import Vision
import ARKit
import Combine

@MainActor
class HandGestureManager: NSObject, ObservableObject {
    @Published var handState = HandState()

    private let processingQueue = DispatchQueue(label: "com.momomo.handgesture", qos: .userInitiated)

    func start() {
        processingQueue.async { [weak self] in
            self?.isProcessingEnabled = true
        }
    }

    func stop() {
        processingQueue.async { [weak self] in
            self?.isProcessingEnabled = false
        }
    }
    
    private var isProcessingEnabled = false
    private var frameCount = 0
    private var gestureHistory: [HandGesture] = []
    private let historySize = 3  // Require 3 consecutive frames for confirmation
    
    // Called by FaceTrackingManager with ARFrame
    nonisolated func processFrame(_ frame: ARFrame) {
        processingQueue.async { [weak self] in
            guard let self = self, self.isProcessingEnabled else { return }
            
            self.frameCount += 1
            guard self.frameCount % 15 == 0 else { return }  // Process every 15th frame (~4fps)
            
            let pixelBuffer = frame.capturedImage
            
            let request = VNDetectHumanHandPoseRequest()
            request.maximumHandCount = 2
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
            
            do {
                try handler.perform([request])
                
                guard let observations = request.results, !observations.isEmpty else {
                    Task { @MainActor in
                        self.handState.detected = false
                        self.handState.gesture = .none
                        self.gestureHistory.removeAll()
                    }
                    return
                }
                
                // Process first hand
                let observation = observations[0]
                let chirality = observation.chirality
                let isLeft = (chirality == .left)
                
                guard let points = try? observation.recognizedPoints(.all) else { return }
                
                let gesture = self.recognizeGesture(from: points)
                
                // Add to history and check for consistency
                self.gestureHistory.append(gesture)
                if self.gestureHistory.count > self.historySize {
                    self.gestureHistory.removeFirst()
                }
                
                // Only update if gesture is consistent across history
                let confirmedGesture: HandGesture
                if self.gestureHistory.count == self.historySize &&
                   self.gestureHistory.allSatisfy({ $0 == gesture }) {
                    confirmedGesture = gesture
                } else {
                    confirmedGesture = self.handState.gesture  // Keep previous
                }
                
                Task { @MainActor in
                    self.handState.detected = true
                    self.handState.gesture = confirmedGesture
                    self.handState.isLeftHand = isLeft
                    self.handState.handPoints = points.values.map { CGPoint(x: CGFloat($0.location.x), y: CGFloat($0.location.y)) }
                }
                
            } catch {
                print("Hand pose detection error: \(error)")
            }
        }
    }
    
    private func recognizeGesture(from points: [VNHumanHandPoseObservation.JointName: VNRecognizedPoint]) -> HandGesture {
        guard let thumb = points[.thumbTip],
              let index = points[.indexTip],
              let middle = points[.middleTip],
              let ring = points[.ringTip],
              let pinky = points[.littleTip],
              let wrist = points[.wrist],
              thumb.confidence > 0.5,  // Increased from 0.3
              index.confidence > 0.5 else {
            return .none
        }
        
        let thumbUp = thumb.location.y > wrist.location.y + 0.15
        let indexUp = index.location.y > wrist.location.y + 0.15
        let middleUp = middle.location.y > wrist.location.y + 0.15
        let ringUp = ring.location.y > wrist.location.y + 0.15
        let pinkyUp = pinky.location.y > wrist.location.y + 0.15
        let ringDown = ring.location.y < wrist.location.y + 0.1
        let pinkyDown = pinky.location.y < wrist.location.y + 0.1
        
        // Thumbs up: thumb up, others down
        if thumbUp && !indexUp && !middleUp && ringDown && pinkyDown {
            return .thumbsUp
        }
        
        // Peace: index + middle up, others down
        if indexUp && middleUp && !thumbUp && ringDown && pinkyDown {
            return .peace
        }
        
        // OK: thumb and index touching, others up
        let thumbIndexDist = hypot(thumb.location.x - index.location.x, thumb.location.y - index.location.y)
        if thumbIndexDist < 0.08 && middleUp && ringUp && pinkyUp {
            return .ok
        }
        
        // Love heart: thumb and index forming angle, others down
        if thumbUp && indexUp && !middleUp && ringDown && pinkyDown {
            let angle = atan2(index.location.y - thumb.location.y, index.location.x - thumb.location.x)
            if abs(angle) > 0.5 && abs(angle) < 2.0 {
                return .love
            }
        }
        
        // Rock: index + pinky up, middle + ring down
        if indexUp && pinkyUp && !middleUp && ringDown {
            return .rock
        }
        
        // Pray: all fingers up and close together
        if thumbUp && indexUp && middleUp && ringUp && pinkyUp {
            let spread = abs(index.location.x - pinky.location.x)
            if spread < 0.15 {
                return .pray
            }
            return .openPalm
        }
        
        // Numbers
        if indexUp && !middleUp && !thumbUp && ringDown && pinkyDown {
            return .one
        }
        
        if indexUp && middleUp && !thumbUp && ringDown && pinkyDown {
            return .two
        }
        
        if thumbUp && indexUp && middleUp && ringDown && pinkyDown {
            return .three
        }
        
        if indexUp && middleUp && ringUp && pinkyUp && !thumbUp {
            return .four
        }
        
        if thumbUp && indexUp && middleUp && ringUp && pinkyUp {
            return .five
        }
        
        // Fist: all fingers down
        if !thumbUp && !indexUp && !middleUp && ringDown && pinkyDown {
            return .fist
        }
        
        return .none
    }
}
