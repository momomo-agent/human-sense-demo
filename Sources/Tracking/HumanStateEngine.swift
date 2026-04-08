import Foundation
import Combine

@MainActor
@Observable
class HumanStateEngine {
    var humanState = HumanState()
    
    private let faceManager: FaceTrackingManager
    private let audioManager: AudioDetectionManager
    
    private var cancellables = Set<AnyCancellable>()
    private var stateDebounceTimer: Timer?
    private var pendingActivity: HumanActivity?
    private var pendingActivityStartTime: Date?
    
    private let debounceInterval: TimeInterval = 0.3  // 300ms
    private var previousJawOpen: Float = 0
    
    init(faceManager: FaceTrackingManager, audioManager: AudioDetectionManager) {
        self.faceManager = faceManager
        self.audioManager = audioManager
        
        // Observe face and audio state changes
        faceManager.$faceState
            .combineLatest(audioManager.$audioState)
            .sink { [weak self] face, audio in
                self?.updateHumanState(face: face, audio: audio)
            }
            .store(in: &cancellables)
    }
    
    private func updateHumanState(face: FaceState, audio: AudioState) {
        humanState.face = face
        humanState.audio = audio
        
        // Determine activity
        let newActivity = inferActivity(face: face, audio: audio)
        
        // Debounce state changes
        if newActivity != humanState.activity {
            if pendingActivity == newActivity {
                // Same pending activity, check if enough time has passed
                if let startTime = pendingActivityStartTime,
                   Date().timeIntervalSince(startTime) >= debounceInterval {
                    humanState.activity = newActivity
                    pendingActivity = nil
                    pendingActivityStartTime = nil
                }
            } else {
                // New pending activity
                pendingActivity = newActivity
                pendingActivityStartTime = Date()
            }
        } else {
            // Activity matches current, reset pending
            pendingActivity = nil
            pendingActivityStartTime = nil
        }
        
        previousJawOpen = face.jawOpen
    }
    
    private func inferActivity(face: FaceState, audio: AudioState) -> HumanActivity {
        // Priority order: absent > eyesClosed > speaking > distracted > listening
        
        if !face.faceDetected {
            return .absent
        }
        
        if face.eyesClosed {
            return .eyesClosed
        }
        
        // Speaking: mouth moving + has voice + looking at screen
        let jawDelta = abs(face.jawOpen - previousJawOpen)
        let mouthMoving = jawDelta > 0.02
        
        if mouthMoving && audio.isSpeaking && face.isLookingAtScreen {
            return .speaking
        }
        
        if !face.isLookingAtScreen {
            return .distracted
        }
        
        return .listening
    }
    
    func start() {
        faceManager.start()
        audioManager.start()
    }
    
    func stop() {
        faceManager.stop()
        audioManager.stop()
    }
}
