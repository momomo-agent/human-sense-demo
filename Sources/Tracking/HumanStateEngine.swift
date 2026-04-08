import Foundation
import Combine

@MainActor
@Observable
class HumanStateEngine {
    var humanState = HumanState()
    // State history for timeline (last 10 seconds at ~10fps = 100 entries)
    var stateHistory: [(date: Date, activity: HumanActivity)] = []

    private let faceManager: FaceTrackingManager
    private let audioManager: AudioDetectionManager

    private var cancellables = Set<AnyCancellable>()
    private var pendingActivity: HumanActivity?
    private var pendingActivityStartTime: Date?
    private let debounceInterval: TimeInterval = 0.3
    private var previousJawOpen: Float = 0
    private var lastHistoryAppend = Date.distantPast

    init(faceManager: FaceTrackingManager, audioManager: AudioDetectionManager) {
        self.faceManager = faceManager
        self.audioManager = audioManager

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

        let newActivity = inferActivity(face: face, audio: audio)

        if newActivity != humanState.activity {
            if pendingActivity == newActivity {
                if let startTime = pendingActivityStartTime,
                   Date().timeIntervalSince(startTime) >= debounceInterval {
                    humanState.activity = newActivity
                    pendingActivity = nil
                    pendingActivityStartTime = nil
                }
            } else {
                pendingActivity = newActivity
                pendingActivityStartTime = Date()
            }
        } else {
            pendingActivity = nil
            pendingActivityStartTime = nil
        }

        // Append to history at ~10fps
        let now = Date()
        if now.timeIntervalSince(lastHistoryAppend) >= 0.1 {
            stateHistory.append((date: now, activity: humanState.activity))
            // Keep only last 10 seconds
            let cutoff = now.addingTimeInterval(-10)
            stateHistory.removeAll { $0.date < cutoff }
            lastHistoryAppend = now
        }

        previousJawOpen = face.jawOpen
    }

    private func inferActivity(face: FaceState, audio: AudioState) -> HumanActivity {
        if !face.faceDetected { return .absent }
        if face.eyesClosed { return .eyesClosed }

        // Speaking: mouth moving + has voice (regardless of gaze direction)
        let jawDelta = abs(face.jawOpen - previousJawOpen)
        let mouthMoving = jawDelta > 0.015 || face.jawOpen > 0.15
        if mouthMoving && audio.isSpeaking { return .speaking }

        // "Talking to device": speaking + looking at screen
        if !face.isLookingAtScreen { return .distracted }

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
