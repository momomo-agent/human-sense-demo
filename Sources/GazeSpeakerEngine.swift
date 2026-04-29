import SwiftUI
import AVFoundation
import FluidAudio
import HumanSenseKit

@MainActor
@Observable
class GazeSpeakerEngine {
    enum Phase {
        case calibration
        case live
    }

    struct DebugInfo {
        var isLookingAtScreen: Bool = false
        var isHeadForward: Bool = false
        var speakerMatch: Bool = false
        var speakerDistance: Float = 1.0
        var audioLevel: Float = -60.0
        var userEmbeddingStatus: String = "未标定"
        var currentJawDelta: Float = 0.0
        var currentJawVelocity: Float = 0.0
        var gazeOnScreen: Float = 0.0
        var headYaw: Float = 0.0
        var headPitch: Float = 0.0
        var faceDistance: Float = 0.0
    }

    struct TranscriptSegment: Identifiable {
        let id = UUID()
        let tokens: [TokenSegment]
        let isFinal: Bool
        let timestamp: Date

        var text: String { tokens.map { $0.text }.joined() }
        var isUserSpeaker: Bool {
            let userCount = tokens.filter { $0.isUserSpeaker }.count
            return userCount > tokens.count / 2
        }
        var score: Float {
            tokens.map { $0.score }.reduce(0, +) / Float(tokens.count)
        }
        var audioTime: Double { tokens.first?.audioTime ?? 0 }
    }

    struct TokenSegment: Identifiable {
        let id = UUID()
        let text: String
        let isUserSpeaker: Bool
        let score: Float
        let audioTime: Double
        let jawDelta: Float
        let jawVelocity: Float
        var gazeOnScreen: Float = 0
        var headYaw: Float = 0
        var headPitch: Float = 0
        var faceDistance: Float = 0

        init(from speakerToken: SpeakerToken) {
            self.text = speakerToken.text
            self.isUserSpeaker = speakerToken.isUserSpeaker
            self.score = speakerToken.score
            self.audioTime = speakerToken.audioTime
            self.jawDelta = speakerToken.jawDelta
            self.jawVelocity = speakerToken.jawVelocity
            self.gazeOnScreen = speakerToken.gazeOnScreen
            self.headYaw = speakerToken.headYaw
            self.headPitch = speakerToken.headPitch
            self.faceDistance = speakerToken.faceDistance
        }

        init(text: String, isUserSpeaker: Bool, score: Float, audioTime: Double,
             jawDelta: Float, jawVelocity: Float) {
            self.text = text
            self.isUserSpeaker = isUserSpeaker
            self.score = score
            self.audioTime = audioTime
            self.jawDelta = jawDelta
            self.jawVelocity = jawVelocity
        }
    }

    // MARK: - UI State

    var phase: Phase = .calibration
    var transcriptSegments: [TranscriptSegment] = []
    var currentTokens: [TokenSegment] = []
    var debugInfo = DebugInfo()
    var calibrationProgress: Float = 0.0
    var isCalibrating = false

    // MARK: - Delegated thresholds (proxy to attributor)

    var speakerThreshold: Float {
        get { attributor?.speakerThreshold ?? 5.75 }
        set { attributor?.speakerThreshold = newValue }
    }
    var perTokenThreshold: Float {
        get { attributor?.perTokenThreshold ?? 5.75 }
        set { attributor?.perTokenThreshold = newValue }
    }
    var scoreWeight: Float {
        get { attributor?.scoreWeight ?? 0.5 }
        set { attributor?.scoreWeight = newValue }
    }
    var jawWeight: Float {
        get { attributor?.jawWeight ?? 1.5 }
        set { attributor?.jawWeight = newValue }
    }
    var jawVelocityWeight: Float {
        get { attributor?.jawVelocityWeight ?? 2.5 }
        set { attributor?.jawVelocityWeight = newValue }
    }
    var timeDeltaWeight: Float {
        get { attributor?.timeDeltaWeight ?? 0.5 }
        set { attributor?.timeDeltaWeight = newValue }
    }
    var contextWeight: Float {
        get { attributor?.contextWeight ?? 0.25 }
        set { attributor?.contextWeight = newValue }
    }
    var jawMargin: Double {
        get { attributor?.jawMargin ?? 0.1 }
        set { attributor?.jawMargin = newValue }
    }
    var noJawPenalty: Float {
        get { attributor?.noJawPenalty ?? 0.5 }
        set { attributor?.noJawPenalty = newValue }
    }
    var enableIncrementalLearning: Bool {
        get { attributor?.enableIncrementalLearning ?? true }
        set { attributor?.enableIncrementalLearning = newValue }
    }
    var learningThreshold: Float {
        get { attributor?.learningThreshold ?? 4.0 }
        set { attributor?.learningThreshold = newValue }
    }
    var learningRate: Float {
        get { attributor?.learningRate ?? 0.3 }
        set { attributor?.learningRate = newValue }
    }
    var learningCount: Int { attributor?.learningCount ?? 0 }
    var currentCalibrationSentence: Int { attributor?.currentCalibrationSentence ?? 0 }
    var calibrationSentences: [String] { attributor?.calibrationSentences ?? [] }

    // MARK: - Private

    private let engine: HumanStateEngine
    private var attributor: GazeSpeakerAttributor?
    private var audioStreamStartTime: Date?

    private let logFileURL: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("speaker_recognition_log.jsonl")
    }()

    // MARK: - Init

    nonisolated init(engine: HumanStateEngine) {
        self.engine = engine

        Task { @MainActor in
            self.attributor = GazeSpeakerAttributor()
            self.setupSTTListener()
            self.syncFromAttributor()
        }
    }

    private func syncFromAttributor() {
        // Sync initial state from attributor
        guard let attributor = attributor else { return }
        if attributor.hasEmbedding {
            phase = .live
            debugInfo.userEmbeddingStatus = "✅ 已加载 (\(attributor.embeddingCount) 个样本)"
        }
    }

    // MARK: - STT Listener

    private func setupSTTListener() {
        audioStreamStartTime = engine.sttManager.audioStreamStartTime

        engine.sttManager.onTokens = { [weak self] tokens, isFinal in
            guard let self = self else { return }
            guard self.phase == .live, !self.isCalibrating, !tokens.isEmpty else { return }

            Task { @MainActor in
                // Delegate to attributor
                guard let attributor = self.attributor else { return }
                let speakerTokens = attributor.processTokens(tokens, isFinal: isFinal)
                let newTokens = speakerTokens.map { TokenSegment(from: $0) }

                if isFinal {
                    self.buildFinalSegments(newTokens)
                    self.currentTokens = []
                } else {
                    self.buildStreamingTokens(newTokens)
                }
            }
        }
    }

    private func buildFinalSegments(_ newTokens: [TokenSegment]) {
        var currentGroup: [TokenSegment] = []
        var currentIsUser: Bool? = nil

        // Check merge with last segment
        if let lastSegment = transcriptSegments.last {
            let timeSinceLastSegment = Date().timeIntervalSince(lastSegment.timestamp)
            if timeSinceLastSegment < 1.0 {
                transcriptSegments.removeLast()
                currentGroup = lastSegment.tokens
                currentIsUser = lastSegment.tokens.last?.isUserSpeaker
            }
        }

        for token in newTokens {
            logTokenRecognition(token: token, isFinal: true)

            if currentIsUser == nil {
                currentIsUser = token.isUserSpeaker
                currentGroup.append(token)
            } else if currentIsUser == token.isUserSpeaker {
                currentGroup.append(token)
            } else {
                if !currentGroup.isEmpty {
                    transcriptSegments.append(TranscriptSegment(
                        tokens: currentGroup, isFinal: true, timestamp: Date()
                    ))
                    if transcriptSegments.count > 20 {
                        transcriptSegments.removeFirst(transcriptSegments.count - 20)
                    }
                }
                currentGroup = [token]
                currentIsUser = token.isUserSpeaker
            }
        }

        if !currentGroup.isEmpty {
            transcriptSegments.append(TranscriptSegment(
                tokens: currentGroup, isFinal: true, timestamp: Date()
            ))
            if transcriptSegments.count > 20 {
                transcriptSegments.removeFirst(transcriptSegments.count - 20)
            }
        }
    }

    private func buildStreamingTokens(_ newTokens: [TokenSegment]) {
        var groupedTokens: [TokenSegment] = []
        var currentGroup: [TokenSegment] = []
        var currentIsUser: Bool? = nil

        for token in newTokens {
            if currentIsUser == nil {
                currentIsUser = token.isUserSpeaker
                currentGroup.append(token)
            } else if currentIsUser == token.isUserSpeaker {
                currentGroup.append(token)
            } else {
                groupedTokens.append(contentsOf: currentGroup)
                currentGroup = [token]
                currentIsUser = token.isUserSpeaker
            }
        }
        groupedTokens.append(contentsOf: currentGroup)

        // v146: preserve prefix when iOS STT corrects earlier tokens
        if !currentTokens.isEmpty,
           let newFirst = groupedTokens.first,
           let oldFirst = currentTokens.first,
           newFirst.audioTime > oldFirst.audioTime {
            let prefix = currentTokens.filter { $0.audioTime < newFirst.audioTime }
            currentTokens = prefix + groupedTokens
        } else {
            currentTokens = groupedTokens
        }
    }

    // MARK: - Audio Processing (delegates to attributor)

    func processAudioBuffer(_ samples: [Float]) {
        // Update audio level for debug
        let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))
        debugInfo.audioLevel = 20 * log10(max(rms, 1e-10))

        let face = engine.humanState.face

        // Record sensor data if live
        if !isCalibrating, phase == .live,
           let startTime = engine.sttManager.audioStreamStartTime {
            let elapsed = Date().timeIntervalSince(startTime)
            attributor?.recordSensorData(face: face, audioStreamElapsed: elapsed)
        }

        // Delegate audio processing
        let elapsed: Double? = engine.sttManager.audioStreamStartTime.map {
            Date().timeIntervalSince($0)
        }
        attributor?.processAudioForEmbedding(samples, face: face, audioStreamElapsed: elapsed)

        // Sync debug info from attributor + face
        syncDebugInfo(face: face)
    }

    private func syncDebugInfo(face: FaceState) {
        debugInfo.isLookingAtScreen = face.isLookingAtScreen
        debugInfo.isHeadForward = face.headOrientation.isFacingForward
        debugInfo.gazeOnScreen = face.isLookingAtScreen ? 1.0 : 0.0
        debugInfo.headYaw = face.headYaw
        debugInfo.headPitch = face.headPitch
        debugInfo.faceDistance = face.distanceFromCamera
        debugInfo.speakerMatch = attributor?.speakerMatch ?? false
        debugInfo.speakerDistance = attributor?.speakerDistance ?? 1.0

        // Sync calibration state
        isCalibrating = attributor?.isCalibrating ?? false
        calibrationProgress = attributor?.calibrationProgress ?? 0

        // Sync phase
        switch attributor?.phase {
        case .calibration: phase = .calibration
        case .live: phase = .live
        case .none: break
        }

        // Update embedding status
        if attributor?.hasEmbedding == true {
            debugInfo.userEmbeddingStatus = "✅ 已标定 (\(attributor?.embeddingCount ?? 0) 个样本)"
        } else if isCalibrating {
            debugInfo.userEmbeddingStatus = "标定中 (\((attributor?.currentCalibrationSentence ?? 0) + 1)/\(attributor?.calibrationSentences.count ?? 0))..."
        }

        // Jaw delta/velocity from recent history
        if let startTime = engine.sttManager.audioStreamStartTime {
            let elapsed = Date().timeIntervalSince(startTime)
            // These are approximations for debug display
            debugInfo.currentJawDelta = face.jawOpen
            debugInfo.currentJawVelocity = 0 // Would need history, approximation OK for debug
        }
    }

    // MARK: - Calibration (delegates to attributor)

    func startCalibration() {
        attributor?.startCalibration()
        isCalibrating = true
        debugInfo.userEmbeddingStatus = "标定中 (1/\(calibrationSentences.count))..."
    }

    func startAdditionalCalibration() {
        attributor?.startAdditionalCalibration()
        isCalibrating = true
        debugInfo.userEmbeddingStatus = "追加标定中..."
    }

    func stopAdditionalCalibration() {
        attributor?.stopAdditionalCalibration()
        isCalibrating = false
        if attributor?.hasEmbedding == true {
            debugInfo.userEmbeddingStatus = "✅ 已标定 (\(attributor?.embeddingCount ?? 0) 个样本)"
        }
    }

    func resetToInitialEmbedding() {
        attributor?.resetToInitialEmbedding()
    }

    func reset() {
        attributor?.reset()
    }

    func deleteEmbedding() {
        attributor?.deleteEmbedding()
        phase = .calibration
        debugInfo.userEmbeddingStatus = "未标定"
    }

    func clearTranscript() {
        transcriptSegments = []
        currentTokens = []
        attributor?.reset()
    }

    // MARK: - Logging

    private func logTokenRecognition(token: TokenSegment, isFinal: Bool) {
        let logEntry: [String: Any] = [
            "timestamp": Date().timeIntervalSince1970,
            "text": token.text,
            "audioTime": token.audioTime,
            "score": token.score,
            "jawDelta": token.jawDelta,
            "jawVelocity": token.jawVelocity,
            "isUserSpeaker": token.isUserSpeaker,
            "isFinal": isFinal,
            "gazeOnScreen": token.gazeOnScreen,
            "headYaw": token.headYaw,
            "headPitch": token.headPitch,
            "faceDistance": token.faceDistance,
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: logEntry),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            if let fileHandle = try? FileHandle(forWritingTo: logFileURL) {
                fileHandle.seekToEndOfFile()
                if let data = (jsonString + "\n").data(using: .utf8) {
                    fileHandle.write(data)
                }
                try? fileHandle.close()
            } else {
                try? (jsonString + "\n").write(to: logFileURL, atomically: true, encoding: .utf8)
            }
        }
    }

    func getLogFileURL() -> URL { logFileURL }

    func clearLog() {
        try? FileManager.default.removeItem(at: logFileURL)
    }
}
