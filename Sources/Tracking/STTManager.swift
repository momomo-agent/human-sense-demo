import Foundation
import Speech
import AVFoundation

@MainActor
class STTManager: NSObject, ObservableObject {
    @Published var segments: [SpeechSegment] = []
    @Published var isListening: Bool = false
    
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    // --- External inputs (set by Engine) ---
    var isLookingAtScreen: Bool = false
    var isSpeaking: Bool = false {
        didSet {
            if isSpeaking {
                speakingOutputEnabled = true
                speakingOffTimer?.invalidate()
                speakingOffTimer = nil
            } else {
                speakingOffTimer?.invalidate()
                speakingOffTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: false) { [weak self] _ in
                    Task { @MainActor in
                        self?.speakingOutputEnabled = false
                    }
                }
            }
        }
    }
    
    // --- Sentence model ---
    
    private struct Sentence {
        let id = UUID()
        var text: String
        let isToScreen: Bool
        let startedLookingAtScreen: Bool
    }
    
    private var sentences: [Sentence] = []
    private var activeSentence: Sentence?
    private let maxSentences = 20
    
    // --- Internal state ---
    private var speakingOutputEnabled: Bool = false
    private var speakingOffTimer: Timer?
    private var speechStartCaptured: Bool = false
    private var taskGeneration: Int = 0
    
    private var lastRecognitionTime: Date?
    private var silenceTimer: Timer?
    private let sentenceGapThreshold: TimeInterval = 1.5
    
    func captureSpeechStartState() {}
    
    func start() {
        guard let recognizer = speechRecognizer, recognizer.isAvailable else { return }
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            guard status == .authorized else { return }
            Task { @MainActor in
                self?.beginListening()
            }
        }
    }
    
    func stop() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        taskGeneration += 1
        
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        
        finalizeActiveSentence()
        isListening = false
    }
    
    func clearSegments() {
        sentences = []
        activeSentence = nil
        rebuildSegments()
    }
    
    // MARK: - Segment Output
    
    private func rebuildSegments() {
        guard speakingOutputEnabled else {
            segments = []
            return
        }
        
        var result: [SpeechSegment] = []
        
        for s in sentences {
            if !result.isEmpty {
                result.append(SpeechSegment(text: " ", isToScreen: s.isToScreen, sentenceStartedLookingAtScreen: s.startedLookingAtScreen))
            }
            result.append(SpeechSegment(id: s.id, text: s.text, isToScreen: s.isToScreen, sentenceStartedLookingAtScreen: s.startedLookingAtScreen))
        }
        
        if let active = activeSentence, !active.text.isEmpty {
            if !result.isEmpty {
                result.append(SpeechSegment(text: " ", isToScreen: active.isToScreen, sentenceStartedLookingAtScreen: active.startedLookingAtScreen))
            }
            result.append(SpeechSegment(id: active.id, text: active.text, isToScreen: active.isToScreen, sentenceStartedLookingAtScreen: active.startedLookingAtScreen))
        }
        
        segments = result
    }
    
    // MARK: - Sentence Lifecycle
    
    private func finalizeActiveSentence() {
        guard let active = activeSentence, !active.text.isEmpty else {
            activeSentence = nil
            return
        }
        sentences.append(active)
        if sentences.count > maxSentences {
            sentences.removeFirst()
        }
        activeSentence = nil
        rebuildSegments()
    }
    
    // MARK: - Audio Engine (start once, never restart)
    
    private func beginListening() {
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        guard recordingFormat.sampleRate > 0 && recordingFormat.channelCount > 0 else {
            print("STT: invalid audio format (simulator?), skipping")
            return
        }
        
        // Audio tap feeds whatever the current recognitionRequest is.
        // Tap installed once, never removed until stop().
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }
        
        audioEngine.prepare()
        try? audioEngine.start()
        isListening = true
        
        startRecognitionTask()
        startSilenceTimer()
    }
    
    // MARK: - Recognition Task (restartable without touching audio engine)
    
    private func startRecognitionTask() {
        taskGeneration += 1
        let myGeneration = taskGeneration
        
        // New sentence — gaze state locked at creation
        activeSentence = Sentence(
            text: "",
            isToScreen: isLookingAtScreen,
            startedLookingAtScreen: isLookingAtScreen
        )
        speechStartCaptured = false
        
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        recognitionRequest = request
        
        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self = self else { return }
            
            Task { @MainActor in
                guard myGeneration == self.taskGeneration else { return }
                
                if let result = result {
                    self.activeSentence?.text = result.bestTranscription.formattedString
                    self.lastRecognitionTime = Date()
                    
                    // Lock gaze on first real text if not yet captured
                    if !self.speechStartCaptured && !result.bestTranscription.formattedString.isEmpty {
                        self.speechStartCaptured = true
                    }
                    
                    if self.speakingOutputEnabled {
                        self.rebuildSegments()
                    }
                    
                    if result.isFinal {
                        self.finalizeActiveSentence()
                        // Start fresh task for next sentence (audio engine stays running)
                        self.startRecognitionTask()
                    }
                }
                
                if error != nil && !result.map(\.isFinal).isTrue {
                    // Error without isFinal — restart task
                    self.finalizeActiveSentence()
                    self.startRecognitionTask()
                }
            }
        }
    }
    
    // MARK: - Silence Detection
    
    private func startSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.3, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.checkSilence()
            }
        }
    }
    
    private func checkSilence() {
        guard let lastTime = lastRecognitionTime else { return }
        guard Date().timeIntervalSince(lastTime) >= sentenceGapThreshold else { return }
        guard activeSentence != nil, !(activeSentence?.text.isEmpty ?? true) else { return }
        
        // Silence detected — finalize sentence, swap to new task
        // No audio engine restart needed — just end the request and start a new one
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        lastRecognitionTime = nil
        
        finalizeActiveSentence()
        startRecognitionTask()
    }
}

// MARK: - Helpers

private extension Optional where Wrapped == Bool {
    var isTrue: Bool { self == true }
}
