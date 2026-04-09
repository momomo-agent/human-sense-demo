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
    
    /// Each sentence is one object with a clear lifecycle.
    private struct Sentence {
        let id = UUID()
        var text: String
        var isToScreen: Bool
        var startedLookingAtScreen: Bool
    }
    
    /// Completed sentences — each was one recognition task's output.
    private var sentences: [Sentence] = []
    
    /// The sentence currently being recognized (nil if no active task).
    private var activeSentence: Sentence?
    
    private let maxSentences = 20
    
    // --- Internal state ---
    private var speakingOutputEnabled: Bool = false
    private var speakingOffTimer: Timer?
    private var speechStartCaptured: Bool = false
    
    /// Generation counter — callbacks from stale tasks are ignored.
    private var taskGeneration: Int = 0
    
    /// Silence-based sentence splitting
    private var lastRecognitionTime: Date?
    private var silenceTimer: Timer?
    private let sentenceGapThreshold: TimeInterval = 1.5
    
    func captureSpeechStartState() {
        // Captured at sentence creation time now, not externally
    }
    
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
        
        // Finalize active sentence if any
        finalizeActiveSentence()
        
        isListening = false
    }
    
    func clearSegments() {
        sentences = []
        activeSentence = nil
        rebuildSegments()
    }
    
    // MARK: - Segment Output
    
    /// Rebuild the published segments array from sentences + active sentence.
    /// Uses each Sentence's stable UUID so SwiftUI doesn't re-render completed sentences.
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
    
    /// Commit the active sentence to the completed list.
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
    
    // MARK: - Recognition
    
    private func beginListening() {
        startRecognitionTask()
        
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        guard recordingFormat.sampleRate > 0 && recordingFormat.channelCount > 0 else {
            print("STT: invalid audio format (simulator?), skipping")
            return
        }
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }
        
        audioEngine.prepare()
        try? audioEngine.start()
        isListening = true
        startSilenceTimer()
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
        
        // Silence gap detected — finalize current sentence, restart task for next sentence
        finalizeActiveSentence()
        
        // Kill current task and start fresh
        taskGeneration += 1
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        lastRecognitionTime = nil
        
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        restartAudioAndRecognition()
    }
    
    /// Start a new recognition task = start a new sentence.
    private func startRecognitionTask() {
        taskGeneration += 1
        let myGeneration = taskGeneration
        
        // New task = new sentence object. startedLookingAtScreen will be
        // captured on first actual speech text, not here.
        activeSentence = Sentence(
            text: "",
            isToScreen: isLookingAtScreen,
            startedLookingAtScreen: isLookingAtScreen
        )
        speechStartCaptured = false
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let request = recognitionRequest else { return }
        request.shouldReportPartialResults = true
        
        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self = self else { return }
            
            // Capture generation at callback creation
            let callbackGeneration = myGeneration
            
            if let result = result {
                Task { @MainActor in
                    // Drop callbacks from stale tasks
                    guard callbackGeneration == self.taskGeneration else { return }
                    
                    // The cumulative text IS this sentence's text — direct assignment
                    self.activeSentence?.text = result.bestTranscription.formattedString
                    // Keep isToScreen live — reflects state at time of speech
                    self.activeSentence?.isToScreen = self.isLookingAtScreen
                    self.lastRecognitionTime = Date()
                    
                    // Capture startedLookingAtScreen once on first real text
                    if !self.speechStartCaptured && !result.bestTranscription.formattedString.isEmpty {
                        self.activeSentence?.startedLookingAtScreen = self.isLookingAtScreen
                        self.speechStartCaptured = true
                    }
                    
                    if self.speakingOutputEnabled {
                        self.rebuildSegments()
                    }
                    
                    if result.isFinal {
                        self.finalizeActiveSentence()
                    }
                }
            }
            
            // Task ended — restart to begin next sentence
            // But only if this task is still current (checkSilence may have already restarted)
            if error != nil || result?.isFinal == true {
                Task { @MainActor in
                    guard callbackGeneration == self.taskGeneration else { return }
                    
                    self.audioEngine.stop()
                    self.audioEngine.inputNode.removeTap(onBus: 0)
                    self.recognitionRequest = nil
                    self.recognitionTask = nil
                    
                    try? await Task.sleep(for: .milliseconds(500))
                    guard self.isListening else { return }
                    
                    self.finalizeActiveSentence()
                    self.restartAudioAndRecognition()
                }
            }
        }
    }
    
    /// Restart audio engine + new recognition task after the previous one ended.
    private func restartAudioAndRecognition() {
        startRecognitionTask()
        
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        guard recordingFormat.sampleRate > 0 && recordingFormat.channelCount > 0 else { return }
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }
        
        audioEngine.prepare()
        try? audioEngine.start()
    }
}
