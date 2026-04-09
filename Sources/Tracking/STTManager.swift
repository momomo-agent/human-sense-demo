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
                // Capture gaze at the moment of speech onset — real-time, no STT delay
                gazeAtSpeechOnset = isLookingAtScreen
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
    
    /// Gaze state captured at the instant isSpeaking becomes true.
    /// Used to determine sentence color instead of STT callback time.
    private var gazeAtSpeechOnset: Bool = false
    
    // --- Sentence model ---
    
    /// A gaze span within a sentence: a run of text with the same gaze direction.
    private struct GazeSpan {
        let id = UUID()
        var charCount: Int       // number of characters in this span
        let isToScreen: Bool
    }
    
    private struct Sentence {
        let id = UUID()
        var text: String
        var startedLookingAtScreen: Bool  // determined on first recognized text, not task creation
        var gazeSpans: [GazeSpan]
    }
    
    private var sentences: [Sentence] = []
    private var activeSentence: Sentence?
    private let maxSentences = 20
    
    /// Track the last known char count so we can attribute new chars to current gaze
    private var lastCharCount: Int = 0
    
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
        // Don't clear segments when speaking stops — keep showing last state.
        // Only skip updating while not speaking.
        guard speakingOutputEnabled else { return }
        
        var result: [SpeechSegment] = []
        
        func appendSentence(_ s: Sentence) {
            if !result.isEmpty {
                result.append(SpeechSegment(
                    text: " ",
                    isToScreen: false,
                    sentenceStartedLookingAtScreen: s.startedLookingAtScreen
                ))
            }
            
            if !s.startedLookingAtScreen {
                // P0: whole sentence is blue
                result.append(SpeechSegment(
                    id: s.id,
                    text: s.text,
                    isToScreen: false,
                    sentenceStartedLookingAtScreen: false
                ))
            } else {
                // P1: split by gaze spans — yellow (looking) / orange (not looking)
                var offset = s.text.startIndex
                for (i, span) in s.gazeSpans.enumerated() {
                    let end = s.text.index(offset, offsetBy: span.charCount, limitedBy: s.text.endIndex) ?? s.text.endIndex
                    let spanText = String(s.text[offset..<end])
                    if !spanText.isEmpty {
                        // Use span's own stable id
                        result.append(SpeechSegment(
                            id: i == 0 ? s.id : span.id,
                            text: spanText,
                            isToScreen: span.isToScreen,
                            sentenceStartedLookingAtScreen: true
                        ))
                    }
                    offset = end
                }
                // Any remaining text (edge case: spans charCount sum < text.count)
                if offset < s.text.endIndex {
                    let remaining = String(s.text[offset...])
                    if !remaining.isEmpty {
                        result.append(SpeechSegment(
                            text: remaining,
                            isToScreen: s.gazeSpans.last?.isToScreen ?? true,
                            sentenceStartedLookingAtScreen: true
                        ))
                    }
                }
            }
        }
        
        for s in sentences { appendSentence(s) }
        if let active = activeSentence, !active.text.isEmpty { appendSentence(active) }
        
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
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }
        
        audioEngine.prepare()
        try? audioEngine.start()
        isListening = true
        
        startRecognitionTask()
        startSilenceTimer()
    }
    
    // MARK: - Recognition Task
    
    private func startRecognitionTask() {
        taskGeneration += 1
        let myGeneration = taskGeneration
        
        // Sentence created with placeholder gaze — will be locked on first recognized text
        activeSentence = Sentence(
            text: "",
            startedLookingAtScreen: false,
            gazeSpans: []
        )
        lastCharCount = 0
        speechStartCaptured = false
        
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        recognitionRequest = request
        
        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self = self else { return }
            
            Task { @MainActor in
                guard myGeneration == self.taskGeneration else { return }
                
                if let result = result {
                    let newText = result.bestTranscription.formattedString
                    let newCharCount = newText.count
                    let addedChars = max(0, newCharCount - self.lastCharCount)
                    
                    self.activeSentence?.text = newText
                    self.lastRecognitionTime = Date()
                    
                    // Lock gaze direction on first recognized text,
                    // using the gaze captured at speech onset (isSpeaking=true moment)
                    if !self.speechStartCaptured && !newText.isEmpty {
                        let looking = self.gazeAtSpeechOnset
                        self.activeSentence?.startedLookingAtScreen = looking
                        if looking {
                            self.activeSentence?.gazeSpans = [GazeSpan(charCount: newCharCount, isToScreen: self.isLookingAtScreen)]
                        }
                        self.lastCharCount = newCharCount
                        self.speechStartCaptured = true
                    } else if addedChars > 0, self.activeSentence?.startedLookingAtScreen == true {
                        // Track gaze spans for sentences that started looking at screen
                        self.updateGazeSpans(addedChars: addedChars)
                        self.lastCharCount = newCharCount
                    } else {
                        self.lastCharCount = newCharCount
                    }
                    
                    if self.speakingOutputEnabled {
                        self.rebuildSegments()
                    }
                    
                    if result.isFinal {
                        self.finalizeActiveSentence()
                        self.startRecognitionTask()
                    }
                }
                
                if error != nil && !(result?.isFinal ?? false) {
                    self.finalizeActiveSentence()
                    self.startRecognitionTask()
                }
            }
        }
    }
    
    /// Attribute newly added characters to the current gaze direction.
    private func updateGazeSpans(addedChars: Int) {
        guard var spans = activeSentence?.gazeSpans, !spans.isEmpty else { return }
        
        let lastSpan = spans[spans.count - 1]
        if lastSpan.isToScreen == isLookingAtScreen {
            // Same direction — extend current span
            spans[spans.count - 1] = GazeSpan(
                charCount: lastSpan.charCount + addedChars,
                isToScreen: lastSpan.isToScreen
            )
        } else {
            // Direction changed — new span
            spans.append(GazeSpan(charCount: addedChars, isToScreen: isLookingAtScreen))
        }
        
        activeSentence?.gazeSpans = spans
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
        
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        lastRecognitionTime = nil
        
        finalizeActiveSentence()
        startRecognitionTask()
    }
}
