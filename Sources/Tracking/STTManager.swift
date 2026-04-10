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
    
    /// AudioDetectionManager receives buffers from our shared tap.
    weak var audioDetectionManager: AudioDetectionManager?
    
    // --- External inputs (set by Engine) ---
    var isLookingAtScreen: Bool = false
    var isSpeaking: Bool = false {
        didSet {
            if isSpeaking {
                gazeAtSpeechOnset = isLookingAtScreen
            }
        }
    }
    
    private var gazeAtSpeechOnset: Bool = false
    
    // --- Sentence model ---
    
    private struct GazeSpan {
        let id = UUID()
        var charCount: Int
        let isToScreen: Bool
    }
    
    private struct Sentence {
        let id = UUID()
        var text: String
        var startedLookingAtScreen: Bool
        var gazeSpans: [GazeSpan]
    }
    
    private var sentences: [Sentence] = []
    private var activeSentence: Sentence?
    private let maxSentences = 20
    private var lastCharCount: Int = 0
    
    // --- Internal state ---
    private var speechStartCaptured: Bool = false
    private var taskGeneration: Int = 0
    
    // --- Silence detection ---
    private var lastRecognitionTime: Date?
    private var silenceTimer: Timer?
    private let sentenceGapThreshold: TimeInterval = 1.5
    
    /// Apple Speech ~60s per-task limit.
    private var taskDurationTimer: Timer?
    private let maxTaskDuration: TimeInterval = 50.0
    
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
        taskDurationTimer?.invalidate()
        taskDurationTimer = nil
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
                result.append(SpeechSegment(
                    id: s.id,
                    text: s.text,
                    isToScreen: false,
                    sentenceStartedLookingAtScreen: false
                ))
            } else {
                var offset = s.text.startIndex
                for (i, span) in s.gazeSpans.enumerated() {
                    let end = s.text.index(offset, offsetBy: span.charCount, limitedBy: s.text.endIndex) ?? s.text.endIndex
                    let spanText = String(s.text[offset..<end])
                    if !spanText.isEmpty {
                        result.append(SpeechSegment(
                            id: i == 0 ? s.id : span.id,
                            text: spanText,
                            isToScreen: span.isToScreen,
                            sentenceStartedLookingAtScreen: true
                        ))
                    }
                    offset = end
                }
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
        
        let texts = result.map { "\($0.text)(\($0.isToScreen ? "Y" : "B"))" }.joined(separator: " | ")
        print("[STT] REBUILD: \(sentences.count) finalized + \(activeSentence?.text.isEmpty == false ? 1 : 0) active → \(result.count) segments: \(texts)")
        
        segments = result
    }
    
    // MARK: - Sentence Lifecycle
    
    private func finalizeActiveSentence() {
        guard let active = activeSentence, !active.text.isEmpty else {
            activeSentence = nil
            return
        }
        print("[STT] FINALIZE: \"\(active.text)\" → sentences[\(sentences.count)]")
        sentences.append(active)
        if sentences.count > maxSentences {
            print("[STT] TRIM: removing oldest sentence, count was \(sentences.count)")
            sentences.removeFirst()
        }
        activeSentence = nil
        rebuildSegments()
    }
    
    // MARK: - Audio Engine
    
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
        
        // Single tap: feeds current recognitionRequest + AudioDetectionManager.
        // recognitionRequest pointer is swapped atomically in splitSentence()
        // before endAudio, so no buffer is ever lost.
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            self.recognitionRequest?.append(buffer)
            Task { @MainActor in
                self.audioDetectionManager?.processBuffer(buffer)
            }
        }
        
        audioEngine.prepare()
        try? audioEngine.start()
        isListening = true
        
        startRecognitionTask()
        startSilenceTimer()
    }
    
    // MARK: - Recognition Task
    
    /// Create a new SFSpeechAudioBufferRecognitionRequest with our standard config.
    private func makeRequest() -> SFSpeechAudioBufferRecognitionRequest {
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.contextualStrings = ["看屏幕", "看别处", "说话", "识别"]
        if #available(iOS 13, *) {
            request.requiresOnDeviceRecognition = speechRecognizer?.supportsOnDeviceRecognition ?? false
        }
        return request
    }
    
    /// Bind a recognition task to a request, routing results to handleResult.
    private func bindTask(to request: SFSpeechAudioBufferRecognitionRequest, generation: Int) {
        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self = self else { return }
            Task { @MainActor in
                guard generation == self.taskGeneration else {
                    if let result = result {
                        print("[STT] STALE gen=\(generation) current=\(self.taskGeneration) text=\"\(result.bestTranscription.formattedString)\" isFinal=\(result.isFinal)")
                    }
                    if let error = error {
                        print("[STT] STALE ERROR gen=\(generation) current=\(self.taskGeneration) error=\(error.localizedDescription)")
                    }
                    return
                }
                
                if let result = result {
                    self.handleResult(result)
                    if result.isFinal {
                        print("[STT] isFinal gen=\(generation) text=\"\(result.bestTranscription.formattedString)\"")
                        self.finalizeActiveSentence()
                        self.startRecognitionTask()
                    }
                }
                
                if error != nil && !(result?.isFinal ?? false) {
                    print("[STT] ERROR gen=\(generation) current=\(self.taskGeneration) error=\(error!.localizedDescription)")
                    self.taskGeneration += 1
                    let gen = self.taskGeneration
                    self.finalizeActiveSentence()
                    if gen == self.taskGeneration {
                        print("[STT] RESTART after error, new gen=\(gen + 1)")
                        self.startRecognitionTask()
                    } else {
                        print("[STT] SKIP restart: gen changed \(gen) → \(self.taskGeneration)")
                    }
                }
            }
        }
    }
    
    /// Start a fresh recognition task. One task = one sentence.
    private func startRecognitionTask() {
        taskGeneration += 1
        let gen = taskGeneration
        print("[STT] START TASK gen=\(gen), sentences=\(sentences.count)")
        
        resetActiveSentence()
        resetTaskDurationTimer()
        
        let request = makeRequest()
        recognitionRequest = request
        bindTask(to: request, generation: gen)
        if recognitionTask == nil {
            print("[STT] ⚠️ recognitionTask is nil! recognizer available=\(speechRecognizer?.isAvailable ?? false)")
        }
    }
    
    /// Finalize current sentence and seamlessly start a new task.
    /// Swaps recognitionRequest BEFORE endAudio so the audio tap never has a gap.
    private func splitSentence() {
        guard activeSentence != nil, !(activeSentence?.text.isEmpty ?? true) else { return }
        
        print("[STT] SPLIT: finalizing \"\(activeSentence?.text ?? "")\" and starting new task")
        
        let oldRequest = recognitionRequest
        let oldTask = recognitionTask
        
        // Finalize with current partial result
        finalizeActiveSentence()
        
        // Bump generation — old task callbacks become no-ops
        taskGeneration += 1
        let gen = taskGeneration
        
        // Create new request and swap pointer — tap immediately writes here
        let newRequest = makeRequest()
        recognitionRequest = newRequest
        
        // Now safe to end old request
        oldRequest?.endAudio()
        oldTask?.cancel()
        
        resetActiveSentence()
        resetTaskDurationTimer()
        bindTask(to: newRequest, generation: gen)
    }
    
    private func resetActiveSentence() {
        activeSentence = Sentence(text: "", startedLookingAtScreen: false, gazeSpans: [])
        lastCharCount = 0
        speechStartCaptured = false
        lastRecognitionTime = nil
    }
    
    private func resetTaskDurationTimer() {
        taskDurationTimer?.invalidate()
        taskDurationTimer = Timer.scheduledTimer(withTimeInterval: maxTaskDuration, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.splitSentence()
            }
        }
    }
    
    // MARK: - Recognition Result Handling
    
    private func handleResult(_ result: SFSpeechRecognitionResult) {
        let newText = result.bestTranscription.formattedString
        let newCharCount = newText.count
        let addedChars = max(0, newCharCount - lastCharCount)
        
        activeSentence?.text = newText
        lastRecognitionTime = Date()
        
        // Lock gaze on first recognized text.
        // Only use gazeAtSpeechOnset if user is actually speaking (isSpeaking=true).
        // Otherwise default to not-looking (blue) to avoid stale gaze state.
        if !speechStartCaptured && !newText.isEmpty {
            let looking = isSpeaking ? gazeAtSpeechOnset : false
            activeSentence?.startedLookingAtScreen = looking
            if looking {
                activeSentence?.gazeSpans = [GazeSpan(charCount: newCharCount, isToScreen: isLookingAtScreen)]
            }
            lastCharCount = newCharCount
            speechStartCaptured = true
        } else if addedChars > 0, activeSentence?.startedLookingAtScreen == true {
            updateGazeSpans(addedChars: addedChars)
            lastCharCount = newCharCount
        } else {
            if newCharCount != lastCharCount, activeSentence?.startedLookingAtScreen == true {
                ensureGazeSpansCoverText(newText)
            }
            lastCharCount = newCharCount
        }
        
        rebuildSegments()
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
        splitSentence()
    }
    
    // MARK: - Gaze Helpers
    
    private func updateGazeSpans(addedChars: Int) {
        guard var spans = activeSentence?.gazeSpans, !spans.isEmpty else { return }
        
        let lastSpan = spans[spans.count - 1]
        if lastSpan.isToScreen == isLookingAtScreen {
            spans[spans.count - 1] = GazeSpan(
                charCount: lastSpan.charCount + addedChars,
                isToScreen: lastSpan.isToScreen
            )
        } else {
            spans.append(GazeSpan(charCount: addedChars, isToScreen: isLookingAtScreen))
        }
        
        activeSentence?.gazeSpans = spans
    }
    
    private func ensureGazeSpansCoverText(_ text: String) {
        guard var spans = activeSentence?.gazeSpans, !spans.isEmpty else { return }
        let totalChars = spans.reduce(0) { $0 + $1.charCount }
        let textCount = text.count
        if totalChars != textCount {
            let lastIdx = spans.count - 1
            let newCount = max(0, spans[lastIdx].charCount + (textCount - totalChars))
            spans[lastIdx] = GazeSpan(charCount: newCount, isToScreen: spans[lastIdx].isToScreen)
            if spans[lastIdx].charCount == 0 && spans.count > 1 {
                spans.removeLast()
            }
            activeSentence?.gazeSpans = spans
        }
    }
}
