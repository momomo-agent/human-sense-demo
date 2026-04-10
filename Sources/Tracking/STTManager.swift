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
                speakingOutputEnabled = true
                speakingOffTimer?.invalidate()
                speakingOffTimer = nil
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
    private var speakingOutputEnabled: Bool = false
    private var speakingOffTimer: Timer?
    private var speechStartCaptured: Bool = false
    private var taskGeneration: Int = 0
    
    // --- Segment-based sentence splitting ---
    /// Number of segments already committed to finalized sentences in the current task.
    /// When Apple adds new segments after a pause, we detect the time gap to split.
    private var committedSegmentCount: Int = 0
    /// Text of segments already finalized as sentences within this task.
    private var committedText: String = ""
    /// Timestamp of the last segment's end (timestamp + duration) for gap detection.
    private var lastSegmentEndTime: TimeInterval = 0
    /// Minimum gap between segments to trigger a sentence break (seconds).
    private let sentenceGapThreshold: TimeInterval = 1.2
    
    /// Timer to finalize the trailing active sentence when recognition goes quiet.
    private var trailingTimer: Timer?
    private let trailingTimeout: TimeInterval = 2.0
    
    /// Apple Speech has a ~60s per-task limit. We track task start time and
    /// proactively restart before hitting it to avoid forced isFinal + lost audio.
    private var taskStartTime: Date?
    private let maxTaskDuration: TimeInterval = 50.0
    private var taskDurationTimer: Timer?
    
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
        trailingTimer?.invalidate()
        trailingTimer = nil
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
    
    // MARK: - Audio Engine (single shared instance)
    
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
        
        // Single tap feeds both STT recognition and audio level detection
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            self.recognitionRequest?.append(buffer)
            // Feed the same buffer to AudioDetectionManager for RMS
            Task { @MainActor in
                self.audioDetectionManager?.processBuffer(buffer)
            }
        }
        
        audioEngine.prepare()
        try? audioEngine.start()
        isListening = true
        
        startRecognitionTask()
    }
    
    // MARK: - Recognition Task
    
    private func startRecognitionTask() {
        taskGeneration += 1
        let myGeneration = taskGeneration
        
        activeSentence = Sentence(text: "", startedLookingAtScreen: false, gazeSpans: [])
        lastCharCount = 0
        speechStartCaptured = false
        committedSegmentCount = 0
        committedText = ""
        lastSegmentEndTime = 0
        
        taskStartTime = Date()
        taskDurationTimer?.invalidate()
        taskDurationTimer = Timer.scheduledTimer(withTimeInterval: maxTaskDuration, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.gracefulTaskRestart()
            }
        }
        
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        // Provide context hints for better Chinese recognition
        request.contextualStrings = ["看屏幕", "看别处", "说话", "识别"]
        // Use on-device recognition when available (lower latency, no network dependency)
        if #available(iOS 13, *) {
            request.requiresOnDeviceRecognition = speechRecognizer?.supportsOnDeviceRecognition ?? false
        }
        recognitionRequest = request
        
        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self = self else { return }
            
            Task { @MainActor in
                guard myGeneration == self.taskGeneration else { return }
                
                if let result = result {
                    self.handleRecognitionResult(result)
                    
                    if result.isFinal {
                        // Apple decided this task is done — finalize and restart
                        self.finalizeActiveSentence()
                        self.startRecognitionTask()
                    }
                }
                
                if error != nil && !(result?.isFinal ?? false) {
                    self.taskGeneration += 1
                    let gen = self.taskGeneration
                    self.finalizeActiveSentence()
                    if gen == self.taskGeneration {
                        self.startRecognitionTask()
                    }
                }
            }
        }
    }
    
    // MARK: - Segment-Based Sentence Splitting
    
    private func handleRecognitionResult(_ result: SFSpeechRecognitionResult) {
        let transcription = result.bestTranscription
        let allSegments = transcription.segments
        
        // Reset trailing timer — we got new data
        trailingTimer?.invalidate()
        trailingTimer = Timer.scheduledTimer(withTimeInterval: trailingTimeout, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.handleTrailingTimeout()
            }
        }
        
        // Check for sentence breaks by examining time gaps between segments.
        // Segments already committed (committedSegmentCount) belong to finalized sentences.
        // We scan from committedSegmentCount forward looking for gaps.
        
        var newSentenceBreakIndex: Int? = nil
        
        if committedSegmentCount > 0 && allSegments.count > committedSegmentCount {
            // Check gap between last committed segment and first new segment
            let firstNewSeg = allSegments[committedSegmentCount]
            if firstNewSeg.timestamp - lastSegmentEndTime >= sentenceGapThreshold {
                newSentenceBreakIndex = committedSegmentCount
            }
        }
        
        // Also scan within the uncommitted range for gaps
        if newSentenceBreakIndex == nil {
            for i in max(committedSegmentCount, 1)..<allSegments.count {
                let prev = allSegments[i - 1]
                let curr = allSegments[i]
                let prevEnd = prev.timestamp + prev.duration
                if curr.timestamp - prevEnd >= sentenceGapThreshold && i > committedSegmentCount {
                    newSentenceBreakIndex = i
                    break  // Take the first break
                }
            }
        }
        
        if let breakIdx = newSentenceBreakIndex {
            // Finalize everything before breakIdx as a sentence
            let priorSegments = allSegments[0..<breakIdx]
            if !priorSegments.isEmpty {
                // Build text from prior segments (more accurate than substring of formattedString)
                let priorText = buildTextFromSegments(Array(priorSegments), fullText: transcription.formattedString)
                
                if !priorText.isEmpty && priorText != committedText {
                    // Update active sentence with the finalized text, then commit
                    let sentenceText = extractNewText(fullText: priorText, committed: committedText)
                    if !sentenceText.isEmpty {
                        activeSentence?.text = sentenceText
                        updateGazeForFullText(sentenceText)
                        finalizeActiveSentence()
                    }
                }
            }
            
            // Update committed state
            committedSegmentCount = breakIdx
            let lastPrior = allSegments[breakIdx - 1]
            lastSegmentEndTime = lastPrior.timestamp + lastPrior.duration
            committedText = buildTextFromSegments(Array(allSegments[0..<breakIdx]), fullText: transcription.formattedString)
            
            // Start new active sentence with remaining segments
            let remainingText = extractNewText(fullText: transcription.formattedString, committed: committedText)
            activeSentence = Sentence(text: "", startedLookingAtScreen: false, gazeSpans: [])
            lastCharCount = 0
            speechStartCaptured = false
            
            if !remainingText.isEmpty {
                updateActiveSentenceText(remainingText)
            }
        } else {
            // No sentence break — update active sentence with text after committed portion
            let newText = extractNewText(fullText: transcription.formattedString, committed: committedText)
            updateActiveSentenceText(newText)
        }
        
        if speakingOutputEnabled {
            rebuildSegments()
        }
    }
    
    /// Build text from a range of segments using their NSRange positions in formattedString.
    private func buildTextFromSegments(_ segs: [SFTranscriptionSegment], fullText: String) -> String {
        guard let first = segs.first, let last = segs.last else { return "" }
        let nsString = fullText as NSString
        let startLoc = first.substringRange.location
        let endLoc = last.substringRange.location + last.substringRange.length
        guard startLoc < nsString.length && endLoc <= nsString.length else { return fullText }
        let range = NSRange(location: startLoc, length: endLoc - startLoc)
        return nsString.substring(with: range)
    }
    
    /// Extract the portion of fullText that comes after committedText.
    private func extractNewText(fullText: String, committed: String) -> String {
        if committed.isEmpty { return fullText }
        // formattedString always starts with committed text
        if fullText.hasPrefix(committed) {
            let remainder = String(fullText.dropFirst(committed.count))
            // Trim leading whitespace from the boundary
            return remainder.trimmingLeadingWhitespace()
        }
        // Fallback: Apple rewrote earlier text (rare but possible)
        return fullText
    }
    
    /// Update the active sentence text and gaze tracking.
    private func updateActiveSentenceText(_ newText: String) {
        let newCharCount = newText.count
        let addedChars = max(0, newCharCount - lastCharCount)
        
        activeSentence?.text = newText
        
        if !speechStartCaptured && !newText.isEmpty {
            let looking = gazeAtSpeechOnset
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
            lastCharCount = newCharCount
        }
    }
    
    /// Reset gaze spans to match the full text (used after sentence break).
    private func updateGazeForFullText(_ text: String) {
        guard activeSentence?.startedLookingAtScreen == true else { return }
        // Ensure gaze spans cover the full text length
        let totalChars = activeSentence?.gazeSpans.reduce(0) { $0 + $1.charCount } ?? 0
        let textCount = text.count
        if totalChars < textCount, var spans = activeSentence?.gazeSpans, !spans.isEmpty {
            spans[spans.count - 1] = GazeSpan(
                charCount: spans[spans.count - 1].charCount + (textCount - totalChars),
                isToScreen: spans[spans.count - 1].isToScreen
            )
            activeSentence?.gazeSpans = spans
        }
    }
    
    /// Called when no new recognition results arrive for `trailingTimeout`.
    /// Finalizes the current active sentence so it doesn't hang forever.
    private func handleTrailingTimeout() {
        guard activeSentence != nil, !(activeSentence?.text.isEmpty ?? true) else { return }
        
        // Commit the active sentence's segments
        committedText = (committedText.isEmpty ? "" : committedText + " ") + (activeSentence?.text ?? "")
        // We don't know exact segment count here, but the next result will recalculate
        
        finalizeActiveSentence()
        
        // Prepare for next sentence within the same task
        activeSentence = Sentence(text: "", startedLookingAtScreen: false, gazeSpans: [])
        lastCharCount = 0
        speechStartCaptured = false
    }
    
    /// Proactively restart the recognition task before Apple's ~60s limit.
    private func gracefulTaskRestart() {
        taskDurationTimer?.invalidate()
        taskDurationTimer = nil
        trailingTimer?.invalidate()
        trailingTimer = nil
        
        taskGeneration += 1
        
        // End audio gracefully (not cancel) so final results can flush
        recognitionRequest?.endAudio()
        // Give a brief moment for final result, then force restart
        let gen = taskGeneration
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            guard let self = self, gen == self.taskGeneration else { return }
            self.recognitionTask?.cancel()
            self.recognitionTask = nil
            self.recognitionRequest = nil
            self.finalizeActiveSentence()
            self.startRecognitionTask()
        }
    }
    
    /// Attribute newly added characters to the current gaze direction.
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
}

// MARK: - String Helpers

private extension String {
    func trimmingLeadingWhitespace() -> String {
        var idx = startIndex
        while idx < endIndex && self[idx].isWhitespace {
            idx = index(after: idx)
        }
        return String(self[idx...])
    }
}
