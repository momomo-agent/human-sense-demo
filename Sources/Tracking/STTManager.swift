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
    
    // --- Segment-based sentence splitting ---
    /// Index of the first segment belonging to the current active sentence.
    /// All segments before this index have been finalized into sentences.
    private var activeSentenceStartSegmentIndex: Int = 0
    /// Timestamp of the last finalized segment's end, for gap detection and resync.
    private var lastFinalizedSegmentEndTime: TimeInterval = 0
    /// End time of the most recent segment seen in any recognition result.
    private var lastSeenSegmentEndTime: TimeInterval = 0
    /// Minimum gap between segments to trigger a sentence break (seconds).
    private let sentenceGapThreshold: TimeInterval = 1.2
    
    /// Timer to finalize the trailing active sentence when recognition goes quiet.
    private var trailingTimer: Timer?
    private let trailingTimeout: TimeInterval = 2.0
    
    /// When trailing timeout finalizes a sentence mid-task, we need to resync
    /// activeSentenceStartSegmentIndex on the next recognition result.
    private var needsSegmentIndexResync: Bool = false
    
    /// Apple Speech has a ~60s per-task limit. We proactively restart before hitting it.
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
    }
    
    // MARK: - Recognition Task
    
    private func startRecognitionTask() {
        taskGeneration += 1
        let myGeneration = taskGeneration
        
        activeSentence = Sentence(text: "", startedLookingAtScreen: false, gazeSpans: [])
        lastCharCount = 0
        speechStartCaptured = false
        activeSentenceStartSegmentIndex = 0
        lastFinalizedSegmentEndTime = 0
        lastSeenSegmentEndTime = 0
        needsSegmentIndexResync = false
        
        taskStartTime = Date()
        taskDurationTimer?.invalidate()
        taskDurationTimer = Timer.scheduledTimer(withTimeInterval: maxTaskDuration, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.gracefulTaskRestart()
            }
        }
        
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.contextualStrings = ["看屏幕", "看别处", "说话", "识别"]
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
        
        guard !allSegments.isEmpty else { return }
        
        // If trailing timeout finalized a sentence mid-task, resync our segment index.
        // All segments whose end time <= lastFinalizedSegmentEndTime belong to finalized sentences.
        if needsSegmentIndexResync {
            needsSegmentIndexResync = false
            var newStart = 0
            for i in 0..<allSegments.count {
                let segEnd = allSegments[i].timestamp + allSegments[i].duration
                if segEnd <= lastFinalizedSegmentEndTime + 0.05 {
                    newStart = i + 1
                }
            }
            activeSentenceStartSegmentIndex = min(newStart, allSegments.count)
            lastCharCount = 0
            speechStartCaptured = false
        }
        
        // Reset trailing timer — we got new data
        trailingTimer?.invalidate()
        trailingTimer = Timer.scheduledTimer(withTimeInterval: trailingTimeout, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.handleTrailingTimeout()
            }
        }
        
        // Scan segments from activeSentenceStartSegmentIndex forward for time gaps.
        // When a gap >= threshold is found, everything before the gap becomes a finalized sentence,
        // and a new active sentence starts from the gap onward.
        
        var lastBreakIndex: Int? = nil
        
        // Check all segment boundaries within the active range
        let scanStart = max(activeSentenceStartSegmentIndex + 1, 1)
        // Guard: if Apple returned fewer segments than expected (e.g. after revision),
        // reset our tracking to avoid out-of-bounds.
        if activeSentenceStartSegmentIndex >= allSegments.count {
            activeSentenceStartSegmentIndex = 0
            lastCharCount = 0
            speechStartCaptured = false
        }
        guard scanStart < allSegments.count else {
            // Only one segment (or fewer than scanStart) — just update active sentence
            let activeText = buildTextFromSegmentRange(
                from: activeSentenceStartSegmentIndex,
                to: allSegments.count,
                segments: allSegments,
                fullText: transcription.formattedString
            )
            updateActiveSentenceText(activeText)
            rebuildSegments()
            return
        }
        for i in scanStart..<allSegments.count {
            let prev = allSegments[i - 1]
            let curr = allSegments[i]
            let prevEnd = prev.timestamp + prev.duration
            if curr.timestamp - prevEnd >= sentenceGapThreshold {
                // Found a gap — finalize everything up to (not including) i
                let sentenceText = buildTextFromSegmentRange(
                    from: activeSentenceStartSegmentIndex,
                    to: i,
                    segments: allSegments,
                    fullText: transcription.formattedString
                )
                
                if !sentenceText.isEmpty {
                    activeSentence?.text = sentenceText
                    ensureGazeSpansCoverText(sentenceText)
                    finalizeActiveSentence()
                }
                
                // New active sentence starts at i
                activeSentenceStartSegmentIndex = i
                lastFinalizedSegmentEndTime = prevEnd
                activeSentence = Sentence(text: "", startedLookingAtScreen: false, gazeSpans: [])
                lastCharCount = 0
                speechStartCaptured = false
                lastBreakIndex = i
            }
        }
        
        // Update active sentence with text from activeSentenceStartSegmentIndex to end
        let activeText = buildTextFromSegmentRange(
            from: activeSentenceStartSegmentIndex,
            to: allSegments.count,
            segments: allSegments,
            fullText: transcription.formattedString
        )
        updateActiveSentenceText(activeText)
        
        // Track the latest segment end time for trailing timeout resync
        if let lastSeg = allSegments.last {
            lastSeenSegmentEndTime = lastSeg.timestamp + lastSeg.duration
        }
        
        rebuildSegments()
    }
    
    /// Build text from a segment index range [from, to) using NSRange positions.
    private func buildTextFromSegmentRange(
        from: Int, to: Int,
        segments: [SFTranscriptionSegment],
        fullText: String
    ) -> String {
        guard from < to, from < segments.count else { return "" }
        let first = segments[from]
        let last = segments[min(to - 1, segments.count - 1)]
        let nsString = fullText as NSString
        let startLoc = first.substringRange.location
        let endLoc = last.substringRange.location + last.substringRange.length
        guard startLoc <= nsString.length && endLoc <= nsString.length && startLoc < endLoc else {
            return ""
        }
        return nsString.substring(with: NSRange(location: startLoc, length: endLoc - startLoc))
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
            // Apple revised text (charCount decreased or unchanged) — reset gaze spans to match
            if newCharCount != lastCharCount, activeSentence?.startedLookingAtScreen == true {
                ensureGazeSpansCoverText(newText)
            }
            lastCharCount = newCharCount
        }
    }
    
    /// Ensure gaze spans total charCount matches the text length.
    /// Called when Apple revises text or before finalizing.
    private func ensureGazeSpansCoverText(_ text: String) {
        guard var spans = activeSentence?.gazeSpans, !spans.isEmpty else { return }
        let totalChars = spans.reduce(0) { $0 + $1.charCount }
        let textCount = text.count
        if totalChars != textCount {
            // Adjust last span to make up the difference
            let diff = textCount - totalChars
            let lastIdx = spans.count - 1
            let newCount = max(0, spans[lastIdx].charCount + diff)
            spans[lastIdx] = GazeSpan(charCount: newCount, isToScreen: spans[lastIdx].isToScreen)
            // If last span went to 0, remove it
            if spans[lastIdx].charCount == 0 && spans.count > 1 {
                spans.removeLast()
            }
            activeSentence?.gazeSpans = spans
        }
    }
    
    /// Called when no new recognition results arrive for `trailingTimeout`.
    private func handleTrailingTimeout() {
        guard activeSentence != nil, !(activeSentence?.text.isEmpty ?? true) else { return }
        // Record the end time of what we're finalizing so resync can skip past it
        lastFinalizedSegmentEndTime = lastSeenSegmentEndTime
        finalizeActiveSentence()
        // Prepare for next sentence within the same task.
        activeSentence = Sentence(text: "", startedLookingAtScreen: false, gazeSpans: [])
        lastCharCount = 0
        speechStartCaptured = false
        needsSegmentIndexResync = true
    }
    
    /// Proactively restart the recognition task before Apple's ~60s limit.
    /// Creates the new request FIRST so no buffers are lost.
    private func gracefulTaskRestart() {
        taskDurationTimer?.invalidate()
        taskDurationTimer = nil
        trailingTimer?.invalidate()
        trailingTimer = nil
        
        // Finalize current sentence
        finalizeActiveSentence()
        
        // Kill old task
        taskGeneration += 1
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        
        // Immediately start new task — no delay, no buffer gap
        startRecognitionTask()
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
