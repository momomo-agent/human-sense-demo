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
    
    // --- Internal state ---
    private var speakingOutputEnabled: Bool = false
    private var speakingOffTimer: Timer?
    private var sentenceStartLookingAtScreen: Bool = false
    private var speechStartCaptured: Bool = false
    private var lastUpdateTime: Date = Date()
    private let sentenceGapThreshold: TimeInterval = 1.5
    
    private var completedSentences: [CompletedSentence] = []
    private var currentSentenceText: String = ""
    private var lastRecognizedText: String = ""
    private let maxCompletedSentences = 20
    
    /// The cumulative formattedString at the point we last finalized a sentence.
    /// We subtract this prefix to get only the new (incremental) text.
    private var finalizedPrefix: String = ""
    
    func captureSpeechStartState() {
        if !speechStartCaptured {
            sentenceStartLookingAtScreen = isLookingAtScreen
            speechStartCaptured = true
        }
    }
    
    func start() {
        guard let recognizer = speechRecognizer, recognizer.isAvailable else { return }
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            guard status == .authorized else { return }
            Task { @MainActor in
                self?.startRecognition()
            }
        }
    }
    
    func stop() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        isListening = false
    }
    
    func clearSegments() {
        segments = []
        completedSentences = []
        currentSentenceText = ""
        lastRecognizedText = ""
        finalizedPrefix = ""
    }
    
    // MARK: - Incremental Text Extraction
    
    /// Given the full cumulative text from the recognizer, strip the
    /// already-finalized prefix and return only the new portion.
    private func extractIncremental(from cumulative: String) -> String {
        guard !finalizedPrefix.isEmpty, cumulative.hasPrefix(finalizedPrefix) else {
            return cumulative
        }
        return String(cumulative.dropFirst(finalizedPrefix.count))
            .trimmingCharacters(in: .whitespaces)
    }
    
    /// Compute the new finalized prefix: everything in cumulative except the trailing newText.
    private func trimmedCumulativePrefix(_ cumulative: String, excluding newText: String) -> String {
        if newText.isEmpty { return cumulative }
        // The prefix is everything before the incremental part
        let prefixEnd = cumulative.count - newText.count
        guard prefixEnd > 0 else { return cumulative }
        let idx = cumulative.index(cumulative.startIndex, offsetBy: min(prefixEnd, cumulative.count))
        return String(cumulative[..<idx]).trimmingCharacters(in: .whitespaces)
    }
    
    // MARK: - Private
    
    private struct CompletedSentence {
        let text: String
        let isToScreen: Bool
        let sentenceStartedLookingAtScreen: Bool
    }
    
    private func rebuildSegments() {
        var newSegments: [SpeechSegment] = []
        
        for sentence in completedSentences {
            if !newSegments.isEmpty {
                newSegments.append(SpeechSegment(
                    text: " ",
                    isToScreen: sentence.isToScreen,
                    sentenceStartedLookingAtScreen: sentence.sentenceStartedLookingAtScreen
                ))
            }
            newSegments.append(SpeechSegment(
                text: sentence.text,
                isToScreen: sentence.isToScreen,
                sentenceStartedLookingAtScreen: sentence.sentenceStartedLookingAtScreen
            ))
        }
        
        if !currentSentenceText.isEmpty {
            if !newSegments.isEmpty {
                newSegments.append(SpeechSegment(
                    text: " ",
                    isToScreen: isLookingAtScreen,
                    sentenceStartedLookingAtScreen: sentenceStartLookingAtScreen
                ))
            }
            newSegments.append(SpeechSegment(
                text: currentSentenceText,
                isToScreen: isLookingAtScreen,
                sentenceStartedLookingAtScreen: sentenceStartLookingAtScreen
            ))
        }
        
        segments = newSegments
    }
    
    private func startRecognition() {
        recognitionTask?.cancel()
        recognitionTask = nil
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { return }
        recognitionRequest.shouldReportPartialResults = true
        
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            
            if let result = result {
                Task { @MainActor in
                    let cumulativeText = result.bestTranscription.formattedString
                    let now = Date()
                    let timeSinceLastUpdate = now.timeIntervalSince(self.lastUpdateTime)
                    
                    // Strip the already-finalized prefix to get only new text
                    let newText = self.extractIncremental(from: cumulativeText)
                    
                    let isNewSentence = self.lastRecognizedText.isEmpty || timeSinceLastUpdate > self.sentenceGapThreshold
                    
                    if isNewSentence && !newText.isEmpty {
                        if !self.currentSentenceText.isEmpty && self.speakingOutputEnabled {
                            self.completedSentences.append(CompletedSentence(
                                text: self.currentSentenceText,
                                isToScreen: self.isLookingAtScreen,
                                sentenceStartedLookingAtScreen: self.sentenceStartLookingAtScreen
                            ))
                            if self.completedSentences.count > self.maxCompletedSentences {
                                self.completedSentences = Array(self.completedSentences.suffix(self.maxCompletedSentences))
                            }
                            // Advance the prefix past the sentence we just finalized
                            self.finalizedPrefix = self.trimmedCumulativePrefix(cumulativeText, excluding: newText)
                        }
                        self.currentSentenceText = ""
                        self.speechStartCaptured = false
                        self.sentenceStartLookingAtScreen = self.isLookingAtScreen
                    }
                    
                    if self.speakingOutputEnabled {
                        self.currentSentenceText = newText
                        self.rebuildSegments()
                    }
                    
                    self.lastRecognizedText = newText
                    self.lastUpdateTime = now
                    
                    if result.isFinal {
                        if !self.currentSentenceText.isEmpty && self.speakingOutputEnabled {
                            self.completedSentences.append(CompletedSentence(
                                text: self.currentSentenceText,
                                isToScreen: self.isLookingAtScreen,
                                sentenceStartedLookingAtScreen: self.sentenceStartLookingAtScreen
                            ))
                        }
                        self.currentSentenceText = ""
                        self.lastRecognizedText = ""
                        self.finalizedPrefix = ""
                        self.speechStartCaptured = false
                    }
                }
            }
            
            if error != nil || result?.isFinal == true {
                self.audioEngine.stop()
                self.audioEngine.inputNode.removeTap(onBus: 0)
                self.recognitionRequest = nil
                self.recognitionTask = nil
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    if self.isListening {
                        self.startRecognition()
                    }
                }
            }
        }
        
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        guard recordingFormat.sampleRate > 0 && recordingFormat.channelCount > 0 else {
            print("STT: invalid audio format (simulator?), skipping")
            return
        }
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        try? audioEngine.start()
        isListening = true
    }
}
