import Foundation
import Speech
import AVFoundation
import SwiftUI

struct SpeechSegment: Identifiable {
    let id = UUID()
    let text: String
    let isToScreen: Bool
    let sentenceStartedLookingAtScreen: Bool  // Was looking at screen when sentence started
}

@MainActor
class STTManager: NSObject, ObservableObject {
    @Published var segments: [SpeechSegment] = []
    @Published var isListening: Bool = false
    
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    private var lastText: String = ""
    var isLookingAtScreen: Bool = false  // Set by external observer
    var isSpeaking: Bool = false {  // Set by external observer (mouth + sound)
        didSet {
            if isSpeaking {
                // Speaking started - immediately enable output
                speakingOutputEnabled = true
                speakingOffTimer?.invalidate()
                speakingOffTimer = nil
            } else {
                // Speaking stopped - delay disabling output by 1s to catch trailing words
                speakingOffTimer?.invalidate()
                speakingOffTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: false) { [weak self] _ in
                    Task { @MainActor in
                        self?.speakingOutputEnabled = false
                    }
                }
            }
        }
    }
    private var speakingOutputEnabled: Bool = false {
        didSet {
            if speakingOutputEnabled && !oldValue {
                // Just became enabled - flush any buffered text
                flushPendingText()
            }
        }
    }
    private var speakingOffTimer: Timer?
    private var pendingTextStart: Int = 0  // lastText index when speaking was off
    private var sentenceStartLookingAtScreen: Bool = false  // Captured at sentence start
    private var lastUpdateTime: Date = Date()  // Track when last text was received
    private let sentenceGapThreshold: TimeInterval = 1.5  // 1.5 seconds gap = new sentence
    private var speechStartCaptured: Bool = false  // Track if we captured speech start state
    
    func captureSpeechStartState() {
        // Called when user starts speaking (from activity state change)
        // This captures the state BEFORE STT recognizes any text
        if !speechStartCaptured {
            sentenceStartLookingAtScreen = isLookingAtScreen
            speechStartCaptured = true
        }
    }
    
    private func flushPendingText() {
        // Output text that was buffered while speakingOutputEnabled was false
        let currentText = lastText
        if currentText.count > pendingTextStart {
            let buffered = String(currentText.dropFirst(pendingTextStart))
            if !buffered.trimmingCharacters(in: .whitespaces).isEmpty {
                segments.append(SpeechSegment(
                    text: buffered,
                    isToScreen: isLookingAtScreen,
                    sentenceStartedLookingAtScreen: sentenceStartLookingAtScreen
                ))
            }
        }
        pendingTextStart = currentText.count
    }
    
    func start() {
        // Check if recognizer is available
        guard let recognizer = speechRecognizer, recognizer.isAvailable else {
            return
        }
        
        // Request authorization
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            guard status == .authorized else {
                return
            }
            
            Task { @MainActor in
                self?.startRecognition()
            }
        }
    }
    
    func stop() {
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        isListening = false
    }
    
    private func startRecognition() {
        // Cancel previous task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { 
            return 
        }
        recognitionRequest.shouldReportPartialResults = true
        
        // Start recognition task
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            
            if let result = result {
                Task { @MainActor in
                    let newText = result.bestTranscription.formattedString
                    let now = Date()
                    let timeSinceLastUpdate = now.timeIntervalSince(self.lastUpdateTime)
                    
                    // Detect new sentence: either first text OR gap > threshold
                    let isNewSentence = self.lastText.isEmpty || timeSinceLastUpdate > self.sentenceGapThreshold
                    
                    if isNewSentence && !newText.isEmpty {
                        // Reset capture flag for new sentence
                        self.speechStartCaptured = false
                        // Capture state for new sentence
                        self.sentenceStartLookingAtScreen = self.isLookingAtScreen
                    }
                    
                    // Only show text when user is speaking (with 1s trailing buffer)
                    if self.speakingOutputEnabled {
                        if isNewSentence && !newText.isEmpty && !self.segments.isEmpty {
                            self.segments.append(SpeechSegment(
                                text: " ",
                                isToScreen: self.isLookingAtScreen,
                                sentenceStartedLookingAtScreen: self.sentenceStartLookingAtScreen
                            ))
                        }
                        
                        if newText.count > self.lastText.count {
                            let addedText = String(newText.dropFirst(self.lastText.count))
                            if !addedText.trimmingCharacters(in: .whitespaces).isEmpty {
                                self.segments.append(SpeechSegment(
                                    text: addedText,
                                    isToScreen: self.isLookingAtScreen,
                                    sentenceStartedLookingAtScreen: self.sentenceStartLookingAtScreen
                                ))
                            }
                        }
                        self.pendingTextStart = newText.count
                    } else {
                        // Not speaking yet - just track position for later flush
                        // pendingTextStart stays where it was, so when speaking
                        // starts, flushPendingText() outputs the buffered text
                    }
                    
                    self.lastText = newText
                    self.lastUpdateTime = now
                    
                    // Reset for next sentence when isFinal
                    if result.isFinal {
                        self.lastText = ""
                        self.pendingTextStart = 0
                        self.sentenceStartLookingAtScreen = false
                        self.speechStartCaptured = false
                    }
                }
            }
            
            if error != nil || result?.isFinal == true {
                self.audioEngine.stop()
                self.audioEngine.inputNode.removeTap(onBus: 0)
                self.recognitionRequest = nil
                self.recognitionTask = nil
                
                // Restart recognition after a short delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    if self.isListening {
                        self.startRecognition()
                    }
                }
            }
        }
        
        // Configure audio session
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try? audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        // Start audio engine
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        try? audioEngine.start()
        isListening = true
    }
}
