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
    private var sentenceStartLookingAtScreen: Bool = false  // Captured at sentence start
    private var lastUpdateTime: Date = Date()  // Track when last text was received
    private let sentenceGapThreshold: TimeInterval = 1.5  // 1.5 seconds gap = new sentence
    
    func start() {
        print("STT: start() called")
        
        // Check if recognizer is available
        guard let recognizer = speechRecognizer, recognizer.isAvailable else {
            print("STT: Speech recognizer not available")
            return
        }
        
        print("STT: Speech recognizer available, requesting authorization...")
        
        // Request authorization
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            print("STT Authorization status: \(status.rawValue)")
            guard status == .authorized else {
                print("Speech recognition not authorized")
                return
            }
            
            Task { @MainActor in
                print("Starting STT recognition...")
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
        print("STT: Starting recognition task...")
        
        // Cancel previous task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { 
            print("STT: Failed to create recognition request")
            return 
        }
        recognitionRequest.shouldReportPartialResults = true
        
        print("STT: Created recognition request")
        
        // Start recognition task
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            
            if let error = error {
                print("STT: Recognition error: \(error)")
            }
            
            if let result = result {
                Task { @MainActor in
                    let newText = result.bestTranscription.formattedString
                    let now = Date()
                    let timeSinceLastUpdate = now.timeIntervalSince(self.lastUpdateTime)
                    
                    print("STT: Recognized text: \(newText), time since last: \(timeSinceLastUpdate)s")
                    
                    // Detect new sentence: either first text OR gap > threshold
                    let isNewSentence = self.lastText.isEmpty || timeSinceLastUpdate > self.sentenceGapThreshold
                    
                    if isNewSentence && !newText.isEmpty {
                        self.sentenceStartLookingAtScreen = self.isLookingAtScreen
                        print("STT: New sentence detected, looking at screen: \(self.sentenceStartLookingAtScreen)")
                        
                        // Add space before new sentence (except first one)
                        if !self.segments.isEmpty {
                            self.segments.append(SpeechSegment(
                                text: " ",
                                isToScreen: self.isLookingAtScreen,
                                sentenceStartedLookingAtScreen: self.sentenceStartLookingAtScreen
                            ))
                        }
                    }
                    
                    // Check if new text was added
                    if newText.count > self.lastText.count {
                        let addedText = String(newText.dropFirst(self.lastText.count))
                        if !addedText.trimmingCharacters(in: .whitespaces).isEmpty {
                            print("STT: Adding segment '\(addedText)', isToScreen: \(self.isLookingAtScreen), sentenceStart: \(self.sentenceStartLookingAtScreen)")
                            self.segments.append(SpeechSegment(
                                text: addedText,
                                isToScreen: self.isLookingAtScreen,
                                sentenceStartedLookingAtScreen: self.sentenceStartLookingAtScreen
                            ))
                        }
                    }
                    
                    self.lastText = newText
                    self.lastUpdateTime = now
                    
                    // Reset for next sentence when isFinal
                    if result.isFinal {
                        print("STT: Sentence ended (isFinal), resetting state")
                        self.lastText = ""
                        self.sentenceStartLookingAtScreen = false
                    }
                }
            }
            
            if error != nil || result?.isFinal == true {
                print("STT: Recognition ended (error: \(error != nil), isFinal: \(result?.isFinal == true))")
                self.audioEngine.stop()
                self.audioEngine.inputNode.removeTap(onBus: 0)
                self.recognitionRequest = nil
                self.recognitionTask = nil
                
                // Restart recognition after a short delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    if self.isListening {
                        print("STT: Restarting recognition...")
                        self.startRecognition()
                    }
                }
            }
        }
        
        print("STT: Recognition task started")
        
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
        print("STT: Audio engine started, isListening = true")
    }
}
