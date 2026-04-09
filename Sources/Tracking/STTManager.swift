import Foundation
import Speech
import AVFoundation
import SwiftUI

struct SpeechSegment: Identifiable {
    let id = UUID()
    let text: String
    let isToScreen: Bool
    var isPartial: Bool = false  // true = still recognizing
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
    private var sentenceStartLooking: Bool = false  // Captured at sentence start
    private var currentSentence: String = ""  // Accumulated text for current sentence
    
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
                    
                    // Capture isLookingAtScreen at sentence start
                    if self.currentSentence.isEmpty {
                        self.sentenceStartLooking = self.isLookingAtScreen
                    }
                    
                    // Update current sentence
                    self.currentSentence = newText
                }
            }
            
            if error != nil || result?.isFinal == true {
                Task { @MainActor in
                    // Create segment for the complete sentence
                    if !self.currentSentence.trimmingCharacters(in: .whitespaces).isEmpty {
                        self.segments.append(SpeechSegment(
                            text: self.currentSentence,
                            isToScreen: self.sentenceStartLooking
                        ))
                    }
                    self.currentSentence = ""
                    self.sentenceStartLooking = false
                }
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
