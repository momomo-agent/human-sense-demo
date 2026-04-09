import Foundation
import Speech
import AVFoundation

@MainActor
class STTManager: ObservableObject {
    @Published var isListening = false
    @Published var currentPartialText: String = ""
    @Published var recognizedSentences: [RecognizedSentence] = []
    @Published var errorMessage: String?
    
    struct RecognizedSentence: Identifiable {
        let id = UUID()
        let text: String
        let colorSeed: Int
    }
    
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    private var lastRecognitionTime: Date?
    private var silenceTimer: Timer?
    private var sentenceCounter = 0
    
    /// Incremented on each restart. Callbacks from stale tasks are ignored.
    private var taskGeneration: Int = 0
    
    private let silenceThreshold: TimeInterval = 1.5
    
    init() {
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN"))
    }
    
    func requestAuthorization() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }
    
    func startListening() {
        guard !isListening else { return }
        
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            
            startRecognitionTask()
            
            let inputNode = audioEngine.inputNode
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
                self?.recognitionRequest?.append(buffer)
            }
            
            audioEngine.prepare()
            try audioEngine.start()
            
            isListening = true
            startSilenceTimer()
            
        } catch {
            errorMessage = "Failed to start: \(error.localizedDescription)"
        }
    }
    
    func stopListening() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        
        // Bump generation so any pending callbacks are ignored
        taskGeneration += 1
        
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        
        // Finalize remaining text
        commitSentence()
        
        recognizedSentences = []
        currentPartialText = ""
        sentenceCounter = 0
        isListening = false
    }
    
    // MARK: - Recognition Task Lifecycle
    
    private func startRecognitionTask() {
        taskGeneration += 1
        let myGeneration = taskGeneration
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let request = recognitionRequest else { return }
        request.shouldReportPartialResults = true
        
        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard let self = self else { return }
                // Ignore callbacks from a previous (stale) task
                guard myGeneration == self.taskGeneration else { return }
                
                if let result = result {
                    self.currentPartialText = result.bestTranscription.formattedString
                    self.lastRecognitionTime = Date()
                }
                
                if error != nil {
                    self.stopListening()
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
        
        if Date().timeIntervalSince(lastTime) >= silenceThreshold {
            commitSentence()
            restartRecognition()
        }
    }
    
    // MARK: - Sentence Management
    
    /// Save current partial text as a finalized sentence.
    private func commitSentence() {
        let text = currentPartialText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        
        recognizedSentences.append(RecognizedSentence(text: text, colorSeed: sentenceCounter))
        sentenceCounter += 1
        currentPartialText = ""
        
        if recognizedSentences.count > 20 {
            recognizedSentences.removeFirst()
        }
    }
    
    /// Kill the current recognition task and start a fresh one.
    /// The generation counter ensures stale callbacks from the dying task
    /// cannot overwrite `currentPartialText` after we've cleared it.
    private func restartRecognition() {
        lastRecognitionTime = nil
        
        // End old task (this may fire one last callback — ignored via generation check)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        
        guard isListening else { return }
        
        // Fresh task — cumulative formattedString starts from ""
        startRecognitionTask()
    }
}
