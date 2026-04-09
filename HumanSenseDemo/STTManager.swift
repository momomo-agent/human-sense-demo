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
        let colorSeed: Int  // Stable color — assigned once, never changes
    }
    
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    private var lastRecognitionTime: Date?
    private var silenceTimer: Timer?
    private var sentenceCounter = 0  // Monotonic counter for stable colors
    
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
            
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let recognitionRequest = recognitionRequest else { return }
            recognitionRequest.shouldReportPartialResults = true
            
            recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
                Task { @MainActor in
                    guard let self = self else { return }
                    
                    if let result = result {
                        let text = result.bestTranscription.formattedString
                        self.currentPartialText = text
                        self.lastRecognitionTime = Date()
                    }
                    
                    if error != nil {
                        self.stopEngine()
                    }
                }
            }
            
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
        stopEngine()
        
        // Finalize any remaining partial text
        finalizeSentence()
        
        recognizedSentences = []
        currentPartialText = ""
        sentenceCounter = 0
    }
    
    // MARK: - Private
    
    private func stopEngine() {
        silenceTimer?.invalidate()
        silenceTimer = nil
        
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        
        isListening = false
    }
    
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
            finalizeSentence()
            restartRecognition()
        }
    }
    
    /// Commit current partial text as a finalized sentence, then clear it.
    /// Because we restart recognition after each sentence, `currentPartialText`
    /// is always relative to the current recognition task — no cumulative drift.
    private func finalizeSentence() {
        let text = currentPartialText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        
        let sentence = RecognizedSentence(text: text, colorSeed: sentenceCounter)
        sentenceCounter += 1
        recognizedSentences.append(sentence)
        currentPartialText = ""
        
        // Keep last 20
        if recognizedSentences.count > 20 {
            recognizedSentences.removeFirst()
        }
    }
    
    /// Tear down the current recognition task and start a fresh one.
    /// This resets the cumulative `formattedString` so the next sentence
    /// starts from scratch — the simplest way to get clean sentence boundaries.
    private func restartRecognition() {
        lastRecognitionTime = nil
        
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        
        guard isListening else { return }
        
        do {
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let recognitionRequest = recognitionRequest else { return }
            recognitionRequest.shouldReportPartialResults = true
            
            recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
                Task { @MainActor in
                    guard let self = self else { return }
                    
                    if let result = result {
                        let text = result.bestTranscription.formattedString
                        self.currentPartialText = text
                        self.lastRecognitionTime = Date()
                    }
                    
                    if error != nil {
                        self.stopEngine()
                    }
                }
            }
            
            // Re-install tap only if needed (engine still running)
            if !audioEngine.isRunning {
                let inputNode = audioEngine.inputNode
                let recordingFormat = inputNode.outputFormat(forBus: 0)
                inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
                    self?.recognitionRequest?.append(buffer)
                }
                try audioEngine.start()
            }
        } catch {
            errorMessage = "Restart failed: \(error.localizedDescription)"
        }
    }
}
