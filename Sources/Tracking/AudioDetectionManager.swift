import Foundation
import AVFoundation
import Combine

@MainActor
class AudioDetectionManager: NSObject, ObservableObject {
    @Published var audioState = AudioState()
    
    private let audioEngine = AVAudioEngine()
    private let speechThreshold: Float = 0.0015
    private var silenceTimer: Timer?
    private let silenceDelay: TimeInterval = 0.5
    
    func start() {
        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        
        guard format.sampleRate > 0 && format.channelCount > 0 else {
            print("Audio: invalid input format (simulator?), skipping")
            return
        }
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let self = self else { return }
            let rms = self.calculateRMS(buffer: buffer)
            
            Task { @MainActor in
                self.audioState.volume = rms
                let aboveThreshold = rms > self.speechThreshold
                
                if aboveThreshold {
                    self.silenceTimer?.invalidate()
                    self.silenceTimer = nil
                    self.audioState.isSpeaking = true
                } else if self.audioState.isSpeaking && self.silenceTimer == nil {
                    // Delay turning off to avoid flicker during brief pauses
                    self.silenceTimer = Timer.scheduledTimer(withTimeInterval: self.silenceDelay, repeats: false) { [weak self] _ in
                        Task { @MainActor in
                            self?.audioState.isSpeaking = false
                            self?.silenceTimer = nil
                        }
                    }
                }
            }
        }
        
        do {
            try audioEngine.start()
        } catch {
            print("Audio engine failed to start: \(error)")
        }
    }
    
    func stop() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
    }
    
    private func calculateRMS(buffer: AVAudioPCMBuffer) -> Float {
        guard let channelData = buffer.floatChannelData else { return 0 }
        let channelDataValue = channelData.pointee
        let frameLength = Int(buffer.frameLength)
        
        var sum: Float = 0
        for i in 0..<frameLength {
            let sample = channelDataValue[i]
            sum += sample * sample
        }
        
        let rms = sqrt(sum / Float(frameLength))
        return rms
    }
}
