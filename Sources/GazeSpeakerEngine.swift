import SwiftUI
import AVFoundation
import FluidAudio
import HumanSenseKit

@Observable
class GazeSpeakerEngine {
    enum Phase {
        case calibration
        case live
    }
    
    struct DebugInfo {
        var gazeStatus: String = "未检测"
        var speakerMatch: Bool = false
        var speakerDistance: Float = 1.0
        var audioLevel: Float = -60.0
        var userEmbeddingStatus: String = "未标定"
    }
    
    var phase: Phase = .calibration
    var transcript: [String] = []
    var debugInfo = DebugInfo()
    var calibrationProgress: Float = 0.0
    var isCalibrating = false
    
    private var userEmbedding: [Float]?
    private let speakerThreshold: Float = 0.65
    private var calibrationAudioBuffers: [[Float]] = []
    private let calibrationDuration: TimeInterval = 5.0
    private var calibrationStartTime: Date?
    
    private let engine: HumanStateEngine
    
    init(engine: HumanStateEngine) {
        self.engine = engine
    }
    
    // MARK: - Calibration
    
    func startCalibration() {
        isCalibrating = true
        calibrationProgress = 0.0
        calibrationAudioBuffers = []
        calibrationStartTime = Date()
        debugInfo.userEmbeddingStatus = "标定中..."
    }
    
    func processCalibrationAudio(_ samples: [Float]) {
        guard isCalibrating else { return }
        guard let startTime = calibrationStartTime else { return }
        
        let elapsed = Date().timeIntervalSince(startTime)
        calibrationProgress = Float(min(elapsed / calibrationDuration, 1.0))
        
        // 收集音频
        calibrationAudioBuffers.append(samples)
        
        // 5 秒后完成标定
        if elapsed >= calibrationDuration {
            finishCalibration()
        }
    }
    
    private func finishCalibration() {
        isCalibrating = false
        
        // 合并所有音频
        let allSamples = calibrationAudioBuffers.flatMap { $0 }
        
        // TODO: 提取 embedding（需要 FluidAudio API）
        // 暂时用随机向量模拟
        userEmbedding = (0..<192).map { _ in Float.random(in: -1...1) }
        
        debugInfo.userEmbeddingStatus = "已标定 (\(allSamples.count / 16000) 秒)"
        phase = .live
        calibrationProgress = 1.0
    }
    
    // MARK: - Live Recognition
    
    func processLiveAudio(_ samples: [Float]) {
        guard phase == .live else { return }
        guard let userEmb = userEmbedding else { return }
        
        // 1. 检查 gaze
        let lookingAtScreen = engine.humanState.lookingAtScreen
        debugInfo.gazeStatus = lookingAtScreen ? "👁️ 看屏幕" : "👀 未看屏幕"
        guard lookingAtScreen else { return }
        
        // 2. 提取当前 embedding
        // TODO: 实际提取（需要 FluidAudio API）
        let currentEmbedding = (0..<192).map { _ in Float.random(in: -1...1) }
        
        // 3. 计算相似度
        let distance = cosineSimilarity(userEmb, currentEmbedding)
        debugInfo.speakerDistance = distance
        debugInfo.speakerMatch = distance < speakerThreshold
        
        guard distance < speakerThreshold else { return }
        
        // 4. 转录（使用 HumanSenseKit 的 STT）
        // TODO: 实际转录
        // 暂时模拟
        if samples.count > 16000 { // 至少 1 秒
            let mockText = "测试转录 \(Date().timeIntervalSince1970.truncatingRemainder(dividingBy: 100))"
            transcript.append(mockText)
        }
    }
    
    func updateAudioLevel(_ level: Float) {
        debugInfo.audioLevel = level
    }
    
    func reset() {
        phase = .calibration
        transcript = []
        userEmbedding = nil
        calibrationProgress = 0.0
        isCalibrating = false
        calibrationAudioBuffers = []
        calibrationStartTime = nil
        debugInfo = DebugInfo()
    }
    
    func clearTranscript() {
        transcript = []
    }
    
    // MARK: - Helpers
    
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 1.0 }
        
        let dotProduct = zip(a, b).map(*).reduce(0, +)
        let normA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let normB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        
        guard normA > 0, normB > 0 else { return 1.0 }
        
        // 返回距离（1 - similarity）
        return 1.0 - (dotProduct / (normA * normB))
    }
}
