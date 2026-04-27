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
    private var embeddingExtractor: SimpleSpeakerEmbeddingExtractor?
    
    private let engine: HumanStateEngine
    private var audioBufferQueue: [[Float]] = []  // 缓存音频用于 embedding 提取
    private let bufferWindowSize: Int = 10  // 保留最近 10 个 buffer（约 1-2 秒）
    private var lastTranscriptUpdate: Date = Date()
    
    init(engine: HumanStateEngine) {
        self.engine = engine
        
        // 初始化 embedding extractor
        do {
            self.embeddingExtractor = try SimpleSpeakerEmbeddingExtractor()
        } catch {
            print("⚠️ Failed to load embedding extractor: \(error)")
        }
        
        // 监听 STT 转录结果
        setupSTTListener()
    }
    
    private func setupSTTListener() {
        // 监听 STTManager 的 segments 更新
        engine.sttManager.onTokens = { [weak self] tokens, isFinal in
            guard let self = self else { return }
            guard self.phase == .live else { return }
            guard self.debugInfo.speakerMatch else { return }  // 只有匹配用户时才显示
            
            // 提取文本
            let text = tokens.map { $0.text }.joined()
            guard !text.isEmpty else { return }
            
            // 避免重复添加
            let now = Date()
            if now.timeIntervalSince(self.lastTranscriptUpdate) > 1.0 {
                Task { @MainActor in
                    self.transcript.append(text)
                    self.lastTranscriptUpdate = now
                }
            }
        }
    }
    
    // MARK: - Audio Processing
    
    func processAudioBuffer(_ samples: [Float]) {
        // 更新音频电平
        let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))
        let db = 20 * log10(max(rms, 1e-10))
        debugInfo.audioLevel = db
        
        // 缓存音频
        audioBufferQueue.append(samples)
        if audioBufferQueue.count > bufferWindowSize {
            audioBufferQueue.removeFirst()
        }
        
        // 根据阶段处理
        switch phase {
        case .calibration:
            if isCalibrating {
                processCalibrationAudio(samples)
            }
        case .live:
            // 合并最近的 buffer 用于 embedding 提取
            let recentSamples = audioBufferQueue.flatMap { $0 }
            processLiveAudio(recentSamples)
        }
    }
    
    // MARK: - Calibration
    
    func startCalibration() {
        isCalibrating = true
        calibrationProgress = 0.0
        calibrationAudioBuffers = []
        calibrationStartTime = Date()
        debugInfo.userEmbeddingStatus = "标定中..."
    }
    
    private func processCalibrationAudio(_ samples: [Float]) {
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
        
        // 提取 embedding
        guard let extractor = embeddingExtractor else {
            debugInfo.userEmbeddingStatus = "❌ 提取器未加载"
            return
        }
        
        do {
            userEmbedding = try extractor.extract(from: allSamples)
            debugInfo.userEmbeddingStatus = "✅ 已标定 (\(allSamples.count / 16000) 秒)"
            phase = .live
            calibrationProgress = 1.0
        } catch {
            debugInfo.userEmbeddingStatus = "❌ 提取失败: \(error)"
            print("Embedding extraction failed: \(error)")
        }
    }
    
    // MARK: - Live Recognition
    
    private func processLiveAudio(_ samples: [Float]) {
        guard phase == .live else { return }
        guard let userEmb = userEmbedding else { return }
        guard let extractor = embeddingExtractor else { return }
        
        // 1. 检查 gaze
        let lookingAtScreen = engine.humanState.lookingAtScreen
        debugInfo.gazeStatus = lookingAtScreen ? "👁️ 看屏幕" : "👀 未看屏幕"
        
        // 2. 提取当前 embedding（需要至少 1 秒音频）
        guard samples.count >= 16000 else {
            debugInfo.speakerMatch = false
            return
        }
        
        let currentEmbedding: [Float]
        do {
            currentEmbedding = try extractor.extract(from: samples)
        } catch {
            // 静默失败，不打印（避免刷屏）
            debugInfo.speakerMatch = false
            return
        }
        
        // 3. 计算相似度
        let distance = cosineSimilarity(userEmb, currentEmbedding)
        debugInfo.speakerDistance = distance
        debugInfo.speakerMatch = distance < speakerThreshold
        
        // 4. STT 转录已通过 onTokens 回调自动处理
        // 只有 speakerMatch=true 时才会显示转录
    }
    
    func reset() {
        phase = .calibration
        transcript = []
        userEmbedding = nil
        calibrationProgress = 0.0
        isCalibrating = false
        calibrationAudioBuffers = []
        calibrationStartTime = nil
        audioBufferQueue = []
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
