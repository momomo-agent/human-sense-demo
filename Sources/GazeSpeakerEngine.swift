import SwiftUI
import AVFoundation
import FluidAudio
import HumanSenseKit

@MainActor
@Observable
class GazeSpeakerEngine {
    enum Phase {
        case calibration
        case live
    }

    struct DebugInfo {
        var isLookingAtScreen: Bool = false
        var isHeadForward: Bool = false
        var speakerMatch: Bool = false
        var speakerDistance: Float = 1.0
        var audioLevel: Float = -60.0
        var userEmbeddingStatus: String = "未标定"
    }

    struct TranscriptSegment: Identifiable {
        let id = UUID()
        let tokens: [TokenSegment]  // 保存原始 tokens
        let isFinal: Bool
        let timestamp: Date

        var text: String {
            tokens.map { $0.text }.joined()
        }

        var isUserSpeaker: Bool {
            // 如果大部分 token 是用户的，就认为是用户
            let userCount = tokens.filter { $0.isUserSpeaker }.count
            return userCount > tokens.count / 2
        }

        var score: Float {
            // 平均 score
            tokens.map { $0.score }.reduce(0, +) / Float(tokens.count)
        }

        var audioTime: Double {
            tokens.first?.audioTime ?? 0
        }
    }

    struct TokenSegment: Identifiable {
        let id = UUID()
        let text: String
        let isUserSpeaker: Bool
        let score: Float
        let audioTime: Double
        let jawDelta: Float  // 时间范围内嘴的变化幅度
    }

    var phase: Phase = .calibration
    var transcriptSegments: [TranscriptSegment] = []
    var currentTokens: [TokenSegment] = []  // 当前正在构建的句子
    var debugInfo = DebugInfo()
    var calibrationProgress: Float = 0.0
    var isCalibrating = false
    var speakerThreshold: Float = 0.7  // 可调节的阈值（finalScore 阈值）
    var jawWeight: Float = 2.0  // jaw 权重系数
    var jawMargin: Double = 0.1  // jaw 时间扩展（秒）

    // 增量学习参数
    var enableIncrementalLearning: Bool = false  // 是否启用增量学习
    var learningThreshold: Float = 0.3  // 学习触发阈值（finalScore < 此值才学习）
    var learningRate: Float = 0.1  // 学习率（新 embedding 的权重）
    var learningCount: Int = 0  // 已学习次数
    var lastLearningTime: Date?  // 最后学习时间
    private var initialEmbedding: [Float]?  // 保存初始标定的 embedding，用于回滚

    // 多句标定
    var currentCalibrationSentence: Int = 0
    let calibrationSentences = [
        "今天天气真不错",
        "我喜欢听音乐",
        "明天一起去吃饭",
        "这个想法很有趣",
        "请帮我打开窗户"
    ]

    private var tokenColorMap: [String: (isUser: Bool, score: Float)] = [:]  // token key -> (isUser, score)
    private var speakerHistory: [(timestamp: Double, distance: Float)] = []  // 记录 speaker 距离历史
    private var jawHistory: [(timestamp: Double, jawOpen: Float)] = []  // 记录 jaw 开合历史
    private var audioStreamStartTime: Date?  // 音频流开始时间

    private var userEmbedding: [Float]? {
        didSet {
            if let embedding = userEmbedding {
                saveEmbedding(embedding)
            }
        }
    }
    private var calibrationAudioBuffers: [[Float]] = []
    private var calibrationEmbeddings: [[Float]] = []  // 存储每句的 embedding
    private let calibrationDuration: TimeInterval = 3.0  // 每句 3 秒
    private var calibrationStartTime: Date?
    nonisolated(unsafe) private var embeddingExtractor: SimpleSpeakerEmbeddingExtractor?

    private let embeddingFileURL: URL = {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsPath.appendingPathComponent("user_speaker_embedding.json")
    }()

    private let engine: HumanStateEngine
    private var audioBufferQueue: [[Float]] = []  // 缓存音频用于 embedding 提取
    private let bufferWindowSize: Int = 10  // 保留最近 10 个 buffer（约 1-2 秒）

    nonisolated init(engine: HumanStateEngine) {
        self.engine = engine

        // 初始化 embedding extractor
        do {
            self.embeddingExtractor = try SimpleSpeakerEmbeddingExtractor()
        } catch {
            print("⚠️ Failed to load embedding extractor: \(error)")
        }

        // 监听 STT 转录结果
        Task { @MainActor in
            setupSTTListener()
            loadEmbedding()
        }
    }

    private func setupSTTListener() {
        // 获取音频流开始时间
        audioStreamStartTime = engine.sttManager.audioStreamStartTime

        // 监听 STTManager 的 segments 更新
        engine.sttManager.onTokens = { [weak self] tokens, isFinal in
            guard let self = self else { return }
            guard self.phase == .live else { return }
            guard !tokens.isEmpty else { return }

            Task { @MainActor in
                // 为每个 token 创建带颜色的 segment
                // 如果是 Stream 阶段且 token 时间跨度内 speaker 有变化，则拆分
                var newTokens: [TokenSegment] = []

                for token in tokens {
                    // 检查这个 token 的时间范围内 speaker 是否变化
                    let splitTokens = self.splitTokenBySpeakerChange(token, isFinal: isFinal)
                    newTokens.append(contentsOf: splitTokens)
                }

                if isFinal {
                    // 句子结束，按 speaker 分组成不同的 segment
                    var currentGroup: [TokenSegment] = []
                    var currentIsUser: Bool? = nil

                    for token in newTokens {
                        if currentIsUser == nil {
                            // 第一个 token
                            currentIsUser = token.isUserSpeaker
                            currentGroup.append(token)
                        } else if currentIsUser == token.isUserSpeaker {
                            // 同一个 speaker，继续累积
                            currentGroup.append(token)
                        } else {
                            // speaker 变化，保存当前组，开始新组
                            if !currentGroup.isEmpty {
                                let segment = TranscriptSegment(
                                    tokens: currentGroup,
                                    isFinal: true,
                                    timestamp: Date()
                                )
                                self.transcriptSegments.append(segment)
                            }
                            currentGroup = [token]
                            currentIsUser = token.isUserSpeaker
                        }
                    }

                    // 保存最后一组
                    if !currentGroup.isEmpty {
                        let segment = TranscriptSegment(
                            tokens: currentGroup,
                            isFinal: true,
                            timestamp: Date()
                        )
                        self.transcriptSegments.append(segment)
                    }

                    self.currentTokens = []
                    // 不清空 tokenColorMap，保持颜色锁定

                    // 增量学习：在 Final 阶段，检查用户说的话
                    if self.enableIncrementalLearning {
                        self.tryIncrementalLearningFromTokens(newTokens)
                    }
                } else {
                    // 句子进行中，也按 speaker 分组（用于实时显示）
                    var groupedTokens: [TokenSegment] = []
                    var currentGroup: [TokenSegment] = []
                    var currentIsUser: Bool? = nil

                    for token in newTokens {
                        if currentIsUser == nil {
                            currentIsUser = token.isUserSpeaker
                            currentGroup.append(token)
                        } else if currentIsUser == token.isUserSpeaker {
                            currentGroup.append(token)
                        } else {
                            // speaker 变化，保存当前组，开始新组
                            groupedTokens.append(contentsOf: currentGroup)
                            currentGroup = [token]
                            currentIsUser = token.isUserSpeaker
                        }
                    }

                    // 保存最后一组
                    groupedTokens.append(contentsOf: currentGroup)
                    self.currentTokens = groupedTokens
                }
            }
        }
    }

    // 根据音频时间查询 speaker 状态，返回 (是否用户, 距离分数)
    private func querySpeakerAtTime(_ audioTime: Double) -> (Bool, Float) {
        // 在历史记录中查找最接近的时间点
        guard !speakerHistory.isEmpty else {
            return (debugInfo.speakerMatch, debugInfo.speakerDistance)  // 没有历史，用当前状态
        }

        // 找到最接近的历史记录
        let closest = speakerHistory.min(by: { abs($0.timestamp - audioTime) < abs($1.timestamp - audioTime) })
        guard let record = closest else {
            return (debugInfo.speakerMatch, debugInfo.speakerDistance)
        }

        // 判断是否匹配
        let isUser = record.distance < speakerThreshold
        return (isUser, record.distance)
    }

    // 计算时间范围内 jaw 的变化幅度（最大值 - 最小值的绝对值）
    private func calculateJawDelta(startTime: Double, endTime: Double) -> Float {
        // 添加可调节的 margin
        let expandedStart = max(0, startTime - jawMargin)
        let expandedEnd = endTime + jawMargin

        let relevantJaw = jawHistory.filter {
            $0.timestamp >= expandedStart && $0.timestamp <= expandedEnd
        }

        guard !relevantJaw.isEmpty else { return 0.0 }

        let jawValues = relevantJaw.map { $0.jawOpen }
        let maxJaw = jawValues.max() ?? 0
        let minJaw = jawValues.min() ?? 0

        return abs(maxJaw - minJaw)
    }

    // 根据 token 时间范围内的 speaker 变化拆分 token
    private func splitTokenBySpeakerChange(_ token: SpeechToken, isFinal: Bool) -> [TokenSegment] {
        let tokenKey = token.text + "_\(token.startTime)_\(token.endTime)"

        // 只在 Final 阶段使用缓存
        if isFinal, let cached = self.tokenColorMap[tokenKey] {
            let jawDelta = calculateJawDelta(startTime: token.startTime, endTime: token.endTime)
            let finalScore = cached.score - jawWeight * jawDelta
            let isUser = finalScore < speakerThreshold
            return [TokenSegment(
                text: token.text,
                isUserSpeaker: isUser,
                score: cached.score,
                audioTime: token.startTime,
                jawDelta: jawDelta
            )]
        }

        // 查询 token 时间范围内的 speaker 历史
        let relevantHistory = speakerHistory.filter {
            $0.timestamp >= token.startTime && $0.timestamp <= token.endTime
        }

        // 如果没有历史记录或只有一个说话人，不拆分
        if relevantHistory.isEmpty {
            let result = querySpeakerAtTime(token.startTime)
            let jawDelta = calculateJawDelta(startTime: token.startTime, endTime: token.endTime)
            let finalScore = result.1 - jawWeight * jawDelta
            let isUser = finalScore < speakerThreshold
            let segment = TokenSegment(
                text: token.text,
                isUserSpeaker: isUser,
                score: result.1,
                audioTime: token.startTime,
                jawDelta: jawDelta
            )
            if isFinal {
                tokenColorMap[tokenKey] = (isUser: isUser, score: result.1)
            }
            return [segment]
        }

        // 检查时间范围内是否有 speaker 变化
        let speakerChanges = relevantHistory.map { $0.distance < speakerThreshold }
        let hasChange = speakerChanges.first != speakerChanges.last

        if !hasChange {
            // 没有变化，不拆分
            let result = querySpeakerAtTime(token.startTime)
            let jawDelta = calculateJawDelta(startTime: token.startTime, endTime: token.endTime)
            let finalScore = result.1 - jawWeight * jawDelta
            let isUser = finalScore < speakerThreshold
            let segment = TokenSegment(
                text: token.text,
                isUserSpeaker: isUser,
                score: result.1,
                audioTime: token.startTime,
                jawDelta: jawDelta
            )
            if isFinal {
                tokenColorMap[tokenKey] = (isUser: isUser, score: result.1)
            }
            return [segment]
        }

        // 有 speaker 变化
        if isFinal {
            // Final 阶段：不拆分，只用开始时间的 speaker
            let result = querySpeakerAtTime(token.startTime)
            let jawDelta = calculateJawDelta(startTime: token.startTime, endTime: token.endTime)
            let finalScore = result.1 - jawWeight * jawDelta
            let isUser = finalScore < speakerThreshold
            let segment = TokenSegment(
                text: token.text,
                isUserSpeaker: isUser,
                score: result.1,
                audioTime: token.startTime,
                jawDelta: jawDelta
            )
            tokenColorMap[tokenKey] = (isUser: isUser, score: result.1)
            return [segment]
        }

        // Stream 阶段且有 speaker 变化，按字符平均拆分
        let duration = token.endTime - token.startTime
        let charCount = token.text.count
        guard charCount > 1 else {
            let result = querySpeakerAtTime(token.startTime)
            let jawDelta = calculateJawDelta(startTime: token.startTime, endTime: token.endTime)
            let finalScore = result.1 - jawWeight * jawDelta
            let isUser = finalScore < speakerThreshold
            return [TokenSegment(
                text: token.text,
                isUserSpeaker: isUser,
                score: result.1,
                audioTime: token.startTime,
                jawDelta: jawDelta
            )]
        }

        let timePerChar = duration / Double(charCount)
        var segments: [TokenSegment] = []
        var currentSpeaker: Bool? = nil
        var currentText = ""
        var currentStartTime = token.startTime

        for (index, char) in token.text.enumerated() {
            let charTime = token.startTime + Double(index) * timePerChar
            let charEndTime = charTime + timePerChar
            let result = querySpeakerAtTime(charTime)
            let jawDelta = calculateJawDelta(startTime: charTime, endTime: charEndTime)
            let finalScore = result.1 - jawWeight * jawDelta
            let isUser = finalScore < speakerThreshold

            if currentSpeaker == nil {
                currentSpeaker = isUser
                currentText.append(char)
            } else if currentSpeaker == isUser {
                currentText.append(char)
            } else {
                // Speaker 变化，保存当前段
                let segmentEndTime = charTime
                let segmentJawDelta = calculateJawDelta(startTime: currentStartTime, endTime: segmentEndTime)
                segments.append(TokenSegment(
                    text: currentText,
                    isUserSpeaker: currentSpeaker!,
                    score: result.1,
                    audioTime: currentStartTime,
                    jawDelta: segmentJawDelta
                ))
                currentText = String(char)
                currentSpeaker = isUser
                currentStartTime = charTime
            }
        }

        // 保存最后一段
        if !currentText.isEmpty, let speaker = currentSpeaker {
            let result = querySpeakerAtTime(currentStartTime)
            let jawDelta = calculateJawDelta(startTime: currentStartTime, endTime: token.endTime)
            segments.append(TokenSegment(
                text: currentText,
                isUserSpeaker: speaker,
                score: result.1,
                audioTime: currentStartTime,
                jawDelta: jawDelta
            ))
        }

        return segments
    }

    // MARK: - Audio Processing

    func processAudioBuffer(_ samples: [Float]) {
        // 更新音频电平
        let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))
        let db = 20 * log10(max(rms, 1e-10))
        debugInfo.audioLevel = db

        // 记录 jaw 数据到历史（每次音频回调都记录，实现高频采样）
        if phase == .live, let startTime = engine.sttManager.audioStreamStartTime {
            let elapsed = Date().timeIntervalSince(startTime)
            let jawOpen = engine.humanState.face.jawOpen
            jawHistory.append((timestamp: elapsed, jawOpen: jawOpen))

            // 只保留最近 10 秒的历史
            let cutoff = elapsed - 10.0
            jawHistory.removeAll { $0.timestamp < cutoff }
        }

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
        calibrationEmbeddings = []
        currentCalibrationSentence = 0
        calibrationStartTime = Date()
        debugInfo.userEmbeddingStatus = "标定中 (1/\(calibrationSentences.count))..."
    }

    private func processCalibrationAudio(_ samples: [Float]) {
        guard isCalibrating else { return }
        guard let startTime = calibrationStartTime else { return }

        let elapsed = Date().timeIntervalSince(startTime)
        let totalProgress = (Float(currentCalibrationSentence) + Float(min(elapsed / calibrationDuration, 1.0))) / Float(calibrationSentences.count)
        calibrationProgress = totalProgress

        // 收集音频
        calibrationAudioBuffers.append(samples)

        // 每句 3 秒后完成
        if elapsed >= calibrationDuration {
            finishCurrentSentence()
        }
    }

    private func finishCurrentSentence() {
        // 合并当前句子的音频
        let allSamples = calibrationAudioBuffers.flatMap { $0 }

        // 提取 embedding
        guard let extractor = embeddingExtractor else {
            debugInfo.userEmbeddingStatus = "❌ 提取器未加载"
            isCalibrating = false
            return
        }

        do {
            let embedding = try extractor.extract(from: allSamples)
            calibrationEmbeddings.append(embedding)

            // 进入下一句
            currentCalibrationSentence += 1

            if currentCalibrationSentence < calibrationSentences.count {
                // 还有下一句，继续标定
                calibrationAudioBuffers = []
                calibrationStartTime = Date()
                debugInfo.userEmbeddingStatus = "标定中 (\(currentCalibrationSentence + 1)/\(calibrationSentences.count))..."
            } else {
                // 所有句子完成，计算平均 embedding
                finishCalibration()
            }
        } catch {
            debugInfo.userEmbeddingStatus = "❌ 提取失败: \(error)"
            print("Embedding extraction failed: \(error)")
            isCalibrating = false
        }
    }

    private func finishCalibration() {
        isCalibrating = false

        // 计算平均 embedding
        guard !calibrationEmbeddings.isEmpty else {
            debugInfo.userEmbeddingStatus = "❌ 无有效数据"
            return
        }

        let embeddingSize = calibrationEmbeddings[0].count
        var avgEmbedding = [Float](repeating: 0, count: embeddingSize)

        for embedding in calibrationEmbeddings {
            for i in 0..<embeddingSize {
                avgEmbedding[i] += embedding[i]
            }
        }

        for i in 0..<embeddingSize {
            avgEmbedding[i] /= Float(calibrationEmbeddings.count)
        }

        userEmbedding = avgEmbedding
        initialEmbedding = avgEmbedding  // 保存初始 embedding
        learningCount = 0  // 重置学习次数
        lastLearningTime = nil
        debugInfo.userEmbeddingStatus = "✅ 已标定 (\(calibrationEmbeddings.count) 句)"
        phase = .live
        calibrationProgress = 1.0
    }
    
    // MARK: - Live Recognition
    
    private func processLiveAudio(_ samples: [Float]) {
        guard phase == .live else { return }
        guard let userEmb = userEmbedding else { return }
        guard let extractor = embeddingExtractor else { return }

        // 1. 更新 gaze 和 head 状态
        let face = engine.humanState.face
        debugInfo.isLookingAtScreen = face.isLookingAtScreen
        debugInfo.isHeadForward = face.headOrientation.isFacingForward

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

        // 4. 记录到历史（用于时间对齐）
        if let startTime = engine.sttManager.audioStreamStartTime {
            let elapsed = Date().timeIntervalSince(startTime)
            speakerHistory.append((timestamp: elapsed, distance: distance))

            // 只保留最近 10 秒的历史
            let cutoff = elapsed - 10.0
            speakerHistory.removeAll { $0.timestamp < cutoff }
        }
    }

    // 增量学习（从 Final tokens 中学习）
    private func tryIncrementalLearningFromTokens(_ tokens: [TokenSegment]) {
        guard let userEmb = userEmbedding else { return }
        guard let extractor = embeddingExtractor else { return }

        // 筛选出用户说的 tokens，且 finalScore 足够低（高置信度）
        let userTokens = tokens.filter { token in
            let finalScore = token.score - jawWeight * token.jawDelta
            return token.isUserSpeaker && finalScore < learningThreshold && token.jawDelta > 0.05
        }

        guard !userTokens.isEmpty else { return }

        // 检查学习间隔（避免过于频繁）
        if let lastTime = lastLearningTime {
            let interval = Date().timeIntervalSince(lastTime)
            guard interval > 5.0 else { return }  // 至少间隔 5 秒
        }

        // 计算这些 tokens 的时间范围，提取对应的音频
        guard let minTime = userTokens.map({ $0.audioTime }).min(),
              let maxTime = userTokens.map({ $0.audioTime }).max() else { return }

        // 从 audioBufferQueue 中提取对应时间的音频（简化：使用最近的音频）
        let recentSamples = audioBufferQueue.flatMap { $0 }
        guard recentSamples.count >= 16000 else { return }  // 至少 1 秒

        // 提取 embedding
        let currentEmbedding: [Float]
        do {
            currentEmbedding = try extractor.extract(from: recentSamples)
        } catch {
            return
        }

        // 执行增量学习：加权平均
        let oldWeight = 1.0 - learningRate
        let newWeight = learningRate

        var updatedEmbedding = [Float](repeating: 0, count: userEmb.count)
        for i in 0..<userEmb.count {
            updatedEmbedding[i] = oldWeight * userEmb[i] + newWeight * currentEmbedding[i]
        }

        userEmbedding = updatedEmbedding
        learningCount += 1
        lastLearningTime = Date()

        print("✅ 增量学习 #\(learningCount): \(userTokens.count) tokens")
    }

    // 回滚到初始 embedding
    func resetToInitialEmbedding() {
        guard let initial = initialEmbedding else { return }
        userEmbedding = initial
        learningCount = 0
        lastLearningTime = nil
        print("🔄 已回滚到初始 embedding")
    }
    
    func reset() {
        phase = .calibration
        transcriptSegments = []
        currentTokens = []
        tokenColorMap = [:]
        speakerHistory = []
        jawHistory = []
        audioStreamStartTime = nil
        userEmbedding = nil
        initialEmbedding = nil
        calibrationProgress = 0.0
        isCalibrating = false
        calibrationAudioBuffers = []
        calibrationEmbeddings = []
        currentCalibrationSentence = 0
        calibrationStartTime = nil
        audioBufferQueue = []
        learningCount = 0
        lastLearningTime = nil
        debugInfo = DebugInfo()
    }

    func clearTranscript() {
        transcriptSegments = []
        currentTokens = []
        tokenColorMap = [:]
        speakerHistory = []
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

    // MARK: - Persistence

    private func saveEmbedding(_ embedding: [Float]) {
        do {
            let data = try JSONEncoder().encode(embedding)
            try data.write(to: embeddingFileURL)
            print("✅ Saved user embedding to \(embeddingFileURL.path)")
        } catch {
            print("⚠️ Failed to save embedding: \(error)")
        }
    }

    private func loadEmbedding() {
        guard FileManager.default.fileExists(atPath: embeddingFileURL.path) else {
            debugInfo.userEmbeddingStatus = "未标定"
            return
        }

        do {
            let data = try Data(contentsOf: embeddingFileURL)
            let embedding = try JSONDecoder().decode([Float].self, from: data)
            userEmbedding = embedding
            phase = .live
            debugInfo.userEmbeddingStatus = "✅ 已加载"
            print("✅ Loaded user embedding from \(embeddingFileURL.path)")
        } catch {
            print("⚠️ Failed to load embedding: \(error)")
            debugInfo.userEmbeddingStatus = "未标定"
        }
    }

    func deleteEmbedding() {
        try? FileManager.default.removeItem(at: embeddingFileURL)
        userEmbedding = nil
        debugInfo.userEmbeddingStatus = "未标定"
        print("🗑️ Deleted user embedding")
    }
}
