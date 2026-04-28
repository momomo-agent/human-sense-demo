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
        var currentJawDelta: Float = 0.0
        var currentJawVelocity: Float = 0.0
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
        let jawVelocity: Float  // 嘴的变化速度（delta/时间）
    }

    var phase: Phase = .calibration
    var transcriptSegments: [TranscriptSegment] = []
    var currentTokens: [TokenSegment] = []  // 当前正在构建的句子
    var debugInfo = DebugInfo()
    var calibrationProgress: Float = 0.0
    var isCalibrating = false
    var speakerThreshold: Float = 0.7  // 可调节的阈值（finalScore 阈值）
    var jawWeight: Float = 0.2  // jaw delta 权重系数（优化后：0.2，原 1.0）
    var jawVelocityWeight: Float = 0.2  // jaw velocity 权重系数（优化后：0.2，原 1.0）
    var jawMargin: Double = 0.1  // jaw 时间扩展（秒）
    var noJawPenalty: Float = 0.5  // 嘴不动的惩罚值

    // 增量学习参数
    var enableIncrementalLearning: Bool = true  // 是否启用增量学习
    var learningThreshold: Float = 0.5  // 学习触发阈值（finalScore < 此值才学习）
    var learningRate: Float = 0.3  // 学习率（新 embedding 的权重）
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

    private var userEmbeddings: [[Float]] = [] {
        didSet {
            if !userEmbeddings.isEmpty {
                saveEmbeddings(userEmbeddings)
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

    private let logFileURL: URL = {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsPath.appendingPathComponent("speaker_recognition_log.jsonl")
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
            guard !self.isCalibrating else { return }  // 标定期间不处理 STT
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
                    
                    // 检查是否需要合并到最后一个 segment（避免长句子被分成多个 segment）
                    var shouldMergeWithLast = false
                    if let lastSegment = self.transcriptSegments.last {
                        let timeSinceLastSegment = Date().timeIntervalSince(lastSegment.timestamp)
                        // 如果距离上次 Final < 1 秒，认为是同一句话
                        if timeSinceLastSegment < 1.0 {
                            shouldMergeWithLast = true
                            // 从最后一个 segment 开始
                            self.transcriptSegments.removeLast()
                            currentGroup = lastSegment.tokens
                            currentIsUser = lastSegment.tokens.last?.isUserSpeaker
                        }
                    }

                    for token in newTokens {
                        // 记录每个 token 到日志
                        self.logTokenRecognition(token: token, isFinal: true)

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
                                
                                // 滑动窗口：只保留最近 20 个 segment（句子）
                                if self.transcriptSegments.count > 20 {
                                    self.transcriptSegments.removeFirst(self.transcriptSegments.count - 20)
                                }
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
                        
                        // 滑动窗口：只保留最近 20 个 segment（句子）
                        if self.transcriptSegments.count > 20 {
                            self.transcriptSegments.removeFirst(self.transcriptSegments.count - 20)
                        }
                    }

                    // 不清空 currentTokens，让它累积，避免长句子被截断
                    // self.currentTokens = []
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

    // 计算最终得分
    // score: 声纹距离（越小越像用户）
    // jawDelta: 嘴变化幅度（越大越可能是用户）
    // jawVelocity: 嘴变化速度（越大越可能是用户）
    // 返回: finalScore（越小越可能是用户）
    private func calculateFinalScore(score: Float, jawDelta: Float, jawVelocity: Float) -> Float {
        // 规则 1: 如果 score 本身就很高，直接判为非用户
        if score > 0.75 {
            return 1.5  // 远大于 threshold (0.7)，确保被判为非用户
        }
        
        // 规则 2: 如果嘴完全不动，直接判为非用户
        if jawDelta < 0.02 && jawVelocity < 0.1 {
            return 1.5  // 远大于 threshold (0.7)，确保被判为非用户
        }
        
        // 否则使用乘法权重
        var jawFactor: Float = 1.0 - jawWeight * jawDelta
        var velocityFactor: Float = 1.0 - jawVelocityWeight * jawVelocity

        // 防止因子变成负数或过小
        jawFactor = max(0.1, jawFactor)
        velocityFactor = max(0.1, velocityFactor)

        let finalScore = score * jawFactor * velocityFactor
        return finalScore
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

    private func calculateJawVelocity(startTime: Double, endTime: Double) -> Float {
        let expandedStart = max(0, startTime - jawMargin)
        let expandedEnd = endTime + jawMargin

        let relevantJaw = jawHistory.filter {
            $0.timestamp >= expandedStart && $0.timestamp <= expandedEnd
        }

        guard relevantJaw.count >= 2 else { return 0.0 }

        // 计算相邻采样点之间的速度，取最大值
        var maxVelocity: Float = 0.0
        for i in 1..<relevantJaw.count {
            let dt = relevantJaw[i].timestamp - relevantJaw[i-1].timestamp
            guard dt > 0 else { continue }
            let dJaw = abs(relevantJaw[i].jawOpen - relevantJaw[i-1].jawOpen)
            let velocity = dJaw / Float(dt)
            maxVelocity = max(maxVelocity, velocity)
        }

        return maxVelocity
    }

    // 根据 token 时间范围内的 speaker 变化拆分 token
    private func splitTokenBySpeakerChange(_ token: SpeechToken, isFinal: Bool) -> [TokenSegment] {
        let tokenKey = token.text + "_\(token.startTime)_\(token.endTime)"

        // 只在 Final 阶段使用缓存
        if isFinal, let cached = self.tokenColorMap[tokenKey] {
            let jawDelta = calculateJawDelta(startTime: token.startTime, endTime: token.endTime)
            let jawVelocity = calculateJawVelocity(startTime: token.startTime, endTime: token.endTime)
            let finalScore = calculateFinalScore(score: cached.score, jawDelta: jawDelta, jawVelocity: jawVelocity)
            let isUser = finalScore < speakerThreshold
            return [TokenSegment(
                text: token.text,
                isUserSpeaker: isUser,
                score: cached.score,
                audioTime: token.startTime,
                jawDelta: jawDelta,
                jawVelocity: jawVelocity
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
            let jawVelocity = calculateJawVelocity(startTime: token.startTime, endTime: token.endTime)
            let finalScore = calculateFinalScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity)
            let isUser = finalScore < speakerThreshold
            let segment = TokenSegment(
                text: token.text,
                isUserSpeaker: isUser,
                score: result.1,
                audioTime: token.startTime,
                jawDelta: jawDelta,
                jawVelocity: jawVelocity
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
            let jawVelocity = calculateJawVelocity(startTime: token.startTime, endTime: token.endTime)
            let finalScore = calculateFinalScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity)
            let isUser = finalScore < speakerThreshold
            let segment = TokenSegment(
                text: token.text,
                isUserSpeaker: isUser,
                score: result.1,
                audioTime: token.startTime,
                jawDelta: jawDelta,
                jawVelocity: jawVelocity
            )
            if isFinal {
                tokenColorMap[tokenKey] = (isUser: isUser, score: result.1)
            }
            return [segment]
        }

        // 有 speaker 变化，按字符平均拆分（Final 和 Stream 都拆分）
        let duration = token.endTime - token.startTime
        let charCount = token.text.count
        guard charCount > 1 else {
            let result = querySpeakerAtTime(token.startTime)
            let jawDelta = calculateJawDelta(startTime: token.startTime, endTime: token.endTime)
            let jawVelocity = calculateJawVelocity(startTime: token.startTime, endTime: token.endTime)
            let finalScore = calculateFinalScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity)
            let isUser = finalScore < speakerThreshold
            return [TokenSegment(
                text: token.text,
                isUserSpeaker: isUser,
                score: result.1,
                audioTime: token.startTime,
                jawDelta: jawDelta,
                jawVelocity: jawVelocity
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
            let jawVelocity = calculateJawVelocity(startTime: charTime, endTime: charEndTime)
            let finalScore = calculateFinalScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity)
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
                let segmentJawVelocity = calculateJawVelocity(startTime: currentStartTime, endTime: segmentEndTime)
                segments.append(TokenSegment(
                    text: currentText,
                    isUserSpeaker: currentSpeaker!,
                    score: result.1,
                    audioTime: currentStartTime,
                    jawDelta: segmentJawDelta,
                    jawVelocity: segmentJawVelocity
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
            let jawVelocity = calculateJawVelocity(startTime: currentStartTime, endTime: token.endTime)
            segments.append(TokenSegment(
                text: currentText,
                isUserSpeaker: speaker,
                score: result.1,
                audioTime: currentStartTime,
                jawDelta: jawDelta,
                jawVelocity: jawVelocity
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

        // 标定期间不记录 jaw 历史，不进行 speaker 检测
        if isCalibrating {
            processCalibrationAudio(samples)
            return
        }

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
            // 已在上面处理
            break
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

    // 追加标定：自由录音模式
    func startAdditionalCalibration() {
        isCalibrating = true
        calibrationProgress = 0.0
        calibrationAudioBuffers = []
        calibrationStartTime = Date()
        debugInfo.userEmbeddingStatus = "追加标定中..."
    }

    func stopAdditionalCalibration() {
        guard isCalibrating else { return }

        // 合并音频
        let allSamples = calibrationAudioBuffers.flatMap { $0 }

        guard !allSamples.isEmpty, let extractor = embeddingExtractor else {
            debugInfo.userEmbeddingStatus = "❌ 无有效数据"
            isCalibrating = false
            return
        }

        do {
            let embedding = try extractor.extract(from: allSamples)
            userEmbeddings.append(embedding)
            debugInfo.userEmbeddingStatus = "✅ 已标定 (\(userEmbeddings.count) 个样本)"
            saveEmbeddings(userEmbeddings)
            print("✅ 追加标定完成，当前共 \(userEmbeddings.count) 个样本")
        } catch {
            debugInfo.userEmbeddingStatus = "❌ 提取失败: \(error)"
            print("Embedding extraction failed: \(error)")
        }

        isCalibrating = false
        calibrationProgress = 0.0
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

        // 直接使用多个 embedding（不取平均）
        guard !calibrationEmbeddings.isEmpty else {
            debugInfo.userEmbeddingStatus = "❌ 无有效数据"
            return
        }

        userEmbeddings = calibrationEmbeddings
        initialEmbedding = calibrationEmbeddings.first  // 保存第一个作为初始 embedding
        learningCount = 0  // 重置学习次数
        lastLearningTime = nil
        debugInfo.userEmbeddingStatus = "✅ 已标定 (\(calibrationEmbeddings.count) 个样本)"
        phase = .live
        calibrationProgress = 1.0
    }
    
    // MARK: - Live Recognition
    
    private func processLiveAudio(_ samples: [Float]) {
        guard phase == .live else { return }
        guard !userEmbeddings.isEmpty else { return }
        guard let extractor = embeddingExtractor else { return }

        // 1. 更新 gaze 和 head 状态
        let face = engine.humanState.face
        debugInfo.isLookingAtScreen = face.isLookingAtScreen
        debugInfo.isHeadForward = face.headOrientation.isFacingForward
        
        // 1.5. 检查 face tracking 状态：如果最近 1 秒嘴巴完全没动，认为 tracking 丢失
        if let startTime = engine.sttManager.audioStreamStartTime {
            let elapsed = Date().timeIntervalSince(startTime)
            let recentStart = max(0, elapsed - 1.0)
            let recentJawDelta = calculateJawDelta(startTime: recentStart, endTime: elapsed)
            
            // 如果最近 1 秒 jaw 完全没动（< 0.005），认为看不到人
            if recentJawDelta < 0.005 {
                debugInfo.speakerMatch = false
                return
            }
        }

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

        // 3. 计算与所有模板的相似度，取最佳匹配
        let distances = userEmbeddings.map { cosineSimilarity($0, currentEmbedding) }
        let distance = distances.min() ?? 1.0
        debugInfo.speakerDistance = distance
        debugInfo.speakerMatch = distance < speakerThreshold

        // 4. 记录到历史（用于时间对齐）
        if let startTime = engine.sttManager.audioStreamStartTime {
            let elapsed = Date().timeIntervalSince(startTime)
            speakerHistory.append((timestamp: elapsed, distance: distance))

            // 计算当前的 jawDelta 和 jawVelocity（最近 0.5 秒）
            let recentStart = max(0, elapsed - 0.5)
            debugInfo.currentJawDelta = calculateJawDelta(startTime: recentStart, endTime: elapsed)
            debugInfo.currentJawVelocity = calculateJawVelocity(startTime: recentStart, endTime: elapsed)

            // 只保留最近 10 秒的历史
            let cutoff = elapsed - 10.0
            speakerHistory.removeAll { $0.timestamp < cutoff }
        }
    }

    // 增量学习（从 Final tokens 中学习）
    private func tryIncrementalLearningFromTokens(_ tokens: [TokenSegment]) {
        guard !userEmbeddings.isEmpty else { return }
        guard let extractor = embeddingExtractor else { return }

        // 筛选出用户说的 tokens，且 finalScore 足够低（高置信度）
        let userTokens = tokens.filter { token in
            let finalScore = calculateFinalScore(score: token.score, jawDelta: token.jawDelta, jawVelocity: token.jawVelocity)
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

        // 执行增量学习：添加新模板（而不是加权平均）
        userEmbeddings.append(currentEmbedding)
        learningCount += 1
        lastLearningTime = Date()

        print("✅ 增量学习 #\(learningCount): \(userTokens.count) tokens")
    }

    // 回滚到初始 embedding
    func resetToInitialEmbedding() {
        guard let initial = initialEmbedding else { return }
        userEmbeddings = [initial]
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
        userEmbeddings = []
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

    func getLogFileURL() -> URL {
        return logFileURL
    }

    func clearLog() {
        try? FileManager.default.removeItem(at: logFileURL)
        print("🗑️ Cleared recognition log")
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

    // MARK: - Logging

    private func logTokenRecognition(token: TokenSegment, isFinal: Bool) {
        let timestamp = Date().timeIntervalSince1970

        let logEntry: [String: Any] = [
            "timestamp": timestamp,
            "text": token.text,
            "audioTime": token.audioTime,
            "score": token.score,
            "jawDelta": token.jawDelta,
            "jawVelocity": token.jawVelocity,
            "isUserSpeaker": token.isUserSpeaker,
            "isFinal": isFinal,
            "speakerThreshold": speakerThreshold,
            "jawWeight": jawWeight,
            "jawVelocityWeight": jawVelocityWeight,
            "noJawPenalty": noJawPenalty,
            "jawMargin": jawMargin,
            // 计算 finalScore
            "finalScore": calculateFinalScore(score: token.score, jawDelta: token.jawDelta, jawVelocity: token.jawVelocity)
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: logEntry),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            // 追加到 JSONL 文件
            if let fileHandle = try? FileHandle(forWritingTo: logFileURL) {
                fileHandle.seekToEndOfFile()
                if let data = (jsonString + "\n").data(using: .utf8) {
                    fileHandle.write(data)
                }
                try? fileHandle.close()
            } else {
                // 文件不存在，创建新文件
                try? (jsonString + "\n").write(to: logFileURL, atomically: true, encoding: .utf8)
            }
        }
    }

    // MARK: - Persistence

    private func saveEmbeddings(_ embeddings: [[Float]]) {
        do {
            let data = try JSONEncoder().encode(embeddings)
            try data.write(to: embeddingFileURL)
            print("✅ Saved \(embeddings.count) user embeddings to \(embeddingFileURL.path)")
        } catch {
            print("⚠️ Failed to save embeddings: \(error)")
        }
    }

    private func loadEmbedding() {
        guard FileManager.default.fileExists(atPath: embeddingFileURL.path) else {
            debugInfo.userEmbeddingStatus = "未标定"
            return
        }

        do {
            let data = try Data(contentsOf: embeddingFileURL)
            let embeddings = try JSONDecoder().decode([[Float]].self, from: data)
            userEmbeddings = embeddings
            initialEmbedding = embeddings.first
            phase = .live
            debugInfo.userEmbeddingStatus = "✅ 已加载 (\(embeddings.count) 个样本)"
            print("✅ Loaded \(embeddings.count) user embeddings from \(embeddingFileURL.path)")
        } catch {
            print("⚠️ Failed to load embeddings: \(error)")
            debugInfo.userEmbeddingStatus = "未标定"
        }
    }

    func deleteEmbedding() {
        try? FileManager.default.removeItem(at: embeddingFileURL)
        userEmbeddings = []
        debugInfo.userEmbeddingStatus = "未标定"
        print("🗑️ Deleted user embedding")
    }
}
