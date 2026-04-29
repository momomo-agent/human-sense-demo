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
        // v137 gaze/head/position
        var gazeOnScreen: Float = 0.0
        var headYaw: Float = 0.0
        var headPitch: Float = 0.0
        var faceDistance: Float = 0.0
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
        // Gaze/Head/Position features (v137)
        var gazeOnScreen: Float = 0  // 时间窗口内看屏幕的比例 [0,1]
        var headYaw: Float = 0       // 平均 head yaw (radians)
        var headPitch: Float = 0     // 平均 head pitch (radians)
        var faceDistance: Float = 0  // 平均距离 (meters)
    }

    var phase: Phase = .calibration
    var transcriptSegments: [TranscriptSegment] = []
    var currentTokens: [TokenSegment] = []  // 当前正在构建的句子
    var debugInfo = DebugInfo()
    var calibrationProgress: Float = 0.0
    var isCalibrating = false
    var speakerThreshold: Float = 4.75  // 投票阈值（autoresearch v136: F1=86.4% on 2702 tokens）
    // 投票权重（autoresearch v112 最优配置: R=99.6%, S=92.1%, F1=87.6%）
    var scoreWeight: Float = 0.5  // 音色匹配度权重（score<0.45 → +0.5）
    var jawWeight: Float = 1.5  // 嘴巴张开幅度权重（jd≥0.05 → +1.5）
    var jawVelocityWeight: Float = 2.5  // 嘴巴运动速度权重（vel≥0.5 → +2.5）
    var timeDeltaWeight: Float = 0.5  // 时间间隔权重（dt>=0.2 → +0.5）
    private var lastTokenAudioTime: Double = 0  // 上一个 token 的 audioTime
    var contextWeight: Float = 0.25  // 上下文信号权重
    var jawMargin: Double = 0.1  // jaw 时间扩展（秒）
    var noJawPenalty: Float = 0.5  // 嘴不动的惩罚值
    // v88 新增特征缓存（two-pass rescue 需要全局 token 信息）
    private var pendingTokenScores: [(index: Int, votes: Float, jawWeight: Float, isHighJW: Bool)] = []

    // 增量学习参数
    var enableIncrementalLearning: Bool = true  // 是否启用增量学习
    var learningThreshold: Float = 4.0  // 学习触发阈值（userScore >= 此值才学习，高置信度）
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
    private var gazeHistory: [(timestamp: Double, onScreen: Bool, yaw: Float, pitch: Float, distance: Float)] = []  // gaze/head/position 历史
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
                    // 上下文平滑：用前后 token 的多数投票修正孤立的误判
                    newTokens = self.smoothSpeakerPredictions(newTokens)
                    
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
    /// v112 speaker prediction with 10-second window zone features
    /// 简化的投票系统：zone 特征是最强信号，finalScore 是最强 penalty
    private func smoothSpeakerPredictions(_ tokens: [TokenSegment]) -> [TokenSegment] {
        guard tokens.count >= 2 else { return tokens }
        
        let N = tokens.count
        
        // 计算 timeDelta
        var timeDeltaArr = [Float](repeating: 0, count: N)
        for i in 1..<N {
            timeDeltaArr[i] = Float(max(0, tokens[i].audioTime - tokens[i-1].audioTime))
        }
        
        func mean(_ a: [Float]) -> Float {
            guard !a.isEmpty else { return 0 }
            return a.reduce(0, +) / Float(a.count)
        }
        
        // jaw efficiency per token
        let jawEff: [Float] = tokens.map { t in
            t.jawDelta > 0.001 ? t.jawVelocity / t.jawDelta : 0
        }
        
        // scoreVelAnti per token
        let scoreVelAnti: [Float] = tokens.map { (1.0 - $0.score) * $0.jawVelocity }
        
        // dtZeroRatio (window=5)
        let dtZeroRatio5: [Float] = (0..<N).map { i in
            let lo = max(0, i - 2)
            let hi = min(N - 1, i + 2)
            var zeroCount: Float = 0
            for j in lo...hi {
                if timeDeltaArr[j] < 0.001 { zeroCount += 1 }
            }
            return zeroCount / Float(hi - lo + 1)
        }
        
        // 10-second window zone features (jdMean, jeMean)
        // 使用 audioTime 做时间窗口
        let jdMean10: [Float] = (0..<N).map { i in
            let t0 = tokens[i].audioTime
            var vals: [Float] = []
            // 向前
            var j = i
            while j >= 0 && t0 - tokens[j].audioTime <= 10 {
                vals.append(tokens[j].jawDelta)
                j -= 1
            }
            // 向后
            j = i + 1
            while j < N && tokens[j].audioTime - t0 <= 10 {
                vals.append(tokens[j].jawDelta)
                j += 1
            }
            return mean(vals)
        }
        
        let jeMean10: [Float] = (0..<N).map { i in
            let t0 = tokens[i].audioTime
            var vals: [Float] = []
            var j = i
            while j >= 0 && t0 - tokens[j].audioTime <= 10 {
                vals.append(jawEff[j])
                j -= 1
            }
            j = i + 1
            while j < N && tokens[j].audioTime - t0 <= 10 {
                vals.append(jawEff[j])
                j += 1
            }
            return mean(vals)
        }
        
        // 10-second window score mean
        let scoreMean10: [Float] = (0..<N).map { i in
            let t0 = tokens[i].audioTime
            var vals: [Float] = []
            var j = i
            while j >= 0 && t0 - tokens[j].audioTime <= 10 {
                vals.append(tokens[j].score)
                j -= 1
            }
            j = i + 1
            while j < N && tokens[j].audioTime - t0 <= 10 {
                vals.append(tokens[j].score)
                j += 1
            }
            return mean(vals)
        }
        
        // finalScore: 使用 score 和 jaw 数据估算
        // 在生产中 finalScore 来自 STT 引擎，这里用 score * jawFactor 近似
        let finalScoreArr: [Float] = tokens.map { t in
            let jw: Float = (t.jawDelta > 0.05 || t.jawVelocity > 0.3) ? 1.0 : 0.2
            let jawFactor = max(Float(0.1), 1.0 - jw * t.jawDelta)
            let velocityFactor = max(Float(0.1), 1.0 - jw * t.jawVelocity)
            let noMovementFactor: Float = (t.jawDelta < 0.02 && t.jawVelocity < 0.1) ? 1.5 : 1.0
            return t.score * jawFactor * velocityFactor * noMovementFactor
        }
        
        // isHighJW
        let isHighJW: [Bool] = tokens.map { $0.jawDelta > 0.05 || $0.jawVelocity > 0.3 }
        
        // Gaze zone features: 10s 窗口内平均 gazeOnScreen / headYaw / faceDistance
        let gazeOnScreen10: [Float] = (0..<N).map { i in
            let t0 = tokens[i].audioTime
            var vals: [Float] = []
            var j = i
            while j >= 0 && t0 - tokens[j].audioTime <= 10 {
                vals.append(tokens[j].gazeOnScreen)
                j -= 1
            }
            j = i + 1
            while j < N && tokens[j].audioTime - t0 <= 10 {
                vals.append(tokens[j].gazeOnScreen)
                j += 1
            }
            return mean(vals)
        }
        
        let headYawAbs10: [Float] = (0..<N).map { i in
            let t0 = tokens[i].audioTime
            var vals: [Float] = []
            var j = i
            while j >= 0 && t0 - tokens[j].audioTime <= 10 {
                vals.append(abs(tokens[j].headYaw))
                j -= 1
            }
            j = i + 1
            while j < N && tokens[j].audioTime - t0 <= 10 {
                vals.append(abs(tokens[j].headYaw))
                j += 1
            }
            return mean(vals)
        }
        
        let faceDistance10: [Float] = (0..<N).map { i in
            let t0 = tokens[i].audioTime
            var vals: [Float] = []
            var j = i
            while j >= 0 && t0 - tokens[j].audioTime <= 10 {
                vals.append(tokens[j].faceDistance)
                j -= 1
            }
            j = i + 1
            while j < N && tokens[j].audioTime - t0 <= 10 {
                vals.append(tokens[j].faceDistance)
                j += 1
            }
            return mean(vals)
        }
        
        let headPitchAbs10: [Float] = (0..<N).map { i in
            let t0 = tokens[i].audioTime
            var vals: [Float] = []
            var j = i
            while j >= 0 && t0 - tokens[j].audioTime <= 10 {
                vals.append(abs(tokens[j].headPitch))
                j -= 1
            }
            j = i + 1
            while j < N && tokens[j].audioTime - t0 <= 10 {
                vals.append(abs(tokens[j].headPitch))
                j += 1
            }
            return mean(vals)
        }
        
        // === 计算投票 ===
        var votes = [Float](repeating: 0, count: N)
        for i in 0..<N {
            votes[i] = calculateUserScore(
                score: tokens[i].score,
                jawDelta: tokens[i].jawDelta,
                jawVelocity: tokens[i].jawVelocity,
                timeDelta: timeDeltaArr[i],
                jawEff: jawEff[i],
                scoreVelAnti: scoreVelAnti[i],
                dtZeroRatio5: dtZeroRatio5[i],
                jdMean10: jdMean10[i],
                jeMean10: jeMean10[i],
                finalScore: finalScoreArr[i],
                isHighJW: isHighJW[i],
                zoneScoreMean: scoreMean10[i],
                gazeOnScreen: gazeOnScreen10[i],
                headYawAbs: headYawAbs10[i],
                headPitchAbs: headPitchAbs10[i],
                faceDistance: faceDistance10[i]
            )
        }
        
        // === 直接阈值判定（v112 不需要 two-pass rescue）===
        var result = tokens
        for i in 0..<N {
            let isUser = votes[i] >= speakerThreshold
            result[i] = TokenSegment(
                text: tokens[i].text,
                isUserSpeaker: isUser,
                score: tokens[i].score,
                audioTime: tokens[i].audioTime,
                jawDelta: tokens[i].jawDelta,
                jawVelocity: tokens[i].jawVelocity,
                gazeOnScreen: tokens[i].gazeOnScreen,
                headYaw: tokens[i].headYaw,
                headPitch: tokens[i].headPitch,
                faceDistance: tokens[i].faceDistance
            )
        }
        
        return result
    }

    // MARK: - v112 Speaker Classification (autoresearch v112: F1=87.6%, R=99.6%, S=92.1%)
    // 10 秒时间窗口 zone 特征 + 投票系统 + finalScore penalty
    // 42 个版本 autoresearch 确认此为当前特征空间天花板
    
    /// 计算单个 token 的投票分（v112 最优配置）
    /// - Parameters:
    ///   - score: 音色匹配距离（越低越像用户）
    ///   - jawDelta: 嘴巴张开幅度
    ///   - jawVelocity: 嘴巴运动速度
    ///   - timeDelta: 与上一个 token 的时间间隔
    ///   - jawEff: jaw efficiency (jawVelocity / jawDelta)
    ///   - scoreVelAnti: (1-score) * jawVelocity
    ///   - dtZeroRatio5: 窗口内 dt≈0 的比例
    ///   - jdMean10: 10 秒窗口内 jawDelta 均值
    ///   - jeMean10: 10 秒窗口内 jawEfficiency 均值
    ///   - finalScore: STT finalScore（AI lip sync 时高）
    ///   - isHighJW: jawWeight > 0.5
    private func calculateUserScore(
        score: Float, jawDelta: Float, jawVelocity: Float, timeDelta: Float = 0,
        jawEff: Float = 0, scoreVelAnti: Float = 0, dtZeroRatio5: Float = 0,
        jdMean10: Float = 0, jeMean10: Float = 0, finalScore: Float = 0,
        isHighJW: Bool = false, zoneScoreMean: Float = 0,
        gazeOnScreen: Float = 1, headYawAbs: Float = 0, headPitchAbs: Float = 0, faceDistance: Float = 0.5
    ) -> Float {
        var votes: Float = 0
        
        // === Gaze gate: 不看屏幕 = 不是对 AI 说话 (v137) ===
        if gazeOnScreen < 0.3 {
            votes -= 4.0  // 强 penalty：大部分时间没看屏幕
        } else if gazeOnScreen < 0.5 {
            votes -= 2.0  // 中等 penalty
        } else if gazeOnScreen >= 0.8 {
            votes += 1.0  // 奖励：稳定看屏幕
        }
        
        // Head yaw 偏转太大 = 脸没朝屏幕
        if headYawAbs > 0.4 {
            votes -= 2.0
        } else if headYawAbs > 0.25 {
            votes -= 1.0
        }
        
        // Head pitch 偏转太大 = 低头/抬头没看屏幕
        if headPitchAbs > 0.4 {
            votes -= 2.0
        } else if headPitchAbs > 0.25 {
            votes -= 1.0
        }
        
        // 距离太远 = 不太可能在对屏幕说话
        if faceDistance > 1.5 {
            votes -= 1.5
        } else if faceDistance > 1.0 {
            votes -= 0.5
        }
        
        // === Zone feature: 10 秒窗口内 jaw 活跃度（最强信号） ===
        if jdMean10 >= 0.03 && jeMean10 >= 5 {
            votes += 4.5
        }
        
        // === 正向投票 ===
        
        // 嘴巴运动速度
        if jawVelocity >= 0.5 {
            votes += 1.5
        } else if jawVelocity >= 0.1 {
            votes += 0.6
        }
        
        // 嘴巴张开幅度
        if jawDelta >= 0.05 {
            votes += 1.5
        } else if jawDelta >= 0.02 {
            votes += 0.6
        }
        
        // jaw efficiency 高 → 嘴动有效率
        if jawEff >= 5 {
            votes += 0.5
        }
        
        // scoreVelAnti: (1-score)*velocity 高 → 音色不匹配且嘴动快
        if scoreVelAnti >= 0.2 {
            votes += 0.5
        }
        
        // 音色不匹配（score 低）
        if score < 0.45 {
            votes += 0.5
        }
        
        // 时间间隔：用户说话有自然间隔
        if timeDelta >= 0.2 {
            votes += 0.5
        }
        
        // dt=0 比例高 → 连续 burst，更像用户说话
        if dtZeroRatio5 >= 0.5 {
            votes += 0.5
        }
        
        // === Penalty ===
        
        // 窗口内 jaw 几乎不动 → 不是用户
        if jdMean10 < 0.005 {
            votes -= 2.0
        }
        
        // 窗口内 jaw efficiency 低 → 不是用户
        if jeMean10 < 1.5 {
            votes -= 1.0
        }
        
        // finalScore 高 → AI 正在说话（最强 penalty）
        if finalScore >= 0.7 {
            votes -= 3.5
        }
        
        // 窗口内平均 score 高 → 周围都是 AI 音色
        if zoneScoreMean >= 0.7 {
            votes -= 1.5
        }
        
        return votes
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

    // 计算时间范围内看屏幕的比例
    private func calculateGazeOnScreenRatio(startTime: Double, endTime: Double) -> Float {
        let expandedStart = max(0, startTime - jawMargin)
        let expandedEnd = endTime + jawMargin
        let relevant = gazeHistory.filter { $0.timestamp >= expandedStart && $0.timestamp <= expandedEnd }
        guard !relevant.isEmpty else { return 0 }
        let onCount = relevant.filter { $0.onScreen }.count
        return Float(onCount) / Float(relevant.count)
    }

    // 计算时间范围内平均 head yaw
    private func calculateMeanHeadYaw(startTime: Double, endTime: Double) -> Float {
        let expandedStart = max(0, startTime - jawMargin)
        let expandedEnd = endTime + jawMargin
        let relevant = gazeHistory.filter { $0.timestamp >= expandedStart && $0.timestamp <= expandedEnd }
        guard !relevant.isEmpty else { return 0 }
        return relevant.map { $0.yaw }.reduce(0, +) / Float(relevant.count)
    }

    // 计算时间范围内平均 head pitch
    private func calculateMeanHeadPitch(startTime: Double, endTime: Double) -> Float {
        let expandedStart = max(0, startTime - jawMargin)
        let expandedEnd = endTime + jawMargin
        let relevant = gazeHistory.filter { $0.timestamp >= expandedStart && $0.timestamp <= expandedEnd }
        guard !relevant.isEmpty else { return 0 }
        return relevant.map { $0.pitch }.reduce(0, +) / Float(relevant.count)
    }

    // 计算时间范围内平均距离
    private func calculateMeanFaceDistance(startTime: Double, endTime: Double) -> Float {
        let expandedStart = max(0, startTime - jawMargin)
        let expandedEnd = endTime + jawMargin
        let relevant = gazeHistory.filter { $0.timestamp >= expandedStart && $0.timestamp <= expandedEnd }
        guard !relevant.isEmpty else { return 0 }
        return relevant.map { $0.distance }.reduce(0, +) / Float(relevant.count)
    }

    // 创建 TokenSegment 并自动填充 gaze/head/position 数据
    private func makeTokenSegment(
        text: String, isUserSpeaker: Bool, score: Float,
        audioTime: Double, endTime: Double,
        jawDelta: Float, jawVelocity: Float
    ) -> TokenSegment {
        var seg = TokenSegment(
            text: text, isUserSpeaker: isUserSpeaker, score: score,
            audioTime: audioTime, jawDelta: jawDelta, jawVelocity: jawVelocity
        )
        seg.gazeOnScreen = calculateGazeOnScreenRatio(startTime: audioTime, endTime: endTime)
        seg.headYaw = calculateMeanHeadYaw(startTime: audioTime, endTime: endTime)
        seg.headPitch = calculateMeanHeadPitch(startTime: audioTime, endTime: endTime)
        seg.faceDistance = calculateMeanFaceDistance(startTime: audioTime, endTime: endTime)
        return seg
    }

    // 根据 token 时间范围内的 speaker 变化拆分 token
    private func splitTokenBySpeakerChange(_ token: SpeechToken, isFinal: Bool) -> [TokenSegment] {
        let tokenKey = token.text + "_\(token.startTime)_\(token.endTime)"
        
        // 计算与上一个 token 的时间间隔
        let dt = Float(max(0, token.startTime - lastTokenAudioTime))
        lastTokenAudioTime = token.startTime

        // 只在 Final 阶段使用缓存
        if isFinal, let cached = self.tokenColorMap[tokenKey] {
            let jawDelta = calculateJawDelta(startTime: token.startTime, endTime: token.endTime)
            let jawVelocity = calculateJawVelocity(startTime: token.startTime, endTime: token.endTime)
            let userScore = calculateUserScore(score: cached.score, jawDelta: jawDelta, jawVelocity: jawVelocity, timeDelta: dt)
            let isUser = userScore >= speakerThreshold
            return [makeTokenSegment(
                text: token.text, isUserSpeaker: isUser, score: cached.score,
                audioTime: token.startTime, endTime: token.endTime,
                jawDelta: jawDelta, jawVelocity: jawVelocity
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
            let userScore = calculateUserScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity, timeDelta: dt)
            let isUser = userScore >= speakerThreshold
            let segment = makeTokenSegment(
                text: token.text, isUserSpeaker: isUser, score: result.1,
                audioTime: token.startTime, endTime: token.endTime,
                jawDelta: jawDelta, jawVelocity: jawVelocity
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
            let userScore = calculateUserScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity, timeDelta: dt)
            let isUser = userScore >= speakerThreshold
            let segment = makeTokenSegment(
                text: token.text, isUserSpeaker: isUser, score: result.1,
                audioTime: token.startTime, endTime: token.endTime,
                jawDelta: jawDelta, jawVelocity: jawVelocity
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
            let jawVelocity = calculateJawVelocity(startTime: token.startTime, endTime: token.endTime)
            let userScore = calculateUserScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity, timeDelta: dt)
            let isUser = userScore >= speakerThreshold
            let segment = makeTokenSegment(
                text: token.text, isUserSpeaker: isUser, score: result.1,
                audioTime: token.startTime, endTime: token.endTime,
                jawDelta: jawDelta, jawVelocity: jawVelocity
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
            let jawVelocity = calculateJawVelocity(startTime: token.startTime, endTime: token.endTime)
            let userScore = calculateUserScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity, timeDelta: dt)
            let isUser = userScore >= speakerThreshold
            return [makeTokenSegment(
                text: token.text, isUserSpeaker: isUser, score: result.1,
                audioTime: token.startTime, endTime: token.endTime,
                jawDelta: jawDelta, jawVelocity: jawVelocity
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
            let userScore = calculateUserScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity, timeDelta: dt)
            let isUser = userScore >= speakerThreshold

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
                segments.append(makeTokenSegment(
                    text: currentText, isUserSpeaker: currentSpeaker!, score: result.1,
                    audioTime: currentStartTime, endTime: segmentEndTime,
                    jawDelta: segmentJawDelta, jawVelocity: segmentJawVelocity
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
            segments.append(makeTokenSegment(
                text: currentText, isUserSpeaker: speaker, score: result.1,
                audioTime: currentStartTime, endTime: token.endTime,
                jawDelta: jawDelta, jawVelocity: jawVelocity
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
            let face = engine.humanState.face
            let jawOpen = face.jawOpen
            jawHistory.append((timestamp: elapsed, jawOpen: jawOpen))
            
            // 记录 gaze/head/position 数据
            gazeHistory.append((
                timestamp: elapsed,
                onScreen: face.isLookingAtScreen,
                yaw: face.headYaw,
                pitch: face.headPitch,
                distance: face.distanceFromCamera
            ))

            // 只保留最近 10 秒的历史
            let cutoff = elapsed - 10.0
            jawHistory.removeAll { $0.timestamp < cutoff }
            gazeHistory.removeAll { $0.timestamp < cutoff }
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
        debugInfo.gazeOnScreen = face.isLookingAtScreen ? 1.0 : 0.0
        debugInfo.headYaw = face.headYaw
        debugInfo.headPitch = face.headPitch
        debugInfo.faceDistance = face.distanceFromCamera
        
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

        // 筛选出用户说的 tokens，且 userScore 足够高（高置信度）
        let userTokens = tokens.filter { token in
            let userScore = calculateUserScore(score: token.score, jawDelta: token.jawDelta, jawVelocity: token.jawVelocity)
            return token.isUserSpeaker && userScore >= learningThreshold && token.jawDelta > 0.05
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
        lastTokenAudioTime = 0
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
        gazeHistory = []
        lastTokenAudioTime = 0
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
            "timeDeltaWeight": timeDeltaWeight,
            "noJawPenalty": noJawPenalty,
            "jawMargin": jawMargin,
            // Gaze/Head/Position (v137)
            "gazeOnScreen": token.gazeOnScreen,
            "headYaw": token.headYaw,
            "headPitch": token.headPitch,
            "faceDistance": token.faceDistance,
            // 计算 userScore
            "userScore": calculateUserScore(score: token.score, jawDelta: token.jawDelta, jawVelocity: token.jawVelocity)
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
