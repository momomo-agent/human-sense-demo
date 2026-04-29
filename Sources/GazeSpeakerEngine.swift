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
    var speakerThreshold: Float = 4.0  // 投票阈值（userScore >= 此值判为用户）
    // 投票权重（autoresearch v88 最优配置: R=97.2%, S=98.9%, F1=96.1%）
    var scoreWeight: Float = 3.0  // 音色匹配度权重（score<0.45 → +3, 0.45-0.5 → +0.75, 0.5-0.72 → +0.25）
    var jawWeight: Float = 0.25  // 嘴巴张开幅度权重
    var jawVelocityWeight: Float = 2.0  // 嘴巴运动速度权重
    var timeDeltaWeight: Float = 1.5  // 时间间隔权重（dt>=0.3 → +1.5, dt>=0.03 → +0.75）
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
    /// Two-pass rescue: borderline token 被周围 user 密度 rescue
    /// Pass 1: 基础投票判定
    /// Pass 2: 对 borderline token（votes 在 low..threshold 之间），检查周围窗口内 user 密度
    ///         jawWeight 高的 token 用激进 rescue（因为 63% 是 user）
    ///         jawWeight 低的 token 用保守 rescue
    private func smoothSpeakerPredictions(_ tokens: [TokenSegment]) -> [TokenSegment] {
        guard tokens.count >= 2 else { return tokens }
        
        // 计算每个 token 的全部特征
        let N = tokens.count
        var timeDeltaArr = [Float](repeating: 0, count: N)
        for i in 1..<N {
            timeDeltaArr[i] = Float(max(0, tokens[i].audioTime - tokens[i-1].audioTime))
        }
        
        // 计算窗口统计量
        func windowStat<T: BinaryFloatingPoint>(_ arr: [T], hw: Int, fn: ([T]) -> T) -> [T] {
            return (0..<N).map { i in
                let lo = max(0, i - hw)
                let hi = min(N - 1, i + hw)
                return fn(Array(arr[lo...hi]))
            }
        }
        
        func mean(_ a: [Float]) -> Float {
            guard !a.isEmpty else { return 0 }
            return a.reduce(0, +) / Float(a.count)
        }
        func stddev(_ a: [Float]) -> Float {
            let m = mean(a)
            return sqrt(a.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Float(a.count))
        }
        
        // dt entropy (window=5)
        let dtEntropy5: [Float] = (0..<N).map { i in
            let lo = max(0, i - 2)
            let hi = min(N - 1, i + 2)
            var bins: [Float] = [0, 0, 0]  // dt<0.001, 0.001-0.1, >=0.1
            for j in lo...hi {
                if timeDeltaArr[j] < 0.001 { bins[0] += 1 }
                else if timeDeltaArr[j] < 0.1 { bins[1] += 1 }
                else { bins[2] += 1 }
            }
            let n = Float(hi - lo + 1)
            var entropy: Float = 0
            for b in bins where b > 0 {
                let p = b / n
                entropy -= p * log2(p)
            }
            return entropy
        }
        
        // burst length
        var burstLen = [Int](repeating: 1, count: N)
        for i in 1..<N {
            if timeDeltaArr[i] < 0.001 { burstLen[i] = burstLen[i-1] + 1 }
        }
        for i in stride(from: N - 2, through: 0, by: -1) {
            if i + 1 < N && timeDeltaArr[i+1] < 0.001 {
                burstLen[i] = max(burstLen[i], burstLen[i+1])
            }
        }
        
        // velocity std (window=5)
        let velArr = tokens.map { $0.jawVelocity }
        let velStd5 = windowStat(velArr, hw: 2, fn: stddev)
        
        // score std (window=5)
        let scoreArr = tokens.map { $0.score }
        let scoreStd5 = windowStat(scoreArr, hw: 2, fn: stddev)
        
        // score gap: |finalScore - score| where finalScore = score * jawFactor * velocityFactor * noMovementFactor
        // jawWeight/jawVelocityWeight are per-token fields (0.2 or 1.0), not engine constants
        let scoreGapArr: [Float] = tokens.map { t in
            // In production, jawWeight comes from ARKit blendshape confidence
            // jawVelocityWeight mirrors jawWeight (same 0.2/1.0 binary)
            let jw = t.jawDelta > 0.05 || t.jawVelocity > 0.3 ? Float(1.0) : Float(0.2)
            let jvw = jw  // jawVelocityWeight mirrors jawWeight
            let jawFactor = max(Float(0.1), 1.0 - jw * t.jawDelta)
            let velocityFactor = max(Float(0.1), 1.0 - jvw * t.jawVelocity)
            let noMovementFactor: Float = (t.jawDelta < 0.02 && t.jawVelocity < 0.1) ? 1.5 : 1.0
            let finalScore = t.score * jawFactor * velocityFactor * noMovementFactor
            return abs(finalScore - t.score)
        }
        
        // score slope (window=5)
        let scoreSlope5: [Float] = (0..<N).map { i in
            let lo = max(0, i - 2)
            let hi = min(N - 1, i + 2)
            let window = Array(scoreArr[lo...hi])
            guard window.count >= 2 else { return 0 }
            let n = Float(window.count)
            let mx = (n - 1) / 2.0
            let my = mean(window)
            var num: Float = 0, den: Float = 0
            for (x, y) in window.enumerated() {
                let xf = Float(x)
                num += (xf - mx) * (y - my)
                den += (xf - mx) * (xf - mx)
            }
            return den > 0 ? num / den : 0
        }
        
        // scoreVelAnti
        let scoreVelAnti = tokens.map { (1.0 - $0.score) * $0.jawVelocity }
        
        // score accel
        let scoreAccel: [Float] = (0..<N).map { i in
            guard i > 0, timeDeltaArr[i] >= 0.001 else { return Float(0) }
            return abs(tokens[i].score - tokens[i-1].score) / timeDeltaArr[i]
        }
        
        // jaw efficiency mean (window=5)
        let jawEff: [Float] = tokens.map { t in
            t.jawDelta > 0.001 ? t.jawVelocity / t.jawDelta : 0
        }
        let jawEffMean5 = windowStat(jawEff, hw: 2, fn: mean)
        
        // jawWeight: 检查每个 token 的 jawWeight（从外部传入或计算）
        // 在生产代码中，jawWeight 来自 ARKit 的 jawOpen blendshape 权重
        // 这里用 jawDelta 作为代理：jawDelta > 0.05 认为是高 jawWeight
        let isHighJW: [Bool] = tokens.map { $0.jawDelta > 0.05 || $0.jawVelocity > 0.3 }
        
        // === Pass 1: 基础投票 ===
        var votes = [Float](repeating: 0, count: N)
        for i in 0..<N {
            votes[i] = calculateUserScore(
                score: tokens[i].score,
                jawDelta: tokens[i].jawDelta,
                jawVelocity: tokens[i].jawVelocity,
                timeDelta: timeDeltaArr[i],
                dtEntropy5: dtEntropy5[i],
                burstLen: burstLen[i],
                velStd5: velStd5[i],
                scoreStd5: scoreStd5[i],
                scoreGap: scoreGapArr[i],
                scoreSlope5: scoreSlope5[i],
                scoreVelAnti: scoreVelAnti[i],
                scoreAccel: scoreAccel[i],
                jawEffMean5: jawEffMean5[i],
                isHighJW: isHighJW[i]
            )
        }
        
        let pass1Pred = votes.map { $0 >= speakerThreshold }
        
        // === Pass 2: Two-pass rescue ===
        // Borderline token 被周围 user 密度 rescue
        var result = tokens
        for i in 0..<N {
            if pass1Pred[i] {
                // Pass 1 已判为 user
                result[i] = TokenSegment(
                    text: tokens[i].text,
                    isUserSpeaker: true,
                    score: tokens[i].score,
                    audioTime: tokens[i].audioTime,
                    jawDelta: tokens[i].jawDelta,
                    jawVelocity: tokens[i].jawVelocity
                )
                continue
            }
            
            // Rescue 条件
            guard tokens[i].jawVelocity >= 0.1 else {
                // 嘴不动的不 rescue
                result[i] = TokenSegment(
                    text: tokens[i].text,
                    isUserSpeaker: false,
                    score: tokens[i].score,
                    audioTime: tokens[i].audioTime,
                    jawDelta: tokens[i].jawDelta,
                    jawVelocity: tokens[i].jawVelocity
                )
                continue
            }
            
            // Asymmetric rescue by jawWeight
            let hw: Int
            let nTh: Float
            let lowThreshold: Float
            
            if isHighJW[i] {
                // jw=1.0: 激进 rescue（63% 是 user）
                hw = 6
                nTh = 0.15
                lowThreshold = -5.0
            } else {
                // jw=0.2: 保守 rescue
                hw = 10
                nTh = 0.6
                lowThreshold = -1.0
            }
            
            // 投票太低的不 rescue
            guard votes[i] >= lowThreshold else {
                result[i] = TokenSegment(
                    text: tokens[i].text,
                    isUserSpeaker: false,
                    score: tokens[i].score,
                    audioTime: tokens[i].audioTime,
                    jawDelta: tokens[i].jawDelta,
                    jawVelocity: tokens[i].jawVelocity
                )
                continue
            }
            
            // 计算周围窗口内 user 密度
            var userCount = 0
            var totalCount = 0
            let lo = max(0, i - hw)
            let hi = min(N - 1, i + hw)
            for j in lo...hi where j != i {
                totalCount += 1
                if pass1Pred[j] { userCount += 1 }
            }
            
            let density = totalCount > 0 ? Float(userCount) / Float(totalCount) : 0
            let rescued = density >= nTh
            
            result[i] = TokenSegment(
                text: tokens[i].text,
                isUserSpeaker: rescued,
                score: tokens[i].score,
                audioTime: tokens[i].audioTime,
                jawDelta: tokens[i].jawDelta,
                jawVelocity: tokens[i].jawVelocity
            )
        }
        
        return result
    }

    // MARK: - v88 Speaker Classification (autoresearch v88: F1=96.1%)
    // 投票系统 + penalty 体系 + two-pass rescue
    // 特征: score, jawDelta, jawVelocity, timeDelta, dtEntropy, burstLen, velStd, scoreStd, scoreGap, scoreSlope, scoreVelAnti, scoreAccel, jawEffMean
    
    /// 计算单个 token 的基础投票分（不含 two-pass rescue）
    /// - Parameters:
    ///   - score: 音色匹配距离（越低越像用户）
    ///   - jawDelta: 嘴巴张开幅度
    ///   - jawVelocity: 嘴巴运动速度
    ///   - timeDelta: 与上一个 token 的时间间隔
    ///   - dtEntropy5: 局部 dt 熔（窗口=5）
    ///   - burstLen: 连续 dt=0 的 burst 长度
    ///   - velStd5: 局部 velocity 标准差
    ///   - scoreStd5: 局部 score 标准差
    ///   - scoreGap: |finalScore - score|
    ///   - scoreSlope5: 局部 score 斜率
    ///   - scoreVelAnti: (1-score)*velocity
    ///   - scoreAccel: score 加速度
    ///   - jawEffMean5: 局部 jaw efficiency 均值
    ///   - isHighJW: jawWeight > 0.5
    private func calculateUserScore(
        score: Float, jawDelta: Float, jawVelocity: Float, timeDelta: Float = 0,
        dtEntropy5: Float = 0, burstLen: Int = 1, velStd5: Float = 0, scoreStd5: Float = 0,
        scoreGap: Float = 0, scoreSlope5: Float = 0, scoreVelAnti: Float = 0,
        scoreAccel: Float = 0, jawEffMean5: Float = 0, isHighJW: Bool = false
    ) -> Float {
        var votes: Float = 0
        
        // === 正向投票（13 条规则） ===
        
        // 1. 音色不匹配（score 低）→ 加分
        if score < 0.45 {
            votes += 3.0
        } else if score < 0.5 {
            votes += 0.75
        } else if score < 0.72 {
            votes += 0.25
        }
        
        // 2. 嘴巴张开幅度大 → 加分
        if jawDelta >= 0.1 {
            votes += 0.25
        } else if jawDelta >= 0.05 {
            votes += 0.125
        }
        
        // 3. 嘴巴运动速度快 → 加分
        if jawVelocity >= 0.5 {
            votes += 4.0
        } else if jawVelocity >= 0.1 {
            votes += 2.0
        } else if jawVelocity >= 0.05 {
            votes += 1.0
        }
        
        // 4. 时间间隔：用户说话有自然间隔，AI 批量到达 dt≈0
        if timeDelta >= 0.3 {
            votes += 1.5
        } else if timeDelta >= 0.03 {
            votes += 0.75
        }
        
        // 5. dt 熔高 → 时间间隔多样，更像用户
        if dtEntropy5 >= 0.725 {
            votes += 1.0
        }
        
        // 6. scoreVelAnti: (1-score)*velocity 高 → 音色不匹配且嘴动快
        if scoreVelAnti >= 0.3 {
            votes += 0.375
        }
        
        // 7. score 加速度高 → 说话人切换信号
        if scoreAccel >= 1.5 {
            votes += 0.75
        }
        
        // 8. jaw efficiency 低 → 嘴动不是纯机械的
        if jawEffMean5 < 4.5 {
            votes += 0.25
        }
        
        // 9. scoreGap 大 → 音色匹配不稳定，更像用户
        if scoreGap >= 0.425 {
            votes += 1.75
        }
        
        // 10. score 下降趋势 → 用户开始说话
        if scoreSlope5 < -0.1 {
            votes += 0.5
        }
        
        // 11. (1-score)*velocity 高 → 另一个角度的用户信号
        let sv = (1.0 - score) * jawVelocity
        if sv >= 0.875 {
            votes += 0.375
        }
        
        // === Penalty 体系（9 条规则） ===
        let isDt0 = timeDelta < 0.001
        
        // P1: 中等 score + dt=0 + 嘴动 → AI lip sync 典型模式
        if score >= 0.3 && score < 0.7 && isDt0 && jawVelocity >= 0.15 {
            votes -= 1.625
        }
        
        // P2: velocity 标准差高 + dt=0 → AI burst 中速度波动
        if velStd5 >= 0.6 && isDt0 {
            votes -= 0.875
        }
        
        // P3: score 标准差低 + dt=0 → AI 音色稳定
        if scoreStd5 < 0.12 && isDt0 {
            votes -= 0.375
        }
        
        // P4: 高投票 + dt=0 + 低 score → 可疑的高分
        if votes >= 4.25 && isDt0 && score < 0.35 {
            votes -= 1.75
        }
        
        // P5: score-velocity mismatch（高 score + 高 velocity + dt=0）
        if score >= 0.7 && jawVelocity >= 0.4 && isDt0 {
            votes -= 2.0
        }
        
        // P6-P7: jawWeight 低时的针对性 penalty（jw=0.2 的 token 90% 是 AI）
        if !isHighJW {
            // P6: dt>0 + 高 score → AI 用自然时间间隔但嘴型匹配太好
            if !isDt0 && score >= 0.75 {
                votes -= 3.0
            }
            // P7: dt=0 + 中 score + 高 velocity → AI burst 中高速嘴动
            if isDt0 && score >= 0.3 && jawVelocity >= 0.5 {
                votes -= 1.5
            }
        }
        
        // P8: jw=0.2 + dt=0 + 高 velocity + 低 score → 典型 lip sync
        if !isHighJW && isDt0 && jawVelocity >= 0.4 && score < 0.4 {
            votes -= 2.0
        }
        
        // P9: jw=0.2 + dt=0 + 短 burst → AI 模式（user 说话 burst 更长）
        if !isHighJW && burstLen <= 2 && isDt0 {
            votes -= 1.75
        }
        
        // P10: jw=0.2 + 低 dt entropy → 时间间隔均匀 = AI 模式
        if !isHighJW && dtEntropy5 < 0.75 {
            votes -= 2.25
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
            let userScore = calculateUserScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity, timeDelta: dt)
            let isUser = userScore >= speakerThreshold
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
            let userScore = calculateUserScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity, timeDelta: dt)
            let isUser = userScore >= speakerThreshold
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

        // 有 speaker 变化
        if isFinal {
            // Final 阶段：不拆分，只用开始时间的 speaker
            let result = querySpeakerAtTime(token.startTime)
            let jawDelta = calculateJawDelta(startTime: token.startTime, endTime: token.endTime)
            let jawVelocity = calculateJawVelocity(startTime: token.startTime, endTime: token.endTime)
            let userScore = calculateUserScore(score: result.1, jawDelta: jawDelta, jawVelocity: jawVelocity, timeDelta: dt)
            let isUser = userScore >= speakerThreshold
            let segment = TokenSegment(
                text: token.text,
                isUserSpeaker: isUser,
                score: result.1,
                audioTime: token.startTime,
                jawDelta: jawDelta,
                jawVelocity: jawVelocity
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
