import SwiftUI
import HumanSenseKit
import AVFoundation
import FluidAudio

struct DiarizationTestView: View {
    @State private var engine: GazeSpeakerEngine
    @State private var audioStreamBridge = SharedAudioStreamBridge()
    @State private var showDebug = true  // 默认显示调试面板
    private let humanEngine: HumanStateEngine
    
    init(humanEngine: HumanStateEngine) {
        self.humanEngine = humanEngine
        _engine = State(initialValue: GazeSpeakerEngine(engine: humanEngine))
    }
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // 主内容区
            if engine.phase == .calibration {
                calibrationView
            } else {
                liveRecognitionView
            }
        }
        .onAppear {
            // 确保 STT 已启动
            if !humanEngine.sttManager.isListening {
                humanEngine.sttManager.start()
            }
            setupAudioStream()
        }
        .onDisappear {
            audioStreamBridge.setActive(false)
        }
        .onReceive(humanEngine.sttManager.$isListening) { isListening in
            if isListening {
                setupAudioStream()
            } else {
                audioStreamBridge.markPipelineStopped()
            }
        }
    }
    
    // MARK: - Audio Stream Setup
    
    private func setupAudioStream() {
        let currentEngine = engine
        audioStreamBridge.setActive(true)

        guard humanEngine.sttManager.isListening else { return }

        audioStreamBridge.connectIfNeeded(to: humanEngine.sttManager) { samples in
            Task { @MainActor in
                currentEngine.processAudioBuffer(samples)
            }
        }
    }
    
    // MARK: - Calibration View
    
    private var calibrationView: some View {
        VStack(spacing: 24) {
            Image(systemName: "person.wave.2")
                .font(.system(size: 60))
                .foregroundStyle(.blue)

            Text("声纹标定")
                .font(.title)
                .foregroundStyle(.white)

            if engine.isCalibrating {
                VStack(spacing: 16) {
                    // 显示当前要念的句子
                    Text("请朗读：")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text(engine.calibrationSentences[engine.currentCalibrationSentence])
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundStyle(.blue)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 12)
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(12)

                    ProgressView(value: engine.calibrationProgress)
                        .progressViewStyle(.linear)
                        .tint(.blue)
                        .frame(maxWidth: 250)

                    Text("\(engine.currentCalibrationSentence + 1)/\(engine.calibrationSentences.count)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                VStack(spacing: 12) {
                    Text("需要朗读 \(engine.calibrationSentences.count) 句话")
                        .foregroundStyle(.secondary)

                    Text("每句约 3 秒")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Button {
                    engine.startCalibration()
                } label: {
                    Text("开始标定")
                        .font(.headline)
                        .foregroundStyle(.white)
                        .frame(width: 200, height: 50)
                        .background(Color.blue)
                        .cornerRadius(12)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    // MARK: - Live Recognition View
    
    private var liveRecognitionView: some View {
        ZStack {
            VStack(spacing: 0) {
                // 上部：我说的话（固定高度）
                VStack(alignment: .leading, spacing: 8) {
                    ScrollViewReader { proxy in
                        ScrollView {
                            VStack(alignment: .leading, spacing: 8) {
                                // 历史记录（只显示用户说的，final）
                                ForEach(engine.transcriptSegments.filter { $0.isUserSpeaker }) { segment in
                                    Text(segment.text)
                                        .foregroundStyle(.green)
                                        .padding(.vertical, 4)
                                        .id(segment.id)
                                }

                                // 当前正在说的话（只显示用户的，streaming 固定透明度）
                                if !engine.currentTokens.isEmpty {
                                    let userTokens = engine.currentTokens.filter { $0.isUserSpeaker }
                                    if !userTokens.isEmpty {
                                        HStack(spacing: 0) {
                                            ForEach(userTokens) { token in
                                                Text(token.text)
                                                    .foregroundStyle(.green.opacity(0.2))
                                            }
                                        }
                                        .id("current")
                                    }
                                }

                                if engine.transcriptSegments.filter({ $0.isUserSpeaker }).isEmpty && engine.currentTokens.filter({ $0.isUserSpeaker }).isEmpty {
                                    Text("等待你说话...")
                                        .foregroundStyle(.secondary)
                                        .frame(maxWidth: .infinity, alignment: .center)
                                        .padding(.vertical, 20)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .onChange(of: engine.transcriptSegments.count) { _, _ in
                            if let lastSegment = engine.transcriptSegments.filter({ $0.isUserSpeaker }).last {
                                withAnimation {
                                    proxy.scrollTo(lastSegment.id, anchor: .bottom)
                                }
                            }
                        }
                        .onChange(of: engine.currentTokens.count) { _, _ in
                            if !engine.currentTokens.filter({ $0.isUserSpeaker }).isEmpty {
                                withAnimation {
                                    proxy.scrollTo("current", anchor: .bottom)
                                }
                            }
                        }
                    }
                }
                .frame(height: 120)
                .padding()
                .background(Color.white.opacity(0.05))

                Divider()
                    .background(Color.white.opacity(0.3))

                // 中部：详细日志（撑满剩余空间）
                VStack(spacing: 0) {
                    // 表头（固定）
                    HStack(spacing: 8) {
                        Text("时间")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .frame(width: 40, alignment: .trailing)

                        Text("文字")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        Text("Δ")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                            .frame(width: 35, alignment: .trailing)

                        Text("V")
                            .font(.caption2)
                            .foregroundStyle(.yellow)
                            .frame(width: 35, alignment: .trailing)

                        Text("分数")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .frame(width: 35, alignment: .trailing)

                        Text("终分")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .frame(width: 35, alignment: .trailing)

                        Text("✓")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .frame(width: 20, alignment: .center)

                        Text("状态")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .frame(width: 30, alignment: .center)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.05))

                    // 滚动内容
                    ScrollViewReader { proxy in
                        ScrollView {
                            VStack(alignment: .leading, spacing: 12) {
                                // 历史记录（按句子分组）
                                ForEach(groupedSegments(), id: \.id) { group in
                                VStack(alignment: .leading, spacing: 2) {
                                    // 句子标题
                                    HStack {
                                        Text("句子 #\(group.id)")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                        Spacer()
                                        Text(group.isFinal ? "Final" : "Stream")
                                            .font(.caption2)
                                            .foregroundStyle(group.isFinal ? .white : .secondary)
                                    }
                                    .padding(.bottom, 2)

                                    // Token 列表
                                    ForEach(group.tokens) { token in
                                        detailRow(
                                            time: token.audioTime,
                                            text: token.text,
                                            jawDelta: token.jawDelta,
                                            jawVelocity: token.jawVelocity,
                                            score: token.score,
                                            isUser: token.isUserSpeaker,
                                            isFinal: group.isFinal
                                        )
                                    }
                                }
                                .padding(8)
                                .background(Color.white.opacity(0.03))
                                .cornerRadius(8)
                                .id("segment-\(group.id)")
                            }

                            // 当前正在说的话（streaming）
                            if !engine.currentTokens.isEmpty {
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack {
                                        Text("句子 #current")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                        Spacer()
                                        Text("Stream")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                    .padding(.bottom, 2)

                                    ForEach(engine.currentTokens) { token in
                                        detailRow(
                                            time: token.audioTime,
                                            text: token.text,
                                            jawDelta: token.jawDelta,
                                            jawVelocity: token.jawVelocity,
                                            score: token.score,
                                            isUser: token.isUserSpeaker,
                                            isFinal: false
                                        )
                                    }
                                }
                                .padding(8)
                                .background(Color.white.opacity(0.03))
                                .cornerRadius(8)
                                .id("current-segment")
                            }

                            if engine.transcriptSegments.isEmpty && engine.currentTokens.isEmpty {
                                Text("等待转录...")
                                    .foregroundStyle(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .center)
                                    .padding(.vertical, 20)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: engine.transcriptSegments.count) { _, _ in
                        if let lastGroup = groupedSegments().last {
                            withAnimation {
                                proxy.scrollTo("segment-\(lastGroup.id)", anchor: .bottom)
                            }
                        }
                    }
                    .onChange(of: engine.currentTokens.count) { _, _ in
                        if !engine.currentTokens.isEmpty {
                            withAnimation {
                                proxy.scrollTo("current-segment", anchor: .bottom)
                            }
                        }
                    }
                }
                }

                // 底部：按钮栏
                HStack(spacing: 12) {
                    // 追加标定按钮
                    Button {
                        if engine.isCalibrating {
                            engine.stopAdditionalCalibration()
                        } else {
                            engine.startAdditionalCalibration()
                        }
                    } label: {
                        HStack {
                            Image(systemName: engine.isCalibrating ? "stop.circle.fill" : "mic.circle")
                            Text(engine.isCalibrating ? "停止录音" : "追加标定")
                        }
                        .font(.caption)
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(engine.isCalibrating ? Color.red : Color.blue)
                        .cornerRadius(8)
                    }

                    // 调试按钮
                    Button {
                        withAnimation {
                            showDebug.toggle()
                        }
                    } label: {
                        HStack {
                            Image(systemName: showDebug ? "info.circle.fill" : "info.circle")
                            Text(showDebug ? "隐藏调试" : "显示调试")
                        }
                        .font(.caption)
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Color.white.opacity(0.1))
                        .cornerRadius(8)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            }

            // 悬浮调试面板（带背景遮罩）
            if showDebug {
                ZStack {
                    // 半透明背景，点击收回
                    Color.black.opacity(0.3)
                        .ignoresSafeArea()
                        .onTapGesture {
                            withAnimation {
                                showDebug = false
                            }
                        }

                    // 调试面板
                    VStack {
                        Spacer()
                        HStack {
                            Spacer()
                            ScrollView {
                                debugPanel
                            }
                            .frame(width: 320)
                            .frame(maxHeight: 600)
                            .padding()
                        }
                    }
                }
                .transition(.move(edge: .trailing).combined(with: .opacity))
            }
        }
    }

    // MARK: - Debug Panel
    
    private var debugPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "ladybug")
                Text("调试信息")
                    .font(.headline)
            }
            .foregroundStyle(.white)

            Divider()
                .background(Color.white.opacity(0.3))

            debugRow(label: "看屏幕", value: engine.debugInfo.isLookingAtScreen ? "✅ 是" : "❌ 否")
            debugRow(label: "头朝前", value: engine.debugInfo.isHeadForward ? "✅ 是" : "❌ 否")
            debugRow(
                label: "Speaker",
                value: engine.debugInfo.speakerMatch ? "✅ 匹配" : "❌ 不匹配"
            )
            debugRow(
                label: "距离",
                value: String(format: "%.2f / %.2f", engine.debugInfo.speakerDistance, engine.speakerThreshold)
            )
            debugRow(
                label: "Jaw Δ",
                value: String(format: "%.3f", engine.debugInfo.currentJawDelta)
            )
            debugRow(
                label: "Jaw V",
                value: String(format: "%.3f", engine.debugInfo.currentJawVelocity)
            )
            debugRow(
                label: "音频",
                value: String(format: "%.1f dB", engine.debugInfo.audioLevel)
            )
            debugRow(label: "用户声纹", value: engine.debugInfo.userEmbeddingStatus)

            // 阈值调节
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("匹配阈值:")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                    Spacer()
                    Text(String(format: "%.2f", engine.speakerThreshold))
                        .foregroundStyle(.white)
                        .font(.caption)
                }
                Slider(value: $engine.speakerThreshold, in: 0.3...0.9, step: 0.05)
                    .tint(.blue)
                Text("← 严格    宽松 →")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            .padding(.top, 8)

            // Jaw 权重调节
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Jaw Δ 权重:")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                    Spacer()
                    Text(String(format: "%.1f", engine.jawWeight))
                        .foregroundStyle(.white)
                        .font(.caption)
                }
                Slider(value: $engine.jawWeight, in: 0.0...5.0, step: 0.1)
                    .tint(.orange)
                Text("← 弱    强 →")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            .padding(.top, 8)

            // Jaw Velocity 权重调节
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Jaw V 权重:")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                    Spacer()
                    Text(String(format: "%.1f", engine.jawVelocityWeight))
                        .foregroundStyle(.white)
                        .font(.caption)
                }
                Slider(value: $engine.jawVelocityWeight, in: 0.0...5.0, step: 0.1)
                    .tint(.yellow)
                Text("← 弱    强 →")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            .padding(.top, 8)

            // 嘴不动惩罚
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("嘴不动惩罚:")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                    Spacer()
                    Text(String(format: "%.2f", engine.noJawPenalty))
                        .foregroundStyle(.white)
                        .font(.caption)
                }
                Slider(value: $engine.noJawPenalty, in: 0.0...2.0, step: 0.1)
                    .tint(.red)
                Text("← 弱    强 →")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            .padding(.top, 8)

            // Jaw Margin 调节
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Jaw Margin:")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                    Spacer()
                    Text(String(format: "%.0fms", engine.jawMargin * 1000))
                        .foregroundStyle(.white)
                        .font(.caption)
                }
                Slider(value: $engine.jawMargin, in: 0.0...0.3, step: 0.01)
                    .tint(.orange)
                Text("← 窄    宽 →")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            .padding(.top, 8)

            Divider()
                .background(Color.white.opacity(0.3))

            // 增量学习
            VStack(alignment: .leading, spacing: 8) {
                Toggle(isOn: $engine.enableIncrementalLearning) {
                    HStack {
                        Text("增量学习")
                            .font(.caption)
                            .foregroundStyle(.white)
                        if engine.learningCount > 0 {
                            Text("(\(engine.learningCount)次)")
                                .font(.caption2)
                                .foregroundStyle(.green)
                        }
                    }
                }
                .tint(.green)

                if engine.enableIncrementalLearning {
                    // 学习阈值
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("学习阈值:")
                                .foregroundStyle(.secondary)
                                .font(.caption)
                            Spacer()
                            Text(String(format: "%.2f", engine.learningThreshold))
                                .foregroundStyle(.white)
                                .font(.caption)
                        }
                        Slider(value: $engine.learningThreshold, in: 0.1...1.0, step: 0.05)
                            .tint(.green)
                    }

                    // 学习率
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("学习率:")
                                .foregroundStyle(.secondary)
                                .font(.caption)
                            Spacer()
                            Text(String(format: "%.2f", engine.learningRate))
                                .foregroundStyle(.white)
                                .font(.caption)
                        }
                        Slider(value: $engine.learningRate, in: 0.05...1.0, step: 0.05)
                            .tint(.green)
                    }

                    // 回滚按钮
                    Button {
                        engine.resetToInitialEmbedding()
                    } label: {
                        Text("回滚到初始")
                            .font(.caption)
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 6)
                            .background(Color.gray)
                            .cornerRadius(6)
                    }
                }
            }
            .padding(.top, 8)

            Divider()
                .background(Color.white.opacity(0.3))

            // 操作按钮
            VStack(spacing: 8) {
                Button {
                    engine.reset()
                } label: {
                    Text("重新标定")
                        .font(.caption)
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(Color.orange)
                        .cornerRadius(6)
                }

                Button {
                    engine.deleteEmbedding()
                    engine.reset()
                } label: {
                    Text("删除标定")
                        .font(.caption)
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(Color.purple)
                        .cornerRadius(6)
                }

                Button {
                    engine.clearTranscript()
                } label: {
                    Text("清空日志")
                        .font(.caption)
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(Color.red)
                        .cornerRadius(6)
                }
            }
        }
        .padding()
        .background(Color.black.opacity(0.85))
        .cornerRadius(12)
        .shadow(radius: 10)
    }
    
    private func debugRow(label: String, value: String) -> some View {
        HStack {
            Text(label + ":")
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .foregroundStyle(.white)
        }
        .font(.caption)
    }

    private func detailRow(time: Double, text: String, jawDelta: Float, jawVelocity: Float, score: Float, isUser: Bool, isFinal: Bool) -> some View {
        // 使用与 GazeSpeakerEngine 相同的计算逻辑（乘法权重）
        var jawFactor: Float = 1.0 - engine.jawWeight * jawDelta
        var velocityFactor: Float = 1.0 - engine.jawVelocityWeight * jawVelocity
        var noMovementFactor: Float = 1.0

        if jawDelta < 0.02 && jawVelocity < 0.1 {
            noMovementFactor = 1.0 + engine.noJawPenalty
        }

        jawFactor = max(0.1, jawFactor)
        velocityFactor = max(0.1, velocityFactor)

        let finalScore = score * jawFactor * velocityFactor * noMovementFactor

        return HStack(spacing: 8) {
            // 时间
            Text(String(format: "%.1fs", time))
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: 40, alignment: .trailing)

            // 文字
            Text(text)
                .font(.caption)
                .foregroundStyle(isUser ? .green : .gray)
                .frame(maxWidth: .infinity, alignment: .leading)

            // JawDelta
            Text(String(format: "%.2f", jawDelta))
                .font(.caption2)
                .foregroundStyle(.orange)
                .frame(width: 35, alignment: .trailing)

            // JawVelocity
            Text(String(format: "%.2f", jawVelocity))
                .font(.caption2)
                .foregroundStyle(.yellow)
                .frame(width: 35, alignment: .trailing)

            // Score
            Text(String(format: "%.2f", score))
                .font(.caption2)
                .foregroundStyle(score < engine.speakerThreshold ? .green : .red)
                .frame(width: 35, alignment: .trailing)

            // FinalScore
            Text(String(format: "%.2f", finalScore))
                .font(.caption2)
                .foregroundStyle(finalScore < engine.speakerThreshold ? .green : .red)
                .frame(width: 35, alignment: .trailing)

            // 用户标记
            Text(isUser ? "✓" : "✗")
                .font(.caption2)
                .foregroundStyle(isUser ? .green : .gray)
                .frame(width: 15)

            // Final 标记
            Text(isFinal ? "F" : "S")
                .font(.caption2)
                .foregroundStyle(isFinal ? .white : .secondary)
                .frame(width: 15)
        }
        .padding(.vertical, 2)
    }

    // 按句子分组
    struct SentenceGroup: Identifiable {
        let id: Int
        let tokens: [GazeSpeakerEngine.TokenSegment]
        let isFinal: Bool
    }

    private func groupedSegments() -> [SentenceGroup] {
        // transcriptSegments 已经按 speaker 分组，每个 segment 包含多个 tokens
        return engine.transcriptSegments.enumerated().map { index, segment in
            SentenceGroup(
                id: index,
                tokens: segment.tokens,
                isFinal: segment.isFinal
            )
        }
    }
}

// 呼吸效果
struct BreathingEffect: ViewModifier {
    @State private var opacity: Double = 0.6

    func body(content: Content) -> some View {
        content
            .opacity(opacity)
            .onAppear {
                withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: true)) {
                    opacity = 1.0
                }
            }
    }
}

private final class SharedAudioStreamBridge {
    private let lock = NSLock()
    private let audioConverter = AudioConverter()
    private weak var audioEngineManager: AudioEngineManager?
    private var isConnected = false
    private var isActive = false
    private var samplesHandler: (@MainActor @Sendable ([Float]) -> Void)?

    func setActive(_ active: Bool) {
        lock.lock()
        isActive = active
        lock.unlock()
    }

    func markPipelineStopped() {
        lock.lock()
        isConnected = false
        audioEngineManager = nil
        samplesHandler = nil
        lock.unlock()
    }

    @MainActor
    func connectIfNeeded(
        to sttManager: STTManager,
        samplesHandler: @escaping @MainActor @Sendable ([Float]) -> Void
    ) {
        guard let audioEngineManager = Self.audioEngineManager(from: sttManager) else {
            print("[Diarization] Unable to connect to STT shared audio stream")
            return
        }

        lock.lock()
        self.samplesHandler = samplesHandler

        let connectedManager = self.audioEngineManager
        if isConnected, connectedManager === audioEngineManager {
            lock.unlock()
            return
        }
        lock.unlock()

        let existingOnBuffer = audioEngineManager.onBuffer
        audioEngineManager.onBuffer = { [weak self, existingOnBuffer] buffer in
            existingOnBuffer?(buffer)
            self?.forward(buffer)
        }

        lock.lock()
        self.audioEngineManager = audioEngineManager
        isConnected = true
        lock.unlock()
    }

    private func forward(_ buffer: AVAudioPCMBuffer) {
        guard let samples = samples(from: buffer), !samples.isEmpty else { return }

        lock.lock()
        let handler = isActive ? samplesHandler : nil
        lock.unlock()

        guard let handler else { return }

        Task { @MainActor in
            handler(samples)
        }
    }

    @MainActor
    private static func audioEngineManager(from sttManager: STTManager) -> AudioEngineManager? {
        Mirror(reflecting: sttManager).children.first { child in
            child.label == "audio"
        }?.value as? AudioEngineManager
    }

    private func samples(from buffer: AVAudioPCMBuffer) -> [Float]? {
        if let convertedSamples = try? audioConverter.resampleBuffer(buffer) {
            return convertedSamples
        }

        guard let channelData = buffer.floatChannelData else { return nil }
        let frameLength = Int(buffer.frameLength)
        guard frameLength > 0 else { return nil }

        return Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
    }
}
