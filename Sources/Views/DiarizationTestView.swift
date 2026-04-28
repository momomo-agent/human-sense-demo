import SwiftUI
import HumanSenseKit
import AVFoundation

struct DiarizationTestView: View {
    @State private var engine: GazeSpeakerEngine
    @State private var showDebug = false
    private let humanEngine: HumanStateEngine
    
    init(humanEngine: HumanStateEngine) {
        self.humanEngine = humanEngine
        _engine = State(initialValue: GazeSpeakerEngine(engine: humanEngine))
    }
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            VStack(spacing: 20) {
                // 标题
                Text("Gaze + Speaker 实验")
                    .font(.title2)
                    .foregroundStyle(.white)
                
                Spacer()
                
                // 主内容区
                Group {
                    if engine.phase == .calibration {
                        calibrationView
                    } else {
                        liveRecognitionView
                    }
                }
                
                Spacer()
                
                // 调试面板
                if showDebug {
                    debugPanel
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                
                // 调试开关
                Button {
                    withAnimation {
                        showDebug.toggle()
                    }
                } label: {
                    HStack {
                        Image(systemName: "info.circle")
                        Text(showDebug ? "隐藏调试" : "显示调试")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                .padding(.bottom, 8)
            }
            .padding()
        }
        .onAppear {
            setupAudioStream()
        }
    }
    
    // MARK: - Audio Stream Setup
    
    private func setupAudioStream() {
        // 从 AudioEngineManager 获取音频 buffer
        let audioEngine = humanEngine.sttManager.audioEngine
        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        
        // 安装 tap（如果还没安装）
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
            // 转换为 Float 数组
            guard let channelData = buffer.floatChannelData else { return }
            let frameLength = Int(buffer.frameLength)
            let samples = Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
            
            // 处理音频
            DispatchQueue.main.async {
                engine.processAudioBuffer(samples)
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
                VStack(spacing: 12) {
                    Text("请对着屏幕说话...")
                        .foregroundStyle(.secondary)
                    
                    ProgressView(value: engine.calibrationProgress)
                        .progressViewStyle(.linear)
                        .tint(.blue)
                        .frame(maxWidth: 200)
                    
                    Text("\(Int(engine.calibrationProgress * 100))%")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                Text("需要 5 秒语音样本")
                    .foregroundStyle(.secondary)
                
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
        VStack(spacing: 20) {
            // 状态指示
            HStack(spacing: 16) {
                Text(engine.debugInfo.gazeStatus)
                    .font(.headline)
                
                if engine.debugInfo.speakerMatch {
                    Text("🗣️ 用户说话")
                        .font(.headline)
                        .foregroundStyle(.green)
                } else {
                    Text("🔇 非用户")
                        .font(.headline)
                        .foregroundStyle(.secondary)
                }
            }
            .padding()
            .background(Color.white.opacity(0.1))
            .cornerRadius(12)
            
            // 转录内容
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    if engine.transcript.isEmpty {
                        Text("等待转录...")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding(.vertical, 40)
                    } else {
                        ForEach(Array(engine.transcript.enumerated()), id: \.offset) { _, text in
                            Text(text)
                                .foregroundStyle(.white)
                                .padding(.vertical, 4)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 300)
            .padding()
            .background(Color.white.opacity(0.05))
            .cornerRadius(12)
            
            // 操作按钮
            HStack(spacing: 16) {
                Button {
                    engine.reset()
                } label: {
                    Text("重新标定")
                        .font(.subheadline)
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Color.orange)
                        .cornerRadius(8)
                }
                
                Button {
                    engine.clearTranscript()
                } label: {
                    Text("清空")
                        .font(.subheadline)
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Color.red)
                        .cornerRadius(8)
                }
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
            
            debugRow(label: "Gaze", value: engine.debugInfo.gazeStatus)
            debugRow(
                label: "Speaker",
                value: engine.debugInfo.speakerMatch ? "✅ 匹配" : "❌ 不匹配"
            )
            debugRow(
                label: "距离",
                value: String(format: "%.2f / %.2f", engine.debugInfo.speakerDistance, 0.65)
            )
            debugRow(
                label: "音频",
                value: String(format: "%.1f dB", engine.debugInfo.audioLevel)
            )
            debugRow(label: "用户声纹", value: engine.debugInfo.userEmbeddingStatus)
        }
        .padding()
        .background(Color.white.opacity(0.1))
        .cornerRadius(12)
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
}
