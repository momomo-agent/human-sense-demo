# HumanSense Demo — iOS App Spec

## 概述

一个 iOS demo app，用前置摄像头实时感知用户状态：看哪里、脸朝哪、嘴是否张开、是否在说话、有没有人声。把这些信号融合成一个完整的"人的状态"可视化出来。

## 技术栈

- **SwiftUI** + **ARKit**（ARFaceTrackingConfiguration）
- **AVAudioEngine**（音频检测）
- 最低 iOS 17，iPhone X+（TrueDepth camera）
- Swift Package Manager，无外部依赖
- 用 XcodeGen 生成 .xcodeproj（`project.yml`）

## 架构

### 数据源层

1. **FaceTrackingManager**（ARKit）
   - 使用 `ARSession` + `ARSessionDelegate`（不用 ARSCNView，纯数据模式更轻）
   - 用 `ARCamera.projectPoint` 做 gaze → 屏幕坐标映射（参考 kyle-fox/ios-eye-tracking）
   - 独立 DispatchQueue 处理 ARKit 回调（参考 IT-Jim 架构）
   - LowPassFilter 平滑 gaze 坐标（α=0.85）
   - 从 ARFaceAnchor 提取：
     - `lookAtPoint` → 屏幕坐标（用户看哪）
     - `transform` → yaw/pitch/roll（头部朝向）
     - `blendShapes` → jawOpen, mouthClose, eyeBlink, eyeLookIn/Out/Up/Down 等（表情状态）
   - **输出**: `FaceState` 数据结构，60fps 更新

2. **AudioDetectionManager**
   - AVAudioEngine → installTap → RMS 音量计算
   - 阈值判断：有声音 / 静音
   - 频谱分析可选（区分人声 vs 噪音）
   - **输出**: `AudioState`（isSpeaking: Bool, volume: Float）

### 融合层

3. **HumanStateEngine**（@Observable）
   - 融合 FaceState + AudioState → HumanState
   - 状态推理规则：
     - `lookingAtScreen`: headYaw < 25° && headPitch < 20°
     - `speaking`: jawOpen 帧间 delta 高 + 有人声
     - `listening`: 看屏幕 + 嘴闭着 + 安静
     - `distracted`: 不看屏幕
     - `absent`: 没检测到脸
     - `eyesClosed`: eyeBlink > 0.8 双眼
   - 状态防抖：状态变化需持续 300ms 以上才切换（防闪烁）

### 展示层

4. **ContentView**（SwiftUI）
   - 顶部：前置摄像头预览（ARView 或 AVCaptureVideoPreviewLayer，小窗口即可）
   - 中间：**状态卡片**
     - 当前状态大字显示（如 "正在看屏幕说话"）
     - 状态图标 + 颜色（绿=注视，黄=分心，红=离开）
   - 下部：**实时数据面板**
     - Gaze 坐标（屏幕上的红点，跟着眼睛走）
     - 头部朝向（yaw/pitch/roll 三个条状指示器）
     - 嘴巴状态（jawOpen 值 + 进度条）
     - 眼睛状态（左右眼 blink 值 + 方向）
     - 音频波形 / 音量指示器
     - 是否在说话的指示灯
   - 布局：一屏看完，不要滚动

## 文件结构

```
human-sense-demo/
├── project.yml              # XcodeGen 配置
├── SPEC.md
├── Sources/
│   ├── App.swift             # @main, SwiftUI App
│   ├── ContentView.swift     # 主界面
│   ├── Tracking/
│   │   ├── FaceTrackingManager.swift    # ARKit 数据采集
│   │   ├── AudioDetectionManager.swift  # 音频检测
│   │   ├── HumanStateEngine.swift       # 状态融合
│   │   └── LowPassFilter.swift          # 平滑滤波器
│   ├── Models/
│   │   ├── FaceState.swift              # 脸部数据结构
│   │   ├── AudioState.swift             # 音频数据结构
│   │   └── HumanState.swift             # 融合状态
│   └── Views/
│       ├── CameraPreview.swift          # 摄像头预览
│       ├── StateCard.swift              # 状态卡片
│       ├── GazeOverlay.swift            # Gaze 红点
│       ├── HeadOrientationView.swift    # 头部朝向
│       ├── BlendShapePanel.swift        # Blend shape 面板
│       └── AudioVisualizerView.swift    # 音频可视化
├── Resources/
│   └── Assets.xcassets/
└── Info.plist                # NSCameraUsageDescription + NSMicrophoneUsageDescription
```

## 关键实现细节

### Gaze 屏幕坐标映射（参考 kyle-fox）

```swift
// 1. face 坐标系 → world 坐标系
let lookAtVector = anchor.transform * SIMD4<Float>(anchor.lookAtPoint, 1)

// 2. 投影到屏幕坐标（Apple 原生 API）
let lookPoint = frame.camera.projectPoint(
    SIMD3<Float>(x: lookAtVector.x, y: lookAtVector.y, z: lookAtVector.z),
    orientation: orientation,
    viewportSize: UIScreen.main.bounds.size
)
```

### 头部朝向提取

```swift
let transform = anchor.transform
let yaw = atan2(transform.columns.0.z, transform.columns.2.z)   // 左右转
let pitch = asin(-transform.columns.1.z)                          // 上下仰
let roll = atan2(transform.columns.1.x, transform.columns.1.y)   // 歪头
```

### 说话检测（视觉+音频融合）

```swift
// 视觉：jawOpen 帧间变化率
let jawDelta = abs(currentJawOpen - previousJawOpen)
let mouthMoving = jawDelta > 0.02  // 嘴在动

// 音频：有人声
let hasVoice = audioState.volume > 0.01

// 融合判断
let isSpeaking = mouthMoving && hasVoice && isLookingAtScreen
```

### LowPassFilter

```swift
struct LowPassFilter {
    var value: CGFloat
    let filterValue: CGFloat  // 0.85 推荐

    mutating func update(with newValue: CGFloat) {
        value = filterValue * value + (1.0 - filterValue) * newValue
    }
}
```

## 性能要求

- 60fps 数据采集不掉帧
- UI 更新限制在 30fps（用 throttle）
- ARKit 回调在独立队列，不阻塞 main thread
- 内存 < 100MB

## 权限

- Info.plist:
  - `NSCameraUsageDescription`: "用于人脸追踪和注视检测"
  - `NSMicrophoneUsageDescription`: "用于语音活动检测"

## 不需要做

- 不需要数据存储/持久化
- 不需要校准流程
- 不需要多人脸支持
- 不需要后置摄像头
- 不需要网络功能
