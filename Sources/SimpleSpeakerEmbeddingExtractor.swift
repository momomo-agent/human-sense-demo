import Foundation
import CoreML
import Accelerate

/// 简化的 Speaker Embedding 提取器
/// 用于 gaze-gated speaker recognition 场景
class SimpleSpeakerEmbeddingExtractor {
    private let model: MLModel
    
    init() throws {
        guard let modelURL = Bundle.main.url(
            forResource: "Embedding",
            withExtension: "mlmodelc"
        ) else {
            throw ExtractorError.modelNotFound
        }
        
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
    }
    
    /// 从音频样本提取 speaker embedding
    /// - Parameter samples: 16kHz mono Float32 音频，至少 1 秒（16000 samples）
    /// - Returns: 256 维 embedding 向量
    func extract(from samples: [Float]) throws -> [Float] {
        // 确保至少有 1 秒音频
        guard samples.count >= 16000 else {
            throw ExtractorError.insufficientAudio
        }
        
        // 截取或填充到 160000 samples (10 秒)
        let targetLength = 160_000
        var audioData = samples
        
        if audioData.count < targetLength {
            // 填充零
            audioData.append(contentsOf: [Float](repeating: 0.0, count: targetLength - audioData.count))
        } else if audioData.count > targetLength {
            // 截取前 10 秒
            audioData = Array(audioData.prefix(targetLength))
        }
        
        // 创建 waveform MLMultiArray (shape: [3, 160000])
        let waveformArray = try MLMultiArray(shape: [3, 160_000], dataType: .float32)
        for i in 0..<targetLength {
            waveformArray[i] = NSNumber(value: audioData[i])
        }
        
        // 创建 mask MLMultiArray (shape: [3, 1000])
        // 使用全 1 mask（假设整段音频都是说话）
        let maskLength = 1000
        let maskArray = try MLMultiArray(shape: [3, maskLength], dataType: .float32)
        for i in 0..<maskLength {
            maskArray[i] = NSNumber(value: 1.0)
        }
        
        // 创建输入
        let input = EmbeddingModelInput(waveform: waveformArray, mask: maskArray)
        
        // 推理
        let output = try model.prediction(from: input)
        
        // 提取 embedding
        guard let embeddingFeature = output.featureValue(for: "embedding"),
              let embeddingArray = embeddingFeature.multiArrayValue else {
            throw ExtractorError.invalidOutput
        }
        
        // 转换为 Float 数组
        var embedding: [Float] = []
        for i in 0..<embeddingArray.count {
            embedding.append(Float(truncating: embeddingArray[i]))
        }
        
        return embedding
    }
    
    enum ExtractorError: Error {
        case modelNotFound
        case insufficientAudio
        case invalidOutput
    }
}

/// CoreML 模型输入（手动实现 MLFeatureProvider）
private class EmbeddingModelInput: MLFeatureProvider {
    let waveform: MLMultiArray
    let mask: MLMultiArray
    
    var featureNames: Set<String> {
        return ["waveform", "mask"]
    }
    
    init(waveform: MLMultiArray, mask: MLMultiArray) {
        self.waveform = waveform
        self.mask = mask
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "waveform":
            return MLFeatureValue(multiArray: waveform)
        case "mask":
            return MLFeatureValue(multiArray: mask)
        default:
            return nil
        }
    }
}
