#!/bin/bash
set -e

# 下载 FluidAudio speaker diarization 模型到 app bundle

REPO="FluidInference/speaker-diarization-coreml"
MODELS_DIR="$(dirname "$0")/../Resources/Models"

mkdir -p "$MODELS_DIR"

echo "📦 下载 speaker diarization 模型..."

# 只下载 embedding 模型（speaker recognition 需要）
# Segmentation 模型用于完整 diarization，暂时不需要

cd "$MODELS_DIR"

# 使用 git lfs 下载（如果有 git lfs）
if command -v git-lfs &> /dev/null; then
    echo "使用 git lfs 下载..."
    git lfs install
    git clone --depth 1 --filter=blob:none --sparse https://huggingface.co/$REPO temp_repo
    cd temp_repo
    git sparse-checkout set Embedding.mlmodelc
    git lfs pull --include="Embedding.mlmodelc/**"
    mv Embedding.mlmodelc ../
    cd ..
    rm -rf temp_repo
else
    echo "❌ 需要 git-lfs 下载模型"
    echo "安装: brew install git-lfs"
    exit 1
fi

echo "✅ 模型下载完成: $MODELS_DIR/Embedding.mlmodelc"
echo "📊 模型大小:"
du -sh Embedding.mlmodelc
