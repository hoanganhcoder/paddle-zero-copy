#!/usr/bin/env bash
set -e
apt-get update
apt-get install -y cmake ninja-build git pkg-config build-essential ffmpeg libavformat-dev libavcodec-dev libavutil-dev
mkdir -p third_party
if [ ! -d third_party/video-sdk-samples ]; then
  git clone --depth 1 https://github.com/NVIDIA/video-sdk-samples.git third_party/video-sdk-samples
fi
if [ ! -f /usr/include/NvInfer.h ] && [ ! -f /usr/lib/x86_64-linux-gnu/libnvinfer.so ]; then
  echo "TensorRT headers/libs not found. On Colab, install TensorRT matching CUDA, or use NVIDIA PyTorch/TensorRT container."
fi
