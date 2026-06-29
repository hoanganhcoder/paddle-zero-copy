#!/usr/bin/env bash
set -euo pipefail
apt-get update
apt-get install -y build-essential cmake ninja-build git wget unzip pkg-config ffmpeg libavformat-dev libavcodec-dev libavutil-dev
python -m pip install -U onnxruntime-gpu modelscope huggingface_hub onnx onnxsim
mkdir -p third_party
cd third_party
ORT_VERSION=${ORT_VERSION:-1.22.0}
ORT_DIR="onnxruntime-linux-x64-gpu-${ORT_VERSION}"
if [ ! -d "$ORT_DIR" ]; then
  wget -q --show-progress "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_DIR}.tgz"
  tar -xzf "${ORT_DIR}.tgz"
fi
rm -rf onnxruntime
ln -s "$ORT_DIR" onnxruntime
cd ..
echo "ORT_ROOT=$(pwd)/third_party/onnxruntime"
