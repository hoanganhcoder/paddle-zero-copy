#!/usr/bin/env bash
set -euo pipefail
export ORT_ROOT=${ORT_ROOT:-$(pwd)/third_party/onnxruntime}
cmake -S . -B build -G Ninja -DORT_ROOT="$ORT_ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
