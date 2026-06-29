#!/usr/bin/env bash
set -euo pipefail
VIDEO=${1:-/content/video.mp4}
OUT=${2:-/content/output.srt}
export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime/lib:${LD_LIBRARY_PATH:-}"
./build/paddle_zero_copy \
  --video "$VIDEO" \
  --out "$OUT" \
  --det models/det.onnx \
  --rec models/rec.onnx \
  --dict models/ppocr_keys_v1.txt \
  --crop 0,760,1920,320 \
  --det-size 960x192 \
  --rec-size 320x48 \
  --diff 0.006 \
  --det-threshold 0.30 \
  --box-threshold 0.50 \
  --rec-threshold 0.45
