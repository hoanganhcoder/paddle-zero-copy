#!/usr/bin/env bash
set -e
mkdir -p models
cat <<'MSG'
Put PP-OCRv5 ONNX models here:
  models/det.onnx
  models/rec.onnx
  models/ppocr_keys_v1.txt

Then build TensorRT engines, example:
  trtexec --onnx=models/det.onnx --saveEngine=models/det.engine --fp16 --minShapes=x:1x3x192x960 --optShapes=x:1x3x192x960 --maxShapes=x:1x3x192x960
  trtexec --onnx=models/rec.onnx --saveEngine=models/rec.engine --fp16 --minShapes=x:1x3x48x320 --optShapes=x:1x3x48x320 --maxShapes=x:1x3x48x320

You may need to inspect real input/output tensor names using:
  polygraphy inspect model models/det.onnx
  polygraphy inspect model models/rec.onnx
and then edit tensor names in src/main.cpp:
  det input/output: x / sigmoid_0.tmp_0
  rec input/output: x / softmax_0.tmp_0
MSG
