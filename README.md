# paddle-zero-copy-ort

GPU streaming subtitle OCR engine for Colab/Linux.

This branch removes TensorRT and uses **ONNX Runtime CUDA C++** so it can build on Colab without TensorRT SDK headers.

Pipeline:

```text
FFmpeg demux
-> FFmpeg CUDA/NVDEC decode, AV_PIX_FMT_CUDA
-> CUDA NV12 ROI crop to RGB
-> CUDA diff map / skip unchanged subtitle frames
-> CUDA resize/normalize det tensor
-> ONNX Runtime CUDA PP-OCRv5 detector
-> DB postprocess on small detector map
-> CUDA crop/resize text boxes
-> ONNX Runtime CUDA PP-OCRv5 recognizer
-> CTC decode
-> temporal subtitle merge
-> SRT
```

Important honesty note:

- It does **not** copy full frames to CPU.
- It keeps video frame, ROI, preprocess tensors on GPU.
- Detector output and recognizer logits are copied back for postprocess/decode.
- This is the Colab-friendly zero-copy-core version. Full GPU DB postprocess can be added later without changing the public pipeline.

## Colab install/build

```bash
%%bash
cd /content
rm -rf paddle-zero-copy-ort
git clone <your-repo-url> paddle-zero-copy-ort
cd paddle-zero-copy-ort
bash scripts/setup_colab.sh
bash scripts/download_models.sh
bash scripts/build.sh
```

If using the zip instead of git:

```bash
%%bash
cd /content
unzip /content/paddle-zero-copy-ort.zip -d /content
cd /content/paddle-zero-copy-ort
bash scripts/setup_colab.sh
bash scripts/download_models.sh
bash scripts/build.sh
```

## Run

```bash
%%bash
cd /content/paddle-zero-copy-ort
bash scripts/run_colab.sh /content/video.mp4 /content/output.srt
```

Manual:

```bash
export LD_LIBRARY_PATH="$(pwd)/third_party/onnxruntime/lib:${LD_LIBRARY_PATH:-}"
./build/paddle_zero_copy \
  --video /content/video.mp4 \
  --out /content/output.srt \
  --det models/det.onnx \
  --rec models/rec.onnx \
  --dict models/ppocr_keys_v1.txt \
  --crop 0,760,1920,320 \
  --det-size 960x192 \
  --rec-size 320x48 \
  --diff 0.006
```

## Tuning

- `--crop x,y,w,h`: subtitle region. Crop as tight as possible.
- `--diff`: lower = more OCR calls, safer. Higher = faster but may skip short subtitle changes.
- `--det-size`: detector tensor size. Bigger = more accurate, slower.
- `--rec-size`: recognizer tensor size.
- `--sim`: text similarity merge threshold.
- `--max-empty-frames`: how many empty OCR frames before closing current subtitle.

## Model files

Expected:

```text
models/det.onnx
models/rec.onnx
models/ppocr_keys_v1.txt
```

`scripts/download_models.sh` downloads `begoshensaid/pp-ocrv5-server` from ModelScope and falls back to PaddleOCR dict.

## Known model caveat

PP-OCR ONNX model output tensor shapes vary by export method. If recognition output class count is not around the dictionary size, edit `src/ocr_engine.cpp` and adjust the recognition output layout logic.

## Why ONNX Runtime CUDA, not TensorRT?

Colab normally has CUDA but not TensorRT development headers (`NvInfer.h`). ONNX Runtime CUDA ships cleanly as a downloadable SDK and builds much more reliably on Colab.
