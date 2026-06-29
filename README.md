# videocr-zero

C++/CUDA/TensorRT streaming OCR engine for subtitle extraction.

Target pipeline:

```text
FFmpeg demux CPU -> NVDEC GPU frame -> CUDA ROI crop/resize/normalize -> TensorRT det -> DB postprocess -> CUDA recognizer crop -> TensorRT rec -> CTC decode -> temporal merger -> SRT
```

This repo is designed for Colab/Linux with NVIDIA GPU. It expects PP-OCRv5 det/rec ONNX models converted to TensorRT engines.

## Colab quick start

```bash
bash scripts/setup_colab.sh
bash scripts/prepare_models.sh
bash scripts/build.sh
./build/videocr_zero --video /content/video.mp4 --out /content/output.srt \
  --det models/det.engine --rec models/rec.engine --dict models/ppocr_keys_v1.txt \
  --crop 0,760,1920,320 --det-size 960,192 --rec-size 320,48
```

## Notes

- CPU does not receive decoded frames or OCR crop images.
- CPU receives small OCR outputs for postprocess/decode and writes SRT.
- The code uses NVIDIA `NvDecoder` from `video-sdk-samples`, downloaded by setup script.
- The exact PP-OCRv5 ONNX export URLs/commands may need adjustment depending on PaddleOCR model release naming.
