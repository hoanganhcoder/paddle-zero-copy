#!/usr/bin/env bash
set -euo pipefail
mkdir -p models
python - <<'PY'
from modelscope import snapshot_download
from pathlib import Path
import shutil
src = Path(snapshot_download('begoshensaid/pp-ocrv5-server'))
dst = Path('models')
print('downloaded', src)
for p in src.rglob('*.onnx'):
    s = str(p).lower()
    if 'det' in s and not (dst/'det.onnx').exists(): shutil.copy(p, dst/'det.onnx')
    if 'rec' in s and not (dst/'rec.onnx').exists(): shutil.copy(p, dst/'rec.onnx')
for p in src.rglob('*.txt'):
    s = str(p).lower()
    if any(k in s for k in ['dict','keys','char']):
        shutil.copy(p, dst/'ppocr_keys_v1.txt'); break
print('models:')
for p in dst.iterdir(): print(p, p.stat().st_size)
PY
if [ ! -f models/ppocr_keys_v1.txt ]; then
  wget -O models/ppocr_keys_v1.txt https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/ppocr_keys_v1.txt
fi
ls -lh models
