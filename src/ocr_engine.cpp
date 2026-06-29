#include "ocr_engine.hpp"
#include <algorithm>

OcrEngine::OcrEngine(const std::string& det_model, const std::string& rec_model, const std::string& dict, int gpu, int det_w, int det_h, int rec_w, int rec_h, int roi_w, int roi_h, float det_thr, float box_thr, float rec_thr)
    : det_(det_model, gpu), rec_(rec_model, gpu), db_(det_w, det_h, roi_w, roi_h, det_thr, box_thr), ctc_(dict), det_w_(det_w), det_h_(det_h), rec_w_(rec_w), rec_h_(rec_h), rec_thr_(rec_thr) {}

std::vector<TextBox> OcrEngine::detect(float* det_tensor_gpu) {
    auto out = det_.infer_gpu_float(det_tensor_gpu, {1,3,det_h_,det_w_});
    // Accept common output layouts: [1,1,H,W] or [1,H,W]
    return db_.run(out);
}

std::vector<OcrLine> OcrEngine::recognize(float* rec_tensor_gpu, const std::vector<TextBox>& boxes) {
    std::vector<OcrLine> lines;
    if (boxes.empty()) return lines;
    int batch = (int)boxes.size();
    auto out = rec_.infer_gpu_float(rec_tensor_gpu, {batch,3,rec_h_,rec_w_});
    int classes = 0, steps = 0;
    // Heuristic for common PP-OCR rec output [B,T,C].
    int total = (int)out.size();
    if (batch > 0) {
        int per = total / batch;
        // ppocr ch dict around 6625 chars; infer classes by trying large dim.
        if (per % 6625 == 0) { classes = 6625; steps = per / classes; }
        else if (per % 6624 == 0) { classes = 6624; steps = per / classes; }
        else if (per % 6623 == 0) { classes = 6623; steps = per / classes; }
        else { classes = 6625; steps = per / classes; }
    }
    if (steps <= 0 || classes <= 0) return lines;
    for (int b=0;b<batch;b++) {
        auto [txt, score] = ctc_.decode(out.data() + (size_t)b*steps*classes, steps, classes);
        if (!txt.empty() && score >= rec_thr_) lines.push_back({boxes[b].box, txt, score});
    }
    return lines;
}

std::vector<OcrLine> OcrEngine::run(float* det_tensor_gpu, float* rec_tensor_gpu, const std::vector<TextBox>& rec_boxes) {
    (void)det_tensor_gpu;
    return recognize(rec_tensor_gpu, rec_boxes);
}
