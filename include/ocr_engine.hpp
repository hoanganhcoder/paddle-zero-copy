#pragma once
#include "common.hpp"
#include "onnx_engine.hpp"
#include "db_postprocess.hpp"
#include "ctc_decoder.hpp"
#include <memory>

class OcrEngine {
public:
    OcrEngine(const std::string& det_model, const std::string& rec_model, const std::string& dict, int gpu, int det_w, int det_h, int rec_w, int rec_h, int roi_w, int roi_h, float det_thr, float box_thr, float rec_thr);
    std::vector<OcrLine> run(float* det_tensor_gpu, float* rec_tensor_gpu, const std::vector<TextBox>& rec_boxes);
    std::vector<TextBox> detect(float* det_tensor_gpu);
    std::vector<OcrLine> recognize(float* rec_tensor_gpu, const std::vector<TextBox>& boxes);
private:
    OnnxCudaEngine det_;
    OnnxCudaEngine rec_;
    DbPostProcessor db_;
    CtcDecoder ctc_;
    int det_w_, det_h_, rec_w_, rec_h_;
    float rec_thr_;
};
