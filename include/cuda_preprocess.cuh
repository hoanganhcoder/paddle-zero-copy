#pragma once
#include "common.hpp"
#include <cuda_runtime.h>
#include <vector>

class CudaPreprocessor {
public:
    CudaPreprocessor(RectI crop, int det_w, int det_h, int rec_w, int rec_h);
    ~CudaPreprocessor();
    float* det_tensor() const { return det_tensor_; }
    float* rec_tensor() const { return rec_tensor_; }
    uint8_t* roi_rgb() const { return roi_rgb_; }
    int roi_w() const { return crop_.w; }
    int roi_h() const { return crop_.h; }
    int det_w() const { return det_w_; }
    int det_h() const { return det_h_; }
    int rec_w() const { return rec_w_; }
    int rec_h() const { return rec_h_; }
    float prepare_det(const GpuFrame& frame, cudaStream_t stream);
    void prepare_rec_batch(const std::vector<TextBox>& boxes, cudaStream_t stream, int max_batch);
private:
    RectI crop_;
    int det_w_, det_h_, rec_w_, rec_h_;
    uint8_t* roi_rgb_ = nullptr;
    uint8_t* prev_roi_rgb_ = nullptr;
    float* det_tensor_ = nullptr;
    float* rec_tensor_ = nullptr;
    unsigned int* diff_sum_ = nullptr;
    size_t roi_bytes_ = 0;
};
