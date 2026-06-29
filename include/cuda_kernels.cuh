#pragma once
#include <cuda_runtime.h>
#include "common.hpp"

void nv12_roi_to_det_tensor(
  const uint8_t* nv12, int pitch, int src_w, int src_h,
  CropRect crop, float* out_nchw, int out_w, int out_h, cudaStream_t stream);

void nv12_box_to_rec_tensor(
  const uint8_t* nv12, int pitch, int src_w, int src_h,
  CropRect roi, DetBox box, float* out_nchw, int out_w, int out_h, cudaStream_t stream);

void gray_signature_from_nv12_roi(
  const uint8_t* nv12, int pitch, int src_w, int src_h,
  CropRect crop, float* sig, int sig_w, int sig_h, cudaStream_t stream);

float signature_similarity(float* a, float* b, int n, cudaStream_t stream);
