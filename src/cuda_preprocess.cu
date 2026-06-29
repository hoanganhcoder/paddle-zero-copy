#include "cuda_preprocess.cuh"
#include <cuda_runtime.h>
#include <cmath>

__device__ inline unsigned char clamp_u8(float v) { return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

__global__ void nv12_crop_to_rgb_kernel(const uint8_t* y, const uint8_t* uv, uint8_t* out, int src_w, int src_h, int pitch_y, int pitch_uv, int cx, int cy, int cw, int ch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cw || yy >= ch) return;
    int sx = cx + x, sy = cy + yy;
    if (sx < 0 || sx >= src_w || sy < 0 || sy >= src_h) return;
    int Y = y[sy * pitch_y + sx];
    int uvx = (sx / 2) * 2;
    int uvy = sy / 2;
    int U = uv[uvy * pitch_uv + uvx + 0] - 128;
    int V = uv[uvy * pitch_uv + uvx + 1] - 128;
    float R = Y + 1.402f * V;
    float G = Y - 0.344136f * U - 0.714136f * V;
    float B = Y + 1.772f * U;
    int o = (yy * cw + x) * 3;
    out[o+0] = clamp_u8(R);
    out[o+1] = clamp_u8(G);
    out[o+2] = clamp_u8(B);
}

__global__ void diff_sum_kernel(const uint8_t* a, const uint8_t* b, unsigned int* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(sum, (unsigned int)abs((int)a[i] - (int)b[i]));
}

__global__ void rgb_resize_norm_chw_kernel(const uint8_t* src, float* dst, int sw, int sh, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    float gx = (x + 0.5f) * sw / dw - 0.5f;
    float gy = (y + 0.5f) * sh / dh - 0.5f;
    int x0 = max(0, min(sw-1, (int)floorf(gx)));
    int y0 = max(0, min(sh-1, (int)floorf(gy)));
    int x1 = min(sw-1, x0+1);
    int y1 = min(sh-1, y0+1);
    float tx = gx - x0, ty = gy - y0;
    int idx00 = (y0*sw+x0)*3, idx01=(y0*sw+x1)*3, idx10=(y1*sw+x0)*3, idx11=(y1*sw+x1)*3;
    int pix = y*dw+x;
    for (int c=0;c<3;c++) {
        float v = (1-tx)*(1-ty)*src[idx00+c] + tx*(1-ty)*src[idx01+c] + (1-tx)*ty*src[idx10+c] + tx*ty*src[idx11+c];
        v = v / 255.0f;
        v = (v - 0.5f) / 0.5f;
        dst[c*dw*dh + pix] = v;
    }
}

__global__ void rgb_crop_resize_norm_batch_kernel(const uint8_t* roi, float* dst, const TextBox* boxes, int sw, int sh, int rec_w, int rec_h, int batch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (x >= rec_w || y >= rec_h || b >= batch) return;
    TextBox tb = boxes[b];
    int bx = max(0, tb.box.x), by = max(0, tb.box.y);
    int bw = max(1, min(sw - bx, tb.box.w));
    int bh = max(1, min(sh - by, tb.box.h));
    float gx = bx + (x + 0.5f) * bw / rec_w - 0.5f;
    float gy = by + (y + 0.5f) * bh / rec_h - 0.5f;
    int x0 = max(0, min(sw-1, (int)floorf(gx)));
    int y0 = max(0, min(sh-1, (int)floorf(gy)));
    int x1 = min(sw-1, x0+1); int y1 = min(sh-1, y0+1);
    float tx = gx - x0, ty = gy - y0;
    int idx00=(y0*sw+x0)*3, idx01=(y0*sw+x1)*3, idx10=(y1*sw+x0)*3, idx11=(y1*sw+x1)*3;
    int plane = rec_w * rec_h;
    int base = b * 3 * plane;
    int pix = y * rec_w + x;
    for (int c=0;c<3;c++) {
        float v = (1-tx)*(1-ty)*roi[idx00+c] + tx*(1-ty)*roi[idx01+c] + (1-tx)*ty*roi[idx10+c] + tx*ty*roi[idx11+c];
        v = v / 255.0f;
        v = (v - 0.5f) / 0.5f;
        dst[base + c*plane + pix] = v;
    }
}

CudaPreprocessor::CudaPreprocessor(RectI crop, int det_w, int det_h, int rec_w, int rec_h): crop_(crop), det_w_(det_w), det_h_(det_h), rec_w_(rec_w), rec_h_(rec_h) {
    roi_bytes_ = (size_t)crop_.w * crop_.h * 3;
    CHECK_CUDA(cudaMalloc(&roi_rgb_, roi_bytes_));
    CHECK_CUDA(cudaMalloc(&prev_roi_rgb_, roi_bytes_));
    CHECK_CUDA(cudaMemset(prev_roi_rgb_, 0, roi_bytes_));
    CHECK_CUDA(cudaMalloc(&det_tensor_, (size_t)3 * det_w_ * det_h_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&rec_tensor_, (size_t)64 * 3 * rec_w_ * rec_h_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&diff_sum_, sizeof(unsigned int)));
}
CudaPreprocessor::~CudaPreprocessor(){ cudaFree(roi_rgb_); cudaFree(prev_roi_rgb_); cudaFree(det_tensor_); cudaFree(rec_tensor_); cudaFree(diff_sum_); }

float CudaPreprocessor::prepare_det(const GpuFrame& frame, cudaStream_t stream) {
    dim3 block(16,16); dim3 grid((crop_.w+15)/16,(crop_.h+15)/16);
    nv12_crop_to_rgb_kernel<<<grid,block,0,stream>>>(frame.y, frame.uv, roi_rgb_, frame.width, frame.height, frame.pitch_y, frame.pitch_uv, crop_.x, crop_.y, crop_.w, crop_.h);
    CHECK_CUDA(cudaMemsetAsync(diff_sum_, 0, sizeof(unsigned int), stream));
    int n = (int)roi_bytes_;
    diff_sum_kernel<<<(n+255)/256,256,0,stream>>>(roi_rgb_, prev_roi_rgb_, diff_sum_, n);
    unsigned int host_sum=0;
    CHECK_CUDA(cudaMemcpyAsync(&host_sum, diff_sum_, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    rgb_resize_norm_chw_kernel<<<dim3((det_w_+15)/16,(det_h_+15)/16),block,0,stream>>>(roi_rgb_, det_tensor_, crop_.w, crop_.h, det_w_, det_h_);
    CHECK_CUDA(cudaMemcpyAsync(prev_roi_rgb_, roi_rgb_, roi_bytes_, cudaMemcpyDeviceToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return (float)host_sum / (float)(roi_bytes_ * 255.0f);
}

void CudaPreprocessor::prepare_rec_batch(const std::vector<TextBox>& boxes, cudaStream_t stream, int max_batch) {
    int batch = std::min((int)boxes.size(), max_batch);
    if (batch <= 0) return;
    TextBox* d_boxes=nullptr;
    CHECK_CUDA(cudaMalloc(&d_boxes, sizeof(TextBox)*batch));
    CHECK_CUDA(cudaMemcpyAsync(d_boxes, boxes.data(), sizeof(TextBox)*batch, cudaMemcpyHostToDevice, stream));
    dim3 block(16,16); dim3 grid((rec_w_+15)/16,(rec_h_+15)/16,batch);
    rgb_crop_resize_norm_batch_kernel<<<grid,block,0,stream>>>(roi_rgb_, rec_tensor_, d_boxes, crop_.w, crop_.h, rec_w_, rec_h_, batch);
    CHECK_CUDA(cudaFree(d_boxes));
}
