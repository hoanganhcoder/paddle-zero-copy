#include "cuda_kernels.cuh"
#include "common.hpp"
#include <cuda_runtime.h>

__device__ float clampf(float v, float a, float b){ return fminf(fmaxf(v,a),b); }

__device__ void nv12_to_rgb_pixel(const uint8_t* nv12, int pitch, int w, int h, int x, int y, float& r, float& g, float& b) {
  x = max(0, min(x, w-1)); y = max(0, min(y, h-1));
  int Y = nv12[y*pitch + x];
  const uint8_t* uv = nv12 + pitch*h;
  int uvx = (x/2)*2; int uvy = y/2;
  int U = uv[uvy*pitch + uvx] - 128;
  int V = uv[uvy*pitch + uvx + 1] - 128;
  float yy = (float)Y;
  r = clampf(yy + 1.402f*V, 0.f, 255.f);
  g = clampf(yy - 0.344136f*U - 0.714136f*V, 0.f, 255.f);
  b = clampf(yy + 1.772f*U, 0.f, 255.f);
}

__global__ void det_kernel(const uint8_t* nv12, int pitch, int sw, int sh, CropRect crop, float* out, int ow, int oh) {
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=ow||y>=oh) return;
  float sx = crop.x + (x + 0.5f) * crop.w / ow;
  float sy = crop.y + (y + 0.5f) * crop.h / oh;
  float r,g,b; nv12_to_rgb_pixel(nv12,pitch,sw,sh,(int)sx,(int)sy,r,g,b);
  int idx = y*ow+x;
  out[0*ow*oh+idx] = (r/255.f - 0.485f)/0.229f;
  out[1*ow*oh+idx] = (g/255.f - 0.456f)/0.224f;
  out[2*ow*oh+idx] = (b/255.f - 0.406f)/0.225f;
}

void nv12_roi_to_det_tensor(const uint8_t* nv12, int pitch, int src_w, int src_h, CropRect crop, float* out_nchw, int out_w, int out_h, cudaStream_t stream) {
  dim3 block(16,16), grid((out_w+15)/16,(out_h+15)/16);
  det_kernel<<<grid,block,0,stream>>>(nv12,pitch,src_w,src_h,crop,out_nchw,out_w,out_h);
}

__global__ void rec_kernel(const uint8_t* nv12, int pitch, int sw, int sh, CropRect roi, DetBox box, float* out, int ow, int oh) {
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=ow||y>=oh) return;
  float bx1 = roi.x + box.x1 * roi.w;
  float by1 = roi.y + box.y1 * roi.h;
  float bw = (box.x2 - box.x1) * roi.w;
  float bh = (box.y2 - box.y1) * roi.h;
  float sx = bx1 + (x + 0.5f) * bw / ow;
  float sy = by1 + (y + 0.5f) * bh / oh;
  float r,g,b; nv12_to_rgb_pixel(nv12,pitch,sw,sh,(int)sx,(int)sy,r,g,b);
  int idx=y*ow+x;
  out[0*ow*oh+idx]=(r/255.f-0.5f)/0.5f;
  out[1*ow*oh+idx]=(g/255.f-0.5f)/0.5f;
  out[2*ow*oh+idx]=(b/255.f-0.5f)/0.5f;
}

void nv12_box_to_rec_tensor(const uint8_t* nv12, int pitch, int src_w, int src_h, CropRect roi, DetBox box, float* out_nchw, int out_w, int out_h, cudaStream_t stream) {
  dim3 block(16,16), grid((out_w+15)/16,(out_h+15)/16);
  rec_kernel<<<grid,block,0,stream>>>(nv12,pitch,src_w,src_h,roi,box,out_nchw,out_w,out_h);
}

__global__ void sig_kernel(const uint8_t* nv12, int pitch, int sw, int sh, CropRect crop, float* sig, int ow, int oh) {
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=ow||y>=oh) return;
  float sx = crop.x + (x+0.5f)*crop.w/ow;
  float sy = crop.y + (y+0.5f)*crop.h/oh;
  int Y = nv12[((int)sy)*pitch + (int)sx];
  sig[y*ow+x] = Y / 255.f;
}
void gray_signature_from_nv12_roi(const uint8_t* nv12, int pitch, int src_w, int src_h, CropRect crop, float* sig, int sig_w, int sig_h, cudaStream_t stream) {
  dim3 block(16,16), grid((sig_w+15)/16,(sig_h+15)/16);
  sig_kernel<<<grid,block,0,stream>>>(nv12,pitch,src_w,src_h,crop,sig,sig_w,sig_h);
}

__global__ void diff_kernel(const float* a, const float* b, float* out, int n) {
  __shared__ float s[256]; int tid=threadIdx.x; int i=blockIdx.x*blockDim.x+tid;
  float v=0; if(i<n) v=fabsf(a[i]-b[i]); s[tid]=v; __syncthreads();
  for(int k=128;k>0;k>>=1){ if(tid<k) s[tid]+=s[tid+k]; __syncthreads(); }
  if(tid==0) atomicAdd(out,s[0]);
}
float signature_similarity(float* a, float* b, int n, cudaStream_t stream) {
  float* d_sum=nullptr; cudaMalloc(&d_sum,sizeof(float)); cudaMemsetAsync(d_sum,0,sizeof(float),stream);
  diff_kernel<<<(n+255)/256,256,0,stream>>>(a,b,d_sum,n);
  float h=0; cudaMemcpyAsync(&h,d_sum,sizeof(float),cudaMemcpyDeviceToHost,stream); cudaStreamSynchronize(stream); cudaFree(d_sum);
  return 1.0f - h / n;
}
