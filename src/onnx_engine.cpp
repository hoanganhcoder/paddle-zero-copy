#include "onnx_engine.hpp"
#include <cuda_runtime.h>
#include <numeric>
#include <stdexcept>
#include <cstring>

OnnxCudaEngine::OnnxCudaEngine(const std::string& model_path, int gpu_id)
    : env_(ORT_LOGGING_LEVEL_WARNING, "paddle_zero_copy"), gpu_id_(gpu_id) {
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = gpu_id_;
    opts_.AppendExecutionProvider_CUDA(cuda_options);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts_);
    auto in = session_->GetInputNameAllocated(0, allocator_);
    auto out = session_->GetOutputNameAllocated(0, allocator_);
    input_name_ = in.get(); output_name_ = out.get();
    input_shape_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    output_shape_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

static size_t numel_of(const std::vector<int64_t>& shape) {
    size_t n=1;
    for (auto v: shape) n *= (v > 0 ? (size_t)v : 1);
    return n;
}

std::vector<float> OnnxCudaEngine::infer_gpu_float(float* d_input, const std::vector<int64_t>& shape) {
    Ort::MemoryInfo cuda_mem("Cuda", OrtAllocatorType::OrtArenaAllocator, gpu_id_, OrtMemTypeDefault);
    size_t input_count = numel_of(shape);
    Ort::Value input = Ort::Value::CreateTensor<float>(cuda_mem, d_input, input_count, shape.data(), shape.size());
    const char* in_names[] = { input_name_.c_str() };
    const char* out_names[] = { output_name_.c_str() };
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    auto out_shape = info.GetShape();
    size_t n = info.GetElementCount();
    float* outp = outputs[0].GetTensorMutableData<float>();
    std::vector<float> host(n);
    // ORT may return CUDA or CPU memory depending on graph; copy via cudaMemcpy first, fallback memcpy is safe only if CPU.
    cudaError_t e = cudaMemcpy(host.data(), outp, n*sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        cudaGetLastError();
        std::memcpy(host.data(), outp, n*sizeof(float));
    }
    return host;
}
