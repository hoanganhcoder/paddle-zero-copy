#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

class OnnxCudaEngine {
public:
    OnnxCudaEngine(const std::string& model_path, int gpu_id);
    std::vector<int64_t> input_shape() const { return input_shape_; }
    std::vector<int64_t> output_shape() const { return output_shape_; }
    const std::string& input_name() const { return input_name_; }
    const std::string& output_name() const { return output_name_; }
    std::vector<float> infer_gpu_float(float* d_input, const std::vector<int64_t>& shape);
private:
    Ort::Env env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    int gpu_id_ = 0;
};
