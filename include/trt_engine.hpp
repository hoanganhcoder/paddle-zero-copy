#pragma once
#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>

class TRTLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override;
};

class TRTEngine {
public:
  explicit TRTEngine(const std::string& engine_path);
  ~TRTEngine();
  void infer(cudaStream_t stream);
  void set_input(const std::string& name, void* ptr);
  void set_output(const std::string& name, void* ptr);
  std::vector<int64_t> shape(const std::string& name) const;
private:
  TRTLogger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* ctx_ = nullptr;
};
