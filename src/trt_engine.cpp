#include "trt_engine.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

void TRTLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
}

static std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("failed to read engine: " + path);
  f.seekg(0, std::ios::end); size_t n = f.tellg(); f.seekg(0);
  std::vector<char> buf(n); f.read(buf.data(), n); return buf;
}

TRTEngine::TRTEngine(const std::string& engine_path) {
  auto data = read_file(engine_path);
  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) throw std::runtime_error("createInferRuntime failed");
  engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
  if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
  ctx_ = engine_->createExecutionContext();
  if (!ctx_) throw std::runtime_error("createExecutionContext failed");
}
TRTEngine::~TRTEngine() { delete ctx_; delete engine_; delete runtime_; }
void TRTEngine::set_input(const std::string& name, void* ptr) { if (!ctx_->setTensorAddress(name.c_str(), ptr)) throw std::runtime_error("set input failed: " + name); }
void TRTEngine::set_output(const std::string& name, void* ptr) { if (!ctx_->setTensorAddress(name.c_str(), ptr)) throw std::runtime_error("set output failed: " + name); }
void TRTEngine::infer(cudaStream_t stream) { if (!ctx_->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed"); }
std::vector<int64_t> TRTEngine::shape(const std::string& name) const {
  auto d = engine_->getTensorShape(name.c_str());
  std::vector<int64_t> s; for (int i=0;i<d.nbDims;++i) s.push_back(d.d[i]); return s;
}
