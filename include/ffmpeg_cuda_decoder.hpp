#pragma once
#include "common.hpp"
#include <string>

struct AVFormatContext;
struct AVCodecContext;
struct AVBufferRef;
struct AVFrame;
struct AVPacket;

class FfmpegCudaDecoder {
public:
    FfmpegCudaDecoder(const std::string& path, int gpu_id);
    ~FfmpegCudaDecoder();
    bool next(GpuFrame& out);
    double fps() const { return fps_; }
    int width() const { return width_; }
    int height() const { return height_; }
private:
    bool receive(GpuFrame& out);
    AVFormatContext* fmt_ = nullptr;
    AVCodecContext* dec_ = nullptr;
    AVBufferRef* hwdev_ = nullptr;
    AVPacket* pkt_ = nullptr;
    AVFrame* frame_ = nullptr;
    int stream_ = -1;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 25.0;
    bool eof_ = false;
};
