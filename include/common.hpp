#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e)); \
    } \
} while (0)

struct RectI { int x=0,y=0,w=0,h=0; };
struct TextBox { RectI box; float score=0.0f; };
struct OcrLine { RectI box; std::string text; float score=0.0f; };

struct GpuFrame {
    uint8_t* y = nullptr;
    uint8_t* uv = nullptr;
    int width = 0;
    int height = 0;
    int pitch_y = 0;
    int pitch_uv = 0;
    double pts = 0.0;
    void* owner = nullptr;
};
