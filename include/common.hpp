#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <sstream>

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ std::ostringstream os; os<<"CUDA error "<<cudaGetErrorString(e)<<" at "<<__FILE__<<":"<<__LINE__; throw std::runtime_error(os.str()); } } while(0)

struct CropRect { int x=0; int y=0; int w=0; int h=0; };
struct Size2D { int w=0; int h=0; };
struct OCRText { std::string text; float score=0.f; };
struct DetBox { float x1,y1,x2,y2; float score; };
