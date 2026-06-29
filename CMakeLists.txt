cmake_minimum_required(VERSION 3.22)
project(videocr_zero LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVUTIL REQUIRED libavutil)

set(TENSORRT_ROOT /usr)
set(VIDEO_SDK_ROOT ${CMAKE_SOURCE_DIR}/third_party/video-sdk-samples)

include_directories(
  include
  ${AVFORMAT_INCLUDE_DIRS}
  ${AVCODEC_INCLUDE_DIRS}
  ${AVUTIL_INCLUDE_DIRS}
  ${VIDEO_SDK_ROOT}/Samples/NvCodec/NvDecoder
  ${VIDEO_SDK_ROOT}/Samples/NvCodec/NvCodec
  ${VIDEO_SDK_ROOT}/Samples/Utils
  /usr/local/cuda/include
  ${TENSORRT_ROOT}/include
)

add_executable(videocr_zero
  src/main.cpp
  src/ffmpeg_demuxer.cpp
  src/trt_engine.cpp
  src/db_postprocess.cpp
  src/ctc_decode.cpp
  src/subtitle_merger.cpp
  src/srt_writer.cpp
  src/cuda_kernels.cu
)

target_link_directories(videocr_zero PRIVATE /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)

target_link_libraries(videocr_zero
  CUDA::cudart
  CUDA::cuda_driver
  nvinfer
  nvonnxparser
  ${AVFORMAT_LIBRARIES}
  ${AVCODEC_LIBRARIES}
  ${AVUTIL_LIBRARIES}
)
