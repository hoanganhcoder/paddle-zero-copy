cmake_minimum_required(VERSION 3.22)
project(paddle_zero_copy_ort LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVUTIL REQUIRED libavutil)

if(NOT DEFINED ORT_ROOT)
  if(DEFINED ENV{ORT_ROOT})
    set(ORT_ROOT $ENV{ORT_ROOT})
  else()
    set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
  endif()
endif()

set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR "${ORT_ROOT}/lib")
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ORT_LIB_DIR} REQUIRED)

add_executable(paddle_zero_copy
  src/main.cpp
  src/args.cpp
  src/ffmpeg_cuda_decoder.cpp
  src/cuda_preprocess.cu
  src/onnx_engine.cpp
  src/db_postprocess.cpp
  src/ocr_engine.cpp
  src/ctc_decoder.cpp
  src/subtitle_merger.cpp
  src/srt_writer.cpp
  src/profiler.cpp
)

target_include_directories(paddle_zero_copy PRIVATE
  include
  ${ORT_INCLUDE_DIR}
  ${AVFORMAT_INCLUDE_DIRS}
  ${AVCODEC_INCLUDE_DIRS}
  ${AVUTIL_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_directories(paddle_zero_copy PRIVATE
  ${AVFORMAT_LIBRARY_DIRS}
  ${AVCODEC_LIBRARY_DIRS}
  ${AVUTIL_LIBRARY_DIRS}
  ${ORT_LIB_DIR}
)

target_compile_options(paddle_zero_copy PRIVATE -O3 -Wall -Wextra)
target_compile_options(paddle_zero_copy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

target_link_libraries(paddle_zero_copy PRIVATE
  ${ONNXRUNTIME_LIB}
  CUDA::cudart
  CUDA::cuda_driver
  ${AVFORMAT_LIBRARIES}
  ${AVCODEC_LIBRARIES}
  ${AVUTIL_LIBRARIES}
)

set_target_properties(paddle_zero_copy PROPERTIES
  BUILD_RPATH "${ORT_LIB_DIR}"
  INSTALL_RPATH "${ORT_LIB_DIR}"
)
