#pragma once
#include "common.hpp"
#include <string>

struct AppArgs {
    std::string video;
    std::string out = "output.srt";
    std::string det_model = "models/det.onnx";
    std::string rec_model = "models/rec.onnx";
    std::string dict = "models/ppocr_keys_v1.txt";
    RectI crop{0,760,1920,320};
    int det_w = 960;
    int det_h = 192;
    int rec_w = 320;
    int rec_h = 48;
    float diff_threshold = 0.006f;
    float det_threshold = 0.30f;
    float box_threshold = 0.50f;
    float rec_threshold = 0.45f;
    float sim_threshold = 0.84f;
    double max_merge_gap = 0.35;
    double min_duration = 0.12;
    int max_empty_frames = 6;
    int gpu = 0;
    bool fp16 = false;
    bool benchmark = true;
};

AppArgs parse_args(int argc, char** argv);
void print_usage();
