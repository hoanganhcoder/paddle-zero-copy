#pragma once
#include "common.hpp"
#include <vector>

class DbPostProcessor {
public:
    DbPostProcessor(int det_w, int det_h, int roi_w, int roi_h, float det_threshold, float box_threshold);
    std::vector<TextBox> run(const std::vector<float>& prob);
private:
    int det_w_, det_h_, roi_w_, roi_h_;
    float det_threshold_, box_threshold_;
};
