#pragma once
#include <vector>
#include "common.hpp"

std::vector<DetBox> db_postprocess_cpu(const float* prob, int w, int h, float thresh, float box_thresh, int max_boxes);
