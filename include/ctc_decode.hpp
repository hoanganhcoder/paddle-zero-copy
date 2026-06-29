#pragma once
#include <string>
#include <vector>
#include "common.hpp"

std::vector<std::string> load_dict(const std::string& path);
OCRText ctc_decode_cpu(const float* logits, int time_steps, int classes, const std::vector<std::string>& dict);
