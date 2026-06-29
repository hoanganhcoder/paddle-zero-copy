#pragma once
#include <string>
#include <vector>
#include <utility>

class CtcDecoder {
public:
    explicit CtcDecoder(const std::string& dict_path);
    std::pair<std::string,float> decode(const float* logits, int time_steps, int classes) const;
private:
    std::vector<std::string> chars_;
};
