#include "ctc_decoder.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>

CtcDecoder::CtcDecoder(const std::string& dict_path) {
    chars_.push_back("");
    std::ifstream f(dict_path);
    std::string line;
    while (std::getline(f,line)) { if(!line.empty() && line.back()=='\r') line.pop_back(); chars_.push_back(line); }
    chars_.push_back(" ");
}

std::pair<std::string,float> CtcDecoder::decode(const float* logits, int time_steps, int classes) const {
    std::string text; float conf_sum=0; int conf_n=0; int last=-1;
    for(int t=0;t<time_steps;t++) {
        const float* row = logits + t*classes;
        int best = 0; float bv=row[0];
        for(int c=1;c<classes;c++) if(row[c]>bv){bv=row[c]; best=c;}
        if(best>0 && best!=last && best < (int)chars_.size()) { text += chars_[best]; conf_sum += bv; conf_n++; }
        last = best;
    }
    float score = conf_n ? conf_sum / conf_n : 0.0f;
    return {text, score};
}
