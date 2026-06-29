#include "ctc_decode.hpp"
#include <fstream>
#include <stdexcept>
#include <algorithm>

std::vector<std::string> load_dict(const std::string& path) {
  std::ifstream f(path); if (!f) throw std::runtime_error("failed to read dict: " + path);
  std::vector<std::string> dict; std::string line; dict.push_back("");
  while (std::getline(f,line)) { if (!line.empty() && line.back()=='\r') line.pop_back(); dict.push_back(line); }
  return dict;
}

OCRText ctc_decode_cpu(const float* logits, int time_steps, int classes, const std::vector<std::string>& dict) {
  std::string out; float score_sum=0; int count=0; int prev=-1;
  for (int t=0;t<time_steps;t++) {
    const float* row = logits + t*classes;
    int best = 0; float bv = row[0];
    for (int c=1;c<classes;c++) if (row[c] > bv) { bv=row[c]; best=c; }
    if (best != 0 && best != prev && best < (int)dict.size()) { out += dict[best]; score_sum += bv; count++; }
    prev = best;
  }
  return {out, count ? score_sum / count : 0.f};
}
