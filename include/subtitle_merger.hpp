#pragma once
#include "common.hpp"
#include <string>
#include <vector>

struct SubtitleItem { double start=0,end=0; std::string text; };

class SubtitleMerger {
public:
    SubtitleMerger(double fps, float sim_threshold, double max_merge_gap, double min_duration, int max_empty_frames);
    void feed(int frame_idx, const std::string& text);
    void extend(int frame_idx);
    void flush(int frame_idx);
    const std::vector<SubtitleItem>& items() const { return items_; }
private:
    double fps_;
    float sim_threshold_;
    double max_merge_gap_;
    double min_duration_;
    int max_empty_frames_;
    bool active_ = false;
    int start_frame_ = 0;
    int last_frame_ = 0;
    int empty_ = 0;
    std::vector<std::string> votes_;
    std::string last_text_;
    std::vector<SubtitleItem> items_;
};
