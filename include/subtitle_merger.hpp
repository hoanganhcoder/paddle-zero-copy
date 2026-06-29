#pragma once
#include <string>
#include <vector>

struct SubtitleItem { double start; double end; std::string text; };

class SubtitleMerger {
public:
  SubtitleMerger(double max_merge_gap, double min_duration, double sim_threshold);
  void feed(double t, const std::string& text);
  void extend(double t);
  void finish(double t);
  const std::vector<SubtitleItem>& items() const { return items_; }
private:
  bool active_ = false;
  double start_ = 0;
  double last_ = 0;
  std::string text_;
  std::vector<SubtitleItem> items_;
  double max_gap_;
  double min_dur_;
  double sim_th_;
  void flush(double end);
};
