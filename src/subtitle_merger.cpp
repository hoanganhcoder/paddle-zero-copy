#include "subtitle_merger.hpp"
#include <algorithm>
#include <vector>

static double sim_ratio(const std::string& a, const std::string& b) {
  if (a.empty() || b.empty()) return 0.0;
  int n=a.size(), m=b.size(); std::vector<int> dp(m+1);
  for(int i=1;i<=n;i++){ int prev=0; for(int j=1;j<=m;j++){ int tmp=dp[j]; if(a[i-1]==b[j-1]) dp[j]=prev+1; else dp[j]=std::max(dp[j],dp[j-1]); prev=tmp; } }
  return (2.0*dp[m])/(n+m);
}

SubtitleMerger::SubtitleMerger(double max_merge_gap, double min_duration, double sim_threshold): max_gap_(max_merge_gap), min_dur_(min_duration), sim_th_(sim_threshold) {}

void SubtitleMerger::feed(double t, const std::string& text) {
  if (text.empty()) return;
  if (!active_) { active_=true; start_=t; last_=t; text_=text; return; }
  if (sim_ratio(text_, text) >= sim_th_) { last_=t; if (text.size() > text_.size()) text_=text; return; }
  flush(t); active_=true; start_=t; last_=t; text_=text;
}
void SubtitleMerger::extend(double t) { if(active_) last_=t; }
void SubtitleMerger::finish(double t) { if(active_) flush(t); }
void SubtitleMerger::flush(double end) {
  double e = std::max(end, start_ + min_dur_);
  if (!items_.empty() && items_.back().text == text_ && start_ - items_.back().end <= max_gap_) items_.back().end = e;
  else items_.push_back({start_, e, text_});
  active_=false; text_.clear();
}
