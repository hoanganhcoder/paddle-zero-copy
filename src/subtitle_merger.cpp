#include "subtitle_merger.hpp"
#include <algorithm>
#include <unordered_map>
#include <regex>

static std::string clean(const std::string& s) { return std::regex_replace(s, std::regex("\\s+"), ""); }
static float sim(const std::string& a, const std::string& b) {
    if(a.empty()||b.empty()) return 0;
    int m=a.size(), n=b.size(); std::vector<int> dp(n+1);
    for(int j=0;j<=n;j++) dp[j]=j;
    for(int i=1;i<=m;i++){ int prev=dp[0]; dp[0]=i; for(int j=1;j<=n;j++){ int tmp=dp[j]; dp[j]=std::min({dp[j]+1, dp[j-1]+1, prev+(a[i-1]==b[j-1]?0:1)}); prev=tmp; }}
    int d=dp[n]; return 1.0f - (float)d / (float)std::max(m,n);
}
static std::string vote(const std::vector<std::string>& v) { std::unordered_map<std::string,int> c; for(auto&s:v)c[s]++; return std::max_element(c.begin(),c.end(),[](auto&a,auto&b){return a.second<b.second;})->first; }

SubtitleMerger::SubtitleMerger(double fps, float sim_threshold, double max_merge_gap, double min_duration, int max_empty_frames)
    : fps_(fps), sim_threshold_(sim_threshold), max_merge_gap_(max_merge_gap), min_duration_(min_duration), max_empty_frames_(max_empty_frames) {}

void SubtitleMerger::feed(int frame_idx, const std::string& raw) {
    std::string text = clean(raw);
    if(!text.empty()) {
        empty_=0;
        if(!active_) { active_=true; start_frame_=last_frame_=frame_idx; votes_={text}; last_text_=text; return; }
        if(sim(text,last_text_) >= sim_threshold_) { votes_.push_back(text); last_frame_=frame_idx; last_text_=text; return; }
        flush(frame_idx-1); active_=true; start_frame_=last_frame_=frame_idx; votes_={text}; last_text_=text; return;
    }
    if(active_ && ++empty_ >= max_empty_frames_) flush(frame_idx-empty_);
}
void SubtitleMerger::extend(int frame_idx) { if(active_) last_frame_=frame_idx; }
void SubtitleMerger::flush(int frame_idx) {
    if(!active_) return;
    std::string best = vote(votes_);
    double st=start_frame_/fps_, en=std::max(frame_idx/fps_, st+min_duration_);
    if(!items_.empty() && items_.back().text==best && st-items_.back().end <= max_merge_gap_) items_.back().end=en;
    else items_.push_back({st,en,best});
    active_=false; votes_.clear(); last_text_.clear(); empty_=0;
}
