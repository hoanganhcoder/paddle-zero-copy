#include "db_postprocess.hpp"
#include <queue>
#include <algorithm>
#include <numeric>

DbPostProcessor::DbPostProcessor(int det_w, int det_h, int roi_w, int roi_h, float det_threshold, float box_threshold)
    : det_w_(det_w), det_h_(det_h), roi_w_(roi_w), roi_h_(roi_h), det_threshold_(det_threshold), box_threshold_(box_threshold) {}

std::vector<TextBox> DbPostProcessor::run(const std::vector<float>& prob) {
    int n = det_w_ * det_h_;
    std::vector<uint8_t> seen(n, 0);
    std::vector<TextBox> boxes;
    auto at=[&](int x,int y){ return y*det_w_+x; };
    int dx[4]={1,-1,0,0}, dy[4]={0,0,1,-1};
    for (int y=0;y<det_h_;y++) for (int x=0;x<det_w_;x++) {
        int id=at(x,y);
        if (seen[id] || prob[id] < det_threshold_) continue;
        int minx=x,maxx=x,miny=y,maxy=y,cnt=0; double sum=0.0;
        std::queue<std::pair<int,int>> q; q.push({x,y}); seen[id]=1;
        while(!q.empty()) {
            auto [cx,cy]=q.front(); q.pop(); int ci=at(cx,cy);
            cnt++; sum += prob[ci]; minx=std::min(minx,cx); maxx=std::max(maxx,cx); miny=std::min(miny,cy); maxy=std::max(maxy,cy);
            for(int k=0;k<4;k++){ int nx=cx+dx[k], ny=cy+dy[k]; if(nx<0||ny<0||nx>=det_w_||ny>=det_h_) continue; int ni=at(nx,ny); if(!seen[ni] && prob[ni]>=det_threshold_) { seen[ni]=1; q.push({nx,ny}); }}
        }
        float score = cnt ? (float)(sum / cnt) : 0;
        if (cnt < 8 || score < box_threshold_) continue;
        int pad_x = 4, pad_y = 3;
        RectI b;
        b.x = std::max(0, (int)((float)minx / det_w_ * roi_w_) - pad_x);
        b.y = std::max(0, (int)((float)miny / det_h_ * roi_h_) - pad_y);
        int x2 = std::min(roi_w_, (int)((float)(maxx+1) / det_w_ * roi_w_) + pad_x);
        int y2 = std::min(roi_h_, (int)((float)(maxy+1) / det_h_ * roi_h_) + pad_y);
        b.w = x2 - b.x; b.h = y2 - b.y;
        if (b.w > 8 && b.h > 6) boxes.push_back({b, score});
    }
    std::sort(boxes.begin(), boxes.end(), [](const TextBox& a, const TextBox& b){ if (std::abs(a.box.y-b.box.y)>10) return a.box.y<b.box.y; return a.box.x<b.box.x; });
    if (boxes.size() > 32) boxes.resize(32);
    return boxes;
}
