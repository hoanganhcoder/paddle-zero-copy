#include "db_postprocess.hpp"
#include <algorithm>

std::vector<DetBox> db_postprocess_cpu(const float* prob, int w, int h, float thresh, float box_thresh, int max_boxes) {
  std::vector<DetBox> boxes;
  int minx=w, miny=h, maxx=0, maxy=0; float maxp=0; bool any=false;
  for (int y=0;y<h;y++) for (int x=0;x<w;x++) {
    float p = prob[y*w+x];
    if (p > thresh) { any=true; minx=std::min(minx,x); miny=std::min(miny,y); maxx=std::max(maxx,x); maxy=std::max(maxy,y); maxp=std::max(maxp,p); }
  }
  if (any && maxp >= box_thresh) boxes.push_back({(float)minx,(float)miny,(float)maxx,(float)maxy,maxp});
  if ((int)boxes.size() > max_boxes) boxes.resize(max_boxes);
  return boxes;
}
