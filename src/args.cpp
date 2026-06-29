#include "args.hpp"
#include <cstdlib>
#include <sstream>

static RectI parse_rect(const std::string& s) {
    RectI r; char c1,c2,c3; std::stringstream ss(s);
    if (!(ss >> r.x >> c1 >> r.y >> c2 >> r.w >> c3 >> r.h) || c1!=',' || c2!=',' || c3!=',') {
        throw std::runtime_error("Bad --crop, expected x,y,w,h");
    }
    return r;
}

static std::pair<int,int> parse_size(const std::string& s) {
    int w,h; char c; std::stringstream ss(s);
    if (!(ss >> w >> c >> h) || (c!='x' && c!=',')) throw std::runtime_error("Bad size, expected WxH");
    return {w,h};
}

void print_usage() {
    std::cout << "paddle_zero_copy --video input.mp4 --out output.srt --det models/det.onnx --rec models/rec.onnx --dict models/ppocr_keys_v1.txt\\n"
              << "  --crop x,y,w,h --det-size 960x192 --rec-size 320x48\\n";
}

AppArgs parse_args(int argc, char** argv) {
    AppArgs a;
    for (int i=1;i<argc;i++) {
        std::string k=argv[i];
        auto val=[&](){ if (i+1>=argc) throw std::runtime_error("Missing value for "+k); return std::string(argv[++i]); };
        if (k=="--video") a.video=val();
        else if (k=="--out") a.out=val();
        else if (k=="--det") a.det_model=val();
        else if (k=="--rec") a.rec_model=val();
        else if (k=="--dict") a.dict=val();
        else if (k=="--crop") a.crop=parse_rect(val());
        else if (k=="--det-size") { auto p=parse_size(val()); a.det_w=p.first; a.det_h=p.second; }
        else if (k=="--rec-size") { auto p=parse_size(val()); a.rec_w=p.first; a.rec_h=p.second; }
        else if (k=="--diff") a.diff_threshold=std::stof(val());
        else if (k=="--det-threshold") a.det_threshold=std::stof(val());
        else if (k=="--box-threshold") a.box_threshold=std::stof(val());
        else if (k=="--rec-threshold") a.rec_threshold=std::stof(val());
        else if (k=="--sim") a.sim_threshold=std::stof(val());
        else if (k=="--max-merge-gap") a.max_merge_gap=std::stod(val());
        else if (k=="--min-duration") a.min_duration=std::stod(val());
        else if (k=="--max-empty-frames") a.max_empty_frames=std::stoi(val());
        else if (k=="--gpu") a.gpu=std::stoi(val());
        else if (k=="--fp16") a.fp16=true;
        else if (k=="--no-benchmark") a.benchmark=false;
        else if (k=="--help" || k=="-h") { print_usage(); std::exit(0); }
        else throw std::runtime_error("Unknown arg: "+k);
    }
    if (a.video.empty()) throw std::runtime_error("Missing --video");
    return a;
}
