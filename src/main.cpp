#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

#include "common.hpp"
#include "ffmpeg_demuxer.hpp"
#include "cuda_kernels.cuh"
#include "trt_engine.hpp"
#include "db_postprocess.hpp"
#include "ctc_decode.hpp"
#include "subtitle_merger.hpp"
#include "srt_writer.hpp"

// Requires NVIDIA video-sdk-samples in third_party. The real NVDEC glue is intentionally kept thin.
// Include path is prepared by setup_colab.sh.
#include "NvDecoder/NvDecoder.h"

struct Args {
  std::string video, out, det, rec, dict;
  CropRect crop{0,760,1920,320};
  Size2D det_size{960,192};
  Size2D rec_size{320,48};
  float ssim = 0.985f;
  float det_thresh = 0.30f;
  float box_thresh = 0.60f;
  float rec_thresh = 0.60f;
};

static std::vector<std::string> split(const std::string& s, char c){ std::vector<std::string> r; std::stringstream ss(s); std::string x; while(std::getline(ss,x,c)) r.push_back(x); return r; }
static Args parse(int argc, char** argv){
  Args a;
  for(int i=1;i<argc;i++){
    std::string k=argv[i]; auto val=[&](){ if(i+1>=argc) throw std::runtime_error("missing value for "+k); return std::string(argv[++i]); };
    if(k=="--video") a.video=val(); else if(k=="--out") a.out=val(); else if(k=="--det") a.det=val(); else if(k=="--rec") a.rec=val(); else if(k=="--dict") a.dict=val();
    else if(k=="--crop"){ auto p=split(val(),','); if(p.size()!=4) throw std::runtime_error("bad crop"); a.crop={std::stoi(p[0]),std::stoi(p[1]),std::stoi(p[2]),std::stoi(p[3])}; }
    else if(k=="--det-size"){ auto p=split(val(),','); a.det_size={std::stoi(p[0]),std::stoi(p[1])}; }
    else if(k=="--rec-size"){ auto p=split(val(),','); a.rec_size={std::stoi(p[0]),std::stoi(p[1])}; }
    else if(k=="--ssim") a.ssim=std::stof(val());
  }
  if(a.video.empty()||a.out.empty()||a.det.empty()||a.rec.empty()||a.dict.empty()) throw std::runtime_error("missing args");
  return a;
}

int main(int argc, char** argv){
  try{
    Args args=parse(argc,argv);
    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t stream; CHECK_CUDA(cudaStreamCreate(&stream));
    CUcontext cu_ctx=nullptr; cuInit(0); cuCtxCreate(&cu_ctx,0,0);

    FFmpegDemuxer demux(args.video);
    cudaVideoCodec codec = cudaVideoCodec_H264;
    if (demux.codec_id() == 173) codec = cudaVideoCodec_HEVC;
    NvDecoder dec(cu_ctx, false, codec, false, false, nullptr, nullptr, false, 0, 0, 1000);

    TRTEngine det(args.det), rec(args.rec);
    auto dict=load_dict(args.dict);
    SubtitleMerger merger(0.35,0.12,0.84);

    size_t det_in_n = 3ull*args.det_size.w*args.det_size.h;
    size_t det_out_n = (size_t)args.det_size.w*args.det_size.h;
    size_t rec_in_n = 3ull*args.rec_size.w*args.rec_size.h;
    int rec_time = 80; int rec_classes = (int)dict.size();

    float *d_det_in=nullptr,*d_det_out=nullptr,*d_rec_in=nullptr,*d_rec_out=nullptr,*d_sig=nullptr,*d_prev_sig=nullptr;
    CHECK_CUDA(cudaMalloc(&d_det_in, det_in_n*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_det_out, det_out_n*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rec_in, rec_in_n*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rec_out, (size_t)rec_time*rec_classes*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sig, 256*48*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_prev_sig, 256*48*sizeof(float)));

    det.set_input("x", d_det_in); det.set_output("sigmoid_0.tmp_0", d_det_out);
    rec.set_input("x", d_rec_in); rec.set_output("softmax_0.tmp_0", d_rec_out);

    std::vector<float> h_det(det_out_n), h_rec((size_t)rec_time*rec_classes);
    std::vector<uint8_t> pkt; int64_t pts=0,dts=0; long long frame_idx=0; int changed=0, skipped=0;
    bool have_prev=false; double fps=demux.fps();

    while(demux.read(pkt,pts,dts)){
      int n=dec.Decode(pkt.data(), (int)pkt.size());
      for(int i=0;i<n;i++){
        uint8_t* frame = dec.GetFrame();
        int pitch = dec.GetDeviceFramePitch();
        int sw = dec.GetWidth(), sh = dec.GetHeight();
        double t = frame_idx / fps;

        gray_signature_from_nv12_roi(frame,pitch,sw,sh,args.crop,d_sig,256,48,stream);
        float sim=0.f;
        if(have_prev) sim=signature_similarity(d_sig,d_prev_sig,256*48,stream);
        if(have_prev && sim >= args.ssim){ merger.extend(t); skipped++; frame_idx++; continue; }
        CHECK_CUDA(cudaMemcpyAsync(d_prev_sig,d_sig,256*48*sizeof(float),cudaMemcpyDeviceToDevice,stream)); have_prev=true;

        nv12_roi_to_det_tensor(frame,pitch,sw,sh,args.crop,d_det_in,args.det_size.w,args.det_size.h,stream);
        det.infer(stream);
        CHECK_CUDA(cudaMemcpyAsync(h_det.data(),d_det_out,det_out_n*sizeof(float),cudaMemcpyDeviceToHost,stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        auto boxes=db_postprocess_cpu(h_det.data(),args.det_size.w,args.det_size.h,args.det_thresh,args.box_thresh,8);
        std::string line;
        float best_score=0.f;
        for(auto& b: boxes){
          b.x1/=args.det_size.w; b.x2/=args.det_size.w; b.y1/=args.det_size.h; b.y2/=args.det_size.h;
          nv12_box_to_rec_tensor(frame,pitch,sw,sh,args.crop,b,d_rec_in,args.rec_size.w,args.rec_size.h,stream);
          rec.infer(stream);
          CHECK_CUDA(cudaMemcpyAsync(h_rec.data(),d_rec_out,h_rec.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));
          CHECK_CUDA(cudaStreamSynchronize(stream));
          auto txt=ctc_decode_cpu(h_rec.data(),rec_time,rec_classes,dict);
          if(txt.score >= args.rec_thresh){ line += txt.text; best_score=std::max(best_score,txt.score); }
        }
        merger.feed(t,line); changed++; frame_idx++;
      }
    }
    merger.finish(frame_idx/fps); write_srt(args.out, merger.items());
    std::cerr << "saved="<<args.out<<" frames="<<frame_idx<<" changed="<<changed<<" skipped="<<skipped<<" segments="<<merger.items().size()<<"\n";
  }catch(const std::exception& e){ std::cerr<<"error: "<<e.what()<<"\n"; return 1; }
  return 0;
}
