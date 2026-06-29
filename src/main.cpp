#include "args.hpp"
#include "ffmpeg_cuda_decoder.hpp"
#include "cuda_preprocess.cuh"
#include "ocr_engine.hpp"
#include "subtitle_merger.hpp"
#include "srt_writer.hpp"
#include "profiler.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

static std::string join_lines(const std::vector<OcrLine>& lines) {
    std::ostringstream o;
    for (auto& l: lines) o << l.text;
    return o.str();
}

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        CHECK_CUDA(cudaSetDevice(args.gpu));
        cudaStream_t stream; CHECK_CUDA(cudaStreamCreate(&stream));

        FfmpegCudaDecoder dec(args.video, args.gpu);
        std::cout << "video " << dec.width() << "x" << dec.height() << " fps=" << dec.fps() << "\n";
        CudaPreprocessor prep(args.crop, args.det_w, args.det_h, args.rec_w, args.rec_h);
        OcrEngine ocr(args.det_model, args.rec_model, args.dict, args.gpu, args.det_w, args.det_h, args.rec_w, args.rec_h, args.crop.w, args.crop.h, args.det_threshold, args.box_threshold, args.rec_threshold);
        SubtitleMerger merger(dec.fps(), args.sim_threshold, args.max_merge_gap, args.min_duration, args.max_empty_frames);
        Profiler prof;

        GpuFrame frame;
        int idx = 0, ocr_calls = 0, skipped = 0;
        while (dec.next(frame)) {
            float diff = 0.0f;
            { Timer t(prof, "prepare+diff"); diff = prep.prepare_det(frame, stream); }
            if (idx > 0 && diff < args.diff_threshold) {
                merger.extend(idx);
                skipped++;
                idx++;
                continue;
            }
            std::vector<TextBox> boxes;
            { Timer t(prof, "detector+db"); boxes = ocr.detect(prep.det_tensor()); }
            std::vector<OcrLine> lines;
            if (!boxes.empty()) {
                { Timer t(prof, "rec-prep"); prep.prepare_rec_batch(boxes, stream, 64); CHECK_CUDA(cudaStreamSynchronize(stream)); }
                { Timer t(prof, "recognizer"); lines = ocr.recognize(prep.rec_tensor(), boxes); }
            }
            std::string text = join_lines(lines);
            merger.feed(idx, text);
            ocr_calls++;
            if ((idx % 500) == 0) {
                std::cout << "frame=" << idx << " diff=" << diff << " boxes=" << boxes.size() << " text=" << text << "\n";
            }
            idx++;
        }
        merger.flush(idx-1);
        write_srt(args.out, merger.items());
        std::cout << "saved: " << args.out << "\n";
        std::cout << "frames=" << idx << " ocr_calls=" << ocr_calls << " skipped=" << skipped << " segments=" << merger.items().size() << "\n";
        if (args.benchmark) prof.print();
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
