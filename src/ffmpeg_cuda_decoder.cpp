#include "ffmpeg_cuda_decoder.hpp"
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}
#include <sstream>

static enum AVPixelFormat get_hw_format(AVCodecContext*, const enum AVPixelFormat* pix_fmts) {
    for (const enum AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == AV_PIX_FMT_CUDA) return *p;
    }
    return AV_PIX_FMT_NONE;
}

FfmpegCudaDecoder::FfmpegCudaDecoder(const std::string& path, int gpu_id) {
    av_log_set_level(AV_LOG_WARNING);
    if (avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr) < 0) throw std::runtime_error("avformat_open_input failed");
    if (avformat_find_stream_info(fmt_, nullptr) < 0) throw std::runtime_error("avformat_find_stream_info failed");
    stream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_ < 0) throw std::runtime_error("No video stream");
    AVStream* st = fmt_->streams[stream_];
    const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!codec) throw std::runtime_error("No decoder");
    dec_ = avcodec_alloc_context3(codec);
    if (!dec_) throw std::runtime_error("avcodec_alloc_context3 failed");
    if (avcodec_parameters_to_context(dec_, st->codecpar) < 0) throw std::runtime_error("parameters_to_context failed");
    std::string dev = std::to_string(gpu_id);
    if (av_hwdevice_ctx_create(&hwdev_, AV_HWDEVICE_TYPE_CUDA, dev.c_str(), nullptr, 0) < 0) {
        throw std::runtime_error("Cannot create CUDA hw device for FFmpeg. Check GPU runtime.");
    }
    dec_->hw_device_ctx = av_buffer_ref(hwdev_);
    dec_->get_format = get_hw_format;
    if (avcodec_open2(dec_, codec, nullptr) < 0) throw std::runtime_error("avcodec_open2 failed");
    width_ = dec_->width; height_ = dec_->height;
    AVRational r = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    if (r.num && r.den) fps_ = av_q2d(r);
    pkt_ = av_packet_alloc(); frame_ = av_frame_alloc();
    if (!pkt_ || !frame_) throw std::runtime_error("packet/frame alloc failed");
}

FfmpegCudaDecoder::~FfmpegCudaDecoder() {
    if (frame_) av_frame_free(&frame_);
    if (pkt_) av_packet_free(&pkt_);
    if (dec_) avcodec_free_context(&dec_);
    if (hwdev_) av_buffer_unref(&hwdev_);
    if (fmt_) avformat_close_input(&fmt_);
}

bool FfmpegCudaDecoder::receive(GpuFrame& out) {
    while (true) {
        int ret = avcodec_receive_frame(dec_, frame_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return false;
        if (ret < 0) throw std::runtime_error("avcodec_receive_frame failed");
        if (frame_->format != AV_PIX_FMT_CUDA) {
            av_frame_unref(frame_);
            throw std::runtime_error("Decoder did not return AV_PIX_FMT_CUDA");
        }
        out.width = frame_->width; out.height = frame_->height;
        out.y = reinterpret_cast<uint8_t*>(frame_->data[0]);
        out.uv = reinterpret_cast<uint8_t*>(frame_->data[1] ? frame_->data[1] : frame_->data[0] + frame_->linesize[0] * frame_->height);
        out.pitch_y = frame_->linesize[0];
        out.pitch_uv = frame_->linesize[1] ? frame_->linesize[1] : frame_->linesize[0];
        double pts = 0.0;
        AVStream* st = fmt_->streams[stream_];
        if (frame_->best_effort_timestamp != AV_NOPTS_VALUE) pts = frame_->best_effort_timestamp * av_q2d(st->time_base);
        out.pts = pts; out.owner = frame_;
        return true;
    }
}

bool FfmpegCudaDecoder::next(GpuFrame& out) {
    if (receive(out)) return true;
    if (eof_) return false;
    while (av_read_frame(fmt_, pkt_) >= 0) {
        if (pkt_->stream_index != stream_) { av_packet_unref(pkt_); continue; }
        int ret = avcodec_send_packet(dec_, pkt_);
        av_packet_unref(pkt_);
        if (ret < 0) throw std::runtime_error("avcodec_send_packet failed");
        if (receive(out)) return true;
    }
    eof_ = true;
    avcodec_send_packet(dec_, nullptr);
    return receive(out);
}
