#include "ffmpeg_demuxer.hpp"
#include <stdexcept>
#include <cstring>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

FFmpegDemuxer::FFmpegDemuxer(const std::string& path) {
  if (avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr) < 0) throw std::runtime_error("failed to open input");
  if (avformat_find_stream_info(fmt_, nullptr) < 0) throw std::runtime_error("failed to find stream info");
  for (unsigned i=0; i<fmt_->nb_streams; ++i) {
    if (fmt_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) { video_stream_ = (int)i; break; }
  }
  if (video_stream_ < 0) throw std::runtime_error("no video stream");
  auto* st = fmt_->streams[video_stream_];
  codec_id_ = st->codecpar->codec_id;
  width_ = st->codecpar->width;
  height_ = st->codecpar->height;
  AVRational fr = av_guess_frame_rate(fmt_, st, nullptr);
  if (fr.num && fr.den) fps_ = av_q2d(fr);
  time_base_ = av_q2d(st->time_base);
  pkt_ = av_packet_alloc();
}

FFmpegDemuxer::~FFmpegDemuxer() {
  if (pkt_) av_packet_free(&pkt_);
  if (fmt_) avformat_close_input(&fmt_);
}

bool FFmpegDemuxer::read(std::vector<uint8_t>& data, int64_t& pts, int64_t& dts) {
  while (av_read_frame(fmt_, pkt_) >= 0) {
    if (pkt_->stream_index == video_stream_) {
      data.assign(pkt_->data, pkt_->data + pkt_->size);
      pts = pkt_->pts;
      dts = pkt_->dts;
      av_packet_unref(pkt_);
      return true;
    }
    av_packet_unref(pkt_);
  }
  return false;
}

int FFmpegDemuxer::codec_id() const { return codec_id_; }
