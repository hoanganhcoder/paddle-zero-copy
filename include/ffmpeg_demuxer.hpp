#pragma once
#include <string>
#include <vector>
#include <cstdint>
extern "C" { struct AVFormatContext; struct AVPacket; }

class FFmpegDemuxer {
public:
  explicit FFmpegDemuxer(const std::string& path);
  ~FFmpegDemuxer();
  bool read(std::vector<uint8_t>& data, int64_t& pts, int64_t& dts);
  int codec_id() const;
  int width() const { return width_; }
  int height() const { return height_; }
  double fps() const { return fps_; }
  double time_base() const { return time_base_; }
private:
  AVFormatContext* fmt_ = nullptr;
  AVPacket* pkt_ = nullptr;
  int video_stream_ = -1;
  int codec_id_ = 0;
  int width_ = 0;
  int height_ = 0;
  double fps_ = 25.0;
  double time_base_ = 0.04;
};
