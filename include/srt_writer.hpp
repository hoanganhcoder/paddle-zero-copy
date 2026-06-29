#pragma once
#include "subtitle_merger.hpp"
#include <string>
void write_srt(const std::string& path, const std::vector<SubtitleItem>& items);
