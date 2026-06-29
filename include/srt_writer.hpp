#pragma once
#include <string>
#include <vector>
#include "subtitle_merger.hpp"
void write_srt(const std::string& path, const std::vector<SubtitleItem>& items);
