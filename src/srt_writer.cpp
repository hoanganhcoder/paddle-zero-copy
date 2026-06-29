#include "srt_writer.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
static std::string ts(double t){ long ms=(long)(t*1000+0.5); int h=ms/3600000; ms%=3600000; int m=ms/60000; ms%=60000; int s=ms/1000; ms%=1000; std::ostringstream o; o<<std::setfill('0')<<std::setw(2)<<h<<":"<<std::setw(2)<<m<<":"<<std::setw(2)<<s<<","<<std::setw(3)<<ms; return o.str(); }
void write_srt(const std::string& path, const std::vector<SubtitleItem>& items){ std::ofstream f(path); for(size_t i=0;i<items.size();i++){ f<<i+1<<"\n"<<ts(items[i].start)<<" --> "<<ts(items[i].end)<<"\n"<<items[i].text<<"\n\n"; }}
