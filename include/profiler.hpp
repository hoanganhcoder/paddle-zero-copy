#pragma once
#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>
class Profiler { public: void add(const std::string& k, double ms){ sum[k]+=ms; count[k]++; } void print(){ for(auto&[k,v]:sum) std::cout<<k<<": avg="<<(v/count[k])<<" ms n="<<count[k]<<"\n"; } private: std::unordered_map<std::string,double> sum; std::unordered_map<std::string,int> count; };
class Timer { public: Timer(Profiler& p, std::string k):p(p),k(std::move(k)),t(std::chrono::high_resolution_clock::now()){} ~Timer(){ auto e=std::chrono::high_resolution_clock::now(); p.add(k,std::chrono::duration<double,std::milli>(e-t).count()); } private: Profiler&p; std::string k; std::chrono::high_resolution_clock::time_point t; };
