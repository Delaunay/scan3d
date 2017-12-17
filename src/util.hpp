#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <random>


namespace util{
#ifndef __linux__
std::mt19937 &random_engine();

// non-negative, double-precision, floating-point values, uniformly distributed
// over the interval [0.0 , 1.0].
double drand48();
#endif

class Chronometer {
public:
	typedef std::chrono::high_resolution_clock::time_point TimePoint;
	typedef std::chrono::high_resolution_clock::duration Duration;

	Chronometer() { start(); }

	TimePoint start() {
		_start = std::chrono::high_resolution_clock::now();
		return _start;
	}

	template <typename T = std::ratio<1, 1>> double time() {
		std::chrono::duration<double, T> dur = std::chrono::high_resolution_clock::now() - _start;
		return dur.count();
	}

private:
	TimePoint _start;
};

std::vector<cv::Mat> readImages(const char *name, int from, int to, double fct);

std::vector<cv::Mat> readImagesFromCam(const std::vector<cv::Mat> &cam, int from, int to);
}