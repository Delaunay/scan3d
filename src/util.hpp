#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::Mat> readImages(const char *name, int from, int to, double fct);

std::vector<cv::Mat> readImagesFromCam(const std::vector<cv::Mat> &cam, int from, int to);
