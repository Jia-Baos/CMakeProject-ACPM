#pragma once
#ifndef __VARIATIONAL_H__
#define __VARIATIONAL_H__

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp> // for variational refinement

void VariationalRefine(const cv::Mat& fixed_image, const cv::Mat& moved_image, cv::Mat& flow_image);
#endif // !__VARIATIONAL_H__
