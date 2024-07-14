#pragma once
#ifndef __ACPMFUNC_H__
#define __ACPMFUNC_H__

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "FlowIO.h"
#include "MyUtils.h"
#include "Demons.h"
#include "Variational.h"
#include "MinimumCoveringCircle.h"

static const int NUM_NEIGHBORS = 8;
static const int NEIGHBOR_DX[NUM_NEIGHBORS] = { 0, 0, 1, -1, -1, -1, 1, 1 };
static const int NEIGHBOR_DY[NUM_NEIGHBORS] = { -1, 1, 0, 0, -1, 1, -1, 1 };

void ImagePadded(const cv::Mat& input1,
	const cv::Mat& input2,
	cv::Mat& output1,
	cv::Mat& output2,
	const int blocks_width,
	const int blocks_height);

void GetGaussPyramid(const cv::Mat& src,
	std::vector<cv::Mat>& gauss_pyramid,
	const int py_layers);

void ConstructPyramid(const cv::Mat& src,
	std::vector<cv::Mat>& pyramid,
	const float py_ratio,
	const int py_layers);

void MakeSeedsAndNeighbors(std::vector<cv::Point2f>& seeds,
	std::vector<std::vector<int>>& neighbors,
	const int w, const int h, const int step);

float Entropy(const cv::Mat& src);

void GetDescs(const cv::Mat& image,
	std::vector<cv::Mat>& descs,
	std::vector<cv::Vec4f>& descs_info,
	const std::vector<cv::Point2f> seeds,
	const int descs_width_min,
	const int descs_width_max,
	const float descs_thresh);

void GetDescs(const cv::Mat& image, 
	std::vector<cv::Mat>& descs, 
	std::vector<cv::Vec4f>& descs_info,
	const std::vector<cv::Point2f> seeds,
	const int descs_width_min, 
	const int descs_width_max,
	const float descs_thresh,
	cv::Mat& score_img);

void GetDescs(const cv::Mat& image,
	std::vector<cv::Mat>& descs,
	const std::vector<cv::Vec4f>& descs_info,
	const std::vector<cv::Point2f> seeds,
	const cv::Mat& flow,
	const cv::Mat& radius);

void MatchDescs(const std::vector<cv::Mat>& descs1,
	const std::vector<cv::Mat>& descs2,
	cv::Mat& flow, const cv::Mat& radius,
	const float data_thresh);

void CrossCheck(const std::vector<cv::Point2f>& seeds,
	cv::Mat& flow_seeds, cv::Mat& good_seeds_flag,
	const int w, const int h,
	const int max_displacement,
	const float descs_thresh);

void FillHoles(cv::Mat& flow, cv::Mat& good_seeds_flag);

void UpdateSearchRadius(const cv::Mat& flow,
	const std::vector<std::vector<int>>& neighbors,
	cv::Mat& radius);

void SpreadRadiusInter(const std::vector<cv::Size>& seeds_size,
	cv::Mat& search_radius, const int iter, const float py_ratio);

void SpreadFlowInter(const std::vector<cv::Size>& seeds_size,
	cv::Mat& flow, const int iter, const float py_ratio);

void RecoverOpticalFlow(const std::vector<cv::Point2f>& seeds,
	const cv::Size& image_size, const cv::Mat& flow,
	cv::Mat& flow_norm, const int seeds_width);

void MovePixels(const cv::Mat& src,
	cv::Mat& dst,
	const cv::Mat& flow,
	const int interpolation);

void DrawPatch(const cv::Mat& src,
	const std::vector<cv::Point2f>& seeds,
	const std::vector<cv::Vec4f>& descs_info);

void DrawGrid(const std::vector<cv::Point2f>& seeds,
	const cv::Mat& flow, cv::Mat& src, const int seeds_width);

void RemoveSpeckles(cv::Mat& flow, const float thresh,
	const int min_area);

void WriteMatchesFile(const std::vector<cv::Point2f>& seeds,
	const cv::Mat& flow_seeds, const std::string matches_path);

#endif // !__ACPMFUNC_H__
