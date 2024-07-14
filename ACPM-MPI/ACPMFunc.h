#pragma once
#ifndef __ACPMFUNC_H__
#define __ACPMFUNC_H__

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "FlowIO.h"
#include "MyUtils.h"
#include "Variational.h"
#include "MinimumCoveringCircle.h"

static const int NUM_NEIGHBORS = 8;
static const int NEIGHBOR_DX[NUM_NEIGHBORS] = { 0, 0, 1, -1, -1, -1, 1, 1 };
static const int NEIGHBOR_DY[NUM_NEIGHBORS] = { -1, 1, 0, 0, -1, 1, -1, 1 };

/*生成图像金字塔及种子点（block的中心点）*/
void ImagePadded(const cv::Mat& src1,
	const cv::Mat& src2,
	cv::Mat& dst1,
	cv::Mat& dst2,
	const int patch_width,
	const int patch_height);

void GetGaussPyramid(const cv::Mat& src,
	std::vector<cv::Mat>& gauss_pyramid,
	const int py_layers);

void MakeSeedsAndNeighbors(std::vector<cv::Point2f>& seeds,
	std::vector<std::vector<int>>& neighbors,
	const int w,
	const int h,
	const int step);

/*模板匹配计算初始光流*/
float Entropy(const cv::Mat& src);

void BorderTest(int& roi_x1, int& roi_y1,
	int& roi_x2, int& roi_y2,
	const int w, const int h);

float TextureTest(const cv::Mat& src,
	int& roi_x1, int& roi_y1,
	int& roi_x2, int& roi_y2,
	const int descs_width_max,
	const float descs_thresh);

void GetDescs(const cv::Mat& src,
	std::vector<cv::Mat>& descs,
	std::vector<cv::Vec4f>& descs_info,
	const std::vector<cv::Point2f>& seeds,
	const int descs_width_min,
	const int descs_width_max,
	const float descs_thresh);

void GetDescs(const cv::Mat& src,
	std::vector<cv::Mat>& descs,
	const std::vector<cv::Vec4f>& descs_info,
	const std::vector<cv::Point2f>& seeds,
	const cv::Mat& flow_seeds,
	const cv::Mat& search_radius);

void MatchDescs(const std::vector<cv::Mat>& descs1,
	const std::vector<cv::Mat>& descs2,
	cv::Mat& flow_seeds,
	const cv::Mat& search_radius,
	const float data_thresh);

/*错误偏移量校正*/
void CrossCheck(const std::vector<cv::Point2f>& seeds,
	cv::Mat& flow_seeds,
	cv::Mat& seeds_flag,
	const cv::Size image_size,
	const int max_displacement);

void GetKlables(const std::vector<cv::Point2f>& seeds,
	cv::Mat& k_labels,
	const cv::Size seeds_size,
	const int seeds_width);

void CrossCheck(const std::vector<cv::Point2f>& seeds,
	cv::Mat& flow_seeds_forward,
	cv::Mat& flow_seeds_backward,
	cv::Mat& seeds_flag,
	const cv::Mat& k_labels,
	const int max_displacement,
	const float check_thresh);

void FillHoles(cv::Mat& seeds_flow,
	cv::Mat& seeds_flag);

/*自动计算搜索半径，层间传播光流*/
void UpdateSearchRadius(const cv::Mat& seeds_flow,
	const std::vector<std::vector<int>>& neighbors,
	cv::Mat& search_radius);

void SpreadRadiusInter(const std::vector<cv::Size>& seeds_size,
	cv::Mat& search_radius,
	const int iter,
	const float py_ratio);

void SpreadFlowInter(const std::vector<cv::Size>& seeds_size,
	cv::Mat& flow_seeds,
	const int iter,
	const float py_ratio);

/*Debug，后处理，保存结果*/
void RecoverOpticalFlow(const std::vector<cv::Point2f>& seeds,
	const cv::Size& image_size,
	const cv::Mat& flow_seeds,
	cv::Mat& flow_image,
	const int seeds_width);

void MovePixels(cv::Mat src,
	cv::Mat& dst,
	cv::Mat& flow,
	int interpolation);

void Drawgrid(const std::vector<cv::Point2f>& seeds,
	const cv::Mat& flow,
	cv::Mat& src,
	const int seeds_width);

void WriteMatchesFile(const std::vector<cv::Point2f>& seeds,
	const cv::Mat& flow_seeds,
	const std::string matches_path);

#endif // !__ACPMFUNC_H__
