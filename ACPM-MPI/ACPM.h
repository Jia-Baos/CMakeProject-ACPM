#pragma once
#ifndef __ACPM_H__
#define __ACPM_H__

#include <iostream>
#include <string>
#include <cmath>

#include "ACPMFunc.h"

struct acpm_params_t
{
	int layers_;								// 金子塔层数
	int min_width_;								// 最小宽度
	int seeds_width_;							// 种子点的大小
	int descs_width_min_;						// 描述子最小值
	int descs_width_max_;						// 描述子最大值
	int max_displacement_;						// 最大位移量
	float py_ratio_;							// 金字塔缩放比例
	float data_thresh_;							// Patch相似性阈值
	float descs_thresh_;						// 描述子阈值
	float check_thresh_;						// 交叉检查阈值

	std::string optflow_path_;					// 光流文件路径
	std::string mathces_path_;					// 匹配点文件路径
	std::string res_image_path_;				// 残差图文件路径
	acpm_params_t() :
		layers_(0),
		min_width_(30),
		seeds_width_(5),
		descs_width_min_(7),
		descs_width_max_(19),
		max_displacement_(100),
		py_ratio_(0.5),
		data_thresh_(0.4),
		descs_thresh_(0.08),
		check_thresh_(3.0),
		optflow_path_(),
		mathces_path_(),
		res_image_path_() {}
};

class ACPM
{
private:
	cv::Mat fixed_image_;						// fixed_image拷贝
	cv::Mat moved_image_;						// moved_image拷贝
	cv::Mat res_image_;							// 残差图
	cv::Mat moved_image_warpped_;				// warped moved_image
	cv::Mat flow_seeds_;						// 种子点的光流
	cv::Mat flow_image_;						// 图像的光流
	cv::Mat search_radius_;						// 种子点光流的搜索半径
	cv::Mat data_cost_;							// 数据代价
	cv::Mat texture_cost_;						// 纹理代价
	cv::Mat seeds_flag_;						// 种子点分类

public:
	acpm_params_t acpm_parames;					// 配置参数结构体

	ACPM();
	ACPM(const acpm_params_t& acpm_parames);
	~ACPM();

	void Compute(const cv::Mat& fixed_image,
		const cv::Mat& moved_image);

	void SetInput(const cv::Mat& fixed_image,
		const cv::Mat& moved_image,
		const float py_ratio);
};

#endif // !__ACPM_H__
