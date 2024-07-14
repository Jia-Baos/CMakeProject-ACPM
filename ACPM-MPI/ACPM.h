#pragma once
#ifndef __ACPM_H__
#define __ACPM_H__

#include <iostream>
#include <string>
#include <cmath>

#include "ACPMFunc.h"

struct acpm_params_t
{
	int layers_;								// ����������
	int min_width_;								// ��С���
	int seeds_width_;							// ���ӵ�Ĵ�С
	int descs_width_min_;						// ��������Сֵ
	int descs_width_max_;						// ���������ֵ
	int max_displacement_;						// ���λ����
	float py_ratio_;							// ���������ű���
	float data_thresh_;							// Patch��������ֵ
	float descs_thresh_;						// ��������ֵ
	float check_thresh_;						// ��������ֵ

	std::string optflow_path_;					// �����ļ�·��
	std::string mathces_path_;					// ƥ����ļ�·��
	std::string res_image_path_;				// �в�ͼ�ļ�·��
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
	cv::Mat fixed_image_;						// fixed_image����
	cv::Mat moved_image_;						// moved_image����
	cv::Mat res_image_;							// �в�ͼ
	cv::Mat moved_image_warpped_;				// warped moved_image
	cv::Mat flow_seeds_;						// ���ӵ�Ĺ���
	cv::Mat flow_image_;						// ͼ��Ĺ���
	cv::Mat search_radius_;						// ���ӵ�����������뾶
	cv::Mat data_cost_;							// ���ݴ���
	cv::Mat texture_cost_;						// �������
	cv::Mat seeds_flag_;						// ���ӵ����

public:
	acpm_params_t acpm_parames;					// ���ò����ṹ��

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
