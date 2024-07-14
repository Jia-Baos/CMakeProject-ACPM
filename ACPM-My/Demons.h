#pragma once
#ifndef __DEMONS_H__
#define __DEMONS_H__

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

struct demons_params_t
{
	int niter_;
	float alpha_;
	float sigma_fluid_;
	float sigma_diffusion_;

	demons_params_t() :
		niter_(30),
		alpha_(0.6),
		sigma_fluid_(0.1),
		sigma_diffusion_(1.0) {}
};

class Demons
{
private:
	cv::Mat fixed_image_;
	cv::Mat moved_image_;

public:
	cv::Mat res_image_;
	cv::Mat moved_image_warpped_;
	demons_params_t demons_params;

	Demons();
	~Demons();
	Demons(const demons_params_t& demons_params);

	void SingleScale(const cv::Mat& fixed_image, const cv::Mat& moved_image);
	void DemonsRefinement(const cv::Mat& fixed_image, const cv::Mat& moved_image, cv::Mat& flow);
	void MultiScale(const cv::Mat& fixed_image, const cv::Mat& moved_image);
};

/// @brief ���߶�Demons�㷨
/// @param S0 �̶�ͼ��
/// @param M0 ����ͼ��
/// @param sx ����ˮƽ����λ�Ƴ�
/// @param sy ������ֱ����λ�Ƴ�
/// @param niter ��������
/// @param alpha �ٶ���ɢϵ��
/// @param sigma_fluid fluid ��������ϵ��
/// @param sigma_diffusion diffusion ��������ϵ��
/// @return ���ƶ�
float DemonsSingle(const cv::Mat& S0,
	const cv::Mat& M0,
	cv::Mat& sx,
	cv::Mat& sy,
	const int niter,
	const float alpha,
	const float sigma_fluid,
	const  float sigma_diffusion);

/// @brief ���ò�ַ������ݶ�
/// @param src ԭͼ��
/// @param Fx ˮƽ�����ݶ�
/// @param Fy ��ֱ�����ݶ�
void ComputeGradient(const cv::Mat& src, cv::Mat& Fx, cv::Mat& Fy);

/// @brief ����Thirion Demonsλ�Ƴ��Ĵ���ʵ��
/// @param S �̶�ͼ��
/// @param M ����ͼ��
/// @param Tx ˮƽ����λ�Ƴ�
/// @param Ty ��ֱ����λ�Ƴ�
/// @param alpha �ٶ���ɢϵ��
void ThirionDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha);

/// @brief ����Active Demonsλ�Ƴ��Ĵ���ʵ��
/// @param S �̶�ͼ��
/// @param M ����ͼ��
/// @param Tx ˮƽ����λ�Ƴ�
/// @param Ty ��ֱ����λ�Ƴ�
/// @param alpha �ٶ���ɢϵ��
void ActiveDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha);

/// @brief ����Symmetric demonsλ�Ƴ��Ĵ���ʵ��
/// @param S �̶�ͼ��
/// @param M ����ͼ��
/// @param Tx ˮƽ����λ�Ƴ�
/// @param Ty ��ֱ����λ�Ƴ�
/// @param alpha �ٶ���ɢϵ��
void SymmetricDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha);

/// @brief ͼ��ƽ��
/// @param src Դͼ��
/// @param dst Ŀ��ͼ��
/// @param sigma ��˹������׼��
void GaussianSmoothing(const cv::Mat& src, cv::Mat& dst, const float sigma);

/// @brief ����ͼ�������
/// @param src Դͼ��
/// @param gauss_pyramid ͼ�������
/// @param layers ͼ�����������
void GaussPyramid(const cv::Mat& src,
	std::vector<cv::Mat>& gauss_pyramid, const int layers = 3);

/// @brief ����ӳ����mask�����ص�λ�þ���ӳ��󳬱߽�������Ϊ0��ͳ�����ƶ�ʱ���ٿ���
/// @param Tx ˮƽ����λ�Ƴ�
/// @param Ty ��ֱ����λ�Ƴ�
/// @param mask ӳ���mask
void ComputeMask(const cv::Mat& Tx, const cv::Mat& Ty, cv::Mat& mask);

/// @brief ���ݵ�ǰ�̶�ͼ��͸���ͼ��������ϵ��
/// @param S �̶�ͼ��
/// @param Mi ����ͼ��
/// @param Mask ӳ���mask
/// @return ���ϵ��
double ComputeCCMask(const cv::Mat& S, const cv::Mat& Mi, const cv::Mat& Mask);

/// @brief ������ӳ��
/// @param src Դͼ��
/// @param dst Ŀ��ͼ��
/// @param Tx ˮƽ����λ�Ƴ�
/// @param Ty ��ֱ����λ�Ƴ�
/// @param interpolation ��ֵ����
void MovePixels(const cv::Mat& src, cv::Mat& dst,
	const cv::Mat& Tx, const cv::Mat& Ty,
	const int interpolation);

/// @brief ��������ʵ��
/// @param vx ˮƽ����λ�Ƴ�
/// @param vy ��ֱ����λ�Ƴ�
void ExpComposite(cv::Mat& vx, cv::Mat& vy);

/// @brief Exp����΢��ͬ��ӳ��ת��ʵ��
/// @param vx ˮƽ����λ�Ƴ�
/// @param vy ��ֱ����λ�Ƴ�
/// @param vx_out ˮƽ����λ�Ƴ��Ż����
/// @param vy_out ��ֱ����λ�Ƴ��Ż����
void ExpField(const cv::Mat& vx, const cv::Mat& vy,
	cv::Mat& vx_out, cv::Mat& vy_out);

#endif // !__DEMONS_H__
