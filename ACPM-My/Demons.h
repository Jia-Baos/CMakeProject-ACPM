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

/// @brief 单尺度Demons算法
/// @param S0 固定图像
/// @param M0 浮动图像
/// @param sx 最优水平方向位移场
/// @param sy 最优竖直方向位移场
/// @param niter 迭代次数
/// @param alpha 速度扩散系数
/// @param sigma_fluid fluid 近似正则化系数
/// @param sigma_diffusion diffusion 近似正则化系数
/// @return 相似度
float DemonsSingle(const cv::Mat& S0,
	const cv::Mat& M0,
	cv::Mat& sx,
	cv::Mat& sy,
	const int niter,
	const float alpha,
	const float sigma_fluid,
	const  float sigma_diffusion);

/// @brief 利用差分法计算梯度
/// @param src 原图像
/// @param Fx 水平方向梯度
/// @param Fy 竖直方向梯度
void ComputeGradient(const cv::Mat& src, cv::Mat& Fx, cv::Mat& Fy);

/// @brief 计算Thirion Demons位移场的代码实现
/// @param S 固定图像
/// @param M 浮动图像
/// @param Tx 水平方向位移场
/// @param Ty 竖直方向位移场
/// @param alpha 速度扩散系数
void ThirionDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha);

/// @brief 计算Active Demons位移场的代码实现
/// @param S 固定图像
/// @param M 浮动图像
/// @param Tx 水平方向位移场
/// @param Ty 竖直方向位移场
/// @param alpha 速度扩散系数
void ActiveDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha);

/// @brief 计算Symmetric demons位移场的代码实现
/// @param S 固定图像
/// @param M 浮动图像
/// @param Tx 水平方向位移场
/// @param Ty 竖直方向位移场
/// @param alpha 速度扩散系数
void SymmetricDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha);

/// @brief 图像平滑
/// @param src 源图像
/// @param dst 目标图像
/// @param sigma 高斯函数标准差
void GaussianSmoothing(const cv::Mat& src, cv::Mat& dst, const float sigma);

/// @brief 构造图像金字塔
/// @param src 源图像
/// @param gauss_pyramid 图像金字塔
/// @param layers 图像金字塔层数
void GaussPyramid(const cv::Mat& src,
	std::vector<cv::Mat>& gauss_pyramid, const int layers = 3);

/// @brief 计算映射后的mask，像素的位置经过映射后超边界则将其置为0，统计相似度时不再考虑
/// @param Tx 水平方向位移场
/// @param Ty 竖直方向位移场
/// @param mask 映射后mask
void ComputeMask(const cv::Mat& Tx, const cv::Mat& Ty, cv::Mat& mask);

/// @brief 根据当前固定图像和浮动图像计算相关系数
/// @param S 固定图像
/// @param Mi 浮动图像
/// @param Mask 映射后mask
/// @return 相关系数
double ComputeCCMask(const cv::Mat& S, const cv::Mat& Mi, const cv::Mat& Mask);

/// @brief 像素重映射
/// @param src 源图像
/// @param dst 目标图像
/// @param Tx 水平方向位移场
/// @param Ty 竖直方向位移场
/// @param interpolation 插值方法
void MovePixels(const cv::Mat& src, cv::Mat& dst,
	const cv::Mat& Tx, const cv::Mat& Ty,
	const int interpolation);

/// @brief 复合运算实现
/// @param vx 水平方向位移场
/// @param vy 竖直方向位移场
void ExpComposite(cv::Mat& vx, cv::Mat& vy);

/// @brief Exp域中微分同胚映射转换实现
/// @param vx 水平方向位移场
/// @param vy 竖直方向位移场
/// @param vx_out 水平方向位移场优化结果
/// @param vy_out 竖直方向位移场优化结果
void ExpField(const cv::Mat& vx, const cv::Mat& vy,
	cv::Mat& vx_out, cv::Mat& vy_out);

#endif // !__DEMONS_H__
