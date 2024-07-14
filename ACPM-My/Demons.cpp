#include "Demons.h"

Demons::Demons(const demons_params_t& demons_params)
{
	this->demons_params = demons_params;

	std::cout << "Initalize the class, own..." << std::endl;
}

Demons::Demons()
{
	std::cout << "Initalize the class, pure..." << std::endl;
}

Demons::~Demons()
{
	std::cout << "Destory the class..." << std::endl;
}

void  Demons::SingleScale(const cv::Mat& fixed_image, const cv::Mat& moved_image)
{
	cv::cvtColor(fixed_image, this->fixed_image_, cv::COLOR_BGR2GRAY);
	cv::cvtColor(moved_image, this->moved_image_, cv::COLOR_BGR2GRAY);

	cv::Mat flow_x = cv::Mat::zeros(this->fixed_image_.size(), CV_32FC1);
	cv::Mat flow_y = cv::Mat::zeros(this->fixed_image_.size(), CV_32FC1);
	float cc = DemonsSingle(this->fixed_image_, this->moved_image_,
		flow_x, flow_y,
		demons_params.niter_,
		demons_params.alpha_,
		demons_params.sigma_fluid_,
		demons_params.sigma_diffusion_);

	MovePixels(this->moved_image_, this->moved_image_warpped_, flow_x, flow_y, cv::INTER_CUBIC);
	this->moved_image_warpped_.convertTo(this->moved_image_warpped_, CV_8UC1);

	cv::absdiff(this->fixed_image_, this->moved_image_warpped_, this->res_image_);
}

void Demons::DemonsRefinement(const cv::Mat& fixed_image, const cv::Mat& moved_image, cv::Mat& flow)
{
	this->fixed_image_ = fixed_image.clone();
	this->moved_image_ = moved_image.clone();

	cv::Mat moved_image_temp;
	std::vector<cv::Mat> flow_split;
	cv::split(flow, flow_split);
	MovePixels(this->moved_image_, moved_image_temp, flow_split[0], flow_split[1], cv::INTER_CUBIC);

	cv::Mat flow_x = cv::Mat::zeros(this->fixed_image_.size(), CV_32FC1);
	cv::Mat flow_y = cv::Mat::zeros(this->fixed_image_.size(), CV_32FC1);
	float cc = DemonsSingle(this->fixed_image_, moved_image_temp,
		flow_x,
		flow_y,
		demons_params.niter_,
		demons_params.alpha_,
		demons_params.sigma_fluid_,
		demons_params.sigma_diffusion_);

	flow_split[0] = flow_split[0] + flow_x;
	flow_split[1] = flow_split[1] + flow_y;
	cv::merge(flow_split, flow);
}

void  Demons::MultiScale(const cv::Mat& fixed_image, const cv::Mat& moved_image)
{
}

/************************************************************************/
/*    Demons Single                                                     */
/************************************************************************/

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
	const  float sigma_diffusion)
{
	// 将参考图像和浮动图像转换为浮点型矩阵
	cv::Mat S, M;
	S0.convertTo(S, CV_32FC1);
	M0.convertTo(M, CV_32FC1);
	cv::Mat M1 = M.clone();

	// 当前最佳相似度
	float cc_min = FLT_MIN;

	// 初始化位移场为0
	cv::Mat vx = cv::Mat::zeros(S.size(), CV_32FC1);
	cv::Mat vy = cv::Mat::zeros(S.size(), CV_32FC1);

	// 用于存储当前最佳位移场，不断进行更新
	cv::Mat sx_min, sy_min;

	for (int i = 0; i < niter; i++)
	{
		// 在此处修改计算位移场的方法
		cv::Mat ux, uy;
		ActiveDemonsForce(S, M1, ux, uy, alpha);
		//ThirionDemonsForce(S, M1, ux, uy, alpha);
		//SymmetricDemonsForce(S, M1, ux, uy, alpha);

		// 高斯滤波对计算的位移场进行平滑
		GaussianSmoothing(ux, ux, sigma_fluid);
		GaussianSmoothing(uy, uy, sigma_fluid);

		// 将位移场累加
		vx = vx + 0.75 * ux;
		vy = vy + 0.75 * uy;

		// 再次利用高斯滤波对计算的位移场进行平滑
		GaussianSmoothing(vx, vx, sigma_diffusion);
		GaussianSmoothing(vy, vy, sigma_diffusion);

		// 将累加的位移场转换为微分同胚映射
		ExpField(vx, vy, sx, sy);

		cv::Mat mask;
		// 计算黑色边缘的mask掩码矩阵
		ComputeMask(sx, sy, mask);
		// 对浮动图像M进行像素重采样
		MovePixels(M, M1, sx, sy, cv::INTER_CUBIC);
		// 计算F、M1的相似度
		float cc_cur = ComputeCCMask(S, M1, mask);

		if (cc_cur > cc_min)
		{
			std::cout << "epoch = " << i << "; cc = " << cc_min << std::endl;
			cc_min = cc_cur;
			sx_min = sx.clone();
			sy_min = sy.clone();
		}
	}

	// 得到当前层的最佳微分同胚映射
	sx = sx_min.clone();
	sy = sy_min.clone();
	return cc_min;
}

/************************************************************************/
/*    Driving force                                                     */
/************************************************************************/

/// @brief 利用差分法计算梯度
/// @param src 原图像
/// @param Fx 水平方向梯度
/// @param Fy 竖直方向梯度
void ComputeGradient(const cv::Mat& src, cv::Mat& Fx, cv::Mat& Fy)
{
	cv::Mat src_board;
	cv::copyMakeBorder(src, src_board, 1, 1, 1, 1, cv::BORDER_CONSTANT);
	Fx = cv::Mat::zeros(src.size(), CV_32FC1);
	Fy = cv::Mat::zeros(src.size(), CV_32FC1);

	for (int i = 0; i < src.rows; i++)
	{
		float* p_Fx = Fx.ptr<float>(i);
		float* p_Fy = Fy.ptr<float>(i);

		for (int j = 0; j < src.cols; j++)
		{
			// 水平方向的梯度
			p_Fx[j] = (src_board.ptr<float>(i + 1)[j + 2] - src_board.ptr<float>(i + 1)[j]) / 2.0;
			// 竖直方向的梯度
			p_Fy[j] = (src_board.ptr<float>(i + 2)[j + 1] - src_board.ptr<float>(i)[j + 1]) / 2.0;
		}
	}
}

/// @brief 计算Thirion Demons位移场的代码实现
/// @param S 固定图像
/// @param M 浮动图像
/// @param Tx 水平方向位移场
/// @param Ty 竖直方向位移场
/// @param alpha 速度扩散系数
void ThirionDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha)
{
	// 求浮动图像M与参考图像S的灰度差场Diff
	cv::Mat Diff = S - M;
	Tx = cv::Mat::zeros(S.size(), CV_32FC1);
	Ty = cv::Mat::zeros(S.size(), CV_32FC1);

	// 求参考图像S的梯度
	cv::Mat Sx, Sy;
	ComputeGradient(S, Sx, Sy);

	// 求浮动图像M的梯度
	cv::Mat Mx, My;
	ComputeGradient(M, Mx, My);

	for (int i = 0; i < S.rows; i++)
	{
		// 参考图像S梯度的指针
		float* p_sx = Sx.ptr<float>(i);
		float* p_sy = Sy.ptr<float>(i);
		// 浮动图像M梯度的指针
		float* p_mx = Mx.ptr<float>(i);
		float* p_my = My.ptr<float>(i);
		// 位移场T的指针
		float* p_tx = Tx.ptr<float>(i);
		float* p_ty = Ty.ptr<float>(i);
		// 灰度差场Diff的指针
		float* p_diff = Diff.ptr<float>(i);

		for (int j = 0; j < S.cols; j++)
		{
			// 原始Demons中只考虑参考图像S形成的驱动力
			float a = p_sx[j] * p_sx[j] + p_sy[j] * p_sy[j] + alpha * alpha * p_diff[j] * p_diff[j];

			// 对分母进行截断处理
			if (a < -0.0000001 || a > 0.0000001)
			{
				p_tx[j] = p_diff[j] * (p_sx[j] / a);
				p_ty[j] = p_diff[j] * (p_sy[j] / a);
			}
		}
	}
}

/// @brief 计算Active Demons位移场的代码实现
/// @param S 固定图像
/// @param M 浮动图像
/// @param Tx 水平方向位移场
/// @param Ty 竖直方向位移场
/// @param alpha 速度扩散系数
void ActiveDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha)
{
	// 求浮动图像M与参考图像S的灰度差场Diff
	cv::Mat Diff = S - M;
	Tx = cv::Mat::zeros(S.size(), CV_32FC1);
	Ty = cv::Mat::zeros(S.size(), CV_32FC1);

	// 求参考图像S的梯度
	cv::Mat Sx, Sy;
	ComputeGradient(S, Sx, Sy);

	// 求浮动图像M的梯度
	cv::Mat Mx, My;
	ComputeGradient(M, Mx, My);

	for (int i = 0; i < S.rows; i++)
	{
		// 参考图像S梯度的指针
		float* p_sx = Sx.ptr<float>(i);
		float* p_sy = Sy.ptr<float>(i);
		// 浮动图像M梯度的指针
		float* p_mx = Mx.ptr<float>(i);
		float* p_my = My.ptr<float>(i);
		// 位移场T的指针
		float* p_tx = Tx.ptr<float>(i);
		float* p_ty = Ty.ptr<float>(i);
		// 灰度差场Diff的指针
		float* p_diff = Diff.ptr<float>(i);

		for (int j = 0; j < S.cols; j++)
		{
			float a1 = p_sx[j] * p_sx[j] + p_sy[j] * p_sy[j] + alpha * alpha * p_diff[j] * p_diff[j];
			float a2 = p_mx[j] * p_mx[j] + p_my[j] * p_my[j] + alpha * alpha * p_diff[j] * p_diff[j];

			// 对分母进行截断处理
			if ((a1 < -0.0000001 || a1 > 0.0000001) && (a2 < -0.0000001 || a2 > 0.0000001))
			{
				p_tx[j] = p_diff[j] * (p_sx[j] / a1 + p_mx[j] / a2);
				p_ty[j] = p_diff[j] * (p_sy[j] / a1 + p_my[j] / a2);
			}
		}
	}
}

/// @brief 计算Symmetric demons位移场的代码实现
/// @param S 固定图像
/// @param M 浮动图像
/// @param Tx 水平方向位移场
/// @param Ty 竖直方向位移场
/// @param alpha 速度扩散系数
void SymmetricDemonsForce(const cv::Mat& S, const cv::Mat& M,
	cv::Mat& Tx, cv::Mat& Ty, const  float alpha)
{
	// 求浮动图像M与参考图像S的灰度差场Diff
	cv::Mat diff = S - M;
	Tx = cv::Mat::zeros(S.size(), CV_32FC1);
	Ty = cv::Mat::zeros(S.size(), CV_32FC1);

	// 求参考图像S的梯度
	cv::Mat Sx, Sy;
	ComputeGradient(S, Sx, Sy);

	// 求浮动图像M的梯度
	cv::Mat Mx, My;
	ComputeGradient(M, Mx, My);

	for (int i = 0; i < S.rows; i++)
	{
		// 参考图像S梯度的指针
		float* p_sx = Sx.ptr<float>(i);
		float* p_sy = Sy.ptr<float>(i);
		// 浮动图像M梯度的指针
		float* p_mx = Mx.ptr<float>(i);
		float* p_my = My.ptr<float>(i);
		// 位移场T的指针
		float* p_tx = Tx.ptr<float>(i);
		float* p_ty = Ty.ptr<float>(i);
		// 灰度差场Diff的指针
		float* p_diff = diff.ptr<float>(i);

		for (int j = 0; j < S.cols; j++)
		{
			float ax = (p_sx[j] + p_mx[j]) * (p_sx[j] + p_mx[j]) + (p_sy[j] + p_my[j]) * (p_sy[j] + p_my[j]) + 4 * alpha * alpha * p_diff[j] * p_diff[j];
			float ay = (p_sx[j] + p_mx[j]) * (p_sx[j] + p_mx[j]) + (p_sy[j] + p_my[j]) * (p_sy[j] + p_my[j]) + 4 * alpha * alpha * p_diff[j] * p_diff[j];

			//float ax = (p_sx[j] + p_mx[j]) * (p_sx[j] + p_mx[j]) + (p_sy[j] + p_my[j]) * (p_sy[j] + p_my[j]) + alpha * alpha * p_diff[j] * p_diff[j];
			//float ay = (p_sx[j] + p_mx[j]) * (p_sx[j] + p_mx[j]) + (p_sy[j] + p_my[j]) * (p_sy[j] + p_my[j]) + alpha * alpha * p_diff[j] * p_diff[j];

			// 对分母进行截断处理
			if ((ax < -0.0000001 || ax > 0.0000001) && (ay < -0.0000001 || ay > 0.0000001))
			{
				p_tx[j] = 2 * p_diff[j] * (p_sx[j] + p_mx[j]) / ax;
				p_ty[j] = 2 * p_diff[j] * (p_sy[j] + p_my[j]) / ay;

				// p_tx[j] = p_diff[j] * (p_sx[j] + p_mx[j]) / ax;
				// p_ty[j] = p_diff[j] * (p_sy[j] + p_my[j]) / ay;
			}
		}
	}
}

/************************************************************************/
/*    Smoothing                                                         */
/************************************************************************/

/// @brief 图像平滑
/// @param src 源图像
/// @param dst 目标图像
/// @param sigma 高斯函数标准差
void GaussianSmoothing(const cv::Mat& src, cv::Mat& dst, const float sigma)
{
	// 向上取整
	int radius = static_cast<int>(std::ceil(3 * sigma));
	// 不论radius为奇还是偶，ksize始终为奇
	int ksize = 2 * radius + 1;

	cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), sigma);
}

/// @brief 构造图像金字塔
/// @param src 源图像
/// @param gauss_pyramid 图像金字塔
/// @param layers 图像金字塔层数
void GaussPyramid(const cv::Mat& src,
	std::vector<cv::Mat>& gauss_pyramid, const int layers)
{
	// 构建图像金字塔，共四层，可直接在此处修改金字塔的层数
	cv::Mat current_img = src;
	gauss_pyramid.emplace_back(current_img);
	for (int i = 0; i < layers - 1; i++)
	{
		cv::Mat temp_img;
		cv::pyrDown(current_img, temp_img, cv::Size(current_img.cols / 2, current_img.rows / 2));
		gauss_pyramid.emplace_back(temp_img);
		current_img = temp_img;
	}
	std::reverse(gauss_pyramid.begin(), gauss_pyramid.end());
}

/************************************************************************/
/*    Metrics                                                           */
/************************************************************************/

/// @brief 计算映射后的mask，像素的位置经过映射后超边界则将其置为0，统计相似度时不再考虑
/// @param Tx 水平方向位移场
/// @param Ty 竖直方向位移场
/// @param mask 映射后mask
void ComputeMask(const cv::Mat& Tx, const cv::Mat& Ty, cv::Mat& mask)
{
	mask = cv::Mat::zeros(Tx.size(), CV_8UC1);

	for (int i = 0; i < Tx.rows; i++)
	{
		const float* p_Tx = Tx.ptr<float>(i);
		const float* p_Ty = Ty.ptr<float>(i);
		uchar* p_mask = mask.ptr<uchar>(i);

		for (int j = 0; j < Tx.cols; j++)
		{
			int x = static_cast<int>(j + p_Tx[j]);
			int y = static_cast<int>(i + p_Ty[j]);

			if (x > 0 && x < Tx.cols && y > 0 && y < Tx.rows)
			{
				p_mask[j] = 255;
			}
		}
	}
}

/// @brief 根据当前固定图像和浮动图像计算相关系数
/// @param S 固定图像
/// @param Mi 浮动图像
/// @param Mask 映射后mask
/// @return 相关系数
double ComputeCCMask(const cv::Mat& S, const cv::Mat& Mi, const cv::Mat& Mask)
{
	float sum1 = 0.0;
	float sum2 = 0.0;
	float sum3 = 0.0;

	for (int i = 0; i < S.rows; i++)
	{
		const float* p_S = S.ptr<float>(i);
		const float* p_Mi = Mi.ptr<float>(i);
		for (int j = 0; j < S.cols; j++)
		{
			// 映射后超边界的点不进行统计
			if (Mask.ptr<uchar>(i)[j])
			{
				float S_value = p_S[j];
				float Mi_value = p_Mi[j];
				sum1 += S_value * Mi_value;
				sum2 += S_value * S_value;
				sum3 += Mi_value * Mi_value;
			}
		}
	}

	// 归一化
	const float result = sum1 / std::sqrt(sum2 * sum3);
	return result;
}

/************************************************************************/
/*    Remap and Exp Composite                                           */
/************************************************************************/

/// @brief 像素重映射
/// @param src 源图像
/// @param dst 目标图像
/// @param Tx 水平方向位移场
/// @param Ty 竖直方向位移场
/// @param interpolation 插值方法
void MovePixels(const cv::Mat& src, cv::Mat& dst,
	const cv::Mat& Tx, const cv::Mat& Ty,
	const int interpolation)
{
	cv::Mat Tx_map(src.size(), CV_32FC1, 0.0);
	cv::Mat Ty_map(src.size(), CV_32FC1, 0.0);

	for (int i = 0; i < src.rows; i++)
	{
		float* p_Tx_map = Tx_map.ptr<float>(i);
		float* p_Ty_map = Ty_map.ptr<float>(i);
		for (int j = 0; j < src.cols; j++)
		{
			p_Tx_map[j] = j + Tx.ptr<float>(i)[j];
			p_Ty_map[j] = i + Ty.ptr<float>(i)[j];
		}
	}

	cv::remap(src, dst, Tx_map, Ty_map, interpolation);
}

/// @brief 复合运算实现
/// @param vx 水平方向位移场
/// @param vy 竖直方向位移场
void ExpComposite(cv::Mat& vx, cv::Mat& vy)
{
	// 复合运算实现
	// 假设x、y方向的位移场分别为Ux、Uy
	// 使用Ux、Uy分别对它们自身进行像素重采样的操作得到Ux'和Uy'，然后再计算Ux+Ux'和Uy+Uy'的运算就是复合运算

	cv::Mat bxp, byp;
	MovePixels(vx, bxp, vx, vy, cv::INTER_CUBIC);
	MovePixels(vy, byp, vx, vy, cv::INTER_CUBIC);

	// 这里运算的本质就是递归乘法运算
	vx = vx + bxp;
	vy = vy + byp;
}

/// @brief Exp域中微分同胚映射转换实现
/// @param vx 水平方向位移场
/// @param vy 竖直方向位移场
/// @param vx_out 水平方向位移场优化结果
/// @param vy_out 竖直方向位移场优化结果
void ExpField(const cv::Mat& vx, const cv::Mat& vy,
	cv::Mat& vx_out, cv::Mat& vy_out)
{
	// 矩阵中对应位置的元素相乘
	cv::Mat normv2 = vx.mul(vx) + vy.mul(vy);

	// 求最大值、最小值
	double minv, maxv;
	cv::Point pt_min, pt_max;
	cv::minMaxLoc(normv2, &minv, &maxv, &pt_min, &pt_max);

	float m = std::sqrt(maxv);
	float n = std::ceil(std::log2(m / 0.5));
	n = n > 0.0 ? n : 0.0;

	float a = std::pow(2.0, -n);

	// 缩放，通过伯德近似可以更好的提高精度
	vx_out = vx * a;
	vy_out = vy * a;

	// n次复合运算，个人理解就是递归乘方运算
	for (int i = 0; i < static_cast<int>(n); i++)
	{
		ExpComposite(vx_out, vy_out);
	}
}
