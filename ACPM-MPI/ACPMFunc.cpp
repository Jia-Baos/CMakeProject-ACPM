#include "ACPMFunc.h"

/*****************************************生成图像金字塔及种子点（block的中心点）*****************************************/
void ImagePadded(const cv::Mat& src1,
	const cv::Mat& src2,
	cv::Mat& dst1,
	cv::Mat& dst2,
	const int patch_width,
	const int patch_height)
{
	const int width_max = src1.cols > src2.cols ? src1.cols : src2.cols;
	const int height_max = src1.rows > src2.rows ? src1.rows : src2.rows;

	// Pad the image avoid the block's size euqals 1
	const int width_max_even = patch_width *
		(static_cast<int>(width_max / patch_width) + 1);
	const int height_max_even = patch_height *
		(static_cast<int>(height_max / patch_height) + 1);

	cv::resize(src1, dst1, cv::Size(width_max_even, height_max_even));
	cv::resize(src2, dst2, cv::Size(width_max_even, height_max_even));
}

void GetGaussPyramid(const cv::Mat& src,
	std::vector<cv::Mat>& gauss_pyramid,
	const int py_layers)
{
	// Construct the gauss pyramid
	cv::Mat current_img = src.clone();
	gauss_pyramid[0] = current_img;

	for (int i = 1; i < py_layers; i++)
	{
		cv::Mat temp_img;
		const int width = std::round(current_img.cols / 2);
		const int height = std::round(current_img.rows / 2);
		cv::pyrDown(current_img, temp_img, cv::Size(width, height));
		gauss_pyramid[i] = temp_img;
		current_img = temp_img;
	}
	std::reverse(gauss_pyramid.begin(), gauss_pyramid.end());
}

void MakeSeedsAndNeighbors(std::vector<cv::Point2f>& seeds,
	std::vector<std::vector<int>>& neighbors,
	const int w,
	const int h,
	const int step)
{
	const int gridw = w / step;
	const int gridh = h / step;
	const int ofsx = (w - (gridw - 1) * step) / 2;
	const int ofsy = (h - (gridh - 1) * step) / 2;
	const int nseeds = gridw * gridh;

	for (int i = 0; i < nseeds; i++)
	{
		const int x = i % gridw;
		const int y = i / gridw;

		// 保证 seed 不会出现在图像边缘上
		const float seedx = static_cast<float>(x * step + ofsx);
		const float seedy = static_cast<float>(y * step + ofsy);
		seeds[i] = cv::Vec2f(seedx, seedy);

		std::vector<int> neighbors_current(NUM_NEIGHBORS, -1);
		for (int n = 0; n < NUM_NEIGHBORS; n++)
		{
			const int nbx = x + NEIGHBOR_DX[n];
			const int nby = y + NEIGHBOR_DY[n];
			if (nbx < 0 || nbx >= gridw || nby < 0 || nby >= gridh) { continue; }

			// 这里的 neighbors 是当前 seed 八邻域中的 seed
			neighbors_current[n] = nby * gridw + nbx;
		}
		neighbors[i] = neighbors_current;
	}
}

/*****************************************模板匹配计算初始光流*****************************************/
float Entropy(const cv::Mat& src)
{
	const int w = src.cols;
	const int h = src.rows;
	cv::Mat temp = src.clone();
	float gray_array[256] = { 0.0 };

	temp.convertTo(temp, CV_8UC1, 255.0);
	// Compute the nums of every gray value
	for (int i = 0; i < h; i++)
	{
		const uchar* temp_poniter = temp.ptr<uchar>(i);
		for (int j = 0; j < w; j++)
		{
			int gray_value = temp_poniter[j];
			gray_array[gray_value]++;
		}
	}

	// Compute the probability of every gray value and entropy
	float entropy = 0;
	float* gray_array_pointer = gray_array;
	for (int i = 0; i < 255; i++)
	{
		float gray_value_prob = *gray_array_pointer / (w * h);
		if (gray_value_prob != 0) {
			entropy = entropy - gray_value_prob * std::log(gray_value_prob);
		}
		gray_array_pointer++;
	}
	return entropy;
}

void BorderTest(int& roi_x1, int& roi_y1,
	int& roi_x2, int& roi_y2,
	const int w, const int h)
{
	roi_x1 = roi_x1 < 0 ? 0 : roi_x1;
	roi_y1 = roi_y1 < 0 ? 0 : roi_y1;
	roi_x2 = roi_x2 > w ? w : roi_x2;
	roi_y2 = roi_y2 > h ? h : roi_y2;
	//std::cout << "Border Test..." << std::endl;
}

float TextureTest(const cv::Mat& src,
	int& roi_x1, int& roi_y1,
	int& roi_x2, int& roi_y2,
	const int descs_width_max,
	const float descs_thresh)
{
	const int w = src.cols;
	const int h = src.rows;
	BorderTest(roi_x1, roi_y1, roi_x2, roi_y2, w, h);
	cv::Mat roi = src(cv::Rect(roi_x1, roi_y1,
		roi_x2 - roi_x1, roi_y2 - roi_y1));
	float texture_cost = std::exp(-Entropy(roi));

	// if the text_cost is smaller than descs_thresh, we think it is a good desc
	const int step = 3;
	while (texture_cost > descs_thresh)
	{
		roi_x1 = roi_x1 - step < 0 ? 0 : roi_x1 - step;
		roi_y1 = roi_y1 - step < 0 ? 0 : roi_y1 - step;
		roi_x2 = roi_x2 + step > w ? w : roi_x2 + step;
		roi_y2 = roi_y2 + step > h ? h : roi_y2 + step;
		if (roi_x2 - roi_x1 >= descs_width_max ||
			roi_y2 - roi_y1 >= descs_width_max ||
			roi_x2 - roi_x1 >= w ||
			roi_y2 - roi_y1 >= h)
		{
			break;
		}
		roi = src(cv::Rect(roi_x1, roi_y1,
			roi_x2 - roi_x1, roi_y2 - roi_y1));
		texture_cost = std::exp(-Entropy(roi));
	}

	return texture_cost;
}

void GetDescs(const cv::Mat& src,
	std::vector<cv::Mat>& descs,
	std::vector<cv::Vec4f>& descs_info,
	const std::vector<cv::Point2f>& seeds,
	const int descs_width_min,
	const int descs_width_max,
	const float descs_thresh)
{
	const int radius = descs_width_min / 2;
	const int nseeds = seeds.size();

	for (int i = 0; i < nseeds; i++)
	{
		int roi_x1 = seeds[i].x - radius;
		int roi_y1 = seeds[i].y - radius;
		int roi_x2 = seeds[i].x + radius + 1;
		int roi_y2 = seeds[i].y + radius + 1;
		const float texture_cost = TextureTest(src, roi_x1, roi_y1,
			roi_x2, roi_y2, descs_width_max, descs_thresh);
		if (texture_cost > descs_thresh) { continue; }
		descs_info[i][0] = seeds[i].x - roi_x1;
		descs_info[i][1] = seeds[i].y - roi_y1;
		descs_info[i][2] = roi_x2 - seeds[i].x;
		descs_info[i][3] = roi_y2 - seeds[i].y;
		const cv::Mat roi = src(cv::Rect(roi_x1, roi_y1,
			roi_x2 - roi_x1, roi_y2 - roi_y1));
		descs[i] = roi;
	}
}

void GetDescs(const cv::Mat& src,
	std::vector<cv::Mat>& descs,
	const std::vector<cv::Vec4f>& descs_info,
	const std::vector<cv::Point2f>& seeds,
	const cv::Mat& flow_seeds,
	const cv::Mat& search_radius)
{
	const int w = flow_seeds.cols;
	const int h = flow_seeds.rows;
	const int nseeds = seeds.size();

	int width_padded = 0;
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(search_radius, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	width_padded += std::ceil(maxVal);

	std::vector<cv::Mat> flow_split;
	cv::split(flow_seeds, flow_split);
	cv::minMaxLoc(flow_split[0], &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	width_padded += std::ceil(maxVal);

	cv::minMaxLoc(flow_split[1], &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	width_padded += std::ceil(maxVal);

	cv::Mat image_padded = src.clone();
	cv::copyMakeBorder(src, image_padded, 2 * width_padded, 2 * width_padded,
		2 * width_padded, 2 * width_padded, cv::BORDER_REPLICATE);

	for (int i = 0; i < nseeds; i++)
	{
		const int x = i % w;
		const int y = i / w;

		if (descs_info[i][0] < eps && descs_info[i][1] < eps &&
			descs_info[i][2] < eps && descs_info[i][3] < eps)
		{
			continue;
		}

		const int flow_x = std::round(flow_seeds.at<cv::Vec2f>(y, x)[0]);
		const int flow_y = std::round(flow_seeds.at<cv::Vec2f>(y, x)[1]);
		const int radius = search_radius.at<float>(y, x);

		int roi_x1 = 2 * width_padded + seeds[i].x - descs_info[i][0] +
			flow_x - radius;
		int roi_y1 = 2 * width_padded + seeds[i].y - descs_info[i][1] +
			flow_y - radius;
		int roi_x2 = 2 * width_padded + seeds[i].x + descs_info[i][2] +
			flow_x + radius + 1;
		int roi_y2 = 2 * width_padded + seeds[i].y + descs_info[i][3] +
			flow_y + radius + 1;

		const cv::Mat roi = image_padded(cv::Rect(roi_x1, roi_y1,
			roi_x2 - roi_x1, roi_y2 - roi_y1));
		descs[i] = roi;
	}
}

float ParabolicInterpolation(const int l, const int c, const int r)
{
	const float offset = 0.5 * (l - r) / (l + r - 2.0 * c);
	return offset;
}

void MatchDescs(const std::vector<cv::Mat>& descs1,
	const std::vector<cv::Mat>& descs2,
	cv::Mat& flow_seeds,
	const cv::Mat& search_radius,
	const float data_thresh)
{
	const int w = flow_seeds.cols;
	const int h = flow_seeds.rows;
	const int nseeds = descs1.size();

	for (int i = 0; i < nseeds; i++)
	{
		const int x = i % w;
		const int y = i / w;

		// 将弱纹理区域种子点的光流置为UNKNOWN_FLOW
		if (descs1[i].rows == 0 || descs1[i].cols == 0 ||
			descs2[i].rows == 0 || descs2[i].cols == 0)
		{
			flow_seeds.at<cv::Vec2f>(y, x) =
				cv::Vec2f(UNKNOWN_FLOW, UNKNOWN_FLOW);
			continue;
		}

		const cv::Mat block1 = descs1[i].clone();
		const cv::Mat block2 = descs2[i].clone();

		// Template matching
		cv::Size result_size = cv::Size(block2.cols - block1.cols + 1,
			block2.rows - block1.rows + 1);
		cv::Mat result = cv::Mat::zeros(result_size, CV_32FC1);
		cv::matchTemplate(block2, block1, result, cv::TM_CCOEFF_NORMED);
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

		// 将遮挡区域种子点的光流置为UNKNOWN_FLOW
		if (maxVal < data_thresh)
		{
			flow_seeds.at<cv::Vec2f>(y, x) =
				cv::Vec2f(UNKNOWN_FLOW, UNKNOWN_FLOW);
			continue;
		}

		// Get the new flow of every block
		const float flow_x_old = flow_seeds.at<cv::Vec2f>(y, x)[0];
		const float flow_y_old = flow_seeds.at<cv::Vec2f>(y, x)[1];
		const int radius = search_radius.at<float>(y, x);
		const float flow_x_new = flow_x_old + maxLoc.x - radius;
		const float flow_y_new = flow_y_old + maxLoc.y - radius;
		flow_seeds.at<cv::Vec2f>(y, x)[0] = flow_x_new;
		flow_seeds.at<cv::Vec2f>(y, x)[1] = flow_y_new;
	}
}

/*****************************************错误偏移量校正*****************************************/
void CrossCheck(const std::vector<cv::Point2f>& seeds,
	cv::Mat& flow_seeds,
	cv::Mat& seeds_flag,
	const cv::Size image_size,
	const int max_displacement)
{
	const int seeds_w = flow_seeds.cols;
	const int seeds_h = flow_seeds.rows;
	const int image_w = image_size.width;
	const int image_h = image_size.height;
	const int seeds_num = seeds_w * seeds_h;

	cv::Vec2f flow_curr;
	cv::Point2f seed_warpped;
	for (int i = 0; i < seeds_num; i++)
	{
		const int x = i % seeds_w;
		const int y = i / seeds_w;

		// 无效光流检查
		flow_curr = flow_seeds.at<cv::Vec2f>(y, x);
		if (flow_curr == cv::Vec2f(UNKNOWN_FLOW, UNKNOWN_FLOW))
		{
			seeds_flag.at<uchar>(y, x) = 255;
			continue;
		}

		// 超边界、超位移阈值光流检查
		seed_warpped = cv::Point2f(seeds[i].x + flow_curr[0],
			seeds[i].y + flow_curr[1]);
		if (seed_warpped.x < 0 || seed_warpped.x >= image_w ||
			seed_warpped.y < 0 || seed_warpped.y >= image_h ||
			std::sqrtf(flow_curr[0] * flow_curr[0] +
				flow_curr[1] * flow_curr[1]) > max_displacement)
		{
			seeds_flag.at<uchar>(y, x) = 255;
			continue;
		}
	}
}

void GetKlables(const std::vector<cv::Point2f>& seeds,
	cv::Mat& k_labels,
	const cv::Size seeds_size,
	const int seeds_width)
{
	const int image_w = k_labels.cols;
	const int image_h = k_labels.rows;
	const int seeds_w = seeds_size.width;
	const int seeds_h = seeds_size.height;
	const int nmatches = seeds_w * seeds_h;
	const int radius = seeds_width / 2;

	for (int i = 0; i < nmatches; i++)
	{
		const int seed_x = i % seeds_w;
		const int seed_y = i / seeds_w;
		for (int dy = -radius; dy <= radius; dy++)
		{
			for (int dx = -radius; dx <= radius; dx++)
			{
				const int x = std::max(0,
					std::min(static_cast<int>(seeds[i].x + dx + 0.5f),
						image_w - 1));
				const int y = std::max(0,
					std::min(static_cast<int>(seeds[i].y + dy + 0.5f),
						image_h - 1));
				k_labels.at<float>(y, x) = i;
			}
		}
	}
}

void CrossCheck(const std::vector<cv::Point2f>& seeds,
	cv::Mat& flow_seeds_forward,
	cv::Mat& flow_seeds_backward,
	cv::Mat& seeds_flag,
	const cv::Mat& k_labels,
	const int max_displacement,
	const float check_thresh)
{
	const int image_w = k_labels.cols;
	const int image_h = k_labels.rows;
	const int seeds_w = flow_seeds_forward.cols;
	const int seeds_h = flow_seeds_forward.rows;
	const int seeds_num = seeds.size();

	int x_forward;
	int y_forward;
	int x_backward;
	int y_backward;
	int seed_forward_warpped_label;
	cv::Vec2f flow_sum;
	cv::Vec2f flow_forward;
	cv::Vec2f flow_backward;
	cv::Point2f seed_forward_warpped;
	for (int i = 0; i < seeds_num; i++)
	{
		x_forward = i % seeds_w;
		y_forward = i / seeds_w;

		// 超边界光流检查，之前填充结果中可能存在超边界的情况
		flow_forward = flow_seeds_forward.at<cv::Vec2f>(y_forward, x_forward);
		seed_forward_warpped = cv::Point2f(seeds[i].x + flow_forward[0], seeds[i].y + flow_forward[1]);
		if (seed_forward_warpped.x < 0 || seed_forward_warpped.x >= image_w ||
			seed_forward_warpped.y < 0 || seed_forward_warpped.y >= image_h)
		{
			seeds_flag.at<uchar>(y_forward, x_forward) = 255;
			continue;
		}

		seed_forward_warpped_label = k_labels.at<float>(static_cast<int>(seed_forward_warpped.y),
			static_cast<int>(seed_forward_warpped.x));
		x_backward = seed_forward_warpped_label % seeds_w;
		y_backward = seed_forward_warpped_label / seeds_w;
		flow_backward = flow_seeds_backward.at<cv::Vec2f>(y_backward, x_backward);

		// 前后一致性检查，用于检查遮挡区域
		flow_sum = flow_forward + flow_backward;
		if (std::sqrtf(flow_sum[0] * flow_sum[0] + flow_sum[1] * flow_sum[1]) > check_thresh)
		{
			seeds_flag.at<uchar>(y_forward, x_forward) = 255;
			continue;
		}

		seeds_flag.at<uchar>(y_forward, x_forward) = 0;
	}
}

void FillHoles(cv::Mat& seeds_flow, cv::Mat& seeds_flag)
{
	const int w = seeds_flow.cols;
	const int h = seeds_flow.rows;

	double maxVal = 0.0;
	cv::Point maxLoc;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,
		cv::Size(3, 3));
	cv::minMaxLoc(seeds_flag, NULL, &maxVal, NULL, &maxLoc, cv::Mat());

	int flag = 0;
	while (maxVal == 255)
	{
		cv::Mat temp = seeds_flag.clone();
		cv::erode(seeds_flag, seeds_flag, element);

		// 实现正向反向交替迭代优化
		int i0 = 0, i1 = h, i2 = 0, i3 = w, step = 1;
		if (flag == 1)
		{
			i0 = h - 1, i1 = -1, i2 = w - 1, i3 = -1, step = -1;
		}
		for (int i = i0; i != i1; i += step)
		{
			for (int j = i2; j != i3; j += step)
			{
				if ((temp.at<uchar>(i, j) - seeds_flag.at<uchar>(i, j)) == 255)
				{
					int candidate_size = 0;
					float flow_x_estimate = 0;
					float flow_y_estimate = 0;
					for (int k = 0; k < NUM_NEIGHBORS; k++)
					{
						const int x = j + NEIGHBOR_DX[k];
						const int y = i + NEIGHBOR_DY[k];
						if (x < 0 || x >= w || y < 0 || y >= h) { continue; }
						else
						{
							if (temp.at<uchar>(y, x) == 0)
							{
								flow_x_estimate += seeds_flow.at<cv::Vec2f>(y, x)[0];
								flow_y_estimate += seeds_flow.at<cv::Vec2f>(y, x)[1];
								++candidate_size;
							}
						}
					}
					seeds_flow.at<cv::Vec2f>(i, j) = cv::Vec2f(flow_x_estimate / candidate_size,
						flow_y_estimate / candidate_size);
					temp.at<uchar>(i, j) = 0;
				}
			}
		}
		flag = flag > 0 ? 0 : 1;
		cv::minMaxLoc(seeds_flag, NULL, &maxVal, NULL, &maxLoc, cv::Mat());
	}
}

/*****************************************自动计算搜索半径，层间传播光流*****************************************/
void UpdateSearchRadius(const cv::Mat& seeds_flow,
	const std::vector<std::vector<int>>& neighbors,
	cv::Mat& search_radius)
{
	const int w = seeds_flow.cols;
	const int h = seeds_flow.rows;
	const int nseeds = neighbors.size();
	std::vector<MyVector> flows(NUM_NEIGHBORS + 1);
	for (int i = 0; i < nseeds; i++)
	{
		const int x = i % w;
		const int y = i / w;
		// 获取当前种子点的光流
		flows[0] = { seeds_flow.at<cv::Vec2f>(y, x)[0],
			seeds_flow.at<cv::Vec2f>(y, x)[1] };

		// 获取种子点邻域中种子点的光流
		int count = 1;
		for (int n = 0; n < NUM_NEIGHBORS; n++)
		{
			const int index = neighbors[i][n];
			if (index >= 0)
			{
				const int x = index % w;
				const int y = index / w;
				flows[count] = { seeds_flow.at<cv::Vec2f>(y, x)[0],
					seeds_flow.at<cv::Vec2f>(y, x)[1] };
				count++;
			}
		}

		Circle circle = MinimumCoveringCircle(flows, count);
		search_radius.at<float>(y, x) = circle.radius;
	}
}

void SpreadRadiusInter(const std::vector<cv::Size>& seeds_size,
	cv::Mat& search_radius,
	const int iter,
	const float py_ratio)
{
	const int w = seeds_size[iter].width;
	const int h = seeds_size[iter].height;
	const int w_new = seeds_size[iter + 1].width;
	const int h_new = seeds_size[iter + 1].height;
	cv::Mat search_radius_old = search_radius.clone();

	// 上采样过程中 当dst.cols=2*src.cols+1或dst.rows=2*src.rows+1时，会报错
	if (h_new == 2 * h + 1 || h_new == 2 * h + 2)
	{
		cv::copyMakeBorder(search_radius_old, search_radius_old, 0, 1, 0, 0, cv::BORDER_REPLICATE);
	}

	if (w_new == 2 * w + 1 || w_new == 2 * w + 2)
	{
		cv::copyMakeBorder(search_radius_old, search_radius_old, 0, 0, 0, 1, cv::BORDER_REPLICATE);
	}

	search_radius_old = search_radius_old / py_ratio;
	cv::pyrUp(search_radius_old, search_radius, cv::Size(w_new, h_new));
}

void SpreadFlowInter(const std::vector<cv::Size>& seeds_size,
	cv::Mat& flow_seeds,
	const int iter,
	const float py_ratio)
{
	const int w = seeds_size[iter].width;
	const int h = seeds_size[iter].height;
	const int w_new = seeds_size[iter + 1].width;
	const int h_new = seeds_size[iter + 1].height;
	cv::Mat flow_seeds_old = flow_seeds.clone();

	// 上采样过程中 当dst.cols=2*src.cols+1或dst.rows=2*src.rows+1时，会报错
	if (h_new == 2 * h + 1 || h_new == 2 * h + 2)
	{
		cv::copyMakeBorder(flow_seeds_old, flow_seeds_old, 0, 1, 0, 0, cv::BORDER_REPLICATE);
	}

	if (w_new == 2 * w + 1 || w_new == 2 * w + 2)
	{
		cv::copyMakeBorder(flow_seeds_old, flow_seeds_old, 0, 0, 0, 1, cv::BORDER_REPLICATE);
	}

	flow_seeds_old = flow_seeds_old / py_ratio;
	cv::pyrUp(flow_seeds_old, flow_seeds, cv::Size(w_new, h_new));
}

/*****************************************Debug，后处理，保存结果*****************************************/
void RecoverOpticalFlow(const std::vector<cv::Point2f>& seeds,
	const cv::Size& image_size,
	const cv::Mat& flow_seeds,
	cv::Mat& flow_image,
	const int seeds_width)
{
	const int seeds_w = flow_seeds.cols;
	const int seeds_h = flow_seeds.rows;
	const int image_w = image_size.width;
	const int image_h = image_size.height;
	const int nmatches = seeds_w * seeds_h;
	const int radius = seeds_width / 2;

	flow_image = cv::Mat::zeros(image_h, image_w, CV_32FC2);
	for (int i = 0; i < nmatches; i++)
	{
		const int seed_x = i % flow_seeds.cols;
		const int seed_y = i / flow_seeds.cols;
		const float fx = flow_seeds.at<cv::Vec2f>(seed_y, seed_x)[0];
		const float fy = flow_seeds.at<cv::Vec2f>(seed_y, seed_x)[1];

		// draw each match as a radius*radius color block, 图像边缘部分并未被填充，
		for (int dy = -radius; dy <= radius; dy++)
		{
			for (int dx = -radius; dx <= radius; dx++)
			{
				const int x = std::max(0,
					std::min(static_cast<int>(seeds[i].x + dx + 0.5f),
						image_w - 1));
				const int y = std::max(0,
					std::min(static_cast<int>(seeds[i].y + dy + 0.5f),
						image_h - 1));
				flow_image.at<cv::Vec2f>(y, x)[0] = fx;
				flow_image.at<cv::Vec2f>(y, x)[1] = fy;
			}
		}
	}
}

void MovePixels(cv::Mat src, cv::Mat& dst,
	cv::Mat& flow, int interpolation)
{
	std::vector<cv::Mat> flow_spilit;
	cv::split(flow, flow_spilit);

	//像素重采样实现
	cv::Mat Tx_map(src.size(), CV_32FC1, 0.0);
	cv::Mat Ty_map(src.size(), CV_32FC1, 0.0);

	for (int i = 0; i < src.rows; i++)
	{
		float* p_Tx_map = Tx_map.ptr<float>(i);
		float* p_Ty_map = Ty_map.ptr<float>(i);
		for (int j = 0; j < src.cols; j++)
		{
			p_Tx_map[j] = j + flow_spilit[0].ptr<float>(i)[j];
			p_Ty_map[j] = i + flow_spilit[1].ptr<float>(i)[j];
		}
	}

	cv::remap(src, dst, Tx_map, Ty_map, interpolation);
}

void Drawgrid(const std::vector<cv::Point2f>& seeds,
	const cv::Mat& flow, cv::Mat& src, const int seeds_width)
{
	const int w = flow.cols;
	const int h = flow.rows;
	const int radius = seeds_width / 2;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			const int index = i * w + j;
			int roi_begin_x = seeds[index].x + std::round(flow.at<cv::Vec2f>(i, j)[0]) - radius;
			int roi_begin_y = seeds[index].y + std::round(flow.at<cv::Vec2f>(i, j)[1]) - radius;
			cv::Point pt1 = cv::Point(roi_begin_x, roi_begin_y);
			cv::Point pt2 = cv::Point(roi_begin_x + seeds_width, roi_begin_y + seeds_width);
			cv::rectangle(src, pt1, pt2, cv::Scalar(1.0), 1, cv::LINE_AA);
		}
	}
}

void WriteMatchesFile(const std::vector<cv::Point2f>& seeds,
	const cv::Mat& flow_seeds,
	const std::string matches_path)
{
	const int w = flow_seeds.cols;
	const int h = flow_seeds.rows;

	std::ofstream outfile(matches_path, std::ios::trunc);
	for (int i = 0; i < w * h; i++)
	{
		const int x = i % w;
		const int y = i / w;
		if (flow_seeds.at<cv::Vec2f>(y, x) != cv::Vec2f(UNKNOWN_FLOW, UNKNOWN_FLOW))
		{
			const cv::Point2f pre = seeds[i];
			const cv::Point2f flow = cv::Point2f(flow_seeds.at<cv::Vec2f>(y, x)[0],
				flow_seeds.at<cv::Vec2f>(y, x)[1]);
			const cv::Point2f curr = pre + flow;

			outfile << static_cast<int>(pre.x) << " "
				<< static_cast<int>(pre.y) << " "
				<< static_cast<int>(curr.x) << " "
				<< static_cast<int>(curr.y) << std::endl;
		}
	}

	outfile.close();
}
