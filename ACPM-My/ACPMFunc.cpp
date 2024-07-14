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
	const int width_max_even = width_max % patch_width > 0 ?
		patch_width * (static_cast<int>(width_max / patch_width) + 1) : patch_width * (static_cast<int>(width_max / patch_width));
	const int height_max_even = height_max % patch_height > 0 ?
		patch_height * (static_cast<int>(height_max / patch_height) + 1) : patch_height * (static_cast<int>(height_max / patch_height));

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

void ConstructPyramid(const cv::Mat& src, std::vector<cv::Mat>& pyramid,
	const float py_ratio, const int py_layers)
{
	float ratio = py_ratio;
	// the ratio cannot be arbitrary numbers
	if (ratio > 0.98f || ratio < 0.4f)
		ratio = 0.75f;

	pyramid[0] = src;
	const float baseSigma = (1 / ratio - 1);	// baseSigma=1
	const int n = static_cast<int>(std::log(0.25) / std::log(ratio));	// n=2
	const float nSigma = n * baseSigma;
	for (int i = 1; i < py_layers; i++)
	{
		cv::Mat foo;
		if (i <= n)
		{
			const float sigma = i * baseSigma;
			const float rate = std::pow(ratio, i);
			cv::GaussianBlur(src, foo, cv::Size(), sigma);
			cv::resize(foo, foo, cv::Size(), rate, rate);
			pyramid[i] = foo;
		}
		else
		{
			cv::GaussianBlur(pyramid[i - n], foo, cv::Size(), nSigma);
			const float rate = std::pow(ratio, i) * src.cols / foo.cols;
			cv::resize(foo, foo, cv::Size(), rate, rate);
			pyramid[i] = foo;
		}
	}
	std::reverse(pyramid.begin(), pyramid.end());
}

void MakeSeedsAndNeighbors(std::vector<cv::Point2f>& seeds,
	std::vector<std::vector<int>>& neighbors,
	const int w, const int h, const int step)
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
			else {
				// 这里的 neighbors 是当前 seed 八邻域中的 seed
				neighbors_current[n] = nby * gridw + nbx;
			}
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
		const uchar* src_poniter = temp.ptr<uchar>(i);
		for (int j = 0; j < w; j++)
		{
			int gray_value = src_poniter[j];
			gray_array[gray_value]++;
		}
	}

	// Compute the probability of every gray value and entropy
	float my_entropy = 0;
	float* gray_array_pointer = gray_array;
	for (int i = 0; i < 255; i++)
	{
		float gray_value_prob = *gray_array_pointer / (w * h);
		if (gray_value_prob != 0)
		{
			my_entropy = my_entropy - gray_value_prob * std::log(gray_value_prob);
		}
		else { my_entropy = my_entropy; }
		gray_array_pointer++;
	}
	return my_entropy;
}

void BorderTest(int& roi_x1, int& roi_y1, int& roi_x2, int& roi_y2, const int w, const int h)
{
	roi_x1 = roi_x1 < 0 ? 0 : roi_x1;
	roi_y1 = roi_y1 < 0 ? 0 : roi_y1;
	roi_x2 = roi_x2 > w ? w : roi_x2;
	roi_y2 = roi_y2 > h ? h : roi_y2;
	//std::cout << "Border Test..." << std::endl;
}

float TextureTest(const cv::Mat& image, int& roi_x1, int& roi_y1, int& roi_x2, int& roi_y2,
	const int descs_width_max, const float descs_thresh)
{
	const int w = image.cols;
	const int h = image.rows;
	BorderTest(roi_x1, roi_y1, roi_x2, roi_y2, w, h);
	cv::Mat roi = image(cv::Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1));
	float texture_cost = std::exp(-Entropy(roi));

	// if the text_cost is smaller than descs_thresh, we think it is a good desc
	const int step = 3;
	while (texture_cost > descs_thresh)
	{
		roi_x1 = roi_x1 - step < 0 ? 0 : roi_x1 - step;
		roi_y1 = roi_y1 - step < 0 ? 0 : roi_y1 - step;
		roi_x2 = roi_x2 + step > w ? w : roi_x2 + step;
		roi_y2 = roi_y2 + step > h ? h : roi_y2 + step;
		if (roi_x2 - roi_x1 >= descs_width_max || roi_y2 - roi_y1 >= descs_width_max ||
			roi_x2 - roi_x1 >= w || roi_y2 - roi_y1 >= h)
		{
			break;
		}
		roi = image(cv::Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1));
		texture_cost = std::exp(-Entropy(roi));
	}

	return texture_cost;
}

void GetDescs(const cv::Mat& image, std::vector<cv::Mat>& descs, std::vector<cv::Vec4f>& descs_info,
	const std::vector<cv::Point2f> seeds, const int descs_width_min, const int descs_width_max,
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
		const float texture_cost = TextureTest(image, roi_x1, roi_y1, roi_x2, roi_y2, descs_width_max, descs_thresh);
		if (texture_cost > descs_thresh) { continue; }
		descs_info[i][0] = seeds[i].x - roi_x1;
		descs_info[i][1] = seeds[i].y - roi_y1;
		descs_info[i][2] = roi_x2 - seeds[i].x;
		descs_info[i][3] = roi_y2 - seeds[i].y;
		const cv::Mat roi = image(cv::Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1));
		descs[i] = roi;
	}
}

void GetDescs(const cv::Mat& image, std::vector<cv::Mat>& descs, std::vector<cv::Vec4f>& descs_info,
	const std::vector<cv::Point2f> seeds, const int descs_width_min, const int descs_width_max,
	const float descs_thresh, cv::Mat& score_img)
{
	const int radius = descs_width_min / 2;
	const int nseeds = seeds.size();

	for (int i = 0; i < nseeds; i++)
	{
		int roi_x1 = seeds[i].x - radius;
		int roi_y1 = seeds[i].y - radius;
		int roi_x2 = seeds[i].x + radius + 1;
		int roi_y2 = seeds[i].y + radius + 1;
		const float texture_cost = TextureTest(image, roi_x1, roi_y1, roi_x2, roi_y2, descs_width_max, descs_thresh);
		if (texture_cost > descs_thresh) { continue; }
		descs_info[i][0] = seeds[i].x - roi_x1;
		descs_info[i][1] = seeds[i].y - roi_y1;
		descs_info[i][2] = roi_x2 - seeds[i].x;
		descs_info[i][3] = roi_y2 - seeds[i].y;
		const cv::Mat roi = image(cv::Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1));
		descs[i] = roi;

		score_img.ptr<float>(i / score_img.cols)[i % score_img.cols] = texture_cost;
	}
}

void GetDescs(const cv::Mat& image, std::vector<cv::Mat>& descs,
	const std::vector<cv::Vec4f>& descs_info, const std::vector<cv::Point2f> seeds,
	const cv::Mat& flow, const cv::Mat& radius)
{
	const int w = flow.cols;
	const int h = flow.rows;
	const int nseeds = seeds.size();

	int width_padded = 0;
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(radius, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	width_padded += 2 * std::ceil(maxVal);

	// 位移量可能为负，所以需要添加绝对值
	std::vector<cv::Mat> flow_split;
	cv::split(cv::abs(flow), flow_split);
	cv::minMaxLoc(flow_split[0], &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	width_padded += std::ceil(maxVal);

	cv::minMaxLoc(flow_split[1], &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	width_padded += std::ceil(maxVal);

	cv::Mat image_padded = image.clone();
	cv::copyMakeBorder(image, image_padded, width_padded, width_padded,
		width_padded, width_padded, cv::BORDER_REPLICATE);

	for (int i = 0; i < nseeds; i++)
	{
		const int x = i % w;
		const int y = i / w;

		if (descs_info[i][0] < eps && descs_info[i][1] < eps &&
			descs_info[i][2] < eps && descs_info[i][3] < eps)
		{
			continue;
		}

		const int flow_x = std::round(flow.at<cv::Vec2f>(y, x)[0]);
		const int flow_y = std::round(flow.at<cv::Vec2f>(y, x)[1]);
		const int search_radius = radius.at<float>(y, x);

		int roi_x1 = width_padded + seeds[i].x - descs_info[i][0] + flow_x - search_radius;
		int roi_y1 = width_padded + seeds[i].y - descs_info[i][1] + flow_y - search_radius;
		int roi_x2 = width_padded + seeds[i].x + descs_info[i][2] + flow_x + search_radius + 1;
		int roi_y2 = width_padded + seeds[i].y + descs_info[i][3] + flow_y + search_radius + 1;

		const cv::Mat roi = image_padded(cv::Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1));
		descs[i] = roi;
	}
}

float ParabolicInterpolation(const int l, const int c, const int r)
{
	const float offset = 0.5 * (l - r) / (l + r - 2.0 * c);
	return offset;
}

void MatchDescs(const std::vector<cv::Mat>& descs1, const std::vector<cv::Mat>& descs2,
	cv::Mat& flow, const cv::Mat& radius, const float data_thresh)
{
	const int w = flow.cols;
	const int h = flow.rows;
	const int nseeds = descs1.size();

	for (int i = 0; i < nseeds; i++)
	{
		const int x = i % w;
		const int y = i / w;

		// 将弱纹理区域种子点的光流置为UNKNOWN_FLOW
		if (descs1[i].rows == 0 || descs1[i].cols == 0 ||
			descs2[i].rows == 0 || descs2[i].cols == 0)
		{
			flow.at<cv::Vec2f>(y, x) = cv::Vec2f(UNKNOWN_FLOW, UNKNOWN_FLOW);
			continue;
		}

		const cv::Mat block1 = descs1[i].clone();
		const cv::Mat block2 = descs2[i].clone();

		// Template matching
		cv::Size result_size = cv::Size(block2.cols - block1.cols + 1, block2.rows - block1.rows + 1);
		cv::Mat result = cv::Mat::zeros(result_size, CV_32FC1);
		cv::matchTemplate(block2, block1, result, cv::TM_CCOEFF_NORMED);
		double minVal, maxVal;
		cv::Point minLoc, maxLoc;
		cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

		// 将遮挡区域种子点的光流置为UNKNOWN_FLOW
		if (maxVal < data_thresh)
		{
			flow.at<cv::Vec2f>(y, x) = cv::Vec2f(UNKNOWN_FLOW, UNKNOWN_FLOW);
			continue;
		}

		// Get the new flow of every block
		const float flow_x_old = flow.at<cv::Vec2f>(y, x)[0];
		const float flow_y_old = flow.at<cv::Vec2f>(y, x)[1];
		const int search_radius = radius.at<float>(y, x);
		float flow_x_new = flow_x_old + maxLoc.x - search_radius;
		float flow_y_new = flow_y_old + maxLoc.y - search_radius;
		flow.at<cv::Vec2f>(y, x)[0] = flow_x_new;
		flow.at<cv::Vec2f>(y, x)[1] = flow_y_new;
	}
}

/*****************************************错误偏移量校正*****************************************/
void CrossCheck(const std::vector<cv::Point2f>& seeds,
	cv::Mat& flow_seeds, cv::Mat& good_seeds_flag,
	const int w, const int h, const int max_displacement,
	const float descs_thresh)
{
	const int seeds_num = seeds.size();
	const int seeds_width = flow_seeds.cols;

	cv::Vec2f flow_curr;
	cv::Point2f seed_warpped;
	for (int i = 0; i < seeds_num; i++)
	{
		const int x = i % seeds_width;
		const int y = i / seeds_width;

		if (flow_seeds.at<cv::Vec2f>(y, x) == cv::Vec2f(UNKNOWN_FLOW, UNKNOWN_FLOW))
		{
			good_seeds_flag.at<uchar>(y, x) = 255;
		}

		flow_curr = flow_seeds.at<cv::Vec2f>(y, x);
		seed_warpped = cv::Point2f(seeds[i].x + flow_curr[0], seeds[i].y + flow_curr[1]);

		if (seed_warpped.x < 0 || seed_warpped.x >= w ||
			seed_warpped.y < 0 || seed_warpped.y >= h ||
			std::sqrtf(flow_curr[0] * flow_curr[0] + flow_curr[1] * flow_curr[1])>max_displacement)
		{
			good_seeds_flag.at<uchar>(y, x) = 255;
		}
	}
}

void FillHoles(cv::Mat& flow, cv::Mat& good_seeds_flag)
{
	const int w = flow.cols;
	const int h = flow.rows;

	double maxVal = 0.0;
	cv::Point maxLoc;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	cv::minMaxLoc(good_seeds_flag, NULL, &maxVal, NULL, &maxLoc, cv::Mat());

	int flag = 0;
	while (maxVal == 255)
	{
		cv::Mat temp = good_seeds_flag.clone();
		cv::erode(good_seeds_flag, good_seeds_flag, element);

		// 正向反向交替迭代优化
		int i0 = 0, i1 = h, i2 = 0, i3 = w, step = 1;
		if (flag == 1)
		{
			i0 = h - 1, i1 = -1, i2 = w - 1, i3 = -1, step = -1;
		}
		for (int i = i0; i != i1; i += step)
		{
			for (int j = i2; j != i3; j += step)
			{
				if ((temp.at<uchar>(i, j) - good_seeds_flag.at<uchar>(i, j)) == 255)
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
								flow_x_estimate += flow.at<cv::Vec2f>(y, x)[0];
								flow_y_estimate += flow.at<cv::Vec2f>(y, x)[1];
								++candidate_size;
							}
						}
					}
					flow.at<cv::Vec2f>(i, j) = cv::Vec2f(flow_x_estimate / candidate_size,
						flow_y_estimate / candidate_size);
					temp.at<uchar>(i, j) = 0;
				}
			}
		}
		flag = flag > 0 ? 0 : 1;
		cv::minMaxLoc(good_seeds_flag, NULL, &maxVal, NULL, &maxLoc, cv::Mat());
	}
}

/*****************************************自动计算搜索半径，层间传播光流*****************************************/
void UpdateSearchRadius(const cv::Mat& flow,
	const std::vector<std::vector<int>>& neighbors,
	cv::Mat& radius)
{
	const int w = flow.cols;
	const int h = flow.rows;
	const int nseeds = neighbors.size();
	std::vector<MyVector> flows(NUM_NEIGHBORS + 1);
	for (int i = 0; i < nseeds; i++)
	{
		const int x = i % w;
		const int y = i / w;
		// 获取当前种子点的光流
		flows[0] = { flow.at<cv::Vec2f>(y, x)[0], flow.at<cv::Vec2f>(y, x)[1] };

		// 获取种子点邻域中种子点的光流
		int count = 1;
		for (int n = 0; n < NUM_NEIGHBORS; n++)
		{
			const int index = neighbors[i][n];
			if (index >= 0)
			{
				const int x = index % w;
				const int y = index / w;
				flows[count] = { flow.at<cv::Vec2f>(y, x)[0], flow.at<cv::Vec2f>(y, x)[1] };
				count++;
			}
		}

		Circle circle = MinimumCoveringCircle(flows, count);
		radius.at<float>(y, x) = circle.radius;
	}
}

void SpreadRadiusInter(const std::vector<cv::Size>& seeds_size,
	cv::Mat& search_radius, const int iter, const float py_ratio)
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
	cv::Mat& flow, const int iter, const float py_ratio)
{
	const int w = seeds_size[iter].width;
	const int h = seeds_size[iter].height;
	const int w_new = seeds_size[iter + 1].width;
	const int h_new = seeds_size[iter + 1].height;
	cv::Mat flow_old = flow.clone();

	// 上采样过程中 当dst.cols=2*src.cols+1或dst.rows=2*src.rows+1时，会报错
	if (h_new == 2 * h + 1 || h_new == 2 * h + 2)
	{
		cv::copyMakeBorder(flow_old, flow_old, 0, 1, 0, 0, cv::BORDER_REPLICATE);
	}

	if (w_new == 2 * w + 1 || w_new == 2 * w + 2)
	{
		cv::copyMakeBorder(flow_old, flow_old, 0, 0, 0, 1, cv::BORDER_REPLICATE);
	}

	flow_old = flow_old / py_ratio;
	cv::pyrUp(flow_old, flow, cv::Size(w_new, h_new));
}

/*****************************************Debug，后处理，保存结果*****************************************/
void RecoverOpticalFlow(const std::vector<cv::Point2f>& seeds, const cv::Size& image_size,
	const cv::Mat& flow, cv::Mat& flow_norm, const int seeds_width)
{
	const int w = image_size.width;
	const int h = image_size.height;
	const int nmatches = seeds.size();
	const int radius = seeds_width / 2;

	flow_norm = cv::Mat::zeros(h, w, CV_32FC2);
	for (int i = 0; i < nmatches; i++)
	{
		const int x = i % flow.cols;
		const int y = i / flow.cols;
		const float fx = flow.at<cv::Vec2f>(y, x)[0];
		const float fy = flow.at<cv::Vec2f>(y, x)[1];

		// draw each match as a radius*radius color block, 图像边缘部分并未被填充，
		for (int dy = -radius; dy <= radius; dy++)
		{
			for (int dx = -radius; dx <= radius; dx++)
			{
				const int x = std::max(0, std::min(static_cast<int>(seeds[i].x + dx + 0.5f), w - 1));
				const int y = std::max(0, std::min(static_cast<int>(seeds[i].y + dy + 0.5f), h - 1));
				flow_norm.at<cv::Vec2f>(y, x)[0] = fx;
				flow_norm.at<cv::Vec2f>(y, x)[1] = fy;
			}
		}
	}
}

void MovePixels(const cv::Mat& src, cv::Mat& dst, const cv::Mat& flow, const int interpolation)
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

	// dst(x, y) = src(mapx(x, y), mapy(x, y))
	cv::remap(src, dst, Tx_map, Ty_map, interpolation);
}

void DrawPatch(const cv::Mat& src,
	const std::vector<cv::Point2f>& seeds,
	const std::vector<cv::Vec4f>& descs_info)
{
	const int w = src.cols;
	const int h = src.rows;
	const int seeds_num = seeds.size();

	cv::Mat dst = src.clone();
	dst.convertTo(dst, CV_8UC1, 255.0);
	cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);

	cv::RNG rng(1314);
	/*for (int index = 0; index < seeds_num; index = index + 1)
	{
		step = rng.uniform(8, 16);

		cv::Point center = cv::Point(seeds[index].x, seeds[index].y);

		cv::Vec3b vec3 = cv::Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		cv::circle(dst, center, 1, vec3, -1);
	}*/

	int step = 1;
	for (int index = 0; index < seeds_num; index = index + step)
	{
		step = rng.uniform(8, 16);

		cv::Point center = cv::Point(seeds[index].x, seeds[index].y);

		cv::Vec3b vec3 = cv::Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

		int roi_x1 = seeds[index].x - descs_info[index][0];
		int roi_y1 = seeds[index].y - descs_info[index][1];
		int roi_x2 = seeds[index].x + descs_info[index][2];
		int roi_y2 = seeds[index].y + descs_info[index][3];

		cv::Point pt1 = cv::Point(roi_x1, roi_y1);
		cv::Point pt2 = cv::Point(roi_x2, roi_y2);

		if (descs_info[index] != cv::Vec4f(0.0, 0.0, 0.0, 0.0))
		{
			cv::circle(dst, center, 1, vec3, -1);
			cv::rectangle(dst, pt1, pt2, vec3, 1, cv::LINE_AA);
		}
	}

	/*std::string path = "D:/Code-VS/Project-ACPM/Project-ACPM/pathces-colored.png";
	cv::imwrite(path, dst);*/
}

void DrawGrid(const std::vector<cv::Point2f>& seeds,
	const cv::Mat& flow,
	cv::Mat& src,
	const int seeds_width)
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

void RemoveSpeckles(cv::Mat& flow, const float thresh, const int min_area)
{
	const int w = flow.cols;
	const int h = flow.rows;

	cv::Mat visited = cv::Mat::zeros(h, w, CV_8UC1);	// 定义标记像素是否访问的数组
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			cv::Vec2f flo = flow.at<cv::Vec2f>(i, j);
			if (visited.at<uchar>(i, j))
			{
				continue;	// 跳过已访问的像素及无效像素
			}

			// 广度优先遍历，区域跟踪
			std::vector<cv::Vec2f> vec;
			vec.emplace_back(cv::Vec2f(j, i));
			visited.at<uchar>(i, j) = 1;

			int cur = 0, next = 0;
			do {
				// 广度优先遍历，区域跟踪
				next = vec.size();
				for (int k = cur; k < next; k++)
				{
					const cv::Vec2f pixel = vec[k];
					const int col = pixel[0];
					const int row = pixel[1];
					const cv::Vec2f flo_base = flow.at<cv::Vec2f>(row, col);
					// 8邻域遍历
					for (int k = 0; k < NUM_NEIGHBORS; k++)
					{
						const int colc = col + NEIGHBOR_DX[k];
						const int rowr = row + NEIGHBOR_DY[k];
						if (rowr >= 0 && rowr < h && colc >= 0 && colc < w)
						{
							const cv::Vec2f flo_neighbor = flow.at<cv::Vec2f>(rowr, colc);
							if (!visited.at<uchar>(rowr, colc) && cv::norm(flo_neighbor - flo_base) <= thresh)
							{
								vec.emplace_back(colc, rowr);
								visited.at<uchar>(rowr, colc) = 1;
							}
						}
					}
				}
				cur = next;
			} while (next < vec.size());

			if (vec.size() <= min_area)
			{
				for (auto iter : vec)
				{
					flow.at<cv::Vec2f>(iter[1], iter[0]) = cv::Vec2f(UNKNOWN_FLOW, UNKNOWN_FLOW);
				}
			}
		}
	}
}

void WriteMatchesFile(const std::vector<cv::Point2f>& seeds, const cv::Mat& flow_seeds,
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
