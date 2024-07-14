#include "ACPM.h"

ACPM::ACPM()
{
	std::cout << "Initalize the Class: ACPM" << std::endl;
}

ACPM::ACPM(const acpm_params_t& acpm_parames)
{
	this->acpm_parames = acpm_parames;
	std::cout << "Initalize the Class: ACPM" << std::endl;
}

ACPM::~ACPM()
{
	std::cout << "Destory the Class: ACPM" << std::endl;
}

/************************************具体方法实现************************************/
void  ACPM::SetInput(const cv::Mat& fixed_image, const cv::Mat& moved_image, const float py_ratio)
{
	// Assert the src has read success and has the same size
	assert(!fixed_image.empty());
	assert(!moved_image.empty());

	// Assert the type of src is CV_8UC1
	cv::Mat fixed_image_gray;
	cv::Mat moved_image_gray;
	ImagePadded(fixed_image, moved_image, fixed_image_gray, moved_image_gray,
		4, 4);

	if (fixed_image.type() == CV_8UC3 || moved_image.type() == CV_8UC3)
	{
		cv::cvtColor(fixed_image_gray, fixed_image_gray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(moved_image_gray, moved_image_gray, cv::COLOR_BGR2GRAY);
	}

	// MyDataSet
	fixed_image_gray.convertTo(this->fixed_image_, CV_32FC1, 1 / 255.0);
	moved_image_gray.convertTo(this->moved_image_, CV_32FC1, 1 / 255.0);

	// the ratio cannot be arbitrary numbers
	if (this->acpm_parames.py_ratio_ > 0.98 || this->acpm_parames.py_ratio_ < 0.4)
	{
		this->acpm_parames.py_ratio_ = 0.75;
	}
	// Change the layer_ according to image's size
	int max_col_row = this->fixed_image_.cols > this->fixed_image_.rows ?
		this->fixed_image_.cols : this->fixed_image_.rows;
	this->acpm_parames.layers_ = std::log(static_cast<float>(this->acpm_parames.min_width_) / max_col_row)
		/ std::log(py_ratio);
}

void ACPM::Compute(const cv::Mat& fixed_image, const cv::Mat& moved_image)
{
	CTime::in_clock();
	// Construct the fixed_image_ and moved_image_
	this->SetInput(fixed_image, moved_image, this->acpm_parames.py_ratio_);

	/************************************************************************/
	/*    Construct the Gauss Pyramid                                       */
	/************************************************************************/
	std::vector<cv::Mat> fixed_image_pyramid(this->acpm_parames.layers_);
	std::vector<cv::Mat> moved_image_pyramid(this->acpm_parames.layers_);
	GetGaussPyramid(this->fixed_image_, fixed_image_pyramid, this->acpm_parames.layers_);
	GetGaussPyramid(this->moved_image_, moved_image_pyramid, this->acpm_parames.layers_);
	//cv::buildPyramid(this->fixed_image_, fixed_image_pyramid, this->acpm_parames.layers_ - 1);
	//cv::buildPyramid(this->moved_image_, moved_image_pyramid, this->acpm_parames.layers_ - 1);
	//ConstructPyramid(this->fixed_image_, fixed_image_pyramid, this->acpm_parames.py_ratio_, this->acpm_parames.layers_);
	//ConstructPyramid(this->moved_image_, moved_image_pyramid, this->acpm_parames.py_ratio_, this->acpm_parames.layers_);

	/************************************************************************/
	/*    Make Seeds and Get Parameters                                     */
	/************************************************************************/
	std::vector<int> max_radius(this->acpm_parames.layers_);							// 金字塔中每一层最大搜索半径
	std::vector<int> max_displacement(this->acpm_parames.layers_);						// 金字塔中每一层最大位移量
	std::vector<cv::Size> seeds_size(this->acpm_parames.layers_);						// 金字塔中每一层种子点的行数和列数
	std::vector<cv::Size> images_size(this->acpm_parames.layers_);						// 金字塔中每一层图像的行数和列数
	std::vector<std::vector<cv::Point2f>> nSeeds(this->acpm_parames.layers_);			// 金字塔中每一层种子点的坐标
	std::vector<std::vector<std::vector<int>>> nNeighbors(this->acpm_parames.layers_);	// 金字塔中每一层每一个种子点的八邻域种子点
	for (int i = 0; i < this->acpm_parames.layers_; i++)
	{
		const float scale = std::pow(this->acpm_parames.py_ratio_, this->acpm_parames.layers_ - i - 1);
		max_radius[i] = std::min(static_cast<int>(std::round(this->acpm_parames.max_displacement_ * scale)), 32);
		max_displacement[i] = static_cast<int>(std::round(this->acpm_parames.max_displacement_ * scale));

		const int w = fixed_image_pyramid[i].cols;
		const int h = fixed_image_pyramid[i].rows;
		images_size[i] = cv::Size(w, h);

		const int seeds_w = w / this->acpm_parames.seeds_width_;
		const int seeds_h = h / this->acpm_parames.seeds_width_;
		seeds_size[i] = cv::Size(seeds_w, seeds_h);

		std::vector<cv::Point2f> seeds(seeds_w * seeds_h);
		std::vector<std::vector<int>> neighbors(seeds_w * seeds_h);
		MakeSeedsAndNeighbors(seeds, neighbors, w, h, this->acpm_parames.seeds_width_);
		nSeeds[i] = seeds;
		nNeighbors[i] = neighbors;
	}

	this->flow_seeds_ = cv::Mat(seeds_size.front(), CV_32FC2, cv::Scalar(0.0, 0.0));
	this->search_radius_ = cv::Mat(seeds_size.front(), CV_32FC1, cv::Scalar(max_radius.front()));
	for (int iter = 0; iter < this->acpm_parames.layers_; ++iter)
	{
		std::cout << "All Layers: " << this->acpm_parames.layers_ << std::endl;
		std::cout << "	Current Layer: " << iter << std::endl;
		cv::Mat fixed_image_curr = fixed_image_pyramid[iter];
		cv::Mat moved_image_curr = moved_image_pyramid[iter];
		std::vector<cv::Mat> fixed_image_descs(seeds_size[iter].width * seeds_size[iter].height);
		std::vector<cv::Mat> moved_image_descs(seeds_size[iter].width * seeds_size[iter].height);
		std::vector<cv::Vec4f> descs_info(seeds_size[iter].width * seeds_size[iter].height);

		/************************************************************************/
		/*    Get Descs                                                         */
		/************************************************************************/
		GetDescs(fixed_image_curr, fixed_image_descs, descs_info, nSeeds[iter],
			this->acpm_parames.descs_width_min_,
			this->acpm_parames.descs_width_max_,
			this->acpm_parames.descs_thresh_);

		//cv::Mat score_img = cv::Mat::zeros(seeds_size[iter], CV_32FC1);
		//GetDescs(fixed_image_curr, fixed_image_descs, descs_info, nSeeds[iter],
		//	this->acpm_parames.descs_width_min_,
		//	this->acpm_parames.descs_width_max_,
		//	this->acpm_parames.descs_thresh_,
		//	score_img);

		//double minVal, maxVal;
		//cv::Point minLoc, maxLoc;
		//cv::minMaxLoc(score_img, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
		//for (size_t i = 0; i < score_img.rows; ++i)
		//{
		//	for (size_t j = 0; j < score_img.cols; ++j)
		//	{
		//		if (score_img.ptr<float>(i)[j] == 0.0f) {
		//			score_img.ptr<float>(i)[j] = maxVal;
		//		}
		//	}
		//}

		//cv::normalize(score_img, score_img, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
		//score_img.convertTo(score_img, CV_8UC1, 255.0, 0.0);
		//// save img
		//cv::applyColorMap(score_img, score_img, cv::COLORMAP_JET);
		//cv::imwrite("colormap.png", score_img);

		GetDescs(moved_image_curr, moved_image_descs, descs_info, nSeeds[iter],
			this->flow_seeds_, this->search_radius_);
		CTime::inter_clock("	Get Descs: ");

		/************************************************************************/
		/*    Get Displacement                                                  */
		/************************************************************************/
		MatchDescs(fixed_image_descs, moved_image_descs, this->flow_seeds_, this->search_radius_,
			this->acpm_parames.data_thresh_);
		CTime::inter_clock("	Get Displacement: ");

		/************************************************************************/
		/*    Post Process                                                      */
		/************************************************************************/
		this->seeds_flag_ = cv::Mat::zeros(seeds_size[iter], CV_8UC1);
		CrossCheck(nSeeds[iter], this->flow_seeds_, this->seeds_flag_,
			images_size[iter].width, images_size[iter].height,
			max_displacement[iter], this->acpm_parames.descs_thresh_);

		//WriteMatchesFile(nSeeds.back(), this->flow_seeds_, "D:/Code-VS/picture/figs/matches-acpm7-pre.txt");

		FillHoles(this->flow_seeds_, this->seeds_flag_);
		CTime::inter_clock("	Post Process: ");

#ifdef _DEBUG
		DrawPatch(fixed_image_curr, nSeeds[iter], descs_info);
		RecoverOpticalFlow(nSeeds[iter], images_size[iter], this->flow_seeds_, this->flow_image_,
			this->acpm_parames.seeds_width_);
		MovePixels(moved_image_curr, this->moved_image_warpped_, this->flow_image_, cv::INTER_CUBIC);
		this->res_image_ = cv::abs(this->moved_image_warpped_ - fixed_image_curr);
		DrawGrid(nSeeds[iter], this->flow_seeds_, moved_image_curr, this->acpm_parames.seeds_width_);
#endif // _DEBUG

		/************************************************************************/
		/*    Compute the Radius, Update Radius and Optflow                     */
		/************************************************************************/
		if (iter != this->acpm_parames.layers_ - 1)
		{
			UpdateSearchRadius(this->flow_seeds_, nNeighbors[iter], this->search_radius_);
			this->search_radius_ = cv::max(1, cv::min(this->search_radius_, max_radius[iter]));
			SpreadRadiusInter(seeds_size, this->flow_seeds_, iter, this->acpm_parames.py_ratio_);
			SpreadFlowInter(seeds_size, this->search_radius_, iter, this->acpm_parames.py_ratio_);
		}

		/*cv::Mat temp1 = this->search_radius_.clone();
		cv::normalize(temp1, temp1, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
		temp1.convertTo(temp1, CV_8UC1);
		cv::applyColorMap(temp1, temp1, cv::COLORMAP_JET);
		cv::imwrite("D:/Code-VS/picture/figs/temp1.png", temp1);*/
		//WriteFlowFile(this->flow_seeds_, "D:/Code-VS/picture/figs/temp2.flo");
	}

	CTime::out_clock("ACPM Time: ");

	/************************************************************************/
	/*    Variational Refinement                                            */
	/************************************************************************/
	cv::medianBlur(this->flow_seeds_, this->flow_seeds_, 5);
	RecoverOpticalFlow(nSeeds.back(), images_size.back(), this->flow_seeds_,
		this->flow_image_, this->acpm_parames.seeds_width_);
	VariationalRefine(this->fixed_image_, this->moved_image_, this->flow_image_);

	CTime::inter_clock("Variational Refinement: ");
	CTime::out_clock("Totoal time: ");

	// Save the res_image, optflow, matches
	if (!this->acpm_parames.res_image_path_.empty())
	{
		MovePixels(this->moved_image_, this->moved_image_warpped_, this->flow_image_, cv::INTER_CUBIC);
		this->res_image_ = cv::abs(this->moved_image_warpped_ - this->fixed_image_);
		this->res_image_.convertTo(this->res_image_, CV_8UC1, 255.0);
		//this->moved_image_warpped_.convertTo(this->moved_image_warpped_, CV_8UC1, 255.0);
		//cv::imwrite("D:/Code-VS/picture/figs/moved_image_warpped.png", this->moved_image_warpped_);
		cv::imwrite(this->acpm_parames.res_image_path_, this->res_image_);
	}

	if (!this->acpm_parames.optflow_path_.empty())
	{
		WriteFlowFile(this->flow_image_, this->acpm_parames.optflow_path_.c_str());
	}
	if (!this->acpm_parames.mathces_path_.empty())
	{
		WriteMatchesFile(nSeeds.back(), this->flow_seeds_, this->acpm_parames.mathces_path_);
	}
}
