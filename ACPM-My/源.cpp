#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "ACPM.h"

std::string ReplaceAll(const std::string& src,
	const std::string& old_value,
	const std::string& new_value)
{
	std::string dst = src;
	// 每次重新定位起始位置，防止上轮替换后的字符串形成新的old_value
	for (std::string::size_type pos(0); pos != std::string::npos; pos += new_value.length())
	{
		if ((pos = dst.find(old_value, pos)) != std::string::npos)
		{
			dst.replace(pos, old_value.length(), new_value);
		}
		else break;
	}
	return dst;
}

int main(int argc, char* argv[])
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	/************************************************************************/
	/*    Parametes for MY-DATASET                                          */
	/************************************************************************/
	acpm_params_t acpm_params;
	acpm_params.min_width_ = 30;
	acpm_params.seeds_width_ = 13;
	acpm_params.descs_width_min_ = 13;
	acpm_params.descs_width_max_ = 31;
	acpm_params.max_displacement_ = 100;
	acpm_params.py_ratio_ = 0.5;
	acpm_params.data_thresh_ = 0.6;
	acpm_params.descs_thresh_ = 0.08;

	/************************************************************************/
	/*    Parametes for MPI-Sintel                                          */
	/************************************************************************/
	/*acpm_params_t acpm_params;
	acpm_params.min_width_ = 30;
	acpm_params.seeds_width_ = 5;
	acpm_params.descs_width_min_ = 7;
	acpm_params.descs_width_max_ = 19;
	acpm_params.max_displacement_ = 100;
	acpm_params.py_ratio_ = 0.5;
	acpm_params.data_thresh_ = 0.4;
	acpm_params.descs_thresh_ = 0.08;*/

    const std::string target_path = "D:\\Code-VS\\picture\\ACPM\\MPI-Sintel\\";
    const std::string fixed_image_path = "D:\\Code-VS\\picture\\ACPM\\MPI-Sintel\\frame_0029.png";
    const std::string moved_image_path = "D:\\Code-VS\\picture\\ACPM\\MPI-Sintel\\frame_0030.png";

    const std::string matches_path = target_path + "matches-acpm7.txt";
    const std::string optflow_path = target_path + "optflow-acpm7.flo";
    const std::string res_image_path = target_path + "res-acpm7.png";
    acpm_params.mathces_path_ = matches_path;
    acpm_params.optflow_path_ = optflow_path;
    acpm_params.res_image_path_ = res_image_path;

    cv::Mat fixed_image = cv::imread(fixed_image_path);
    cv::Mat moved_image = cv::imread(moved_image_path);

    ACPM* acpm = new ACPM(acpm_params);
    acpm->Compute(fixed_image, moved_image);
    delete acpm;

	/*const std::string fixed_image_path = "F:\\DataSet\\dataset2\\template\\template.png";
	const std::string save_path = "F:\\DataSet\\dataset2\\matches-acpm13\\";
	std::filesystem::path file_path = "F:\\DataSet\\dataset2\\data";
	for (auto iter : std::filesystem::directory_iterator(file_path))
	{
		std::cout << iter.path().string() << std::endl;
		std::string moved_image_path = iter.path().string();
		std::string matches_path = save_path + iter.path().filename().string();
		matches_path = ReplaceAll(matches_path, "png", "txt");
		acpm_params.mathces_path_ = matches_path;

		cv::Mat fixed_image = cv::imread(fixed_image_path);
		cv::Mat moved_image = cv::imread(moved_image_path);

		ACPM* acpm = new ACPM(acpm_params);
		acpm->Compute(fixed_image, moved_image);
		delete acpm;
	}*/

	return 0;
}