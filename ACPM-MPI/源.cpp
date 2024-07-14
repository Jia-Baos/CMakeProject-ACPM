#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "ACPM.h"

int main(int argc, char* argv[])
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	//const std::string fixed_image_path = "D:/Code-VS/Project-ACPM/Project-ACPM/MLRTM/dataset2-test1.jpg";
	//const std::string moved_image_path = "D:/Code-VS/Project-ACPM/Project-ACPM/MLRTM/dataset2-test2.jpg";
	//const std::string fixed_image_path = "D:/Code-VS/Project-ACPM/Project-ACPM/Middlebury/Dimetrodon/frame10.png";
	//const std::string moved_image_path = "D:/Code-VS/Project-ACPM/Project-ACPM/Middlebury/Dimetrodon/frame11.png";
	const std::string fixed_image_path = "D:/Code-VS/Project-ACPM/Project-ACPM/MPI-Sintel/frame_0017.png";
	const std::string moved_image_path = "D:/Code-VS/Project-ACPM/Project-ACPM/MPI-Sintel/frame_0018.png";

	std::string temp_path = fixed_image_path;
	temp_path.erase(temp_path.find_last_of('.'), temp_path.length());
	const std::string matches_path = temp_path + "_matches.txt";
	const std::string optflow_path = temp_path + "_optflow.flo";
	const std::string res_image_path = temp_path + "_res_image.png";

	/************************************************************************/
	/*    Parametes for MPI-Sintel                                          */
	/************************************************************************/
	acpm_params_t acpm_params;
	acpm_params.min_width_ = 30;
	acpm_params.seeds_width_ = 3;
	acpm_params.descs_width_min_ = 13;
	acpm_params.descs_width_max_ = 25;
	acpm_params.max_displacement_ = 100;
	acpm_params.py_ratio_ = 0.5;
	acpm_params.data_thresh_ = 0.6;
	acpm_params.descs_thresh_ = 0.08;
	acpm_params.check_thresh_ = 3.0;
	acpm_params.mathces_path_ = matches_path;
	acpm_params.optflow_path_ = optflow_path;
	acpm_params.res_image_path_ = res_image_path;

	/************************************************************************/
	/*    Parametes for MY-DATASET                                          */
	/************************************************************************/
	/*acpm_params_t acpm_params;
	acpm_params.min_width_ = 30;
	acpm_params.seeds_width_ = 7;
	acpm_params.descs_width_min_ = 13;
	acpm_params.descs_width_max_ = 31;
	acpm_params.max_displacement_ = 100;
	acpm_params.py_ratio_ = 0.5;
	acpm_params.data_thresh_ = 0.6;
	acpm_params.descs_thresh_ = 0.08;
	acpm_params.mathces_path_ = matches_path;
	acpm_params.optflow_path_ = optflow_path;
	acpm_params.res_image_path_ = res_image_path;*/

	cv::Mat fixed_image = cv::imread(fixed_image_path);
	cv::Mat moved_image = cv::imread(moved_image_path);

	ACPM* acpm = new ACPM(acpm_params);
	acpm->Compute(fixed_image, moved_image);
	delete acpm;

	return 0;
}