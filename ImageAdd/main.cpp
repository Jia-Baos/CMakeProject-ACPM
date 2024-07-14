#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

int main()
{
	std::string fixed_image_path =
		"E:\\paper4-dataset\\template.png";
	std::string moved_image_path =
		"E:\\paper4-dataset\\test-png\\204.png";

	cv::Mat fixed_image = cv::imread(fixed_image_path);
	cv::Mat moved_image = cv::imread(moved_image_path);
	//cv::cvtColor(fixed_image, fixed_image, cv::COLOR_BGR2GRAY);
	//cv::cvtColor(moved_image, moved_image, cv::COLOR_BGR2GRAY);

	cv::Mat result = cv::Mat::zeros(fixed_image.size(), fixed_image.type());
	cv::addWeighted(fixed_image, 0.5, moved_image, 0.5, 0.0, result);

	/*cv::resize(moved_image, moved_image,
		cv::Size(fixed_image.rows, fixed_image.rows), cv::INTER_CUBIC);
	cv::copyMakeBorder(moved_image, moved_image, 0, 0,
		(fixed_image.cols - moved_image.cols) / 2,
		(fixed_image.cols - moved_image.cols) / 2,
		cv::BORDER_CONSTANT,
		cv::Scalar(255, 255, 255));*/

	cv::namedWindow("result", cv::WINDOW_NORMAL);
	cv::imshow("result", cv::abs(fixed_image - moved_image));
	cv::imwrite("E:\\paper4-dataset\\tested-res.png", result);

	cv::waitKey();
}
