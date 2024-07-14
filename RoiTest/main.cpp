#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

int main() {

	std::string img_path = "D:\\Code-VS\\picture\\ImageRegistration\\graffiti.png";
	cv::Mat img = cv::imread(img_path);

	cv::Mat roi = img(cv::Rect(0, 0, img.cols / 2, img.rows / 2));
	cv::Mat mat_revise = cv::Mat(img.rows / 2, img.cols / 2, CV_8UC3,cv::Scalar(0, 0, 0));
	roi = mat_revise;
	cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
	cv::imshow("result", img);
	cv::waitKey();
}