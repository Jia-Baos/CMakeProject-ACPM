#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
	std::string img_path = "D:\\Code-VS\\CMakeProject-Test\\LightAdjust\\123.png";

	cv::Mat img = cv::imread(img_path, 0);

	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

	float thresh = minVal + 0.8 * (maxVal - minVal);
	img.setTo(0, img < thresh);

	cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
	cv::imshow("img", img);
	cv::waitKey();

	return 0;
}
