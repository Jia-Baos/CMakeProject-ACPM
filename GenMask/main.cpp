#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

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

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
	
	std::filesystem::path file_path = "E:\\paper1-dataset\\test";
	for (auto iter : std::filesystem::directory_iterator(file_path))
	{
		std::cout << iter.path().string() << std::endl;

		std::string defect_img_path = iter.path().string();
		std::string defect_no_img_path =
			ReplaceAll(defect_img_path, "test", "abnormal-original");
		std::string mask_img_path =
			ReplaceAll(defect_img_path, "test", "mask");
		cv::Mat defect_img = cv::imread(defect_img_path);
		cv::Mat defect_no_img = cv::imread(defect_no_img_path);
		cv::Mat mask = cv::abs(defect_img - defect_no_img);
		cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);

		// 10
		cv::threshold(mask, mask, 50, 255, cv::THRESH_BINARY);
		cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
		cv::morphologyEx(mask, mask, cv::MORPH_OPEN, element);
		
		cv::imwrite(mask_img_path, mask);

		/*cv::namedWindow("mask", cv::WINDOW_NORMAL);
		cv::imshow("mask", mask);
		cv::waitKey();*/
	}
}
