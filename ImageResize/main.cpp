#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

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
	std::cout << 123 << std::endl;

	const std::string fixed_image_path = "E:\\paper2-dataset\\1template.jpg";
	const std::string moved_image_path = "E:\\paper2-dataset\\1tested.jpg";

	cv::Mat fixed_image = cv::imread(fixed_image_path);
	cv::Mat moved_image = cv::imread(moved_image_path);

	std::string fixed_image_resized_path = 
		ReplaceAll(fixed_image_path, "jpg", "png");
	std::string moved_image_resized_path =
		ReplaceAll(moved_image_path, "jpg", "png");

	cv::Mat fixed_image_resized;
	cv::Mat moved_image_resized;
	ImagePadded(fixed_image, moved_image, fixed_image_resized, moved_image_resized,
		4, 4);

	cv::imwrite(fixed_image_resized_path, fixed_image_resized);
	cv::imwrite(moved_image_resized_path, moved_image_resized);
}
