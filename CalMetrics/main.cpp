#include <iostream>
#include <string>
#include <tuple>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

// first four bytes, should be the same in little endian
#define TAG_FLOAT 202021.25 // check for this when READING the file

// read a flow file into 2-band image
void ReadFlowFile(cv::Mat& img, const char* filename)
{
	if (filename == NULL)
		throw "ReadFlowFile: empty filename";

	const char* dot = strrchr(filename, '.');
	if (strcmp(dot, ".flo") != 0)
		throw "ReadFlowFile: extension .flo expected";

	FILE* stream;
	fopen_s(&stream, filename, "rb");
	if (stream == 0)
		throw "ReadFlowFile: could not open";

	int width, height;
	float tag;

	if ((int)fread(&tag, sizeof(float), 1, stream) != 1 ||
		(int)fread(&width, sizeof(int), 1, stream) != 1 ||
		(int)fread(&height, sizeof(int), 1, stream) != 1)
		throw "ReadFlowFile: problem reading file";

	if (tag != TAG_FLOAT) // simple test for correct endian-ness
		throw "ReadFlowFile: wrong tag (possibly due to big-endian machine?)";

	// another sanity check to see that integers were read correctly (99999 should do the trick...)
	if (width < 1 || width > 99999)
		throw "ReadFlowFile: illegal width";

	if (height < 1 || height > 99999)
		throw "ReadFlowFile: illegal height";

	img = cv::Mat(cv::Size(width, height), CV_32FC2);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float* img_pointer = img.ptr<float>(i, j);
			fread(&img_pointer[0], sizeof(float), 1, stream);
			fread(&img_pointer[1], sizeof(float), 1, stream);
		}
	}

	if (fgetc(stream) != EOF)
		throw "ReadFlowFile: file is too long";

	fclose(stream);
}

// 光流场重映射
void MovePixels(const cv::Mat& src, cv::Mat& dst, const cv::Mat& flow, const int interpolation)
{
	std::vector<cv::Mat> flow_spilit;
	cv::split(flow, flow_spilit);

	// 像素重采样实现
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

std::tuple<cv::Mat, cv::Mat> ThresholdDisplay(const cv::Mat src, const int thresh = 180)
{
	cv::Mat tmp = src.clone();
	cv::threshold(tmp, tmp, thresh, 255, cv::THRESH_BINARY);

	cv::RNG rng(10086);
	cv::Mat out;
	int number = cv::connectedComponents(tmp, out, 8, CV_16U);
	std::vector<cv::Vec3b> colors;
	for (int i = 0; i < number; i++)
	{
		// 使用均匀分布的随机数确定颜色
		cv::Vec3b vec3 = cv::Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		colors.push_back(vec3);
	}

	// 以不同的颜色标记处不同的连通域
	cv::Mat dst = src.clone();
	cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int label = out.at<uint16_t>(i, j);
			if (0 == label)
			{
				continue;
			}
			dst.at<cv::Vec3b>(i, j) = colors[label];
		}
	}
	return std::tuple(std::make_tuple(tmp, dst));
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
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	/*for (size_t bin_threshold = 0; bin_threshold <= 255; bin_threshold+=10) {
		std::cout << "----------------"
			<< bin_threshold << "----------------" << std::endl;*/

		float mTP = 0.0f;
		float mFP = 0.0f;
		float mTN = 0.0f;
		float mFN = 0.0f;
		float mIOU = 0.0f;
		int index_sum = 0;
		// binary_threshold from 60 to 240

		std::string metric_path = "E:\\paper4-dataset\\metric.txt";
		std::string mask_dir = "E:\\paper4-dataset\\mask";
		std::string res_dir = "E:\\paper4-dataset\\res-ricflow";
		std::string opflow_dir = "E:\\paper4-dataset\\optflow-ricflow";
		std::string fixed_image_path = "E:\\paper4-dataset\\template.png";

		std::filesystem::path file_path = "E:\\paper4-dataset\\test-png";
		for (auto iter : std::filesystem::directory_iterator(file_path)) {
			std::string moved_image_path = iter.path().string();

			std::string mask_image_path = mask_dir + "\\"
				+ iter.path().filename().string();

			std::string res_image_path = res_dir + "\\"
				+ iter.path().filename().string();

			std::string optflow_path = opflow_dir + "\\"
				+ iter.path().filename().string();
			optflow_path = ReplaceAll(optflow_path, "png", "flo");

			std::cout << "moved_image: " << moved_image_path << std::endl;

			// 读入图像
			cv::Mat fixed_img = cv::imread(fixed_image_path);
			cv::Mat moved_img = cv::imread(moved_image_path);
			//cv::Mat mask_img = cv::imread(mask_image_path);

			cv::cvtColor(fixed_img, fixed_img, cv::COLOR_BGR2GRAY);
			cv::cvtColor(moved_img, moved_img, cv::COLOR_BGR2GRAY);
			//cv::cvtColor(mask_img, mask_img, cv::COLOR_BGR2GRAY);

			cv::Mat optflow;
			ReadFlowFile(optflow, optflow_path.c_str());

			/*std::cout << "fixed_img size: " << fixed_img.size()
				<< ", moved_img size: " << moved_img.size()
				<< ", optflow size: " << optflow.size() << std::endl;*/

			cv::Mat moved_img_warpped;
			MovePixels(moved_img, moved_img_warpped, optflow, cv::INTER_CUBIC);
			cv::Mat res = cv::abs(fixed_img - moved_img_warpped);
			cv::imwrite(res_image_path, res);

			//std::tuple<cv::Mat, cv::Mat> res_collect = ThresholdDisplay(res, 106);
			//

			//// mask映射过程插值使得部分像素值非0或者255
			//MovePixels(mask_img, mask_img, optflow, cv::INTER_CUBIC);
			//cv::threshold(mask_img, mask_img, 200, 255, cv::THRESH_BINARY);

			//cv::Mat predicted = std::move(std::get<0>(res_collect));
			//cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
			//cv::morphologyEx(predicted, predicted, cv::MORPH_OPEN, element,
			//	cv::Point(-1, -1), 1);

			//long long int TP = 0;
			//long long int FP = 0;
			//long long int TN = 0;
			//long long int FN = 0;

			//for (size_t i = 0; i < mask_img.rows; ++i) {
			//	for (size_t j = 0; j < mask_img.cols; ++j) {
			//		if (mask_img.ptr<uchar>(i)[j] == 255 &&
			//			predicted.ptr<uchar>(i)[j] == 255) {
			//			TP += 1;
			//		}
			//		if (mask_img.ptr<uchar>(i)[j] == 0 &&
			//			predicted.ptr<uchar>(i)[j] == 255)
			//		{
			//			FP += 1;
			//		}
			//		if (mask_img.ptr<uchar>(i)[j] == 0 &&
			//			predicted.ptr<uchar>(i)[j] == 0)
			//		{
			//			TN += 1;
			//		}
			//		if (mask_img.ptr<uchar>(i)[j] == 255 &&
			//			predicted.ptr<uchar>(i)[j] == 0)
			//		{
			//			FN += 1;
			//		}
			//	}
			//}

			//mTP += TP;
			//mFP += FP;
			//mTN += TN;
			//mFN += FN;
			//++index_sum;
			//mIOU += (TP * 1.0) / (TP + FP + FN);
			//std::cout << "IOU: " << (TP * 1.0) / (TP + FP + FN) << std::endl;
		}

		//mTP /= index_sum;
		//mFP /= index_sum;
		//mTN /= index_sum;
		//mFN /= index_sum;
		//mIOU /= index_sum;
		////mTP *= 1000;
		////mFN *= 1000;
		//float precison = mTP / (mTP + mFP);
		//float recall = mTP / (mTP + mFN);
		//float f1_score = (2 * precison * recall) / (precison + recall);
		//float new_metric = mFP / std::max(mFP + mTP, (mTP + mFN) * 10);

		//std::cout << "precison: " << std::fixed <<
		//	std::setprecision(6) << precison << std::endl;
		//std::cout << "recall: " << std::fixed <<
		//	std::setprecision(6) << recall << std::endl;
		//std::cout << "F1-score: " << std::fixed <<
		//	std::setprecision(6) << f1_score << std::endl;
		//std::cout << "mIOU: " << std::fixed <<
		//	std::setprecision(6) << mIOU << std::endl;

		/*float TPR = mTP / (mTP + mFN);
		float FPR = mFP / (mFP + mTN);*/

	/*	std::ofstream outfile(metric_path, std::ios::out | std::ios::app);
		outfile << new_metric << " "
			<< recall << std::endl;
		outfile.close();
	}*/

	/*std::string fixed_image_path = "D:\\Code-VS\\picture\\defect1\\template2-resized.png";
	std::string moved_image_path = "D:\\Code-VS\\picture\\defect1\\17.png";
	std::string optflow_path = "D:\\Code-VS\\picture\\defect1\\ACPMFlow\\17.flo";

	cv::Mat fixed_img = cv::imread(fixed_image_path);
	cv::Mat moved_img = cv::imread(moved_image_path);

	cv::cvtColor(fixed_img, fixed_img, cv::COLOR_BGR2GRAY);
	cv::cvtColor(moved_img, moved_img, cv::COLOR_BGR2GRAY);

	cv::Mat optflow;
	ReadFlowFile(optflow, optflow_path.c_str());

	std::cout << "fixed_img size: " << fixed_img.size() << std::endl;
	std::cout << "moved_img size: " << moved_img.size() << std::endl;
	std::cout << "optflow size: " << optflow.size() << std::endl;

	cv::Mat moved_img_warpped;
	MovePixels(moved_img, moved_img_warpped, optflow, cv::INTER_CUBIC);
	cv::Mat res = cv::abs(fixed_img - moved_img_warpped);

	std::tuple<cv::Mat,cv::Mat> res_collect = ThresholdDisplay(res);
	cv::imwrite("D:\\Code-VS\\picture\\defect1\\ACPMFlow\\17-res-img.png",
		std::get<0>(res_collect));
	cv::imwrite("D:\\Code-VS\\picture\\defect1\\ACPMFlow\\17-res-dis.png",
		std::get<1>(res_collect));
	cv::namedWindow("Display", cv::WINDOW_NORMAL);
	cv::imshow("Display", std::get<1>(res_collect));*/
}
