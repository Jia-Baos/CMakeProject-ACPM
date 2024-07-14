#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

//����ͼ�����ֱ����ͶӰ��������һ��ͼ��
cv::Mat VerProj(const cv::Mat& src)
{
	const int height = src.rows;
	const int width = src.cols;

	// ͳ����Ч���ص������Ŀ�����ڵ���
	int max_col = 0, max_num = INT_MIN;
	// ͳ����Ч���ص���С��Ŀ�����ڵ���
	int min_col = 0, min_num = INT_MAX;

	// �洢ÿһ�е���Ч��������
	std::vector<int> projArray(width, 0);

	// �����������أ�ͳ��ÿһ����Ч������
	for (size_t j = 0; j < width; ++j) {
		int sum{};
		for (size_t i = 0; i < height; ++i) {
			if (src.ptr<uchar>(i)[j] != 0) { ++sum; }
		}

		projArray[j] = sum;

		if (sum > max_num) {
			max_num = sum;
			max_col = j;
		}
		if (sum < min_num) {
			min_num = sum;
			min_col = j;
		}
	}

	// ���������ƴ�ֱͶӰͼ��
	cv::Mat projImg(height, width, CV_8UC1, cv::Scalar(255));

	for (size_t j = 0; j < width; ++j) {
		cv::line(projImg, cv::Point(j, height - projArray[j]), cv::Point(j, height - 1), cv::Scalar::all(0));
	}

	return  projImg;
}

//����ͼ���ˮƽ����ͶӰ��������һ��ͼ��
cv::Mat HorProj(const cv::Mat& src)
{
	const int height = src.rows;
	const int width = src.cols;

	// ͳ����Ч���ص������Ŀ�����ڵ���
	int max_row = 0, max_num = INT_MIN;
	// ͳ����Ч���ص���С��Ŀ�����ڵ���
	int min_row = 0, min_num = INT_MAX;

	// �洢ÿһ�е���Ч��������
	std::vector<int> projArray(height, 0);

	// �����������أ�ͳ��ÿһ����Ч������
	for (size_t i = 0; i < height; ++i) {
		int sum{};
		for (size_t j = 0; j < width; ++j) {
			if (src.ptr<uchar>(i)[j] != 0) { ++sum; }
		}

		projArray[i] = sum;

		if (sum > max_num) {
			max_num = sum;
			max_row = i;
		}
		if (sum < min_num) {
			min_num = sum;
			min_row = i;
		}
	}

	//���������ƴ�ֱͶӰͼ��
	cv::Mat projImg(height, width, CV_8UC1, cv::Scalar(255));

	for (int row = 0; row < height; ++row){
		cv::line(projImg, cv::Point(0, row), cv::Point(projArray[row], row), cv::Scalar::all(0));
	}
	return  projImg;
}

int main(int argc, char* argv)
{
	std::string image_path = "D:\\Code-VS\\picture\\ImageRegistration\\kpi_winter.png";

	cv::Mat img = cv::imread(image_path, 0);

	cv::threshold(img, img, 100, 255, cv::THRESH_BINARY);
	cv::Mat dst = HorProj(img);

	cv::namedWindow("img", cv::WINDOW_NORMAL);
	cv::imshow("img", dst);

	cv::waitKey();
	return 0;
}
