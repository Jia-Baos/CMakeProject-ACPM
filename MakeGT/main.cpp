#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

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

int main(int argc, char* argv[]) {
	std::string tmpl_path = "D:\\Code-VS\\picture\\2023-11-10\\13-OK.png";
	std::string test_path = "D:\\Code-VS\\picture\\2023-11-10\\13-NG.png";
	std::string optflow_path = "D:\\Code-VS\\picture\\2023-11-10\\13-NG.flo";
	std::string res_path = "D:\\Code-VS\\picture\\2023-11-10\\13-NG-res.png";

	cv::Mat tmpl = cv::imread(tmpl_path);
	cv::Mat test = cv::imread(test_path);

	cv::Mat res = cv::abs(tmpl - test);

	cv::cvtColor(res, res, cv::COLOR_BGR2GRAY);
	//res.setTo(255, res > 100);

	cv::Mat optflow{}, res_warpped{};
	ReadFlowFile(optflow, optflow_path.c_str());
	MovePixels(res, res_warpped, optflow, cv::INTER_CUBIC);

	cv::imwrite(res_path, res_warpped);
}
