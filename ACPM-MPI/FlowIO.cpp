#pragma once
// flow_io.cpp
//
// read and write our simple .flo flow file format

// ".flo" file format used for optical flow evaluation
//
// Stores 2-band float image for horizontal (u) and vertical (v) flow components.
// Floats are stored in little-endian order.
// A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
//
//  bytes  contents
//
//  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
//          (just a sanity check that floats are represented correctly)
//  4-7     width as an integer
//  8-11    height as an integer
//  12-end  data (width*height*2*4 bytes total)
//          the float values for u and v, interleaved, in row order, i.e.,
//          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
//

// first four bytes, should be the same in little endian
#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file

#include <iostream>
#include <cmath>
#include <exception>
#include <opencv2/opencv.hpp>

#include "flowIO.h"

// return whether flow vector is unknown
bool unknown_flow(float u, float v)
{
	return (fabs(u) > UNKNOWN_FLOW_THRESH)
		|| (fabs(v) > UNKNOWN_FLOW_THRESH)
		|| isnan(u) || isnan(v);
}

bool unknown_flow(float* f)
{
	return unknown_flow(f[0], f[1]);
}

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

// write a 2-band image into flow file
void WriteFlowFile(cv::Mat& img, const char* filename)
{
	if (filename == NULL)
		throw "WriteFlowFile: empty filename";

	const char* dot = strrchr(filename, '.');
	if (dot == NULL)
		throw "WriteFlowFile: extension required in filename";

	if (strcmp(dot, ".flo") != 0)
		throw "WriteFlowFile: filename '%s' should have extension";

	int width = img.cols;
	int height = img.rows;
	int nBands = img.channels();

	if (nBands != 2)
		throw "WriteFlowFile: image must have 2 bands";

	FILE* stream;
	fopen_s(&stream, filename, "wb");
	if (stream == 0)
		throw "WriteFlowFile: could not open";

	// write the header
	fprintf(stream, TAG_STRING);
	if ((int)fwrite(&width, sizeof(int), 1, stream) != 1 ||
		(int)fwrite(&height, sizeof(int), 1, stream) != 1)
		throw "WriteFlowFile: problem writing header";

	// write the data
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float* img_pointer = img.ptr<float>(i, j);
			fwrite(&img_pointer[0], sizeof(float), 1, stream);
			fwrite(&img_pointer[1], sizeof(float), 1, stream);
		}
	}

	fclose(stream);
}

int test() {
	try
	{
		cv::Mat img;
		char* filename = (char*)"D:/Code-VS/Project-ACPM/Project-ACPM/frame_0029.flo";
		char* filename_temp = (char*)"D:/Code-VS/Project-ACPM/Project-ACPM/frame_0029_copy.flo";

		//WriteFlowFile(img, filename);
		ReadFlowFile(img, filename);
		WriteFlowFile(img, filename_temp);
	}
	catch (const char* exception)
	{
		std::cout << exception << std::endl;
		exit(1);
	}

	return 0;
}
