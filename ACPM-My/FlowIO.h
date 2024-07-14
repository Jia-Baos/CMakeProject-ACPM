#pragma
#ifndef __FLOWIO_H__
#define __FLOWIO_H__

// flowIO.h

// the "official" threshold - if the absolute value of either
// flow component is greater, it's considered unknown
#define UNKNOWN_FLOW_THRESH 1e9

// value to use to represent unknown flow
#define UNKNOWN_FLOW 1e10

// return whether flow vector is unknown
bool unknown_flow(float u, float v);
bool unknown_flow(float* f);

#include <iostream>
#include <cmath>
#include <exception>
#include <opencv2/opencv.hpp>

// read a flow file into 2-band image
void ReadFlowFile(cv::Mat& img, const char* filename);

// write a 2-band image into flow file
void WriteFlowFile(cv::Mat& img, const char* filename);

// test file
int test();

#endif // !__FLOWIO_H__