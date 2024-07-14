#pragma once
#ifndef __MINIMUMCOVERINGCIRCLE_H__
#define __MINIMUMCOVERINGCIRCLE_H__

#include <iostream>
#include <vector>
#include <cmath>
#include <utility>

const int N = 100010;
const double eps = 1e-6;
const double PI = 3.1415926;

#define my_x first
#define my_y second
typedef std::pair<double, double> MyVector;

struct Circle
{
	MyVector center;
	double radius;
	Circle() : center(std::pair<double, double>(0.0, 0.0)), radius(0) {}
	Circle(const std::pair<double, double>& center, const double radius) : center(center), radius(radius) {}
};

void FuncTest();
Circle MinimumCoveringCircle(const std::vector<MyVector>& points, const int npoints);

#endif // !__MINIMUMCOVERINGCIRCLE_H__
