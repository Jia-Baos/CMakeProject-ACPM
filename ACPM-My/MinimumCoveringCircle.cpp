#include "MinimumCoveringCircle.h"

// ����+�������
MyVector operator+(const MyVector& p1, const MyVector& p2)
{
	return { p1.my_x + p2.my_x, p1.my_y + p2.my_y };
}

// ����-�������
MyVector operator-(const MyVector& p1, const MyVector& p2)
{
	return { p1.my_x - p2.my_x, p1.my_y - p2.my_y };
}

// ���
double operator*(const MyVector& p1, const MyVector& p2)
{
	return { p1.my_x * p2.my_y - p1.my_y * p2.my_x };
}

// ����
MyVector operator*(const MyVector& p1, const double a)
{
	return { p1.my_x * a, p1.my_y * a };
}

// ����
MyVector operator/(const MyVector& p1, const double a)
{
	return { p1.my_x / a, p1.my_y / a };
}

// �жϰ뾶��С
int Judge(const double a, const double b)
{
	if (std::abs(a - b) < eps)
	{
		return 0;
	}
	if (a < b)
	{
		return -1;
	}
	return 1;
}

// ������ת��a��ʾ˳ʱ����ת�Ļ���ֵ
MyVector Rotate(const MyVector& p1, const double a)
{
	return { p1.my_x * cos(a) + p1.my_y * sin(a), -p1.my_x * sin(a) + p1.my_y * cos(a) };
}

// ������֮�����
double Distance(const MyVector& p1, const MyVector& p2)
{
	const double dx = p1.my_x - p2.my_x;
	const double dy = p1.my_y - p2.my_y;
	return std::sqrt(dx * dx + dy * dy);
}

// ����ֱ�ߵĽ���
MyVector Intersection(const MyVector& p1, const MyVector& v1, const MyVector& p2, const MyVector& v2)
{
	MyVector result;
	if (v1 * v2 != 0)
	{
		const MyVector temp = p1 - p2;
		const double ratio = (v2 * temp) / (v1 * v2);
		result = p1 + v1 * ratio;
	}
	return result;
}

// ������֮�����ߵ��д���
std::pair<MyVector, MyVector> MidPerpendicular(const MyVector& p1, const MyVector& p2)
{
	const MyVector mid = (p1 + p2) / 2;
	// ��ֱʱ��������ת���ή�;���
	// const MyVector rotate = Rotate(p1 - p2, PI / 2);
	const MyVector rotate = { -p1.my_y + p2.my_y, p1.my_x - p2.my_x };
	return { mid, rotate };
}

// ��֪���������Բ
Circle MiniCircle(MyVector p1, MyVector p2, MyVector p3)
{
	auto m1 = MidPerpendicular(p1, p2);
	auto m2 = MidPerpendicular(p1, p3);
	MyVector center_point = Intersection(m1.my_x, m1.my_y, m2.my_x, m2.my_y);
	double radius = Distance(center_point, p1);
	return { center_point, radius };
}

void FuncTest()
{
	MyVector p1(2.0, 2.0);
	MyVector v1(1.0, 1.0);
	MyVector p2(2.0, -2.0);
	MyVector v2(1.0, -1.0);
	MyVector p3(0.0, 0.0);

	double value;
	Circle circle;
	std::pair<double, double> result;
	std::pair<MyVector, MyVector> combination;

	// �ӷ�����
	result = p1 + p2;
	std::cout << "(2.0,2.0) + (2.0,-2.0): " << result.my_x << " " << result.my_y << std::endl;

	// ��������
	result = p1 - p2;
	std::cout << "(2.0,2.0) - (2.0,-2.0): " << result.my_x << " " << result.my_y << std::endl;

	// ��˲���
	value = p1 * p2;
	std::cout << "(2.0,2.0) * (2.0,-2.0): " << value << std::endl;

	// ���˲���
	result = p1 * 1.0;
	std::cout << "(2.0,2.0) * 1.0: " << result.my_x << " " << result.my_y << std::endl;

	// ��������
	result = p1 / 1.0;
	std::cout << "(2.0,2.0) / 1.0: " << result.my_x << " " << result.my_y << std::endl;

	// Judge��������
	value = Judge(2.0, 3.0);
	std::cout << "2.0, 3.0 (-1 means not int circle): " << value << std::endl;

	// ������ת����
	result = Rotate(v1, PI / 2);
	std::cout << "(1.0,1.0) rotate PI/2: " << result.my_x << " " << result.my_y << std::endl;

	// ����֮��������
	value = Distance(p1, p2);
	std::cout << "(2.0,2.0) (2.0,-2.0) Distance: " << value << std::endl;

	// ��ֱ��֮�佻�����
	result = Intersection(p1, v1, p2, v2);
	std::cout << "p1(2.0,2.0), v1(1.0,1.0), p2(2.0,-2.0), v2(1.0,-1.0) Cross Point: " << result.my_x << " " << result.my_y << std::endl;

	// ����֮���д��߲���
	combination = MidPerpendicular(p1, p2);
	std::cout << "p1(2.0,2.0), p2(2.0,-2.0) Mid Perpendicular: "
		<< "Point: " << combination.my_x.my_x << " " << combination.my_x.my_y
		<< " Vec: " << combination.my_y.my_x << " " << combination.my_y.my_y << std::endl;

	// �������Բ����
	circle = MiniCircle(p1, p2, p3);
	std::cout << "p1(2.0,2.0), p2(2.0,-2.0), p3(0.0,0.0) Mini Circle: "
		<< "Point: " << circle.center.my_x << " " << circle.center.my_y
		<< " Radius: " << circle.radius << std::endl;
}

Circle MinimumCoveringCircle(const std::vector<MyVector>& points, const int npoints)
{
	// ��ʼ��Բ��Բ��Ϊpoints[0]���뾶Ϊ0
	Circle circle(points[0], 0.0);
	for (int i = 1; i < npoints; i++)
	{
		if (Judge(circle.radius, Distance(circle.center, points[i])) == -1)
		{
			// ���points[i]��Բ���ⲿ������ָ��Բ�ĵ�λ��
			circle = { points[i], 0 };
			for (int j = 0; j < i; j++)
			{
				if (Judge(circle.radius, Distance(circle.center, points[j])) == -1)
				{
					// ���points[j]��Բ���ⲿ������points[i]��points[j]����һ����СԲ
					circle = { (points[i] + points[j]) / 2, Distance(points[i], points[j]) / 2 };
					for (int k = 0; k < j; k++)
					{
						// ���points[k]��Բ���ⲿ������points[i]��points[j]��points[k]����һ����СԲ
						if (Judge(circle.radius, Distance(circle.center, points[k])) == -1)
						{
							circle = MiniCircle(points[i], points[j], points[k]);
						}
					}
				}
			}
		}
	}
	return circle;
}