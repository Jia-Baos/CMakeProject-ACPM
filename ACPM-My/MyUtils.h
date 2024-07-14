#pragma once
#ifndef __MYUTILS_H__
#define __MYUTILS_H__

#include <iostream>
#include <chrono>
#include <string>

#if defined(_MSC_VER)
#include <direct.h>
#define GetCurrentDir _getcwd
#elif defined(__unix__)
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

class CTime
{
private:
	static CTime* my_time;
	std::chrono::steady_clock::time_point in_time;
	std::chrono::steady_clock::time_point inter_time;
	std::chrono::steady_clock::time_point out_time;
public:
	CTime() {}
	~CTime() {}
	CTime(const CTime& single_instance) {}
	const CTime& operator=(const CTime& single_instance) {}
	static CTime* GetCTime();
	static void DeleteCTime();
	static void in_clock();	// in time
	static void inter_clock(const std::string& message);	// inter time
	static void out_clock(const std::string& message);	// out time
};

std::string& ReplaceAll1(std::string& src,
	const std::string& old_value,
	const std::string& new_value);

/// @brief
/// @return
std::string CurrentDrectory();

#endif // !__MYUTILS_H__
