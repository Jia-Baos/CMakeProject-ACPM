#include "MyUtils.h"

std::string& ReplaceAll(std::string& src, const std::string& old_value, const std::string& new_value)
{
	// 每次重新定位起始位置，防止上轮替换后的字符串形成新的old_value
	for (std::string::size_type pos(0); pos != std::string::npos; pos += new_value.length())
	{
		if ((pos = src.find(old_value, pos)) != std::string::npos)
		{
			src.replace(pos, old_value.length(), new_value);
		}
		else break;
	}
	return src;
}

std::string CurrentDrectory()
{
	char* buffer = nullptr;
	buffer = _getcwd(NULL, 0);
	std::string current_working_directory;
	if (buffer == nullptr)
	{
		std::cerr << "Error message: _getcwd error" << std::endl;
	}
	else
	{
		current_working_directory.assign(buffer, strlen(buffer));
		current_working_directory = ReplaceAll(current_working_directory, "\\", "/");
	}

	return current_working_directory;
}

CTime* CTime::my_time = new CTime();

CTime* CTime::GetCTime()
{
	return my_time;
}

void CTime::DeleteCTime()
{
	if (my_time != nullptr)
	{
		delete my_time;
		my_time = nullptr;
	}
}

void CTime::in_clock()
{
	my_time->in_time = std::chrono::steady_clock::now();
	my_time->inter_time = std::chrono::steady_clock::now();
}

void CTime::inter_clock(const std::string& message)
{
	my_time->out_time = std::chrono::steady_clock::now();
	double spend_time = std::chrono::duration<double>(my_time->out_time - my_time->inter_time).count();
	my_time->inter_time = my_time->out_time;
	std::cout << message << spend_time << std::endl;
}

void CTime::out_clock(const std::string& message)
{
	my_time->out_time = std::chrono::steady_clock::now();
	double spend_time = std::chrono::duration<double>(my_time->out_time - my_time->in_time).count();
	std::cout << message << spend_time << std::endl;
}
