#pragma once
#include <iostream>
#include <time.h>
#include "time_stamp.h"
using namespace std;



int times_stamp()//时间戳函数,获取当前系统时间
{
	time_t rawtime;
	struct tm timeinfo;
	time(&rawtime);

	localtime_s(&timeinfo, &rawtime);

	int Year = timeinfo.tm_year + 1900;
	int Mon = timeinfo.tm_mon + 1;
	int Day = timeinfo.tm_mday;
	int Hour = timeinfo.tm_hour;
	int Min = timeinfo.tm_min;
	int Second = timeinfo.tm_sec;
	cout << "=== == == == == == == == == == == == == == == == == == == == ==" << endl;
	cout << Year << ":" << Mon << ":" << Day << "-" << Hour << ":" << Min << ":" << Second << endl;
	return 0;
}
