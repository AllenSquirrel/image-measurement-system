#pragma once
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <string.h>
#include <windows.h>
#include <thread>
#include <future>
#include<ctime>


using namespace std;





cv::Point3f uv2xyz(cv::Point2f uvLeft, cv::Point2f uvRight);//��С���˷������ά����
