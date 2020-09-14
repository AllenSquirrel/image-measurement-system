#pragma once
#include "time_stamp.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>



using namespace std;


vector<cv::Point> match_feature1();  //ÂÖÀªĞÎ×´ÌØÕ÷Æ¥Åä1



vector<cv::Point> match_feature2(); //ÂÖÀªĞÎ×´ÌØÕ÷Æ¥Åä2



vector<cv::Point> match_feature3();  //ÂÖÀªĞÎ×´ÌØÕ÷Æ¥Åä3


vector<cv::Point> match_feature4();   //ÂÖÀªĞÎ×´ÌØÕ÷Æ¥Åä4
