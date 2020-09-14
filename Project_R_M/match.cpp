#include "time_stamp.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>



using namespace std;


vector<cv::Point> match_feature1()    //轮廓形状特征匹配1
{
	int max_contours = 1;
	int num;
	cv::Mat img_gray;
	cv::Mat mid_filer2;

	cv::Mat img = cv::imread("F:\\visual c++&&opencv\\source\\moudle\\1.png", CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//确保黑中找白
	adaptiveThreshold(img_gray, img_gray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 25, 5);//自适应阈值分割

	medianBlur(img_gray, mid_filer2, 3);//中值滤波
	dilate(img_gray, img_gray, cv::Mat()); erode(img_gray, img_gray, cv::Mat());//膨胀和腐蚀处理，有效消除高亮噪声
	//medianBlur(img_gray, mid_filer2, 3);//中值滤波
	bitwise_not(img_gray, img_gray);

	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<cv::Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, cv::Scalar(0, 255, 0), 2, 8);

	cv::namedWindow("目标模板1", cv::WINDOW_NORMAL);
	imshow("目标模板1", draw_img);

	//             max_conours
	for (int i = 0; i < size(contours1); i++)
	{
		if (size(contours1[i]) > max_contours)
		{
			max_contours = size(contours1[i]);
			num = i;
		}
	}
	return contours1[num];
}


vector<cv::Point> match_feature2()    //轮廓形状特征匹配2
{
	int max_contours = 1;
	int num;
	cv::Mat img_gray;
	cv::Mat mid_filer2;

	cv::Mat img = cv::imread("F:\\visual c++&&opencv\\source\\moudle\\2.png", CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//确保黑中找白
	adaptiveThreshold(img_gray, img_gray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 25, 5);//自适应阈值分割

	medianBlur(img_gray, mid_filer2, 3);//中值滤波
	dilate(img_gray, img_gray, cv::Mat()); erode(img_gray, img_gray, cv::Mat());//膨胀和腐蚀处理，有效消除高亮噪声
	//medianBlur(img_gray, mid_filer2, 3);//中值滤波
	bitwise_not(img_gray, img_gray);

	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<cv::Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, cv::Scalar(0, 255, 0), 2, 8);

	cv::namedWindow("目标模板2", cv::WINDOW_NORMAL);
	imshow("目标模板2", draw_img);

	//             max_conours
	for (int i = 0; i < size(contours1); i++)
	{
		if (size(contours1[i]) > max_contours)
		{
			max_contours = size(contours1[i]);
			num = i;
		}
	}
	return contours1[num];
}


vector<cv::Point> match_feature3()    //轮廓形状特征匹配3
{
	int max_contours = 1;
	int num;
	cv::Mat img_gray;
	cv::Mat mid_filer2;

	cv::Mat img = cv::imread("F:\\visual c++&&opencv\\source\\moudle\\3.png", CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//确保黑中找白
	adaptiveThreshold(img_gray, img_gray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 25, 5);//自适应阈值分割

	medianBlur(img_gray, mid_filer2, 3);//中值滤波
	dilate(img_gray, img_gray, cv::Mat()); erode(img_gray, img_gray, cv::Mat());//膨胀和腐蚀处理，有效消除高亮噪声
	//medianBlur(img_gray, mid_filer2, 3);//中值滤波
	bitwise_not(img_gray, img_gray);

	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<cv::Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, cv::Scalar(0, 255, 0), 2, 8);

	cv::namedWindow("目标模板3", cv::WINDOW_NORMAL);
	imshow("目标模板3", draw_img);

	//             max_conours
	for (int i = 0; i < size(contours1); i++)
	{
		if (size(contours1[i]) > max_contours)
		{
			max_contours = size(contours1[i]);
			num = i;
		}
	}
	return contours1[num];
}


vector<cv::Point> match_feature4()    //轮廓形状特征匹配4
{
	int max_contours = 1;
	int num;
	cv::Mat img_gray;
	cv::Mat mid_filer2;

	cv::Mat img = cv::imread("F:\\visual c++&&opencv\\source\\moudle\\4.png", CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//确保黑中找白
	adaptiveThreshold(img_gray, img_gray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 25, 5);//自适应阈值分割

	medianBlur(img_gray, mid_filer2, 3);//中值滤波
	dilate(img_gray, img_gray, cv::Mat()); erode(img_gray, img_gray, cv::Mat());//膨胀和腐蚀处理，有效消除高亮噪声
	//medianBlur(img_gray, mid_filer2, 3);//中值滤波
	bitwise_not(img_gray, img_gray);

	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<cv::Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, cv::Scalar(0, 255, 0), 2, 8);

	cv::namedWindow("目标模板4", cv::WINDOW_NORMAL);
	imshow("目标模板4", draw_img);

	//             max_conours
	for (int i = 0; i < size(contours1); i++)
	{
		if (size(contours1[i]) > max_contours)
		{
			max_contours = size(contours1[i]);
			num = i;
		}
	}
	return contours1[num];
}