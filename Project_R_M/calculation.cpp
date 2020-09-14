#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <fstream>


using namespace std;


//******************************************************************

#define fx_l 654.53369  //左眼标定焦距
#define fy_l 656.99896

#define cx_l 667.29262  //左眼图像物理中心点坐标
#define cy_l 385.50491

#define cx_r 671.90655  //右眼图像物理中心点坐标
#define cy_r 450.14218

#define fx_r 787.37707  //右眼标定焦距
#define fy_r 783.00536

#define R {0.01297;0.02971;0.00340}  //标定旋转矩阵
#define T {-404.97875;-87.07289;272.21939}  //标定平移矩阵

#define r1 0.01297
#define r4 0.02971
#define r7 0.00340

#define tx -404.97875
#define ty -87.07289
#define tz 272.21939
//******************************************************************



//左相机内参数矩阵
float leftIntrinsic[3][3] = { 680.85971,			 0,		664.11201,
									  0,	681.10996,		399.06642,
									  0,			 0,				1 };
//左相机畸变系数
float leftDistortion[1][5] = { -0.16422, 0.00177, -0.00045, 0.00054, 0 };
//左相机旋转矩阵
float leftRotation[3][3] = { 1,		        0,		        0,
								   0,		        1,		        0,
								   0,		        0,		        1 };
//左相机平移向量
float leftTranslation[1][3] = { 0,0,0 };

//右相机内参数矩阵
float rightIntrinsic[3][3] = { 677.28468,			 0,		673.29250,
										0,	677.72044,		408.16192,
										0,			 0,				1 };
//右相机畸变系数
float rightDistortion[1][5] = { -0.16044, 0.00379, -0.00034, -0.00083, 0 };
//右相机旋转矩阵
float rightRotation[3][3] = { 1,		        0.0005441,		        -0.0040,
							 -0.00057997,		        1,		        -0.0090,
							 0.0040,		        0.0090,		              1 };
//右相机平移向量
float rightTranslation[1][3] = { -119.8714, 0.2209, -0.0451 };





cv::Point3f uv2xyz(cv::Point2f uvLeft, cv::Point2f uvRight)
{
	ofstream outfile("data.txt", ios::app);//得到的坐标数据保存在“data.txt”

	//  [u1]      |X|					  [u2]      |X|
	//Z*|v1| = Ml*|Y|					Z*|v2| = Mr*|Y|
	//  [ 1]      |Z|					  [ 1]      |Z|
	//			  |1|								|1|
	cv::Mat mLeftRotation = cv::Mat(3, 3, CV_32F, leftRotation);
	cv::Mat mLeftTranslation = cv::Mat(3, 1, CV_32F, leftTranslation);
	cv::Mat mLeftRT = cv::Mat(3, 4, CV_32F);//左相机M矩阵
	hconcat(mLeftRotation, mLeftTranslation, mLeftRT);
	cv::Mat mLeftIntrinsic = cv::Mat(3, 3, CV_32F, leftIntrinsic);
	cv::Mat mLeftM = mLeftIntrinsic * mLeftRT;
	//cout<<"左相机M矩阵 = "<<endl<<mLeftM<<endl;

	cv::Mat mRightRotation = cv::Mat(3, 3, CV_32F, rightRotation);
	cv::Mat mRightTranslation = cv::Mat(3, 1, CV_32F, rightTranslation);
	cv::Mat mRightRT = cv::Mat(3, 4, CV_32F);//右相机M矩阵
	hconcat(mRightRotation, mRightTranslation, mRightRT);
	cv::Mat mRightIntrinsic = cv::Mat(3, 3, CV_32F, rightIntrinsic);
	cv::Mat mRightM = mRightIntrinsic * mRightRT;
	//cout<<"右相机M矩阵 = "<<endl<<mRightM<<endl;

	//最小二乘法A矩阵
	cv::Mat A = cv::Mat(4, 3, CV_32F);
	A.at<float>(0, 0) = uvLeft.x * mLeftM.at<float>(2, 0) - mLeftM.at<float>(0, 0);
	A.at<float>(0, 1) = uvLeft.x * mLeftM.at<float>(2, 1) - mLeftM.at<float>(0, 1);
	A.at<float>(0, 2) = uvLeft.x * mLeftM.at<float>(2, 2) - mLeftM.at<float>(0, 2);

	A.at<float>(1, 0) = uvLeft.y * mLeftM.at<float>(2, 0) - mLeftM.at<float>(1, 0);
	A.at<float>(1, 1) = uvLeft.y * mLeftM.at<float>(2, 1) - mLeftM.at<float>(1, 1);
	A.at<float>(1, 2) = uvLeft.y * mLeftM.at<float>(2, 2) - mLeftM.at<float>(1, 2);

	A.at<float>(2, 0) = uvRight.x * mRightM.at<float>(2, 0) - mRightM.at<float>(0, 0);
	A.at<float>(2, 1) = uvRight.x * mRightM.at<float>(2, 1) - mRightM.at<float>(0, 1);
	A.at<float>(2, 2) = uvRight.x * mRightM.at<float>(2, 2) - mRightM.at<float>(0, 2);

	A.at<float>(3, 0) = uvRight.y * mRightM.at<float>(2, 0) - mRightM.at<float>(1, 0);
	A.at<float>(3, 1) = uvRight.y * mRightM.at<float>(2, 1) - mRightM.at<float>(1, 1);
	A.at<float>(3, 2) = uvRight.y * mRightM.at<float>(2, 2) - mRightM.at<float>(1, 2);

	//最小二乘法B矩阵
	cv::Mat B = cv::Mat(4, 1, CV_32F);
	B.at<float>(0, 0) = mLeftM.at<float>(0, 3) - uvLeft.x * mLeftM.at<float>(2, 3);
	B.at<float>(1, 0) = mLeftM.at<float>(1, 3) - uvLeft.y * mLeftM.at<float>(2, 3);
	B.at<float>(2, 0) = mRightM.at<float>(0, 3) - uvRight.x * mRightM.at<float>(2, 3);
	B.at<float>(3, 0) = mRightM.at<float>(1, 3) - uvRight.y * mRightM.at<float>(2, 3);

	cv::Mat XYZ = cv::Mat(3, 1, CV_32F);
	//采用SVD最小二乘法求解XYZ
	solve(A, B, XYZ, cv::DECOMP_SVD);

	//cout<<"空间坐标为 = "<<endl<<XYZ<<endl;

	//世界坐标系中坐标
	cv::Point3f world;
	world.x = XYZ.at<float>(0, 0);
	world.y = XYZ.at<float>(1, 0);
	world.z = XYZ.at<float>(2, 0);
	if (world.z > 0)
	{
		outfile << "X:" << world.x / 1000 << " " << "Y:" << -world.y / 1000 << " " << "Z:" << world.z / 1000 << "\n";
		outfile.close();//关闭文件，保存文件
	}
	
	return world;
}