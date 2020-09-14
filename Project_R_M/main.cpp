#include "match.h"
#include "time_stamp.h"
#include "calculation.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <string.h>
#include <windows.h>
#include <thread>
#include <future>
#include<ctime>

using namespace cv;
using namespace std;


#define threshold_diff1 25 //设置简单帧差法阈值
#define threshold_diff2 25 //设置简单帧差法阈值
//************************************************************************************************************************************

int main(int argc, unsigned char* argv[])
{

	//开4个线程，用于模板匹配匹配处理，节省处理时间

	std::packaged_task<vector<Point>()>mypt1(match_feature1);//函数mythread通过packaged_task包装
	std::thread feature_obj1(std::ref(mypt1));
	std::packaged_task<vector<Point>()>mypt2(match_feature2);
	std::thread feature_obj2(std::ref(mypt2));
	std::packaged_task<vector<Point>()>mypt3(match_feature3);
	std::thread feature_obj3(std::ref(mypt3));
	std::packaged_task<vector<Point>()>mypt4(match_feature4);
	std::thread feature_obj4(std::ref(mypt4));

	//match_feature();
	clock_t startTime, endTime;

	double x_l, y_l;//图像坐标系下目标二维平面坐标,(左上角为坐标原点)
	double x_r, y_r;

	double x_l_, y_l_;//图像坐标系下目标二维平面坐标,(中心点为坐标原点)
	double x_r_, y_r_;//图像坐标系下目标二维平面坐标,(中心点为坐标原点)


	double X, Y, Z;//左眼相机坐标系下目标三维空间坐标


	Mat frame_l;
	Mat frame_r;
	Mat img_src1_l, img_src2_l, img_src3_l;//左眼3帧法需要3帧图片
	Mat img_src1_r, img_src2_r, img_src3_r;//右眼3帧法需要3帧图片

	Mat img_dst_l, gray1_l, gray2_l, gray3_l;
	Mat img_dst_r, gray1_r, gray2_r, gray3_r;

	Mat gray_diff1_l, gray_diff2_l;//存储2次相减的图片
	Mat gray_diff1_r, gray_diff2_r;//存储2次相减的图片

	Mat gray_diff11_l, gray_diff12_l;
	Mat gray_diff11_r, gray_diff12_r;

	Mat gray_diff21_l, gray_diff22_l;
	Mat gray_diff21_r, gray_diff22_r;

	Mat gray_l, gray_r;//用来显示前景的
	Mat mid_filer_l;   //中值滤波法后的照片
	Mat mid_filer_r;   //中值滤波法后的照片
	bool pause = false;


	VideoCapture vido_file_l("F:\\visual c++&&opencv\\source\\text_l.avi");//在这里改相应的文件名
	VideoCapture vido_file_r("F:\\visual c++&&opencv\\source\\text_r.avi");//在这里改相应的文件名
	namedWindow("foreground_l", WINDOW_AUTOSIZE);
	namedWindow("foreground_r", WINDOW_AUTOSIZE);


	//---------------------------------------------------------------------
	//获取视频的宽度、高度、帧率、总的帧数
	int frameH_l = vido_file_l.get(CV_CAP_PROP_FRAME_HEIGHT); //获取帧高
	int frameW_l = vido_file_l.get(CV_CAP_PROP_FRAME_WIDTH);  //获取帧宽
	int fps_l = vido_file_l.get(CV_CAP_PROP_FPS);          //获取帧率
	int numFrames_l = vido_file_l.get(CV_CAP_PROP_FRAME_COUNT);  //获取整个帧数
	int num_l = numFrames_l;
	cout << "Left:" << endl;
	printf("video's \nwidth = %d\t height = %d\n video's FPS = %d\t nums = %d\n", frameW_l, frameH_l, fps_l, numFrames_l);
	//---------------------------------------------------------------------

	int frameH_r = vido_file_r.get(CV_CAP_PROP_FRAME_HEIGHT); //获取帧高
	int frameW_r = vido_file_r.get(CV_CAP_PROP_FRAME_WIDTH);  //获取帧宽
	int fps_r = vido_file_r.get(CV_CAP_PROP_FPS);          //获取帧率
	int numFrames_r = vido_file_r.get(CV_CAP_PROP_FRAME_COUNT);  //获取整个帧数
	int num_r = numFrames_r;
	cout << "Right:" << endl;
	printf("video's \nwidth = %d\t height = %d\n video's FPS = %d\t nums = %d\n", frameW_r, frameH_r, fps_r, numFrames_r);

	feature_obj1.join();
	feature_obj2.join();
	feature_obj3.join();
	feature_obj4.join();

	std::future<vector<Point>>result1 = mypt1.get_future();//std::future对象result通过借助packaged_task类的对象mypt来保存线程入口函数返回值（future类和packaged_task类绑定）
	std::future<vector<Point>>result2 = mypt2.get_future();//std::future对象result通过借助packaged_task类的对象mypt来保存线程入口函数返回值（future类和packaged_task类绑定）
	std::future<vector<Point>>result3 = mypt3.get_future();//std::future对象result通过借助packaged_task类的对象mypt来保存线程入口函数返回值（future类和packaged_task类绑定）
	std::future<vector<Point>>result4 = mypt4.get_future();//std::future对象result通过借助packaged_task类的对象mypt来保存线程入口函数返回值（future类和packaged_task类绑定）

	std::shared_future<vector<Point>>result_s1(std::move(result1));
	std::shared_future<vector<Point>>result_s2(std::move(result2));
	std::shared_future<vector<Point>>result_s3(std::move(result3));
	std::shared_future<vector<Point>>result_s4(std::move(result4));
	while (1)
	{

		startTime = clock();//计时开始
		vido_file_l >> frame_l;
		vido_file_r >> frame_r;
		//Mat matRotation = getRotationMatrix2D(Point(frame.cols / 2, frame.rows / 2), 270, 1);//获取图像中心点旋转矩阵
		//Mat matRotatedFrame;// Rotate the image
		//warpAffine(frame, matRotatedFrame, matRotation, frame.size());

		/*apture >> frame;*/
		//imshow("src", matRotatedFrame);
		if (!false)
		{
			vido_file_l >> img_src1_l;
			vido_file_r >> img_src1_r;
			if (&img_src1_l == nullptr || &img_src1_r == nullptr)
			{
				printf("获取帧失败");
				break;
			}
			cvtColor(img_src1_l, gray1_l, CV_BGR2GRAY);
			cvtColor(img_src1_r, gray1_r, CV_BGR2GRAY);

			waitKey(33);//考虑到pc机处理速度，每隔33ms获取一帧图像，并将其转化为灰度图像分别处理

			vido_file_l >> img_src2_l;
			vido_file_r >> img_src2_r;
			if (&img_src2_l == nullptr || &img_src2_r == nullptr)
			{
				printf("获取帧失败");
				break;
			}
			cvtColor(img_src2_l, gray2_l, CV_BGR2GRAY);
			cvtColor(img_src2_r, gray2_r, CV_BGR2GRAY);

			waitKey(33);

			vido_file_l >> img_src3_l;
			vido_file_r >> img_src3_r;
			if (&img_src3_l == nullptr || &img_src3_r == nullptr) //需要判断视频结束时，获取帧失败的情况
			{
				printf("处理结束");
				break;
			}
			cvtColor(img_src3_l, gray3_l, CV_BGR2GRAY);
			cvtColor(img_src3_r, gray3_r, CV_BGR2GRAY);

			Sobel(gray1_l, gray1_l, CV_8U, 1, 0, 3, 0.4, 128);//sobel算子计算混合图像差分，由于sobel算子结合了Gaussian平滑和微分，所以，其结果或多或少对噪声有一定鲁棒性
			Sobel(gray1_r, gray1_r, CV_8U, 1, 0, 3, 0.4, 128);

			Sobel(gray2_l, gray2_l, CV_8U, 1, 0, 3, 0.4, 128);
			Sobel(gray2_r, gray2_r, CV_8U, 1, 0, 3, 0.4, 128);

			Sobel(gray3_l, gray3_l, CV_8U, 1, 0, 3, 0.4, 128);
			Sobel(gray3_r, gray3_r, CV_8U, 1, 0, 3, 0.4, 128);


			subtract(gray2_l, gray1_l, gray_diff11_l);//第二帧减第一帧
			subtract(gray2_r, gray1_r, gray_diff11_r);

			subtract(gray1_l, gray2_l, gray_diff12_l);
			subtract(gray1_r, gray2_r, gray_diff12_r);

			add(gray_diff11_l, gray_diff12_l, gray_diff1_l);
			add(gray_diff11_r, gray_diff12_r, gray_diff1_r);

			subtract(gray3_l, gray2_l, gray_diff21_l);//第三帧减第二帧
			subtract(gray3_r, gray2_r, gray_diff21_r);

			subtract(gray2_l, gray3_l, gray_diff22_l);
			subtract(gray2_r, gray3_r, gray_diff22_r);

			add(gray_diff21_l, gray_diff22_l, gray_diff2_l);
			add(gray_diff21_r, gray_diff22_r, gray_diff2_r);


			for (int i = 0; i < gray_diff1_l.rows; i++)
				for (int j = 0; j < gray_diff1_l.cols; j++)
				{
					if (abs(gray_diff1_l.at<unsigned char>(i, j)) >= threshold_diff1)//这里模板参数一定要用unsigned char，否则就一直报错
						gray_diff1_l.at<unsigned char>(i, j) = 255;            //第一次相减阈值处理
					else gray_diff1_l.at<unsigned char>(i, j) = 0;

					if (abs(gray_diff2_l.at<unsigned char>(i, j)) >= threshold_diff2)//第二次相减阈值处理
						gray_diff2_l.at<unsigned char>(i, j) = 255;
					else gray_diff2_l.at<unsigned char>(i, j) = 0;
				}
			for (int i = 0; i < gray_diff1_r.rows; i++)
				for (int j = 0; j < gray_diff1_r.cols; j++)
				{
					if (abs(gray_diff1_r.at<unsigned char>(i, j)) >= threshold_diff1)//这里模板参数一定要用unsigned char，否则就一直报错
						gray_diff1_r.at<unsigned char>(i, j) = 255;            //第一次相减阈值处理
					else gray_diff1_r.at<unsigned char>(i, j) = 0;

					if (abs(gray_diff2_r.at<unsigned char>(i, j)) >= threshold_diff2)//第二次相减阈值处理
						gray_diff2_r.at<unsigned char>(i, j) = 255;
					else gray_diff2_r.at<unsigned char>(i, j) = 0;
				}

			bitwise_and(gray_diff1_l, gray_diff2_l, gray_l);//三帧差法第三步，差分图像进行与运算
			bitwise_and(gray_diff1_r, gray_diff2_r, gray_r);

			dilate(gray_l, gray_l, Mat()); erode(gray_l, gray_l, Mat());//膨胀和腐蚀处理，有效消除高亮噪声
			dilate(gray_r, gray_r, Mat()); erode(gray_r, gray_r, Mat());

			medianBlur(gray_l, mid_filer_l, 3);//中值滤波
			medianBlur(gray_r, mid_filer_r, 3);//中值滤波
			//GaussianBlur(gray, mid_filer, Size(3, 3), 0, 0);


			//Mat matRotation1 = getRotationMatrix2D(Point(mid_filer.cols / 2, mid_filer.rows / 2), 270, 1);
			//Mat matRotatedFrame1;// Rotate the image
			//warpAffine(mid_filer, matRotatedFrame1, matRotation1, mid_filer.size());
			//apture >> frame;
			imshow("foreground_l", mid_filer_l);
			imshow("foreground_r", mid_filer_r);


			//int cnts,nums;
			//(matRotatedFrame1,cnts,CV_RETR_EXTERNAL);
			vector<vector<Point>> contours_l;
			vector<Vec4i> hierarchy_l;
			vector<vector<Point>> contours_r;
			vector<Vec4i> hierarchy_r;

			vector <Point> point_l;
			vector <Point> point_r;

			findContours(mid_filer_l, contours_l, hierarchy_l, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//寻找轮廓
			findContours(mid_filer_r, contours_r, hierarchy_r, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//寻找轮廓

			Rect bt_point_l, bt_point_r;//rect类存贮所创建矩形对象的左上角和右下角坐标（图像坐标系下）
			Point p1_l, p2_l, p1_r, p2_r;

			Mat imageContours_l = Mat::zeros(mid_filer_l.size(), CV_8UC1);
			Mat imageContours_r = Mat::zeros(mid_filer_r.size(), CV_8UC1);

			Mat Contours_l = Mat::zeros(mid_filer_l.size(), CV_8UC1);  //绘制  
			Mat Contours_r = Mat::zeros(mid_filer_r.size(), CV_8UC1);  //绘制  


			line(frame_l, Point(frameW_l / 2 - 60, frameH_l / 2), Point(frameW_l / 2 + 60, frameH_l / 2), Scalar(0, 0, 255), 2, 8);
			line(frame_l, Point(frameW_l / 2, frameH_l / 2 - 60), Point(frameW_l / 2, frameH_l / 2 + 60), Scalar(255, 0, 0), 2, 8);

			line(frame_r, Point(frameW_r / 2 - 60, frameH_r / 2), Point(frameW_r / 2 + 60, frameH_r / 2), Scalar(0, 0, 255), 2, 8);
			line(frame_r, Point(frameW_r / 2, frameH_r / 2 - 60), Point(frameW_r / 2, frameH_r / 2 + 60), Scalar(255, 0, 0), 2, 8);



			for (int i = 0; i < contours_l.size(); i++)
			{

				x_l = 0, y_l = 0;
				x_l_ = 0, y_l_ = 0;
				//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数,根据轮廓像素点个数


				double a0 = matchShapes(result_s1.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a1 = matchShapes(result_s2.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a2 = matchShapes(result_s3.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a3 = matchShapes(result_s4.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;

				double epsilon1 = 0.01*arcLength(contours_l[i], true);//形状逼近，排除背景矩形（人，车），圆形（球，人），梯形（车）等
				approxPolyDP(Mat(contours_l[i]), point_l, epsilon1, true);
				int corners1 = point_l.size();
				if ((a0 < 1 || a1 < 1 || a2 < 1 || a3 < 1) && corners1 >= 5 && (contours_l[i].size() > 20 && contours_l[i].size() < 300))
				{
					//drawContours(frame, contours, i, Scalar(0, 255, 0), 2, 8);
					bt_point_l = boundingRect(contours_l[i]);
					p1_l.x = bt_point_l.x;
					p1_l.y = bt_point_l.y;
					p2_l.x = bt_point_l.x + bt_point_l.width;
					p2_l.y = bt_point_l.y + bt_point_l.height;
					rectangle(frame_l, p1_l, p2_l, Scalar(0, 255, 0), 2);//矩形框出ROI区域


					//获取ROI矩形框中心坐标    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   用于后续目标坐标计算

					x_l = (p2_l.x - p1_l.x) / 2 + p1_l.x;
					y_l = (p2_l.y - p1_l.y) / 2 + p1_l.y;
					if (x_l >= frameW_l / 2)
					{
						if (y_l <= frameH_l / 2)//第一象限
						{
							x_l_ = (x_l - frameW_l / 2);
							y_l_ = (frameH_l / 2 - y_l);
						}
						else //第四象限
						{
							x_l_ = (x_l - frameW_l / 2);
							y_l_ = -(y_l - frameH_l / 2);
						}
					}
					else
					{
						if (y_l <= frameH_l / 2)//第二象限
						{
							x_l_ = -(frameW_l / 2 - x_l);
							y_l_ = (frameH_l / 2 - y_l);
						}
						else //第三象限
						{
							x_l_ = -(frameW_l / 2 - x_l);
							y_l_ = -(y_l - frameH_l / 2);
						}
					}

					//空间坐标获取



					//获取时间戳，打印坐标信息
					/*time_stamp();
					cout << "左眼图像坐标系下目标坐标：" << "x_l:" << x_l << "y_l:" << y_l << endl;*/

				}

			}
			for (int i = 0; i < contours_r.size(); i++)
			{
				x_r = 0, y_r = 0;
				x_r_ = 0, y_r_ = 0;
				//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数,根据轮廓像素点个数


				double a0 = matchShapes(result_s1.get(), contours_r[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a1 = matchShapes(result_s2.get(), contours_r[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a2 = matchShapes(result_s3.get(), contours_r[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a3 = matchShapes(result_s4.get(), contours_r[i], CV_CONTOURS_MATCH_I3, 0) - 3;

				double epsilon2 = 0.01 * arcLength(contours_r[i], true);
				approxPolyDP(Mat(contours_r[i]), point_r, epsilon2, true);
				int corners2 = point_r.size();
				if ((a0 < 1 || a1 < 1 || a2 < 1 || a3 < 1) && corners2 >= 5 && (contours_r[i].size() > 20 && contours_r[i].size() < 300))
				{
					//drawContours(frame, contours, i, Scalar(0, 255, 0), 2, 8);
					bt_point_r = boundingRect(contours_r[i]);
					p1_r.x = bt_point_r.x;
					p1_r.y = bt_point_r.y;
					p2_r.x = bt_point_r.x + bt_point_r.width;
					p2_r.y = bt_point_r.y + bt_point_r.height;
					rectangle(frame_r, p1_r, p2_r, Scalar(0, 255, 0), 2);//矩形框出ROI区域


					//获取ROI矩形框中心坐标    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   用于后续目标坐标计算

					x_r = (p2_r.x - p1_r.x) / 2 + p1_r.x;
					y_r = (p2_r.y - p1_r.y) / 2 + p1_r.y;
					if (x_r >= frameW_r / 2)
					{
						if (y_r <= frameH_r / 2)//第一象限
						{
							x_r_ = (x_r - frameW_r / 2);
							y_r_ = (frameH_r / 2 - y_r);
						}
						else //第四象限
						{
							x_r_ = (x_r - frameW_r / 2);
							y_r_ = -(y_r - frameH_r / 2);
						}
					}
					else
					{
						if (y_r <= frameH_r / 2)//第二象限
						{
							x_r_ = -(frameW_r / 2 - x_r);
							y_r_ = (frameH_r / 2 - y_r);
						}
						else //第三象限
						{
							x_r_ = -(frameW_r / 2 - x_r);
							y_r_ = -(y_r - frameH_r / 2);
						}
					}

					//空间坐标获取



					//获取时间戳，打印坐标信息
					if (x_l != 0 && y_l != 0 && x_r != 0 && y_r != 0)
					{

						//引入双目视觉数学测量模型
						//计算目标点在左眼相机坐标系下的三维坐标（X,Y,Z）
						Point3f worldPoint;
						worldPoint = uv2xyz(Point2f(x_l, y_l), Point2f(x_r, y_r));

						X = worldPoint.x / 1000;
						Y = -worldPoint.y / 1000;
						Z = worldPoint.z / 1000;


						times_stamp();
						cout << "左眼图像坐标系下目标坐标：" << "x_l:" << x_l_ << " y_l:" << y_l_ << endl;
						cout << "右眼图像坐标系下目标坐标：" << "x_r:" << x_r_ << " y_r:" << y_r_ << endl;
						cout << "==*-*-*-*-*-*-*-*-*-*-*-*-*-*-*==" << endl;
						cout << "相机坐标系下三维坐标：" << "X:" << X << "m" << " Y:" << Y << "m" << " Z:" << Z << "m" << endl;
					}
				}

			}
		}
		namedWindow("左眼跟踪识别监视器", WINDOW_AUTOSIZE);
		imshow("左眼跟踪识别监视器", frame_l);
		namedWindow("右眼跟踪识别监视器", WINDOW_AUTOSIZE);
		imshow("右眼跟踪识别监视器", frame_r);

		if (cvWaitKey(33) >= 0)
			break;

		/*paint(x, y, z);*/

		endTime = clock();//计时结束
		cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	}
	return 0;
}



