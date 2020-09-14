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


#define threshold_diff1 25 //���ü�֡���ֵ
#define threshold_diff2 25 //���ü�֡���ֵ
//************************************************************************************************************************************

int main(int argc, unsigned char* argv[])
{

	//��4���̣߳�����ģ��ƥ��ƥ�䴦����ʡ����ʱ��

	std::packaged_task<vector<Point>()>mypt1(match_feature1);//����mythreadͨ��packaged_task��װ
	std::thread feature_obj1(std::ref(mypt1));
	std::packaged_task<vector<Point>()>mypt2(match_feature2);
	std::thread feature_obj2(std::ref(mypt2));
	std::packaged_task<vector<Point>()>mypt3(match_feature3);
	std::thread feature_obj3(std::ref(mypt3));
	std::packaged_task<vector<Point>()>mypt4(match_feature4);
	std::thread feature_obj4(std::ref(mypt4));

	//match_feature();
	clock_t startTime, endTime;

	double x_l, y_l;//ͼ������ϵ��Ŀ���άƽ������,(���Ͻ�Ϊ����ԭ��)
	double x_r, y_r;

	double x_l_, y_l_;//ͼ������ϵ��Ŀ���άƽ������,(���ĵ�Ϊ����ԭ��)
	double x_r_, y_r_;//ͼ������ϵ��Ŀ���άƽ������,(���ĵ�Ϊ����ԭ��)


	double X, Y, Z;//�����������ϵ��Ŀ����ά�ռ�����


	Mat frame_l;
	Mat frame_r;
	Mat img_src1_l, img_src2_l, img_src3_l;//����3֡����Ҫ3֡ͼƬ
	Mat img_src1_r, img_src2_r, img_src3_r;//����3֡����Ҫ3֡ͼƬ

	Mat img_dst_l, gray1_l, gray2_l, gray3_l;
	Mat img_dst_r, gray1_r, gray2_r, gray3_r;

	Mat gray_diff1_l, gray_diff2_l;//�洢2�������ͼƬ
	Mat gray_diff1_r, gray_diff2_r;//�洢2�������ͼƬ

	Mat gray_diff11_l, gray_diff12_l;
	Mat gray_diff11_r, gray_diff12_r;

	Mat gray_diff21_l, gray_diff22_l;
	Mat gray_diff21_r, gray_diff22_r;

	Mat gray_l, gray_r;//������ʾǰ����
	Mat mid_filer_l;   //��ֵ�˲��������Ƭ
	Mat mid_filer_r;   //��ֵ�˲��������Ƭ
	bool pause = false;


	VideoCapture vido_file_l("F:\\visual c++&&opencv\\source\\text_l.avi");//���������Ӧ���ļ���
	VideoCapture vido_file_r("F:\\visual c++&&opencv\\source\\text_r.avi");//���������Ӧ���ļ���
	namedWindow("foreground_l", WINDOW_AUTOSIZE);
	namedWindow("foreground_r", WINDOW_AUTOSIZE);


	//---------------------------------------------------------------------
	//��ȡ��Ƶ�Ŀ�ȡ��߶ȡ�֡�ʡ��ܵ�֡��
	int frameH_l = vido_file_l.get(CV_CAP_PROP_FRAME_HEIGHT); //��ȡ֡��
	int frameW_l = vido_file_l.get(CV_CAP_PROP_FRAME_WIDTH);  //��ȡ֡��
	int fps_l = vido_file_l.get(CV_CAP_PROP_FPS);          //��ȡ֡��
	int numFrames_l = vido_file_l.get(CV_CAP_PROP_FRAME_COUNT);  //��ȡ����֡��
	int num_l = numFrames_l;
	cout << "Left:" << endl;
	printf("video's \nwidth = %d\t height = %d\n video's FPS = %d\t nums = %d\n", frameW_l, frameH_l, fps_l, numFrames_l);
	//---------------------------------------------------------------------

	int frameH_r = vido_file_r.get(CV_CAP_PROP_FRAME_HEIGHT); //��ȡ֡��
	int frameW_r = vido_file_r.get(CV_CAP_PROP_FRAME_WIDTH);  //��ȡ֡��
	int fps_r = vido_file_r.get(CV_CAP_PROP_FPS);          //��ȡ֡��
	int numFrames_r = vido_file_r.get(CV_CAP_PROP_FRAME_COUNT);  //��ȡ����֡��
	int num_r = numFrames_r;
	cout << "Right:" << endl;
	printf("video's \nwidth = %d\t height = %d\n video's FPS = %d\t nums = %d\n", frameW_r, frameH_r, fps_r, numFrames_r);

	feature_obj1.join();
	feature_obj2.join();
	feature_obj3.join();
	feature_obj4.join();

	std::future<vector<Point>>result1 = mypt1.get_future();//std::future����resultͨ������packaged_task��Ķ���mypt�������߳���ں�������ֵ��future���packaged_task��󶨣�
	std::future<vector<Point>>result2 = mypt2.get_future();//std::future����resultͨ������packaged_task��Ķ���mypt�������߳���ں�������ֵ��future���packaged_task��󶨣�
	std::future<vector<Point>>result3 = mypt3.get_future();//std::future����resultͨ������packaged_task��Ķ���mypt�������߳���ں�������ֵ��future���packaged_task��󶨣�
	std::future<vector<Point>>result4 = mypt4.get_future();//std::future����resultͨ������packaged_task��Ķ���mypt�������߳���ں�������ֵ��future���packaged_task��󶨣�

	std::shared_future<vector<Point>>result_s1(std::move(result1));
	std::shared_future<vector<Point>>result_s2(std::move(result2));
	std::shared_future<vector<Point>>result_s3(std::move(result3));
	std::shared_future<vector<Point>>result_s4(std::move(result4));
	while (1)
	{

		startTime = clock();//��ʱ��ʼ
		vido_file_l >> frame_l;
		vido_file_r >> frame_r;
		//Mat matRotation = getRotationMatrix2D(Point(frame.cols / 2, frame.rows / 2), 270, 1);//��ȡͼ�����ĵ���ת����
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
				printf("��ȡ֡ʧ��");
				break;
			}
			cvtColor(img_src1_l, gray1_l, CV_BGR2GRAY);
			cvtColor(img_src1_r, gray1_r, CV_BGR2GRAY);

			waitKey(33);//���ǵ�pc�������ٶȣ�ÿ��33ms��ȡһ֡ͼ�񣬲�����ת��Ϊ�Ҷ�ͼ��ֱ���

			vido_file_l >> img_src2_l;
			vido_file_r >> img_src2_r;
			if (&img_src2_l == nullptr || &img_src2_r == nullptr)
			{
				printf("��ȡ֡ʧ��");
				break;
			}
			cvtColor(img_src2_l, gray2_l, CV_BGR2GRAY);
			cvtColor(img_src2_r, gray2_r, CV_BGR2GRAY);

			waitKey(33);

			vido_file_l >> img_src3_l;
			vido_file_r >> img_src3_r;
			if (&img_src3_l == nullptr || &img_src3_r == nullptr) //��Ҫ�ж���Ƶ����ʱ����ȡ֡ʧ�ܵ����
			{
				printf("�������");
				break;
			}
			cvtColor(img_src3_l, gray3_l, CV_BGR2GRAY);
			cvtColor(img_src3_r, gray3_r, CV_BGR2GRAY);

			Sobel(gray1_l, gray1_l, CV_8U, 1, 0, 3, 0.4, 128);//sobel���Ӽ�����ͼ���֣�����sobel���ӽ����Gaussianƽ����΢�֣����ԣ����������ٶ�������һ��³����
			Sobel(gray1_r, gray1_r, CV_8U, 1, 0, 3, 0.4, 128);

			Sobel(gray2_l, gray2_l, CV_8U, 1, 0, 3, 0.4, 128);
			Sobel(gray2_r, gray2_r, CV_8U, 1, 0, 3, 0.4, 128);

			Sobel(gray3_l, gray3_l, CV_8U, 1, 0, 3, 0.4, 128);
			Sobel(gray3_r, gray3_r, CV_8U, 1, 0, 3, 0.4, 128);


			subtract(gray2_l, gray1_l, gray_diff11_l);//�ڶ�֡����һ֡
			subtract(gray2_r, gray1_r, gray_diff11_r);

			subtract(gray1_l, gray2_l, gray_diff12_l);
			subtract(gray1_r, gray2_r, gray_diff12_r);

			add(gray_diff11_l, gray_diff12_l, gray_diff1_l);
			add(gray_diff11_r, gray_diff12_r, gray_diff1_r);

			subtract(gray3_l, gray2_l, gray_diff21_l);//����֡���ڶ�֡
			subtract(gray3_r, gray2_r, gray_diff21_r);

			subtract(gray2_l, gray3_l, gray_diff22_l);
			subtract(gray2_r, gray3_r, gray_diff22_r);

			add(gray_diff21_l, gray_diff22_l, gray_diff2_l);
			add(gray_diff21_r, gray_diff22_r, gray_diff2_r);


			for (int i = 0; i < gray_diff1_l.rows; i++)
				for (int j = 0; j < gray_diff1_l.cols; j++)
				{
					if (abs(gray_diff1_l.at<unsigned char>(i, j)) >= threshold_diff1)//����ģ�����һ��Ҫ��unsigned char�������һֱ����
						gray_diff1_l.at<unsigned char>(i, j) = 255;            //��һ�������ֵ����
					else gray_diff1_l.at<unsigned char>(i, j) = 0;

					if (abs(gray_diff2_l.at<unsigned char>(i, j)) >= threshold_diff2)//�ڶ��������ֵ����
						gray_diff2_l.at<unsigned char>(i, j) = 255;
					else gray_diff2_l.at<unsigned char>(i, j) = 0;
				}
			for (int i = 0; i < gray_diff1_r.rows; i++)
				for (int j = 0; j < gray_diff1_r.cols; j++)
				{
					if (abs(gray_diff1_r.at<unsigned char>(i, j)) >= threshold_diff1)//����ģ�����һ��Ҫ��unsigned char�������һֱ����
						gray_diff1_r.at<unsigned char>(i, j) = 255;            //��һ�������ֵ����
					else gray_diff1_r.at<unsigned char>(i, j) = 0;

					if (abs(gray_diff2_r.at<unsigned char>(i, j)) >= threshold_diff2)//�ڶ��������ֵ����
						gray_diff2_r.at<unsigned char>(i, j) = 255;
					else gray_diff2_r.at<unsigned char>(i, j) = 0;
				}

			bitwise_and(gray_diff1_l, gray_diff2_l, gray_l);//��֡������������ͼ�����������
			bitwise_and(gray_diff1_r, gray_diff2_r, gray_r);

			dilate(gray_l, gray_l, Mat()); erode(gray_l, gray_l, Mat());//���ͺ͸�ʴ������Ч������������
			dilate(gray_r, gray_r, Mat()); erode(gray_r, gray_r, Mat());

			medianBlur(gray_l, mid_filer_l, 3);//��ֵ�˲�
			medianBlur(gray_r, mid_filer_r, 3);//��ֵ�˲�
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

			findContours(mid_filer_l, contours_l, hierarchy_l, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//Ѱ������
			findContours(mid_filer_r, contours_r, hierarchy_r, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//Ѱ������

			Rect bt_point_l, bt_point_r;//rect��������������ζ�������ϽǺ����½����꣨ͼ������ϵ�£�
			Point p1_l, p2_l, p1_r, p2_r;

			Mat imageContours_l = Mat::zeros(mid_filer_l.size(), CV_8UC1);
			Mat imageContours_r = Mat::zeros(mid_filer_r.size(), CV_8UC1);

			Mat Contours_l = Mat::zeros(mid_filer_l.size(), CV_8UC1);  //����  
			Mat Contours_r = Mat::zeros(mid_filer_r.size(), CV_8UC1);  //����  


			line(frame_l, Point(frameW_l / 2 - 60, frameH_l / 2), Point(frameW_l / 2 + 60, frameH_l / 2), Scalar(0, 0, 255), 2, 8);
			line(frame_l, Point(frameW_l / 2, frameH_l / 2 - 60), Point(frameW_l / 2, frameH_l / 2 + 60), Scalar(255, 0, 0), 2, 8);

			line(frame_r, Point(frameW_r / 2 - 60, frameH_r / 2), Point(frameW_r / 2 + 60, frameH_r / 2), Scalar(0, 0, 255), 2, 8);
			line(frame_r, Point(frameW_r / 2, frameH_r / 2 - 60), Point(frameW_r / 2, frameH_r / 2 + 60), Scalar(255, 0, 0), 2, 8);



			for (int i = 0; i < contours_l.size(); i++)
			{

				x_l = 0, y_l = 0;
				x_l_ = 0, y_l_ = 0;
				//contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���,�����������ص����


				double a0 = matchShapes(result_s1.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a1 = matchShapes(result_s2.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a2 = matchShapes(result_s3.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a3 = matchShapes(result_s4.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;

				double epsilon1 = 0.01*arcLength(contours_l[i], true);//��״�ƽ����ų��������Σ��ˣ�������Բ�Σ����ˣ������Σ�������
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
					rectangle(frame_l, p1_l, p2_l, Scalar(0, 255, 0), 2);//���ο��ROI����


					//��ȡROI���ο���������    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   ���ں���Ŀ���������

					x_l = (p2_l.x - p1_l.x) / 2 + p1_l.x;
					y_l = (p2_l.y - p1_l.y) / 2 + p1_l.y;
					if (x_l >= frameW_l / 2)
					{
						if (y_l <= frameH_l / 2)//��һ����
						{
							x_l_ = (x_l - frameW_l / 2);
							y_l_ = (frameH_l / 2 - y_l);
						}
						else //��������
						{
							x_l_ = (x_l - frameW_l / 2);
							y_l_ = -(y_l - frameH_l / 2);
						}
					}
					else
					{
						if (y_l <= frameH_l / 2)//�ڶ�����
						{
							x_l_ = -(frameW_l / 2 - x_l);
							y_l_ = (frameH_l / 2 - y_l);
						}
						else //��������
						{
							x_l_ = -(frameW_l / 2 - x_l);
							y_l_ = -(y_l - frameH_l / 2);
						}
					}

					//�ռ������ȡ



					//��ȡʱ�������ӡ������Ϣ
					/*time_stamp();
					cout << "����ͼ������ϵ��Ŀ�����꣺" << "x_l:" << x_l << "y_l:" << y_l << endl;*/

				}

			}
			for (int i = 0; i < contours_r.size(); i++)
			{
				x_r = 0, y_r = 0;
				x_r_ = 0, y_r_ = 0;
				//contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���,�����������ص����


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
					rectangle(frame_r, p1_r, p2_r, Scalar(0, 255, 0), 2);//���ο��ROI����


					//��ȡROI���ο���������    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   ���ں���Ŀ���������

					x_r = (p2_r.x - p1_r.x) / 2 + p1_r.x;
					y_r = (p2_r.y - p1_r.y) / 2 + p1_r.y;
					if (x_r >= frameW_r / 2)
					{
						if (y_r <= frameH_r / 2)//��һ����
						{
							x_r_ = (x_r - frameW_r / 2);
							y_r_ = (frameH_r / 2 - y_r);
						}
						else //��������
						{
							x_r_ = (x_r - frameW_r / 2);
							y_r_ = -(y_r - frameH_r / 2);
						}
					}
					else
					{
						if (y_r <= frameH_r / 2)//�ڶ�����
						{
							x_r_ = -(frameW_r / 2 - x_r);
							y_r_ = (frameH_r / 2 - y_r);
						}
						else //��������
						{
							x_r_ = -(frameW_r / 2 - x_r);
							y_r_ = -(y_r - frameH_r / 2);
						}
					}

					//�ռ������ȡ



					//��ȡʱ�������ӡ������Ϣ
					if (x_l != 0 && y_l != 0 && x_r != 0 && y_r != 0)
					{

						//����˫Ŀ�Ӿ���ѧ����ģ��
						//����Ŀ����������������ϵ�µ���ά���꣨X,Y,Z��
						Point3f worldPoint;
						worldPoint = uv2xyz(Point2f(x_l, y_l), Point2f(x_r, y_r));

						X = worldPoint.x / 1000;
						Y = -worldPoint.y / 1000;
						Z = worldPoint.z / 1000;


						times_stamp();
						cout << "����ͼ������ϵ��Ŀ�����꣺" << "x_l:" << x_l_ << " y_l:" << y_l_ << endl;
						cout << "����ͼ������ϵ��Ŀ�����꣺" << "x_r:" << x_r_ << " y_r:" << y_r_ << endl;
						cout << "==*-*-*-*-*-*-*-*-*-*-*-*-*-*-*==" << endl;
						cout << "�������ϵ����ά���꣺" << "X:" << X << "m" << " Y:" << Y << "m" << " Z:" << Z << "m" << endl;
					}
				}

			}
		}
		namedWindow("���۸���ʶ�������", WINDOW_AUTOSIZE);
		imshow("���۸���ʶ�������", frame_l);
		namedWindow("���۸���ʶ�������", WINDOW_AUTOSIZE);
		imshow("���۸���ʶ�������", frame_r);

		if (cvWaitKey(33) >= 0)
			break;

		/*paint(x, y, z);*/

		endTime = clock();//��ʱ����
		cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	}
	return 0;
}



