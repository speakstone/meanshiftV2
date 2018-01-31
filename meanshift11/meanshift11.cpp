// meanshift11.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "cv.h"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\video\tracking.hpp>
#include <iostream>
#include <vector>
#include "imgproc/imgproc.hpp"
#include "video/tracking.hpp"
#include<opencv2/imgproc/types_c.h>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<iostream>
//#include<opencv2/nonfree/features2d.hpp>
//#include<opencv2/legacy/legacy.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/calib3d/calib3d.hpp>


using namespace cv;
using namespace std;

class Package
{
private:

public:
	Rect MeanShitfMath(const Mat &OrgInImage, const Mat &PreInImage, Rect &ROIrect);
};

Rect Package::MeanShitfMath(const Mat &OrgInImage, const Mat &PreInImage, Rect &ROIrect)
{
	Mat orgimg = OrgInImage;
	Mat preimg = PreInImage;

	/*先对输入的原始图像进行感兴趣区域的采集*/
	//先获取感兴趣区域的图像
	Mat bgrROI = orgimg(ROIrect);

	//将感兴趣区域的BGR空间转换为HSV空间
	Mat hsvROI;
	cvtColor(bgrROI, hsvROI, CV_BGR2HSV);
	//获取感兴趣区域的HSV空间的S空间
	vector<Mat>hsvROIvector;
	split(hsvROI, hsvROIvector);
	//将S通道的阀值化
	threshold(hsvROIvector[1], hsvROIvector[1], 65, 255.0, THRESH_BINARY);
	//计算S通道的直方图
	float hranges[2];
	const float* ranges[1];
	int channels[1];
	MatND hsvROIhist;
	int histSize[1];
	hranges[0] = 0.0;
	hranges[1] = 180.0;
	ranges[0] = hranges;
	channels[0] = 0;
	histSize[0] = 256;
	calcHist(&hsvROIvector[1], 1, channels, Mat(), hsvROIhist, 1, histSize, ranges);

	//归一化直方图
	normalize(hsvROIhist, hsvROIhist, 1.0);

	/*在输入的第二幅图像中进行均值漂移算法*/
	Mat prehsv;
	cvtColor(preimg, prehsv, CV_BGR2HSV);
	vector<Mat>prehsvvector;
	split(prehsv, prehsvvector);
	/*threshold(prehsvvector[1], prehsvvector[1], 65, 255.0, THRESH_BINARY);*/
	//在第二幅图上获取感兴趣区域的直方图的反投影
	Mat result;
	calcBackProject(&(prehsvvector[1]), 1, channels, hsvROIhist, result, ranges, 255.0);
	threshold(result, result, 255 * (-1.0f), 255.0, THRESH_BINARY);
	bitwise_and(result, prehsvvector[1], result);

	/*rectangle(preimg, ROIrect, Scalar(0, 0, 255));*/
	//meanshift算法
	TermCriteria criteria(TermCriteria::MAX_ITER, 10, 0.01);
	meanShift(result, ROIrect, criteria);
	/*rectangle(preimg, ROIrect, Scalar(0, 255, 0));*/

	return ROIrect;
}

Package P;

Point coord;//储存初始坐标
Rect sqart;//储存矩形框的起始坐标以及长度和宽度
bool draw;
bool flag = 0;//这个标志位是用在如果要将矩形标定的部分单独显示在一个窗口时使用的
Mat frame, frame_org;
Mat dst;//感兴趣区域图像

Rect ROIrect, ROIrect_org;

void onMouse(int event, int x, int y, int flags, void *param)
{
	//显示鼠标的当前坐标
	/*cout << "Event:" << event << endl;
	cout << "x=" << x << "     " << "y=" << y << endl;
	cout << "flags:" << endl;
	cout << "param" << param << endl;*/

	//这个if必须放在switch之前
	if (draw)
	{
		//用MIN得到左上点作为矩形框的其实坐标，如果不加这个，画矩形时只能向一个方向进行
		sqart.x = MIN(x, coord.x);
		sqart.y = MIN(y, coord.y);
		sqart.width = abs(coord.x - x);
		sqart.height = abs(coord.y - y);
		//防止矩形区域超出图像的范围
		sqart &= Rect(0, 0, frame.cols, frame.rows);
	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		coord = Point(x, y);
		sqart = Rect(x, y, 0, 0);
		draw = true;
		break;

	case CV_EVENT_LBUTTONUP:
		draw = false;
		flag = 1;
		break;
	}
}

int main()
{
	char c;
	int start = 0;
	//VideoCapture capture(0);
	VideoCapture capture("E:\\论文视频库\\swa.mp4");
	namedWindow("Mouse", 1);

	setMouseCallback("Mouse", onMouse, 0);

	//由于视频序列的每一帧都在跟新，所以不会出现连环嵌套的状况
	while (1)
	{
		capture >> frame;
		if (frame.empty())
			return -1;

		//将矩形框得到矩形区域用另一个窗口显示
		if ((flag == 1) && sqart.height > 0 && sqart.width > 0)
		{
			dst = frame(Rect(sqart.x, sqart.y, sqart.width, sqart.height));
			namedWindow("dst");
			imshow("dst", dst);

			ROIrect_org = Rect(sqart.x, sqart.y, sqart.width, sqart.height);
			frame_org = frame;

			start = 1;
			flag = 0;
		}

		rectangle(frame, sqart, Scalar(0, 0, 255), 3);

		if (start == 1)
		{
			ROIrect = P.MeanShitfMath(frame_org, frame, ROIrect_org);
			rectangle(frame, ROIrect, Scalar(255, 0, 0), 3);
			frame_org = frame;
			ROIrect_org = ROIrect;
		}

		imshow("Mouse", frame);

		c = waitKey(20);
		if (c == 27)
			break;
	}

	return 0;
}