// meanshift11.cpp : �������̨Ӧ�ó������ڵ㡣
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

	/*�ȶ������ԭʼͼ����и���Ȥ����Ĳɼ�*/
	//�Ȼ�ȡ����Ȥ�����ͼ��
	Mat bgrROI = orgimg(ROIrect);

	//������Ȥ�����BGR�ռ�ת��ΪHSV�ռ�
	Mat hsvROI;
	cvtColor(bgrROI, hsvROI, CV_BGR2HSV);
	//��ȡ����Ȥ�����HSV�ռ��S�ռ�
	vector<Mat>hsvROIvector;
	split(hsvROI, hsvROIvector);
	//��Sͨ���ķ�ֵ��
	threshold(hsvROIvector[1], hsvROIvector[1], 65, 255.0, THRESH_BINARY);
	//����Sͨ����ֱ��ͼ
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

	//��һ��ֱ��ͼ
	normalize(hsvROIhist, hsvROIhist, 1.0);

	/*������ĵڶ���ͼ���н��о�ֵƯ���㷨*/
	Mat prehsv;
	cvtColor(preimg, prehsv, CV_BGR2HSV);
	vector<Mat>prehsvvector;
	split(prehsv, prehsvvector);
	/*threshold(prehsvvector[1], prehsvvector[1], 65, 255.0, THRESH_BINARY);*/
	//�ڵڶ���ͼ�ϻ�ȡ����Ȥ�����ֱ��ͼ�ķ�ͶӰ
	Mat result;
	calcBackProject(&(prehsvvector[1]), 1, channels, hsvROIhist, result, ranges, 255.0);
	threshold(result, result, 255 * (-1.0f), 255.0, THRESH_BINARY);
	bitwise_and(result, prehsvvector[1], result);

	/*rectangle(preimg, ROIrect, Scalar(0, 0, 255));*/
	//meanshift�㷨
	TermCriteria criteria(TermCriteria::MAX_ITER, 10, 0.01);
	meanShift(result, ROIrect, criteria);
	/*rectangle(preimg, ROIrect, Scalar(0, 255, 0));*/

	return ROIrect;
}

Package P;

Point coord;//�����ʼ����
Rect sqart;//������ο����ʼ�����Լ����ȺͿ��
bool draw;
bool flag = 0;//�����־λ���������Ҫ�����α궨�Ĳ��ֵ�����ʾ��һ������ʱʹ�õ�
Mat frame, frame_org;
Mat dst;//����Ȥ����ͼ��

Rect ROIrect, ROIrect_org;

void onMouse(int event, int x, int y, int flags, void *param)
{
	//��ʾ���ĵ�ǰ����
	/*cout << "Event:" << event << endl;
	cout << "x=" << x << "     " << "y=" << y << endl;
	cout << "flags:" << endl;
	cout << "param" << param << endl;*/

	//���if�������switch֮ǰ
	if (draw)
	{
		//��MIN�õ����ϵ���Ϊ���ο����ʵ���꣬������������������ʱֻ����һ���������
		sqart.x = MIN(x, coord.x);
		sqart.y = MIN(y, coord.y);
		sqart.width = abs(coord.x - x);
		sqart.height = abs(coord.y - y);
		//��ֹ�������򳬳�ͼ��ķ�Χ
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
	VideoCapture capture("E:\\������Ƶ��\\swa.mp4");
	namedWindow("Mouse", 1);

	setMouseCallback("Mouse", onMouse, 0);

	//������Ƶ���е�ÿһ֡���ڸ��£����Բ����������Ƕ�׵�״��
	while (1)
	{
		capture >> frame;
		if (frame.empty())
			return -1;

		//�����ο�õ�������������һ��������ʾ
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