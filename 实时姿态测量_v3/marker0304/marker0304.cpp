
#include <Eigen/Core>
#include <Eigen/LU>
#include "pch.h"
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <Eigen/Dense>
#include <iostream>
#include "Marker.hpp"
#include <opencv2\core\eigen.hpp>
#include <cmath>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <librealsense2/hpp/rs_processing.hpp>
#include <librealsense2/hpp/rs_types.hpp>
#include <librealsense2/hpp/rs_sensor.hpp>
#define PI 3.1415926

using namespace cv;
using namespace std;
using namespace Eigen;

typedef vector<Point> PointsVector;
typedef vector<PointsVector> ContoursVector;
vector<Marker> markers;
vector<Point2f> m_markerCorners2d;
vector<Point3f> m_markerCorners3d;
Size markerSize = Size(100, 100);

void findCandidates(const ContoursVector& contours, vector<Marker>& detectedMarkers);
float perimeter(const vector<Point2f> &a);
void recognizeMarkers(const Mat& grayscale, vector<Marker>& detectedMarkers);
void recognizeMarkers1(const Mat& grayscale, vector<Marker>& detectedMarkers);
void estimatePosition(vector<Marker>& detectedMarkers, Mat_<float>& camMatrix, Mat_<float>& distCoeff);
void findPointsOrder(vector<Marker>& detectedMarkers);
void getEulerAngles(cv::Mat matrix);
bool isRotationMatrix(cv::Mat &R);
void getEulerAngles2(cv::Mat &R, cv::Mat &t, cv::Mat &euler_angles);

Mat src;
int main(int argc, char *argv[])
{
	Mat_<float> intrinsMatrix = Mat::eye(3, 3, CV_64F);
	Mat_<float> distCoeff = Mat::zeros(5, 1, CV_64F);
	FileStorage fs("leftVectors.yml", FileStorage::READ);
	if (!fs.isOpened())
	{
		cout << "Could not open the configuration file!" << endl;
		exit(1);
	}
	fs["IntrinsicMatrix"] >> intrinsMatrix;
	fs["Distortion"] >> distCoeff;
	fs.release();

	m_markerCorners2d.push_back(Point2f(0, 0));
	m_markerCorners2d.push_back(Point2f(99, 0));
	m_markerCorners2d.push_back(Point2f(99, 99));
	m_markerCorners2d.push_back(Point2f(0, 99));

	m_markerCorners3d.push_back(Point3f(-2.5f, -2.5f, 0));
	m_markerCorners3d.push_back(Point3f(+2.5f, -2.5f, 0));
	m_markerCorners3d.push_back(Point3f(+2.5f, +2.5f, 0));
	m_markerCorners3d.push_back(Point3f(-2.5f, +2.5f, 0));

	rs2::pipeline pipe;
	rs2::config cfg;
	cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
	rs2::pipeline_profile selection = pipe.start(cfg);
	bool stop = false;
	int flag = 0;
	while (!stop)
	{
		flag++;
		//if(flag==50)
		{ 
		rs2::frameset frames;
		frames = pipe.wait_for_frames();
		auto color_frame = frames.get_color_frame();
		Mat color(Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
		src = color;
		//src = imread("43_Color.png");
		if (src/*frame*/.empty()) {
			cerr << "ERROR: could not grab a camera frame!" << endl;
			exit(1);
		}
		Mat grayscale, threshImg;
		ContoursVector myContour, allContours;
		cvtColor(src, grayscale, CV_BGR2GRAY);
		threshold(grayscale, threshImg, 70, 255, THRESH_BINARY_INV);
		//imshow("二值化", threshImg);
		//cvWaitKey(0);
		findContours(threshImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		myContour.clear();
		for (size_t i = 0; i < allContours.size(); i++)
		{
			int contourSize = allContours[i].size();
			if (contourSize > grayscale.cols / 25)/* 这里有一个筛选条件 */
			{
				myContour.push_back(allContours[i]);
			}
		}
		findCandidates(myContour, markers);
		recognizeMarkers1(grayscale, markers);
		findPointsOrder(markers);
		/*
		for (size_t i = 0; i < markers.size(); i++) {
			line(grayscale, markers[i].points[0], markers[i].points[1], Scalar(255), 2);
			line(grayscale, markers[i].points[1], markers[i].points[2], Scalar(255), 2);
			line(grayscale, markers[i].points[2], markers[i].points[3], Scalar(255), 2);
			line(grayscale, markers[i].points[3], markers[i].points[0], Scalar(255), 2);
		}
		imshow("检测结果", grayscale);
		cvWaitKey(0);
		*/

		//recognizeMarkers(grayscale, markers);
		if (markers.size() == 0) 
			imshow("线夹姿态测量", src);

		estimatePosition(markers, intrinsMatrix, distCoeff);
		flag = 0;
		}
		if (waitKey(10) >= 0)
			stop = true;
	}
	system("PAUSE");
	return 0;
}

void findCandidates(const ContoursVector& contours, vector<Marker>& detectedMarkers) {
	vector<Point> approxCurve;//返回结果为多边形，用点集表示//相似形状
	vector<Marker> possibleMarkers;//可能的标记

	for (size_t i = 0; i < contours.size(); i++) {
		double eps = contours[i].size()*0.05;
		approxPolyDP(contours[i], approxCurve, eps, true);

		if (approxCurve.size() != 4)
			continue;

		if (!isContourConvex(approxCurve)) 
			continue;
		//cout << approxCurve.size() << endl;
		float minDist = 1e5;
		float maxDist = 0.0;
		for (int i = 0; i < 4; i++) {
			Point side = approxCurve[i] - approxCurve[(i + 1) % 4];
			float squaredSideLength = side.dot(side);
			minDist = min(minDist, squaredSideLength);
			maxDist = max(maxDist, squaredSideLength);
		}

		if (minDist < src.cols/15 * src.cols/15)
			continue;
		//if (maxDist > src.cols / 5 * src.rows / 5)
		//	continue;

		Marker m;
		for (int i = 0; i < 4; i++) 
			m.points.push_back(Point2f(approxCurve[i].x, approxCurve[i].y));/* 这里给二维图像上的点赋值，单位应该是像素吧 */
		Point v1 = m.points[1] - m.points[0];
		Point v2 = m.points[2] - m.points[0];
		double o = (v1.x*v2.y) - (v1.y*v2.x);//两个向量叉乘积
		if (o < 0.0)
			swap(m.points[1], m.points[3]);
		possibleMarkers.push_back(m);
	}

	vector< pair<int , int > > tooNearCandidates;
	//cout << "luowei2" << possibleMarkers.size() << endl;
	for (size_t i = 0; i < possibleMarkers.size(); i++) {
		const Marker& m1 = possibleMarkers[i];
		for (size_t j = i+1; j < possibleMarkers.size(); j++) {
			const Marker& m2 = possibleMarkers[j];
			float distSquared = 0.0;
			for (int c = 0; c < 4; c++) {
				Point v = m1.points[c] - m2.points[c];
				distSquared = v.dot(v);
			}
			distSquared /= 4;

			if (distSquared < 100)
				tooNearCandidates.push_back(pair<int, int>(i, j));
		}
	}

	vector<bool> removalMask(possibleMarkers.size(), false);
	for (size_t i = 0; i < tooNearCandidates.size(); i++) {
		float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
		float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);
		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;
		removalMask[removalIndex] = true;
	}
	detectedMarkers.clear();
	for (size_t i = 0; i < possibleMarkers.size(); i++) {
		if (!removalMask[i])
			detectedMarkers.push_back(possibleMarkers[i]);
	}
	//cout << "flag2" << endl;
}

float perimeter(const vector<Point2f> &a)//求多边形周长。
{
	float sum = 0, dx, dy;
	for (size_t i = 0; i < a.size(); i++)
	{
		size_t i2 = (i + 1) % a.size();

		dx = a[i].x - a[i2].x;
		dy = a[i].y - a[i2].y;

		sum += sqrt(dx*dx + dy * dy);
	}

	return sum;
}

void recognizeMarkers1(const Mat& grayscale, vector<Marker>& detectedMarkers) {
	Mat canonicalMarkerImage;
	vector<Marker> goodMarkers;
	for (size_t i = 0; i < detectedMarkers.size(); i++) {
		Marker& marker = detectedMarkers[i];
		marker.markerTransform = getPerspectiveTransform(marker.points, m_markerCorners2d);
		warpPerspective(grayscale, canonicalMarkerImage, marker.markerTransform, markerSize);
		Mat markerImage = grayscale.clone();
		Mat markerSubImage = markerImage(boundingRect(marker.points));
		threshold(canonicalMarkerImage, canonicalMarkerImage, 100, 255, THRESH_BINARY | THRESH_OTSU);
		int cellSize = canonicalMarkerImage.rows / 7;
		/****** 0418@液晶谷 给Marker类加一个成员onepointflag ************/
		if (countNonZero(canonicalMarkerImage(Rect(cellSize, cellSize, cellSize, cellSize))) > (cellSize*cellSize) / 2)
			marker.no1Pointflag = 1;
		else if (countNonZero(canonicalMarkerImage(Rect(cellSize * 5, cellSize, cellSize, cellSize))) > (cellSize*cellSize) / 2)
			marker.no1Pointflag = 2;
		else if (countNonZero(canonicalMarkerImage(Rect(cellSize * 5, cellSize * 5, cellSize, cellSize))) > (cellSize*cellSize) / 2)
			marker.no1Pointflag = 3;
		else if (countNonZero(canonicalMarkerImage(Rect(cellSize, cellSize * 5, cellSize, cellSize))) > (cellSize*cellSize) / 2)
			marker.no1Pointflag = 4;
		/***************** 假如因为光线原因找不到白块可咋办？ ****************************/
		bool blackedge = true;
		int blackcount = 0;
		for (int y = 0; y < 7; y++) {
			int inc = 6;
			if (y == 0 || y == 6)
				inc = 1;
			for (int x = 0; x < 7; x += inc) {
				int cellX = x * cellSize;
				int cellY = y * cellSize;
				Mat cell = canonicalMarkerImage(Rect(cellX, cellY, cellSize, cellSize));
				int nZ = countNonZero(cell);
				if (nZ > (cellSize*cellSize) / 2)
				{
					blackedge = false;
				}
			}
		}
		/************* 20190416 于熊猫液晶谷 ***************************/
		for (int y = 0; y < 7; y++) {
			for (int x = 0; x < 7; x++) {
				int cellX = x * cellSize;
				int cellY = y * cellSize;
				Mat cell = canonicalMarkerImage(Rect(cellX, cellY, cellSize, cellSize));
				int nZ = countNonZero(cell);
				if (nZ < (cellSize*cellSize) / 2)
				{
					blackcount++;
				}
			}
		}
		/**************** 哈哈哈好饿啊怎么还不下班 **********************************/
		/* 检测到的标记是黑边且不是全黑的时候才判定为标记 */
		if (blackedge == true && blackcount < 49)
			goodMarkers.push_back(marker);
	}
	detectedMarkers = goodMarkers;
}

void recognizeMarkers(const Mat& grayscale, vector<Marker>& detectedMarkers) {
	Mat canonicalMarkerImage;
	//char name[20] = "";
	vector<Marker> goodMarkers;
	cout << "检测到" << detectedMarkers.size() << "个候选标记\n" << endl;
	for (size_t i = 0; i < detectedMarkers.size(); i++) {
		//cout << "flag1" << endl;
		Marker& marker = detectedMarkers[i];
		Mat markerTransform = getPerspectiveTransform(marker.points, m_markerCorners2d);
		cout << "\n" << markerTransform << endl;
		warpPerspective(grayscale, canonicalMarkerImage, markerTransform, markerSize);



		Mat markerImage = grayscale.clone();
		marker.drawContour(markerImage);
		Mat markerSubImage = markerImage(boundingRect(marker.points));

		//imshow("Source marker" + ToString(i), markerSubImage);
		char ttt[30]; sprintf_s(ttt, "sourcemarker%d.png", i);
		imwrite(ttt, markerSubImage);

		//imshow("Marker " + ToString(i), canonicalMarkerImage);
		char luowei[30]; sprintf_s(luowei, "Marker%d.png", i);
		imwrite(luowei, canonicalMarkerImage);


		int nRotations;
		int id = Marker::getMarkerId(canonicalMarkerImage, nRotations);
		
		if (id == -1) {
			continue;
		}
		cout << "ID = " << id << endl;

		if (id != -2) {
			marker.id = id;

			rotate(marker.points.begin(), marker.points.begin() + 4 - nRotations, marker.points.end());
			goodMarkers.push_back(marker);
		}
	}
	//cout << "kunsile" << goodMarkers.size() << endl;
	if (goodMarkers.size() > 0) {
		vector<Point2f> preciseCorners(4 * goodMarkers.size());
		for (size_t i = 0; i < goodMarkers.size(); i++) {
			Marker& marker = goodMarkers[i];
			for (int c = 0; c < 4; c++) {
				preciseCorners[i * 4 + c] = marker.points[c];
			}
		}

		TermCriteria termCriteria = TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 30, 0.01);
		cornerSubPix(grayscale, preciseCorners, cvSize(5, 5), cvSize(-1, -1), termCriteria);

		for (size_t i = 0; i < goodMarkers.size(); i++) {
			Marker& marker = goodMarkers[i];
			for (int c = 0; c < 4; c++) {
				marker.points[c] = preciseCorners[i * 4 + c];
			}
		}
	}

	Mat markerCornersMat(grayscale.size(), grayscale.type());
	markerCornersMat = Scalar(0);
	for (size_t i = 0; i < goodMarkers.size(); i++)
		goodMarkers[i].drawContour(markerCornersMat, Scalar(255));
	imwrite("refine.jpg", grayscale*0.5 + markerCornersMat);
	detectedMarkers = goodMarkers;
}

void estimatePosition(vector<Marker>& detectedMarkers, Mat_<float>& camMatrix, Mat_<float>& distCoeff) {
	for (size_t i = 0; i < detectedMarkers.size(); i++)
	{
		vector<Point3f> reference_ObjectPoint;
		vector<Point2f> reference_ImagePoint;
		reference_ObjectPoint.push_back(Point3f(0.0, 0.0, 0.0));
		reference_ObjectPoint.push_back(Point3f(5.0, 0.0, 0.0));
		reference_ObjectPoint.push_back(Point3f(0.0, 5.0, 0.0));
		reference_ObjectPoint.push_back(Point3f(0.0, 0.0, 5.0));
		
		//cout << "\n\n\n\n\n\n跳跳糖" << detectedMarkers.size() << endl;
		Marker& m = detectedMarkers[i];

		Mat Rvec;
		Mat_<float> Tvec;//Mat_<float>对应的是CV_32F
		Mat raux, taux;
		solvePnPRansac(m_markerCorners3d, m.points, camMatrix, distCoeff, raux, taux);
		projectPoints(reference_ObjectPoint, raux, taux, camMatrix, distCoeff, reference_ImagePoint);
		line(src, reference_ImagePoint[0], reference_ImagePoint[1], Scalar(0,0,255), 2);
		line(src, reference_ImagePoint[0], reference_ImagePoint[2], Scalar(0,255,0), 2);
		line(src, reference_ImagePoint[0], reference_ImagePoint[3], Scalar(255,0,0), 2);
		imshow("线夹姿态测量", src);
		raux.convertTo(Rvec, CV_32F);//转换Mat的保存类型，输出Rvec
		taux.convertTo(Tvec, CV_32F);
		Mat_<float> rotMat(3, 3);
		Rodrigues(Rvec, rotMat);
		cout << "旋转矩阵是：\n" << rotMat << endl;
		float theta_z = atan2(rotMat[1][0], rotMat[0][0]) * 180 / PI;
		float theta_y = atan2(-rotMat[2][0], sqrt(rotMat[2][0] * rotMat[2][0] + rotMat[2][2] * rotMat[2][2])) * 180 / PI;
		float theta_x = atan2(rotMat[2][1], rotMat[2][2]) * 180 / PI;
		//getEulerAngles(rotMat);
		cout << "姿态角：theta_z:" << theta_z << "  theta_y:" << theta_y << "  theta_x:" << theta_x << endl;
		cout << "\n" << endl;
		
		Matrix<float, Dynamic, Dynamic> R_n;
		Matrix<float, Dynamic, Dynamic> T_n;
		cv2eigen(rotMat, R_n);
		cv2eigen(Tvec, T_n);
		Vector3f P_oc;//目标在相机坐标系中的坐标
		P_oc = -R_n.inverse()*T_n;
		cout << "目标在相机坐标系中的坐标:\n" << P_oc <<
			"\n"<< "算出来距离是：" << sqrt(P_oc(0)* P_oc(0)+ P_oc(1) * P_oc(1)+ P_oc(2) * P_oc(2))<<"\n"
			<< "相机在目标坐标系中的坐标:\n" << T_n << 
			"\n" << "算出来距离是：" << sqrt(T_n(0) * T_n(0) + T_n(1) * T_n(1) + T_n(2) * T_n(2)) << "\n" << endl;
		cout << "-------------------------------------------------" << "\n" << endl;
	}
}
/* 确定四个点的起点 */
void findPointsOrder(vector<Marker>& detectedMarkers) {
	for (int i = 0; i < detectedMarkers.size(); i++) {
		int fflag;	float minLen = 1e5;
		Marker& marker = detectedMarkers[i];
		Matrix<float, Dynamic, Dynamic> perspectiveT;
		cv2eigen(marker.markerTransform, perspectiveT);
		vector<Point2f> perspectivePoint;
		for (int i = 0; i < 4; i++) {
			perspectivePoint.push_back(Point2f(marker.points[i].x*perspectiveT(0, 0) + marker.points[i].y*perspectiveT(0, 1) + perspectiveT(0, 2), marker.points[i].x*perspectiveT(1, 0) + marker.points[i].y*perspectiveT(1, 1) + perspectiveT(1, 2)));
		}
		switch (marker.no1Pointflag)
		{
		case 1:
			for (int j = 0; j < 4; j++) {
				if (abs(perspectivePoint[j].x) < 10.0 && abs(perspectivePoint[j].y) < 10.0)
					fflag = j;
			}
			break;
		case 2:
			for (int j = 0; j < 4; j++) {
				if (abs(perspectivePoint[j].x) - abs(perspectivePoint[j].y) > 50)
					fflag = j;
			}
			break;
		case 3:
			for (int j = 0; j < 4; j++) {
				if (abs(perspectivePoint[j].x) > 40.0 && abs(perspectivePoint[j].y) > 40.0)
					fflag = j;
			}
			break;
		case 4:
			for (int j = 0; j < 4; j++) {
				if (abs(perspectivePoint[j].y) - abs(perspectivePoint[j].x) > 50)
					fflag = j;
			}
			break;
		}

		if (fflag == 1) {
			Point2f temp;
			temp = marker.points[0];
			marker.points[0] = marker.points[1];
			marker.points[1] = marker.points[2];
			marker.points[2] = marker.points[3];
			marker.points[3] = temp;
		}
		if (fflag == 2) {
			Point2f temp1, temp2;
			temp1 = marker.points[0];
			temp2 = marker.points[1];
			marker.points[0] = marker.points[2];
			marker.points[1] = marker.points[3];
			marker.points[2] = temp1;
			marker.points[3] = temp2;
		}
		if (fflag == 3) {
			Point2f temp;
			temp = marker.points[3];
			marker.points[3] = marker.points[2];
			marker.points[2] = marker.points[1];
			marker.points[1] = marker.points[0];
			marker.points[0] = temp;
		}
	}
}

bool isRotationMatrix(cv::Mat &R)
{
	cv::Mat R_t;
	cv::transpose(R, R_t);
	cv::Mat shouldBeIdentity = R_t * R;
	cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());
	return norm(I, shouldBeIdentity) < 1e-6;
}

void getEulerAngles(cv::Mat matrix)
{

	//assert(isRotationMatrix(matrix));

	float sy = sqrt(matrix.at<double>(0, 0)*matrix.at<double>(0, 0) + matrix.at<double>(1, 0)*matrix.at<double>(1, 0));

	bool singular = sy < 1e-6;
	float theta_x, theta_y, theta_z;
	if (!singular)
	{
		theta_x = atan2(matrix.at<double>(2, 1), matrix.at<double>(2, 2));
		theta_x= theta_x*180.0/PI ;
		theta_y = atan2(-matrix.at<double>(2, 0), sy);
		theta_y= theta_y*180.0/PI ;
		theta_z = atan2(matrix.at<double>(1, 0), matrix.at<double>(0, 0));
		theta_z= theta_z*180.0/PI ;
	}
	else
	{
		theta_x = atan2(-matrix.at<double>(1, 2), matrix.at<double>(1, 1));
		//theta_x= theta_x*180.0/3.1416 ;
		theta_y = atan2(-matrix.at<double>(2, 0), sy);
		//theta_y= theta_y*180.0/3.1416 ;
		theta_z = 0;
		//theta_z= theta_z*180.0/3.1416 ;
	}
	cout << "theta_x:" << theta_x << ",theta_y:" << theta_y << ",theta_z:" << theta_z << endl;
}

void getEulerAngles2(cv::Mat &R, cv::Mat &t, cv::Mat &euler_angles)
{
	Mat camMatrix, rotMatrix, transVect, theta_x, theta_y, theta_z;
	Mat rotation_vec;
	Mat projMatrix = Mat(3, 4, CV_64FC1);
	//cv::Mat euler_angles 	= cv::Mat(3,1,CV_64FC1);
	Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
	Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
	Mat out_translation = cv::Mat(4, 1, CV_64FC1);
	hconcat(R, t, projMatrix);//将R、t拼接维投影矩阵 3*4
	decomposeProjectionMatrix(projMatrix, out_intrinsics, out_rotation, out_translation,
		cv::noArray(), cv::noArray(), cv::noArray(), euler_angles);
	//将投影矩阵分解为旋转矩阵和相机(内参)矩阵
	
}

