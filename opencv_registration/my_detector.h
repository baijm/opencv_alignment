#ifndef _MY_DETECTOR_H
#define _MY_DETECTOR_H

//#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/features2d.hpp>

//#include <nlopt.hpp>

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#include "util.h"

// �����������װ
class MyDetector
{
private:
	cv::Ptr<cv::FeatureDetector> detector;

public:
	// �������������ʼ��detector
	MyDetector(std::string type);

	// ���������
	void detect(cv::Mat &im, std::vector<cv::KeyPoint> &kps);

	// ������������
	void compute(cv::Mat &im, std::vector<cv::KeyPoint> &kps, cv::Mat &des);
};

// ƥ�����������
class MyMatchFilter
{
public:
	// ����ƥ��
	virtual void filter(std::vector<cv::DMatch> &matches) = 0;
};

// ����������ı�������(����SIFT)
class DistPortionFilter : public MyMatchFilter
{
private:
	// DMatch�е���С���������ֵ
	float min_dist_start, max_dist_start; 
	// ���������
	float portion;

public:
	DistPortionFilter(float min_d = 100, float max_d = 0, float p = 0.31)
		:min_dist_start(min_d), max_dist_start(max_d), portion(p) {};

	void filter(std::vector<cv::DMatch> &matches);
};

// ���ݾ�����ֵ����(����BRISK)
class DistThresFilter : public MyMatchFilter
{
private:
	// ������ֵ
	int thres;

public:
	DistThresFilter(int th = 90)
		:thres(th) {};

	void filter(std::vector<cv::DMatch> &matches);
};

// ����λ�ù���(������������)
class PosPortionFilter : public MyMatchFilter
{
private:
	// λ�ñ�����ֵ
	float thres;
	// ͼ��ߴ�
	cv::Size test_size, ref_size;

public:
	PosPortionFilter(cv::Size ts, cv::Size rs, float th = 0.2)
		:test_size(ts), ref_size(rs), thres(th) {};

	void filter(std::vector<cv::DMatch> &matches);

	void filter(std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &test_kps, std::vector<cv::KeyPoint> &ref_kps);
};

// ���������Ļ���
class MyAffineEstimator 
{
public:
	// ���ص�inlier
	std::vector<cv::DMatch> inliers;

	// ͼ����������
	cv::Point2f test_center, ref_center;

	// ���������, �����Ƿ�ɹ�
	virtual bool estimate_affine_matrix(
		std::vector<cv::KeyPoint> &test_kps, 
		std::vector<cv::KeyPoint> &ref_kps, 
		std::vector<cv::DMatch> &matches,
		cv::Mat &A_mat) = 0;
};

/*
// ����������4������������
class Point4AffineEstimator : public MyAffineEstimator
{
private:
double reproj_thres;

public:
Point4AffineEstimator(double r_th = 3.0) : reproj_thres(r_th) {};

bool estimate_affine_matrix(
std::vector<cv::KeyPoint> &test_kps,
std::vector<cv::KeyPoint> &ref_kps,
std::vector<cv::DMatch> &matches,
cv::Mat &A_mat);
};
*/

// ��RANSAC��������
class RansacAffineEstimator : public MyAffineEstimator
{
private:
	double reproj_thres;

public:
	RansacAffineEstimator(double r_th = 3.0) : reproj_thres(r_th) {};

	bool estimate_affine_matrix(
		std::vector<cv::KeyPoint> &test_kps, 
		std::vector<cv::KeyPoint> &ref_kps, 
		std::vector<cv::DMatch> &matches,
		cv::Mat &A_mat);
};

#endif // !_MY_DETECTOR_H
