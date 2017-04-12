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

// 特征检测器封装
class MyDetector
{
private:
	cv::Ptr<cv::FeatureDetector> detector;

public:
	// 根据特征种类初始化detector
	MyDetector(std::string type);

	// 检测特征点
	void detect(cv::Mat &im, std::vector<cv::KeyPoint> &kps);

	// 计算特征向量
	void compute(cv::Mat &im, std::vector<cv::KeyPoint> &kps, cv::Mat &des);
};

// 匹配过滤器基类
class MyMatchFilter
{
public:
	// 过滤匹配
	virtual void filter(std::vector<cv::DMatch> &matches) = 0;
};

// 根据最大距离的比例过滤(用于SIFT)
class DistPortionFilter : public MyMatchFilter
{
private:
	// DMatch中的最小和最大距离初值
	float min_dist_start, max_dist_start; 
	// 最大距离比例
	float portion;

public:
	DistPortionFilter(float min_d = 100, float max_d = 0, float p = 0.31)
		:min_dist_start(min_d), max_dist_start(max_d), portion(p) {};

	void filter(std::vector<cv::DMatch> &matches);
};

// 根据距离阈值过滤(用于BRISK)
class DistThresFilter : public MyMatchFilter
{
private:
	// 距离阈值
	int thres;

public:
	DistThresFilter(int th = 90)
		:thres(th) {};

	void filter(std::vector<cv::DMatch> &matches);
};

// 根据位置过滤(用于所有特征)
class PosPortionFilter : public MyMatchFilter
{
private:
	// 位置比例阈值
	float thres;
	// 图像尺寸
	cv::Size test_size, ref_size;

public:
	PosPortionFilter(cv::Size ts, cv::Size rs, float th = 0.2)
		:test_size(ts), ref_size(rs), thres(th) {};

	void filter(std::vector<cv::DMatch> &matches);

	void filter(std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &test_kps, std::vector<cv::KeyPoint> &ref_kps);
};

// 求解仿射矩阵的基类
class MyAffineEstimator 
{
public:
	// 返回的inlier
	std::vector<cv::DMatch> inliers;

	// 图像中心坐标
	cv::Point2f test_center, ref_center;

	// 求解仿射矩阵, 返回是否成功
	virtual bool estimate_affine_matrix(
		std::vector<cv::KeyPoint> &test_kps, 
		std::vector<cv::KeyPoint> &ref_kps, 
		std::vector<cv::DMatch> &matches,
		cv::Mat &A_mat) = 0;
};

/*
// 用上下左右4个点求仿射矩阵
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

// 用RANSAC求仿射矩阵
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
