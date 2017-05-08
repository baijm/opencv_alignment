#ifndef _MY_DETECTOR_H
#define _MY_DETECTOR_H

#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#include "util.h"

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

	virtual bool estimate_affine_matrix(
		std::vector<cv::Point2f> &test_pts,
		std::vector<cv::Point2f> &ref_pts,
		cv::Mat &A_mat) = 0;
};

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

	bool estimate_affine_matrix(
		std::vector<cv::Point2f> &test_pts,
		std::vector<cv::Point2f> &ref_pts,
		cv::Mat &A_mat);
};

#endif // !_MY_DETECTOR_H
