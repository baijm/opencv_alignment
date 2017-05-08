#ifndef _MY_DETECTOR_H
#define _MY_DETECTOR_H

#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#include "util.h"

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

	virtual bool estimate_affine_matrix(
		std::vector<cv::Point2f> &test_pts,
		std::vector<cv::Point2f> &ref_pts,
		cv::Mat &A_mat) = 0;
};

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

	bool estimate_affine_matrix(
		std::vector<cv::Point2f> &test_pts,
		std::vector<cv::Point2f> &ref_pts,
		cv::Mat &A_mat);
};

#endif // !_MY_DETECTOR_H
