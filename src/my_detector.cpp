#include "my_detector.h"

// 用RANSAC求仿射矩阵
bool RansacAffineEstimator::estimate_affine_matrix(
	std::vector<cv::KeyPoint> &test_kps, std::vector<cv::KeyPoint> &ref_kps,
	std::vector<cv::DMatch> &matches,
	cv::Mat &A_mat)
{
	// 如果小于3个则返回
	if (matches.size() < 3)
	{
		std::cout << "less than 3 points, abort estimating affine matrix" << std::endl;
		return false;
	}

	// 求图像中心
	float left_x_test = 0, right_x_test = 0, top_y_test = 0, bottom_y_test = 0;
	float left_x_ref = 0, right_x_ref = 0, top_y_ref = 0, bottom_y_ref = 0;
	std::vector<cv::Point2f> test_pts, ref_pts;

	for (int i = 0; i < matches.size(); i++)
	{
		Point2f test_p = test_kps[matches[i].queryIdx].pt;
		Point2f ref_p = ref_kps[matches[i].trainIdx].pt;

		test_pts.push_back(test_p);
		ref_pts.push_back(ref_p);

		left_x_test = (test_p.x < left_x_test || left_x_test == 0) ? test_p.x : left_x_test;
		right_x_test = (test_p.x > right_x_test) ? test_p.x : right_x_test;
		top_y_test = (test_p.y < top_y_test || top_y_test == 0) ? test_p.y : top_y_test;
		bottom_y_test = (test_p.y > bottom_y_test) ? test_p.y : bottom_y_test;

		left_x_ref = (ref_p.x < left_x_ref || left_x_ref == 0) ? ref_p.x : left_x_ref;
		right_x_ref = (ref_p.x > right_x_ref) ? ref_p.x : right_x_ref;
		top_y_ref = (ref_p.y < top_y_ref || top_y_ref == 0) ? ref_p.y : top_y_ref;
		bottom_y_ref = (ref_p.y > bottom_y_ref) ? ref_p.y : bottom_y_ref;
	}

	test_center.x = (right_x_test + left_x_test) / 2;
	test_center.y = (bottom_y_test + top_y_test) / 2;
	ref_center.x = (right_x_ref + left_x_ref) / 2;
	ref_center.y = (bottom_y_ref + top_y_ref) / 2;

	// 特征点坐标 - 中心坐标
	for (int i = 0; i < test_pts.size(); i++)
	{
		test_pts[i].x -= test_center.x;
		test_pts[i].y -= test_center.y;
	}
	for (int i = 0; i < ref_pts.size(); i++)
	{
		ref_pts[i].x -= ref_center.x;
		ref_pts[i].y -= ref_center.y;
	}

	// 求仿射矩阵
	std::vector<uchar> stat;
	A_mat = estimateAffine2D(test_pts, ref_pts, stat,
		RANSAC, reproj_thres);
	//std::cout << "found " << countNonZero(stat) << " inliers" << std::endl;

	// 保存inliers
	inliers.clear();
	for (int i = 0; i < matches.size(); i++)
	{
		if (stat[i] != 0)
		{
			inliers.push_back(matches[i]);
		}
	}

	return true;
}

bool RansacAffineEstimator::estimate_affine_matrix(
	std::vector<cv::Point2f> &test_pts,
	std::vector<cv::Point2f> &ref_pts,
	cv::Mat &A_mat)
{
	// 如果test_pts和ref_pts个数不同则返回
	if (test_pts.size() != ref_pts.size())
	{
		std::cout << "test_pts.size() != ref_pts.size(), abort estimating affine matrix" << std::endl;
		return false;
	}

	// 如果小于3个则返回
	if (test_pts.size() < 3)
	{
		std::cout << "less than 3 points, abort estimating affine matrix" << std::endl;
		return false;
	}

	// 求图像中心
	float left_x_test = 0, right_x_test = 0, top_y_test = 0, bottom_y_test = 0;
	float left_x_ref = 0, right_x_ref = 0, top_y_ref = 0, bottom_y_ref = 0;
	vector<Point2f> test_pts_copy(test_pts), ref_pts_copy(ref_pts);

	for (int i = 0; i < test_pts_copy.size(); i++)
	{
		Point2f test_p = test_pts_copy[i];
		Point2f ref_p = ref_pts_copy[i];

		left_x_test = (test_p.x < left_x_test || left_x_test == 0) ? test_p.x : left_x_test;
		right_x_test = (test_p.x > right_x_test) ? test_p.x : right_x_test;
		top_y_test = (test_p.y < top_y_test || top_y_test == 0) ? test_p.y : top_y_test;
		bottom_y_test = (test_p.y > bottom_y_test) ? test_p.y : bottom_y_test;

		left_x_ref = (ref_p.x < left_x_ref || left_x_ref == 0) ? ref_p.x : left_x_ref;
		right_x_ref = (ref_p.x > right_x_ref) ? ref_p.x : right_x_ref;
		top_y_ref = (ref_p.y < top_y_ref || top_y_ref == 0) ? ref_p.y : top_y_ref;
		bottom_y_ref = (ref_p.y > bottom_y_ref) ? ref_p.y : bottom_y_ref;
	}

	test_center.x = (right_x_test + left_x_test) / 2;
	test_center.y = (bottom_y_test + top_y_test) / 2;
	ref_center.x = (right_x_ref + left_x_ref) / 2;
	ref_center.y = (bottom_y_ref + top_y_ref) / 2;

	// 特征点坐标 - 中心坐标
	for (int i = 0; i < test_pts_copy.size(); i++)
	{
		test_pts_copy[i].x -= test_center.x;
		test_pts_copy[i].y -= test_center.y;
	}
	for (int i = 0; i < ref_pts_copy.size(); i++)
	{
		ref_pts_copy[i].x -= ref_center.x;
		ref_pts_copy[i].y -= ref_center.y;
	}

	// 求仿射矩阵
	std::vector<uchar> stat;
	A_mat = estimateAffine2D(test_pts_copy, ref_pts_copy, stat,
		RANSAC, reproj_thres);

	// 保存inliers
	inliers.clear();
	for (int i = 0; i < test_pts_copy.size(); i++)
	{
		if (stat[i] != 0)
		{
			inliers.push_back(DMatch(i, i, 0));
		}
	}

	return true;
}
