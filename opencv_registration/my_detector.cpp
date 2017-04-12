#include "my_detector.h"

// �������������ʼ��detector
MyDetector::MyDetector(std::string type)
{
	if (type == "SIFT")
	{
		detector = cv::xfeatures2d::SIFT::create();
	}
	if (type == "BRISK")
	{
		detector = cv::BRISK::create();
	}
}

// ���������
void MyDetector::detect(cv::Mat &im, std::vector<cv::KeyPoint> &kps)
{
	detector->detect(im, kps);
}

// ������������
void MyDetector::compute(cv::Mat &im, std::vector<cv::KeyPoint> &kps, cv::Mat &des)
{
	detector->compute(im, kps, des);
}

// ƥ�������
// ����������ı�������(����SIFT)
void DistPortionFilter::filter(std::vector<cv::DMatch> &matches)
{
	// ����ƥ���е�������С����
	float min_dist = min_dist_start, max_dist = max_dist_start;
	for (std::vector<cv::DMatch>::iterator ite = matches.begin(); ite != matches.end(); ite++)
	{
		min_dist = std::min(min_dist, ite->distance);
		max_dist = std::max(max_dist, ite->distance);
	}
	std::cout << "filter by portion of maximum distance :"
		<< " max_dist = " << max_dist
		<< ", min_dist = " << min_dist << std::endl;
	std::cout << "\t"
		<< matches.size() << " matches before filter";
	
	for (std::vector<cv::DMatch>::iterator ite = matches.begin(); ite != matches.end(); )
	{
		if (ite->distance > portion * max_dist)
		{
			ite = matches.erase(ite);
		}
		else
		{
			ite++;
		}
	}
	std::cout << "\t"
		<< matches.size() << " matches after filter" << std::endl;
}

// ���ݾ�����ֵ����(����BRISK)
void DistThresFilter::filter(std::vector<cv::DMatch> &matches)
{
	std::cout << "filter by distance threshold : "
		<< matches.size() << " matches before filter";

	for (std::vector<cv::DMatch>::iterator ite = matches.begin(); ite != matches.end(); )
	{
		if (ite->distance > thres)
		{
			ite = matches.erase(ite);
		}
		else
		{
			ite++;
		}
	}
	std::cout << "\t"
		<< matches.size() << " matches after filter" << std::endl;
}

// ����λ�ù���(������������)
void PosPortionFilter::filter(std::vector<cv::DMatch> &matches)
{
}

void PosPortionFilter::filter(std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &test_kps, std::vector<cv::KeyPoint> &ref_kps)
{
	std::cout << "filter by distance threshold : "
		<< matches.size() << " matches before filter";

	// ɸѡƥ��: ����������λ�������ͼ��ߴ�ı���, ������̫����ȥ��
	for (std::vector<cv::DMatch>::iterator ite = matches.begin(); ite != matches.end(); )
	{
		cv::Point2f test_p(
			test_kps[ite->queryIdx].pt.x / test_size.width,
			test_kps[ite->queryIdx].pt.y / test_size.height
		);
		cv::Point2f ref_p(
			ref_kps[ite->trainIdx].pt.x / ref_size.width,
			ref_kps[ite->trainIdx].pt.y / ref_size.height
		);

		if (std::abs(test_p.x - ref_p.x) > thres
			|| std::abs(test_p.y - ref_p.y) > thres)
		{
			matches.erase(ite);
		}
		else
		{
			ite++;
		}
	}

	std::cout << "\t"
		<< matches.size() << " matches after filter" << std::endl;
}

// ����������4������������
/*
bool Point4AffineEstimator::estimate_affine_matrix(
	std::vector<cv::KeyPoint> &test_kps, std::vector<cv::KeyPoint> &ref_kps,
	std::vector<cv::DMatch> &matches,
	cv::Mat &A_mat)
{
	// ���С��3���򷵻�
	if (matches.size() < 3)
	{
		std::cout << "less than 3 points, abort estimating affine matrix" << std::endl;
		return false;
	}

	// ѡ����������4��ƥ��
	float left_x_test = 0, right_x_test = 0, top_y_test = 0, bottom_y_test = 0;
	float left_x_ref = 0, right_x_ref = 0, top_y_ref = 0, bottom_y_ref = 0;
	int left_match_idx = -1, right_match_idx = -1, top_match_idx = -1, bottom_match_idx = -1;
	for (int i = 0; i < matches.size(); i++)
	{
		if (test_kps[matches[i].queryIdx].pt.x < left_x_test || left_x_test == 0)
		{
			left_x_test = test_kps[matches[i].queryIdx].pt.x;
			left_x_ref = ref_kps[matches[i].trainIdx].pt.x;
			left_match_idx = i;
		}
		if (test_kps[matches[i].queryIdx].pt.x > right_x_test)
		{
			right_x_test = test_kps[matches[i].queryIdx].pt.x;
			right_x_ref = ref_kps[matches[i].trainIdx].pt.x;
			right_match_idx = i;
		}
		if (test_kps[matches[i].queryIdx].pt.y < top_y_test || top_y_test == 0)
		{
			top_y_test = test_kps[matches[i].queryIdx].pt.y;
			top_y_ref = ref_kps[matches[i].trainIdx].pt.y;
			top_match_idx = i;
		}
		if (test_kps[matches[i].queryIdx].pt.y > bottom_y_test)
		{
			bottom_y_test = test_kps[matches[i].queryIdx].pt.y;
			bottom_y_ref = ref_kps[matches[i].trainIdx].pt.y;
			bottom_match_idx = i;
		}
	}

	// �������MATCHES_NUM��������
	if ((left_match_idx == right_match_idx) || (left_match_idx == top_match_idx) || (left_match_idx == bottom_match_idx)
		|| (right_match_idx == top_match_idx) || (right_match_idx == bottom_match_idx)
		|| (top_match_idx == bottom_match_idx))
	{
		std::cout << "less than 4 distinct points, abort estimating affine matrix" << std::endl;
		return false;
	}

	// ����inliers
	inliers.clear();
	inliers.push_back(matches[left_match_idx]);
	inliers.push_back(matches[right_match_idx]);
	inliers.push_back(matches[top_match_idx]);
	inliers.push_back(matches[bottom_match_idx]);

	// ��ͼ������
	test_center.x = (right_x_test + left_x_test) / 2;
	test_center.y = (bottom_y_test + top_y_test) / 2;
	ref_center.x = (right_x_ref + left_x_ref) / 2;
	ref_center.y = (bottom_y_ref + top_y_ref) / 2;

	// ���������� - ��������
	std::vector<cv::Point2f> test_pts, ref_pts;
	cv::KeyPoint::convert(test_kps, test_pts);
	cv::KeyPoint::convert(ref_kps, ref_pts);
	std::vector<cv::Point2f> test_3(3), ref_3(3);
	for (int i = 0; i < inliers.size(); i++)
	{
		test_pts[inliers[i].queryIdx].x -= test_center.x;
		test_pts[inliers[i].queryIdx].y -= test_center.y;
		ref_pts[inliers[i].trainIdx].x -= ref_center.x;
		ref_pts[inliers[i].trainIdx].y -= ref_center.y;

		if (i < 3)
		{
			test_3[i] = test_pts[inliers[i].queryIdx];
			ref_3[i] = ref_pts[inliers[i].trainIdx];
		}
	}

	// ���������ֵ
	A_mat = cv::getAffineTransform(test_3, ref_3);

	// ��MATCHES_NUM��ƥ���Ż�
	std::vector<double> a(6);
	a[0] = A_mat.at<double>(0, 0);
	a[1] = A_mat.at<double>(0, 1);
	a[2] = A_mat.at<double>(0, 2);
	a[3] = A_mat.at<double>(1, 0);
	a[4] = A_mat.at<double>(1, 1);
	a[5] = A_mat.at<double>(1, 2);

	ObjectiveFunctionData* obj_func_data =
		new ObjectiveFunctionData(inliers, test_pts, ref_pts);
	nlopt::opt opt(nlopt::LD_MMA, 6);
	opt.set_min_objective(obj_func, obj_func_data);
	opt.set_ftol_abs(0.5);
	opt.set_stopval(reproj_thres * inliers.size());
	double min_f;
	nlopt::result result = opt.optimize(a, min_f);
	if (result < 0)
	{
		std::cout << "optimization failed" << std::endl;
		return false;
	}
	else
	{
		std::cout << "optimization succeeded, value of objective function = " << min_f << std::endl;

		A_mat.at<double>(0, 0) = a[0];
		A_mat.at<double>(0, 1) = a[1];
		A_mat.at<double>(0, 2) = a[2];
		A_mat.at<double>(1, 0) = a[3];
		A_mat.at<double>(1, 1) = a[4];
		A_mat.at<double>(1, 2) = a[5];
		
		return true;
	}
}
*/

// ��RANSAC��������
bool RansacAffineEstimator::estimate_affine_matrix(
	std::vector<cv::KeyPoint> &test_kps, std::vector<cv::KeyPoint> &ref_kps,
	std::vector<cv::DMatch> &matches,
	cv::Mat &A_mat)
{
	// ���С��3���򷵻�
	if (matches.size() < 3)
	{
		std::cout << "less than 3 points, abort estimating affine matrix" << std::endl;
		return false;
	}

	// ��ͼ������
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

	// ���������� - ��������
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

	// ��������
	std::vector<uchar> stat;
	A_mat = estimateAffine2D(test_pts, ref_pts, stat,
		RANSAC, reproj_thres);
	//std::cout << "found " << countNonZero(stat) << " inliers" << std::endl;

	// ����inliers
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