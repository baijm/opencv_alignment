#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>

#include "my_detector.h"

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;


/*
最小化目标函数 \sum_{i=1}^N ||Mp_{i} - p_{i'}||_2
其中N为匹配数, p_{i}为测试图像中特征点坐标, p_{i'}为模板图像中特征点坐标
*/

const int MATCHES_NUM = 4;
const float PORTION_DIFF_THRES = 0.15;

int main()
{
	// 图像位置
	string test_im_dir = "D:/datasets/shelf/patch/query/easy/crop";
	string ref_im_dir = "D:/datasets/shelf/patch/ref/all/crop";

	// 图像名列表
	string test_name_file = "D:/datasets/shelf/patch/query/easy/img_list_bbx_crop.txt";
	string ref_name_file = "D:/datasets/shelf/patch/ref/all/img_list_3_per_class.txt";

	// 结果目录
	string res_dir = "D:/datasets/shelf/patch/query/easy/alignment/sift/cross_no_filter_reject_strict";
	string res_kp_dir = res_dir + "/keypoint"; // 保存特征提取结果
	string res_match_dir = res_dir + "/match"; // 保存匹配结果
	string res_align_dir = res_dir + "/result"; // 保存对齐结果
	string res_crop_dir = res_dir + "/crop"; // 保存裁剪结果

	// 特征类型
	string feature = "SIFT";
	// 特征点检测方法
	MyDetector detector(feature);

	// 匹配方法
	Ptr<DescriptorMatcher> matcher;
	if (feature == "BRISK")
	{
		matcher = BFMatcher::create(NORM_HAMMING, true);
	}
	else
	{
		matcher = BFMatcher::create(NORM_L2, true);
	}

	// 匹配筛选方法
	bool use_filter = false;
	MyMatchFilter *filter = NULL;
	if (use_filter)
	{
		if (feature == "BRISK")
		{
			filter = new DistThresFilter();
		}
		else
		{
			filter = new DistPortionFilter();
		}
	}

	// 求解仿射矩阵的方法
	string estimate = "RANSAC";
	MyAffineEstimator *estimator = NULL;
	if (estimate == "RANSAC")
	{
		estimator = new RansacAffineEstimator();
	}
	else
	{
		estimator = new Point4AffineEstimator();
	}

	/************************************************************************/
	/* 准备工作 : 检查目录是否存在, 读测试和模板图像名, 新建结果目录                                                                     */
	/************************************************************************/
	// 检查图像目录
	fs::path test_im_path(test_im_dir);
	if (!fs::exists(test_im_path))
	{
		cout << "test_im_dir not exist" << endl;
		return -1;
	}
	
	fs::path ref_im_path(ref_im_dir);
	if (!fs::exists(ref_im_path))
	{
		cout << "ref_im_dir not exist" << endl;
		return -1;
	}
	
	// 检查图像名列表文件是否存在, 如果存在则读图像名列表
	vector<string> test_names;
	fs::path test_name_path(test_name_file);
	if (!fs::exists(test_name_path) || !fs::is_regular_file(test_name_path))
	{
		cout << "test_name_file not exist or is not regular file" << endl;
		return -1;
	}
	else
	{
		ifstream txt(test_name_file);
		string line;
		while (getline(txt, line))
		{
			test_names.push_back(line);
		}
	}

	vector<string> ref_names;
	fs::path ref_name_path(ref_name_file);
	if (!fs::exists(ref_name_path) || !fs::is_regular_file(ref_name_path))
	{
		cout << "ref_name_file not exist or is not regular file" << endl;
		return -1;
	}
	else
	{
		ifstream txt(ref_name_file);
		string line;
		while (getline(txt, line))
		{
			ref_names.push_back(line);
		}
	}

	// 新建结果文件夹
	fs::path res_kp_path(res_kp_dir);
	if (!fs::exists(res_kp_path))
	{
		fs::create_directories(res_kp_path);
		cout << "res_kp_dir not exist, created" << endl;
	}

	fs::path res_match_path(res_match_dir);
	if (!fs::exists(res_match_path))
	{
		fs::create_directories(res_match_path);
		cout << "res_match_dir not exist, created" << endl;
	}

	fs::path res_align_path(res_align_dir);
	if (!fs::exists(res_align_path))
	{
		fs::create_directories(res_align_path);
		cout << "res_align_path not exist, created" << endl;
	}

	fs::path res_crop_path(res_crop_dir);
	if (!fs::exists(res_crop_path))
	{
		fs::create_directories(res_crop_path);
		cout << "res_crop_path not exist, created" << endl;
	}

	/************************************************************************/
	/* 对齐                                                                     */
	/************************************************************************/
	// 对每幅测试图像
	for (vector<string>::const_iterator test_iter = test_names.begin(); test_iter != test_names.end(); test_iter++)
	{
		// 读图像
		Mat test_im_c = imread(test_im_dir + '/' + *test_iter + ".jpg");
		Mat test_im_g;
		cvtColor(test_im_c, test_im_g, CV_BGR2GRAY);

		// 图像尺寸
		Size test_im_size = test_im_c.size();

		// 检测特征点
		vector<KeyPoint> test_kps;
		detector.detect(test_im_g, test_kps);

		// 画出所有特征点并保存
		Mat test_kps_im;
		drawKeypoints(test_im_c, test_kps, test_kps_im);
		fs::path test_kps_im_file = res_kp_path
			/ (*test_iter + ".jpg");
		imwrite(test_kps_im_file.string(), test_kps_im);

		// 计算特征向量
		Mat test_des;
		detector.compute(test_im_g, test_kps, test_des);

		// 对每幅模板图像
		for (vector<string>::const_iterator ref_iter = ref_names.begin(); ref_iter != ref_names.end(); ref_iter++)
		{
			cout << "test image : " << *test_iter << "\tref image : " << *ref_iter << endl;

			// 读图像
			Mat ref_im_c = imread(ref_im_dir + '/' + *ref_iter + ".jpg");
			Mat ref_im_g;
			cvtColor(ref_im_c, ref_im_g, CV_BGR2GRAY);

			// 图像尺寸
			Size ref_im_size = ref_im_c.size();

			// 检测特征点
			vector<KeyPoint> ref_kps;
			detector.detect(ref_im_g, ref_kps);

			cout << "\ttest image has " << test_kps.size() << " keypoints, ref image has " << ref_kps.size() << " keypoints" << endl;

			// 画出所有特征点并保存
			Mat ref_kps_im;
			drawKeypoints(ref_im_c, ref_kps, ref_kps_im);
			fs::path ref_kps_im_file = res_kp_path
				/ (*ref_iter + ".jpg");
			imwrite(ref_kps_im_file.string(), ref_kps_im);

			// 计算特征向量
			Mat ref_des;
			detector.compute(ref_im_g, ref_kps, ref_des);

			// 匹配
			vector<DMatch> matches;
			matcher->match(test_des, ref_des, matches);
			cout << "\t" << matches.size() << " matches" << endl;

			// 画出全部特征点匹配关系并保存
			Mat match_all_im;
			drawMatches(test_im_c, test_kps,
				ref_im_c, ref_kps,
				matches, match_all_im);
			fs::path match_all_im_file = res_match_path
				/ (*test_iter + "_" + *ref_iter + "_all.jpg");
			imwrite(match_all_im_file.string(), match_all_im);

			// 筛选匹配
			if (use_filter)
			{
				filter->filter(matches);
			}
			
			// 画出筛选后的特征点匹配关系并保存
			Mat match_filt_im;
			drawMatches(test_im_c, test_kps,
				ref_im_c, ref_kps,
				matches, match_filt_im);
			fs::path match_filt_im_file = res_match_path
				/ (*test_iter + "_" + *ref_iter + "_filter.jpg");
			imwrite(match_filt_im_file.string(), match_filt_im);

			/************************************************************************/
			/* 求仿射矩阵                                                                     */
			/************************************************************************/
			Mat A_mat;
			cout << "\t";
			if (!estimator->estimate_affine_matrix(test_kps, ref_kps, matches, A_mat))
			{
				cout << "failed to estimate affine matrix" << endl;
			}
			Point2f test_center = estimator->test_center;
			Point2f ref_center = estimator->ref_center;

			// 画出inliers并保存
			Mat match_inlier_im;
			drawMatches(test_im_c, test_kps,
				ref_im_c, ref_kps,
				estimator->inliers, match_inlier_im);
			fs::path match_inlier_im_file = res_match_path
				/ (*test_iter + "_" + *ref_iter + "_inliers.jpg");
			imwrite(match_inlier_im_file.string(), match_inlier_im);

			/************************************************************************/
			/* 变换测试图像                                                                     */
			/************************************************************************/
			// 变换前, 测试图像原点移动到中心
			Mat T_mat_pre = Mat::zeros(3, 3, CV_64F);
			T_mat_pre.at<double>(0, 0) = T_mat_pre.at<double>(1, 1) = T_mat_pre.at<double>(2, 2) = 1;
			T_mat_pre.at<double>(0, 2) = -test_center.x;
			T_mat_pre.at<double>(1, 2) = -test_center.y;

			// 优化后的仿射矩阵
			Mat A_mat_h = Mat::zeros(3, 3, A_mat.type());
			A_mat_h.at<double>(0, 0) = A_mat.at<double>(0, 0);
			A_mat_h.at<double>(0, 1) = A_mat.at<double>(0, 1);
			A_mat_h.at<double>(0, 2) = A_mat.at<double>(0, 2);
			A_mat_h.at<double>(1, 0) = A_mat.at<double>(1, 0);
			A_mat_h.at<double>(1, 1) = A_mat.at<double>(1, 1);
			A_mat_h.at<double>(1, 2) = A_mat.at<double>(1, 2);
			A_mat_h.at<double>(2, 2) = 1;

			// 变换后, 测试图像原点移动到模板图像中心
			Mat T_mat_post = Mat::zeros(3, 3, CV_64F);
			T_mat_post.at<double>(0, 0) = T_mat_post.at<double>(1, 1) = T_mat_post.at<double>(2, 2) = 1;
			T_mat_post.at<double>(0, 2) = ref_center.x;
			T_mat_post.at<double>(1, 2) = ref_center.y;

			// 组合变换矩阵
			Mat M_mat_h = T_mat_post * (A_mat_h * T_mat_pre);
			Mat M_mat = M_mat_h(Range(0, 2), Range(0, 3));

			// 仿射变换
			Mat M_im;
			warpAffine(test_im_c, M_im, M_mat, ref_im_size);
			namedWindow("after affine");
			imshow("after affine", M_im);
			fs::path align_im_file = res_align_path / (*test_iter + "_" + *ref_iter + ".jpg");
			imwrite(align_im_file.string(), M_im);
		
			/************************************************************************/
			/* 拒绝不合理的变换, 用变换后的裁剪测试图像和模板图像                                                                     */
			/************************************************************************/
			cout << "\t";

			// 求测试图像4个角变换后的坐标
			vector<Point2f> test_corners(4);
			test_corners[0] = Point2f(0, 0); // 左上
			test_corners[1] = Point2f(0, test_im_size.height); // 左下
			test_corners[2] = Point2f(test_im_size.width, test_im_size.height); // 右下
			test_corners[3] = Point2f(test_im_size.width, 0); // 右上
			transform(test_corners, test_corners, M_mat);

			// 如果x方向顺序不对则拒绝
			if (!(test_corners[0].x < min(test_corners[2].x, test_corners[3].x))
				||
				!(test_corners[1].x < min(test_corners[2].x, test_corners[3].x)))
			{
				cout << "rejected because wrong relative position in x direction" << endl;
				continue;
			}

			// 如果y方向顺序不对则拒绝
			if (!(test_corners[0].y < min(test_corners[1].y, test_corners[2].y))
				||
				!(test_corners[3].y < min(test_corners[1].y, test_corners[2].y)))
			{
				cout << "rejected because wrong relative position in y direction" << endl;
				continue;
			}

			// 裁剪
			vector<int> xs(4), ys(4);
			for (int ci = 0; ci < test_corners.size(); ci++)
			{
				xs[ci] = test_corners[ci].x;
				ys[ci] = test_corners[ci].y;
			}
			sort(xs.begin(), xs.end());
			sort(ys.begin(), ys.end());

			int start_r = min(max(ys[1], 0), ref_im_size.height);
			int start_c = min(max(xs[1], 0), ref_im_size.width);
			int end_r = max(min(ys[2], ref_im_size.height), 0);
			int end_c = max(min(xs[2], ref_im_size.width), 0);

			Mat test_crop_im = M_im(Range(start_r, end_r), Range(start_c, end_c));
			fs::path test_crop_file = res_crop_path / (*test_iter + "_" + *ref_iter + "_test_crop.jpg");
			imwrite(test_crop_file.string(), test_crop_im);

			Mat ref_crop_im = ref_im_c(Range(start_r, end_r), Range(start_c, end_c));
			fs::path ref_crop_file = res_crop_path / (*test_iter + "_" + *ref_iter + "_ref_crop.jpg");
			imwrite(ref_crop_file.string(), ref_crop_im);

			waitKey();
			cout << endl;
		}
	}

	destroyAllWindows();

	return 0;
}
