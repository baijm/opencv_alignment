#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>

//#include <nlopt.hpp>

#include <iostream>
#include <fstream>

#include "util.h"

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;


/*
最小化目标函数 \sum_{i=1}^N ||Mp_{i} - p_{i'}||_2
其中N为匹配数, p_{i}为测试图像中特征点坐标, p_{i'}为模板图像中特征点坐标
*/
/*
int opt_iters = 0;

// 目标函数
// a = {a11, a12, a13, a21, a22, a23}
double obj_func(const vector<double> &a, vector<double> &grad, void *func_data)
{
opt_iters++;

ObjectiveFunctionData *data = reinterpret_cast<ObjectiveFunctionData*>(func_data);
// TODO : 用引用给引用赋值会怎么样??
vector<MatchKpsSim> *matches = data->matches;
vector<Point2f> *test_pts = data->test_pts;
vector<Point2f> *ref_pts = data->tmpl_pts;

if (!grad.empty())
{
for (int gi = 0; gi < grad.size(); gi++)
{
grad[gi] = 0;
}
}

// 求目标函数的值
double val = 0;
for (int i = 0; i < matches->size(); i++)
{
Point2f p_test = (*test_pts)[(*matches)[i].test_pid];
Point2f p_ref = (*ref_pts)[(*matches)[i].tmpl_pid];

Point2f p_test_new(
a[0] * p_test.x + a[1] * p_test.y + a[2],
a[3] * p_test.x + a[4] * p_test.y + a[5]
);

double diff_x_i = p_test_new.x - p_ref.x;
double diff_y_i = p_test_new.y - p_ref.y;
double dist_i = sqrt(pow(diff_x_i, 2) + pow(diff_y_i, 2));

val += dist_i;

if (!grad.empty() && dist_i > 0)
{
grad[0] += (diff_x_i / dist_i)*p_test.x;
grad[1] += (diff_x_i / dist_i)*p_test.y;
grad[2] += (diff_x_i / dist_i);
grad[3] += (diff_y_i / dist_i)*p_test.x;
grad[4] += (diff_y_i / dist_i)*p_test.y;
grad[5] += (diff_y_i / dist_i);
}
}

return val;
}

// main中代码片段
vector<double> a(6);
a[0] = A_mat.at<double>(0, 0);
a[1] = A_mat.at<double>(0, 1);
a[2] = A_mat.at<double>(0, 2);
a[3] = A_mat.at<double>(1, 0);
a[4] = A_mat.at<double>(1, 1);
a[5] = A_mat.at<double>(1, 2);

ObjectiveFunctionData* obj_func_data =
new ObjectiveFunctionData(all_inliers, test_kps, ref_kps);
nlopt::opt opt(nlopt::LD_MMA, 6);
opt.set_min_objective(obj_func, obj_func_data);
opt.set_ftol_abs(0.5);
opt.set_stopval(3.0 * matches.size());

double min_f;
opt_iters = 0;
nlopt::result result = opt.optimize(a, min_f);
*/

const double RANSAC_REPROJ_THRES = 3.0;

int main()
{
	// 特征点txt位置
	string test_kp_dir = "D:/datasets/shelf/patch/query/easy/crop_brisk"; // 测试
	string ref_kp_dir = "D:/datasets/shelf/patch/ref/all/crop_brisk"; // 模板

	// 图像位置
	string test_im_dir = "D:/datasets/shelf/patch/query/easy/crop";
	string ref_im_dir = "D:/datasets/shelf/patch/ref/all/crop";

	// word txt位置
	string word_dir = "D:/datasets/shelf/patch/query/easy/word/crop2crop_3";

	// 图像名列表
	string test_name_file = "D:/datasets/shelf/patch/query/easy/img_list_bbx_crop.txt";
	string ref_name_file = "D:/datasets/shelf/patch/ref/all/img_list_3_per_class.txt";

	// 结果目录
	string res_dir = "D:/datasets/shelf/patch/query/easy/alignment/brisk/ransac/center0_combine_thres3";
	string res_match_dir = res_dir + "/match"; // 保存匹配结果
	string res_align_dir = res_dir + "/result"; // 保存对齐结果
	string res_crop_dir = res_dir + "/crop"; // 保存裁剪结果

	/************************************************************************/
	/* 准备工作 : 检查目录是否存在, 读测试和模板图像名, 新建结果目录                                                                     */
	/************************************************************************/
	// 检查特征点txt目录
	fs::path test_kp_path(test_kp_dir);
	if (!fs::exists(test_kp_path))
	{
		cout << "test_kp_dir not exist" << endl;
		return -1;
	}
	
	fs::path ref_kp_path(ref_kp_dir);
	if (!fs::exists(ref_kp_path))
	{
		cout << "ref_kp_dir not exist" << endl;
		return -1;
	}
	
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
	
	// 检查word txt目录是否存在
	fs::path word_path(word_dir);
	if (!fs::exists(word_path))
	{
		cout << "word_dir not exist" << endl;
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
		cout << "test image : " << *test_iter;

		// 读图像
		Mat test_im = imread(test_im_dir + '/' + *test_iter + ".jpg");

		// 读特征点坐标
		vector<Point2f> test_kps;
		load_kp_pos_txt(test_kp_dir + '/' + *test_iter + "_kp.txt", test_kps);
		cout << ", " << test_kps.size() << " keypoints" << endl;

		// 对每幅模板图像
		for (vector<string>::const_iterator ref_iter = ref_names.begin(); ref_iter != ref_names.end(); ref_iter++)
		{
			cout << "\tref image : " << *ref_iter;

			// 读图像
			Mat ref_im = imread(ref_im_dir + '/' + *ref_iter + ".jpg");

			// 读特征点坐标
			vector<Point2f> ref_kps;
			load_kp_pos_txt(ref_kp_dir + '/' + *ref_iter + "_kp.txt", ref_kps);
			cout << ", " << ref_kps.size() << " keypoints";

			// 读特征点匹配关系
			vector<MatchKpsSim> matches;
			load_match_txt(word_dir + '/' + *test_iter + '_' + *ref_iter + ".txt", matches);
			cout << ", " << matches.size() << " matches";

			// 画出特征点匹配关系
			{
				// (Point2f转换成KeyPoint)
				vector<KeyPoint> test_keypoints;
				vector<KeyPoint> ref_keypoints;
				KeyPoint::convert(test_kps, test_keypoints);
				KeyPoint::convert(ref_kps, ref_keypoints);
				// (MatchKpsSim转换成DMatch)
				vector<DMatch> dmatches;
				for (vector<MatchKpsSim>::const_iterator ite = matches.begin(); ite != matches.end(); ite++)
				{
					dmatches.push_back(DMatch(ite->test_pid, ite->tmpl_pid, 0));
				}
				// 用drawMatches画出来
				Mat match_im;
				drawMatches(test_im, test_keypoints,
					ref_im, ref_keypoints,
					dmatches, match_im);
				fs::path match_im_file = res_match_path / (*test_iter + "_" + *ref_iter + ".jpg");
				imwrite(match_im_file.string(), match_im);
			}

			// 计算测试图像中心坐标 & 模板图像中心坐标, 保存特征点对坐标
			float left_test = test_im.cols, left_ref = ref_im.cols;
			float right_test = 0, right_ref = 0;
			float top_test = test_im.rows, top_ref = ref_im.rows;
			float bottom_test = 0, bottom_ref = 0;

			vector<Point2f> test_kps_new, ref_kps_new;

			for (int mi = 0; mi < matches.size(); mi++)
			{
				Point2f test_p = test_kps[matches[mi].test_pid];
				Point2f ref_p = ref_kps[matches[mi].tmpl_pid];

				test_kps_new.push_back(test_p);
				ref_kps_new.push_back(ref_p);

				left_test = min(left_test, test_p.x);
				right_test = max(right_test, test_p.x);
				top_test = min(top_test, test_p.y);
				bottom_test = max(bottom_test, test_p.y);

				left_ref = min(left_ref, ref_p.x);
				right_ref = max(right_ref, ref_p.x);
				top_ref = min(top_ref, ref_p.y);
				bottom_ref = max(bottom_ref, ref_p.y);
			}
			Point2f test_center((right_test - left_test) / 2, (bottom_test - top_test) / 2);
			Point2f ref_center((right_ref - left_ref) / 2, (bottom_ref - top_ref) / 2);
		
			// 测试图像特征点坐标 - 测试图像中心坐标
			for (vector<Point2f>::iterator ite = test_kps_new.begin(); ite != test_kps_new.end(); ite++)
			{
				ite->x = ite->x - test_center.x;
				ite->y = ite->y - test_center.y;
			}

			// 模板图像特征点坐标 - 模板图像中心坐标
			for (vector<Point2f>::iterator ite = ref_kps_new.begin(); ite != ref_kps_new.end(); ite++)
			{
				ite->x = ite->x - ref_center.x;
				ite->y = ite->y - ref_center.y;
			}

			/************************************************************************/
			/* RANSAC求仿射矩阵                                                                     */
			/************************************************************************/
			vector<uchar> stat;
			Mat A_mat = estimateAffine2D(test_kps_new, ref_kps_new, stat,
				RANSAC, RANSAC_REPROJ_THRES);
			cout << ", " << countNonZero(Mat(stat)) << " inliers";
			Mat A_mat_h = Mat::zeros(3, 3, CV_64F);
			A_mat_h.at<double>(0, 0) = A_mat.at<double>(0, 0);
			A_mat_h.at<double>(0, 1) = A_mat.at<double>(0, 1);
			A_mat_h.at<double>(0, 2) = A_mat.at<double>(0, 2);
			A_mat_h.at<double>(1, 0) = A_mat.at<double>(1, 0);
			A_mat_h.at<double>(1, 1) = A_mat.at<double>(1, 1);
			A_mat_h.at<double>(1, 2) = A_mat.at<double>(1, 2);
			A_mat_h.at<double>(2, 2) = 1;

			/************************************************************************/
			/* 变换测试图像                                                                     */
			/************************************************************************/
			// 变换前, 测试图像原点移动到中心
			Mat T_mat_pre = Mat::zeros(3, 3, CV_64F);
			T_mat_pre.at<double>(0, 0) = T_mat_pre.at<double>(1, 1) = T_mat_pre.at<double>(2, 2) = 1;
			T_mat_pre.at<double>(0, 2) = -test_center.x;
			T_mat_pre.at<double>(1, 2) = -test_center.y;

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
			warpAffine(test_im, M_im, M_mat, ref_im.size());
			namedWindow("after affine");
			imshow("after affine", M_im);
			fs::path align_im_file = res_align_path / (*test_iter + "_" + *ref_iter + ".jpg");
			imwrite(align_im_file.string(), M_im);

			// 画出inlier匹配
			{
				// (Point2f转换成KeyPoint)
				vector<Point2f> test_points, ref_points;
				vector<KeyPoint> test_keypoints, ref_keypoints;
				for (int i = 0; i < stat.size(); i++)
				{
					if (stat[i] != 0)
					{
						test_points.push_back(
							Point2f(test_kps_new[i].x + test_center.x, 
								test_kps_new[i].y+test_center.y)
						);
						ref_points.push_back(
							Point2f(ref_kps_new[i].x + ref_center.x,
								ref_kps_new[i].y + ref_center.y)
						);
					}
				}
				KeyPoint::convert(test_points, test_keypoints);
				KeyPoint::convert(ref_points, ref_keypoints);

				// (转换成DMatch)
				vector<DMatch> dmatches;
				for (int mi = 0; mi < test_keypoints.size(); mi++)
				{
					dmatches.push_back(DMatch(mi, mi, 0));
				}

				// 用drawMatches画出来
				Mat match_im;
				drawMatches(test_im, test_keypoints,
					ref_im, ref_keypoints,
					dmatches, match_im);
				fs::path match_im_file = res_match_path / (*test_iter + "_" + *ref_iter + "_inlier.jpg");
				imwrite(match_im_file.string(), match_im);
			}

			// 用变换后的裁剪测试图像和模板图像
			vector<Point2f> test_corners(4);
			test_corners[0] = Point2f(0, 0);
			test_corners[1] = Point2f(0, test_im.rows);
			test_corners[2] = Point2f(test_im.cols, test_im.rows);
			test_corners[3] = Point2f(test_im.cols, 0);

			transform(test_corners, test_corners, M_mat);
			vector<int> xs(4), ys(4);
			for (int ci = 0; ci < test_corners.size(); ci++)
			{
				xs[ci] = test_corners[ci].x;
				ys[ci] = test_corners[ci].y;
			}
			sort(xs.begin(), xs.end());
			sort(ys.begin(), ys.end());

			int start_r = min(max(ys[1], 0), ref_im.rows);
			int start_c = min(max(xs[1], 0), ref_im.cols);
			int end_r = max(min(ys[2], ref_im.rows), 0);
			int end_c = max(min(xs[2], ref_im.cols), 0);

			Mat test_crop_im = M_im(Range(start_r, end_r), Range(start_c, end_c));
			fs::path test_crop_file = res_crop_path / (*test_iter + "_" + *ref_iter + "_test_crop.jpg");
			imwrite(test_crop_file.string(), test_crop_im);

			Mat ref_crop_im = ref_im(Range(start_r, end_r), Range(start_c, end_c));
			fs::path ref_crop_file = res_crop_path / (*test_iter + "_" + *ref_iter + "_ref_crop.jpg");
			imwrite(ref_crop_file.string(), ref_crop_im);

			waitKey();

			cout << endl;
		}
	}

	destroyAllWindows();

	return 0;
}

/*

*/