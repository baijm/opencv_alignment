#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <boost/filesystem.hpp>

#include <nlopt.hpp>

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

// TODO : 约束

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
	string res_dir = "D:/datasets/shelf/patch/query/easy/alignment";
	string res_match_dir = res_dir + "/match"; // 保存匹配结果
	string res_align_dir = res_dir + "/result"; // 保存对齐结果

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

	/************************************************************************/
	/* 对齐                                                                     */
	/************************************************************************/
	// 对每幅测试图像
	for (vector<string>::const_iterator test_iter = test_names.begin(); test_iter != test_names.end(); test_iter++)
	{
		cout << "test image : " << *test_iter << endl;

		// 读图像
		Mat test_im = imread(test_im_dir + '/' + *test_iter + ".jpg");

		// 读特征点坐标
		vector<Point2f> test_kps;
		load_kp_pos_txt(test_kp_dir + '/' + *test_iter + "_kp.txt", test_kps);

		// 计算中心坐标
		float left = test_im.cols, right = 0, top = test_im.rows, bottom = 0;
		for (vector<Point2f>::const_iterator ite = test_kps.begin(); ite != test_kps.end(); ite++)
		{
			left = min(left, ite->x);
			right = max(right, ite->x);
			top = min(top, ite->y);
			bottom = max(bottom, ite->y);
		}
		Point2f test_center((right - left) / 2, (bottom - top) / 2);

		// 特征点坐标-中心坐标
		for (vector<Point2f>::iterator ite = test_kps.begin(); ite != test_kps.end(); ite++)
		{
			ite->x -= test_center.x;
			ite->y -= test_center.y;
		}

		// 对每幅模板图像
		for (vector<string>::const_iterator ref_iter = ref_names.begin(); ref_iter != ref_names.end(); ref_iter++)
		{
			cout << "\tref image : " << *ref_iter;

			// 读图像
			Mat ref_im = imread(ref_im_dir + '/' + *ref_iter + ".jpg");

			// 读特征点坐标
			vector<Point2f> ref_kps;
			load_kp_pos_txt(ref_kp_dir + '/' + *ref_iter + "_kp.txt", ref_kps);

			// 读特征点匹配关系
			vector<MatchKpsSim> matches;
			load_match_txt(word_dir + '/' + *test_iter + '_' + *ref_iter + ".txt", matches);

			// 计算中心坐标
			float left = ref_im.cols, right = 0, top = ref_im.rows, bottom = 0;
			for (vector<Point2f>::const_iterator ite = ref_kps.begin(); ite != ref_kps.end(); ite++)
			{
				left = min(left, ite->x);
				right = max(right, ite->x);
				top = min(top, ite->y);
				bottom = max(bottom, ite->y);
			}
			Point2f ref_center((right - left) / 2, (bottom - top) / 2);

			// 特征点坐标-中心坐标
			for (vector<Point2f>::iterator ite = ref_kps.begin(); ite != ref_kps.end(); ite++)
			{
				ite->x -= ref_center.x;
				ite->y -= ref_center.y;
			}

			/************************************************************************/
			/* RANSAC求仿射矩阵                                                                     */
			/************************************************************************/
			const int AFFINE_MIN_MATCHES = 3;
			const int RANSAC_MAX_ITERS = 1000;
			const double AFFINE_REPROJ_THRESH = 3;
			const int RANSAC_MIN_INLIERS = 1;

			Mat A_mat_best(2, 3, CV_64F);
			double best_err = numeric_limits<double>::max();

			for (int ite = 0; ite < RANSAC_MAX_ITERS; ite++)
			{
				cout << "\t\tRANSAC iter = " << ite << endl;
				// 随机选3对点
				vector<int> maybe_inliers;
				get_n_idx(AFFINE_MIN_MATCHES, 0, matches.size() - 1, maybe_inliers);
				
				vector<Point2f> test_sample, ref_sample;
				for (int i = 0; i < AFFINE_MIN_MATCHES; i++)
				{
					test_sample.push_back(test_kps[matches[maybe_inliers[i]].test_pid]);
					ref_sample.push_back(ref_kps[matches[maybe_inliers[i]].tmpl_pid]);
				}

				// 测试是否共线
				if (is_colinear_3(test_sample, ref_sample))
				{
					continue;
				}

				// 求仿射矩阵
				Mat A_mat = getAffineTransform(test_sample, ref_sample);

				// 求没用来求仿射矩阵的匹配中的inliers
				vector<int> also_inliers;
				for (int mi = 0; mi < matches.size(); mi++)
				{
					// 如果没用来求仿射矩阵
					if (find(maybe_inliers.begin(), maybe_inliers.end(), mi) == maybe_inliers.end())
					{
						// 测试图像中特征点坐标
						Point2f p_test = test_kps[matches[mi].test_pid];

						// 模板图像中特征点坐标
						Point2f p_ref = ref_kps[matches[mi].tmpl_pid];

						// 计算测试图像中特征点投影后的坐标
						Point2f p_test_proj(
							A_mat.at<double>(0, 0) * p_test.x + A_mat.at<double>(0, 1) * p_test.y + A_mat.at<double>(0, 2),
							A_mat.at<double>(1, 0) * p_test.x + A_mat.at<double>(1, 1) * p_test.y + A_mat.at<double>(1, 2)
							);

						// 计算投影误差
						double proj_err = sqrt(pow(p_test_proj.x - p_ref.x, 2) + pow(p_test_proj.y - p_ref.y, 2));

						// 如果误差小于阈值, 加入inliers
						if (proj_err <= AFFINE_REPROJ_THRESH)
						{
							also_inliers.push_back(mi);
						}
					}
				}

				// 如果没用来求仿射矩阵的匹配中inliers数量超过阈值, 则用全部inliers求参数
				if (also_inliers.size() >= RANSAC_MIN_INLIERS)
				{
					// 用全部inliers求参数
					vector<MatchKpsSim> all_inliers;
					for (int i = 0; i < maybe_inliers.size(); i++)
					{
						all_inliers.push_back(matches[maybe_inliers[i]]);
					}
					for (int i = 0; i < also_inliers.size(); i++)
					{
						all_inliers.push_back(matches[also_inliers[i]]);
					}

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

					if (result > 0)
					{
						// 优化成功
						// 如果误差比当前最好的小, 则更新A_mat_best和best_err
						if (min_f < best_err)
						{
							best_err = min_f;
							A_mat_best.at<double>(0, 0) = a[0];
							A_mat_best.at<double>(0, 1) = a[1];
							A_mat_best.at<double>(0, 2) = a[2];
							A_mat_best.at<double>(1, 0) = a[3];
							A_mat_best.at<double>(1, 1) = a[4];
							A_mat_best.at<double>(1, 2) = a[5];
						}
					}
					else
					{
						// 优化失败
						continue;
					}
				}
			}

			/************************************************************************/
			/* 变换测试图像                                                                     */
			/************************************************************************/
			// 把测试图像中心与模板图像中心对齐
			Mat T(2, 3, A_mat_best.type());
			T.at<double>(0, 0) = T.at<double>(1, 1) = 1;
			T.at<double>(0, 1) = T.at<double>(1, 0) = 0;
			T.at<double>(0, 2) = -test_center.x + ref_center.x;
			T.at<double>(1, 2) = -test_center.y + ref_center.y;
			Mat T_im;
			warpAffine(test_im, T_im, T, ref_im.size());
			namedWindow("after aligning centers");
			imshow("after aligning centers", T_im);

			// 仿射变换
			Mat A_im;
			warpAffine(T_im, A_im, A_mat_best, ref_im.size());
			namedWindow("after affine");
			imshow("after affine", A_im);
			//fs::path align_im_file = res_align_path / (*test_iter + "_" + *ref_iter + ".jpg");
			//imwrite(align_im_file.string(), A_im);


			waitKey();

			cout << endl;
		}
	}

	destroyAllWindows();

	return 0;
}

/*
// 画出特征点匹配关系
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
imshow("matches", match_im);
*/