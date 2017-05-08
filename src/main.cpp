#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>    
#include <boost/date_time/posix_time/posix_time.hpp>

#include <iostream>
#include <fstream>
#include <unordered_map>

#include "my_detector.h"
#include "util.h"

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		cout << "invalid argument number" << endl;
		return -1;
	}

	/*****************************************************************
	* 读配置文件
	******************************************************************/
	string config_file = argv[1];
	FileStorage conf_fs(config_file, FileStorage::READ);
	if (!conf_fs.isOpened())
	{
		cout << "failed to open file config.yml" << endl;
		return -1;
	}

	// ------------------ 保存选项 -----------------------------------
	// 全部匹配和inlier画在图像上的结果
	bool save_match;
	conf_fs["save_match"] >> save_match;
	// 保存未裁剪的结果
	bool save_align;
	conf_fs["save_align"] >> save_align;
	// 保存裁剪后的测试图像
	bool save_crop;
	conf_fs["save_crop"] >> save_crop;

	// -------------- 测试图像相关 -----------------------------------
	// 测试图像目录
	string test_im_dir;
	conf_fs["test_im_dir"] >> test_im_dir;
	// 测试图像名列表文件
	string test_name_file;
	conf_fs["test_name_file"] >> test_name_file;
	// 测试图像recurrent pattern粗分类特征匹配结果目录
	string test_rp_match_dir;
	conf_fs["test_rp_match_dir"] >> test_rp_match_dir;

	// -------------- 模板图像相关 -----------------------------------
	// 模板图像目录
	string tmpl_im_dir;
	conf_fs["tmpl_im_dir"] >> tmpl_im_dir;
	// 模板图像名列表文件
	string tmpl_name_file;
	conf_fs["tmpl_name_file"] >> tmpl_name_file;
	// 模板图像valid region目录
	string tmpl_valid_dir;
	conf_fs["tmpl_valid_dir"] >> tmpl_valid_dir;

	// ------------------- 结果目录 ----------------------------------
	// 保存在图像上画出特征匹配的结果
	string res_match_dir;
	if (save_match)
	{
		conf_fs["res_match_dir"] >> res_match_dir;
	}
	// 保存未裁剪的测试图像对齐结果
	string res_align_dir;
	if (save_align)
	{
		conf_fs["res_align_dir"] >> res_align_dir;
	}
	// 保存裁剪后的测试图像对齐结果
	string res_crop_dir;
	if (save_crop)
	{
		conf_fs["res_crop_dir"] >> res_crop_dir;
	}
	// 保存把模板图像有效区域变换到测试图像产生的包围盒坐标
	string res_box_dir;
	conf_fs["res_box_dir"] >> res_box_dir;
	
	conf_fs.release();

	/*******************************************************************
	*	初始化
	********************************************************************/
	// 求解仿射矩阵的方法
	MyAffineEstimator *estimator = new RansacAffineEstimator();

	// 检查图像目录
	fs::path test_im_path(test_im_dir);
	if (!fs::exists(test_im_path))
	{
		cout << "test_im_dir " << test_im_dir << " not exist" << endl;
		return -1;
	}
	
	fs::path tmpl_im_path(tmpl_im_dir);
	if (!fs::exists(tmpl_im_path))
	{
		cout << "tmpl_im_dir " << tmpl_im_dir << " not exist" << endl;
		return -1;
	}
	
	// 检查RP匹配目录
	fs::path test_rp_match_path(test_rp_match_dir);
	if (!fs::exists(test_rp_match_path))
	{
		cout << "test_rp_match_dir " << test_rp_match_dir << " not exist" << endl;
		return -1;
	}

	// 检查模板图像valid region目录
	fs::path tmpl_valid_path(tmpl_valid_dir);
	if (!fs::exists(tmpl_valid_path))
	{
		cout << "tmpl_valid_dir " << tmpl_valid_dir << " not exist" << endl;
		return -1;
	}

	// 读测试图像名列表
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

	// 读模板图像名列表
	vector<string> tmpl_names;
	fs::path tmpl_name_path(tmpl_name_file);
	if (!fs::exists(tmpl_name_path) || !fs::is_regular_file(tmpl_name_path))
	{
		cout << "tmpl_name_file not exist or is not regular file" << endl;
		return -1;
	}
	else
	{
		ifstream txt(tmpl_name_file);
		string line;
		while (getline(txt, line))
		{
			tmpl_names.push_back(line);
		}
	}

	// 检查结果目录
	fs::path res_match_path(res_match_dir);
	if (save_match && !fs::exists(res_match_path))
	{
		fs::create_directories(res_match_path);
		cout << "res_match_dir not exist, created" << endl;
	}
	fs::path res_align_path(res_align_dir);
	if (save_align && !fs::exists(res_align_path))
	{
		fs::create_directories(res_align_path);
		cout << "res_align_path not exist, created" << endl;
	}
	fs::path res_crop_path(res_crop_dir);
	if (save_crop && !fs::exists(res_crop_path))
	{
		fs::create_directories(res_crop_path);
		cout << "res_crop_path not exist, created" << endl;
	}
	fs::path res_box_path(res_box_dir);
	if (!fs::exists(res_box_path))
	{
		fs::create_directories(res_box_path);
		cout << "res_box_path not exist, created" << endl;
	}

	// log
	string now_str = boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time());
	ofstream res_log("log_" + now_str + ".txt");

	// 模板图像名->彩色图像映射
	unordered_map<string, Mat> ref_name2imc;

	// 模板图像名->灰度图像映射
	unordered_map<string, Mat> ref_name2img;
	
	// 模板图像名->valid region坐标映射
	unordered_map<string, RegionCoords> ref_name2valid;

	/************************************************************************/
	/* 对齐                                                                     */
	/************************************************************************/
	// 对每幅测试图像
	for (vector<string>::const_iterator test_iter = test_names.begin(); test_iter != test_names.end(); test_iter++)
	{
		cout << "test_img " << *test_iter << " : "  << endl;
		res_log << "test_img " << *test_iter << " : " << endl;

		// 读图像
		Mat test_im_c = imread(test_im_dir + '/' + *test_iter + ".jpg", IMREAD_COLOR);
		Mat test_im_g;
		cvtColor(test_im_c, test_im_g, CV_BGR2GRAY);

		// 图像尺寸
		Size test_im_size = test_im_c.size();

		// 变换后的模板图像有效区域
		vector<RegionCoords> trans_valid_regions;
		// - 保存到文件
		fs::path trans_tmpl_valid_file = res_box_path / (*test_iter + ".txt");
		ofstream trans_tmpl_valid_txt(trans_tmpl_valid_file.string(), ios::out);
		// - 画在图像上
		Mat trans_tmpl_valid_im = test_im_c.clone();

		// 与所有模板图像对齐
		for (vector<string>::const_iterator tmpl_iter = tmpl_names.begin(); tmpl_iter != tmpl_names.end(); tmpl_iter++)
		{
			cout << "\t\t template image : " << *tmpl_iter << " ";
			res_log << "\t\t template image : " << *tmpl_iter << " ";

			// 读图像
			Mat ref_im_c, ref_im_g;
			if (ref_name2imc.find(*tmpl_iter) == ref_name2imc.end())
			{
				ref_im_c = imread(tmpl_im_dir + '/' + *tmpl_iter + ".jpg", IMREAD_COLOR);
				cvtColor(ref_im_c, ref_im_g, CV_BGR2GRAY);
				ref_name2imc[*tmpl_iter] = ref_im_c;
				ref_name2img[*tmpl_iter] = ref_im_g;
			}
			else
			{
				ref_im_c = ref_name2imc[*tmpl_iter];
				ref_im_g = ref_name2img[*tmpl_iter];
			}

			// 图像尺寸
			Size ref_im_size = ref_im_c.size();

			// 读valid region坐标
			RegionCoords ref_valid;
			if (ref_name2valid.find(*tmpl_iter) == ref_name2valid.end())
			{
				ref_valid = load_region_txt(tmpl_valid_dir + "/" + *tmpl_iter + ".txt");
				ref_name2valid[*tmpl_iter] = ref_valid;
			}
			else
			{
				ref_valid = ref_name2valid[*tmpl_iter];
			}

			// 读recurrent pattern匹配结果
			// - 先读validID_模板图像名.txt
			vector<int> valid_ids;
			ifstream valid_txt(test_rp_match_dir +
				"/" + *test_iter +
				"/" + "validID_" + *tmpl_iter + ".txt");
			if (!valid_txt)
			{
				cout << test_rp_match_dir +
					"/" + *test_iter +
					"/" + "validID_" + *tmpl_iter + ".txt" << " not exist" << endl;
				res_log << test_rp_match_dir +
					"/" + *test_iter +
					"/" + "validID_" + *tmpl_iter + ".txt" << " not exist" << endl;

				return -1;
			}
			else
			{
				string line;
				while (getline(valid_txt, line))
				{
					valid_ids.push_back(atoi(line.c_str()));
				}
				valid_txt.close();
			}

			// -- 如果只有一行0, 则测试图像与该模板图像无匹配, 继续检查下一幅模板图像
			if (valid_ids.size() == 1 && valid_ids[0] == 0)
			{
				cout << "no match result" << endl;
				res_log << "no match result" << endl;
				continue;
			}

			// -- 否则, 列表中有非0数字k, 则再读模板图像名_k.txt
			cout << valid_ids.size() << " match results" << endl;
			res_log << valid_ids.size() << " match results" << endl;
			for (vector<int>::iterator valid_iter = valid_ids.begin(); valid_iter != valid_ids.end(); valid_iter++)
			{
				cout << "\t\t\t validID " << *valid_iter << " : ";
				res_log << "\t\t\t validID " << *valid_iter << " : ";

				// 读模板图像名_k.txt
				vector<Point2f> test_pts, ref_pts;
				load_match_pts_txt(test_rp_match_dir +
					"/" + *test_iter +
					"/" + *tmpl_iter + "_" + to_string(*valid_iter) + ".txt", test_pts, ref_pts);

				// 求仿射矩阵
				Mat A_mat;
				if (!estimator->estimate_affine_matrix(test_pts, ref_pts, A_mat))
				{
					cout << "failed to estimate affine matrix" << endl;
					res_log << "failed to estimate affine matrix" << endl;
					continue;
				}
				cout << "estimating affine matrix succeed" << endl;
				res_log << "estimating affine matrix succeed" << endl;
				Point2f test_center = estimator->test_center;
				Point2f ref_center = estimator->ref_center;

				// 画出全部特征点匹配关系inliers并保存 (可选)
				if (save_match)
				{
					Mat match_all_im;
					vector<KeyPoint> test_kps, ref_kps;
					cv::KeyPoint::convert(test_pts, test_kps);
					cv::KeyPoint::convert(ref_pts, ref_kps);
					vector<DMatch> matches;
					for (int pi = 0; pi < test_kps.size(); pi++)
					{
						matches.push_back(DMatch(pi, pi, 0));
					}

					drawMatches(test_im_c, test_kps,
						ref_im_c, ref_kps,
						matches, match_all_im);
					fs::path match_all_im_file = res_match_path
						/ (*test_iter + "_" + *tmpl_iter + "_" + to_string(*valid_iter) + "_all.jpg");
					cv::imwrite(match_all_im_file.string(), match_all_im);

					Mat match_inlier_im;
					drawMatches(test_im_c, test_kps,
						ref_im_c, ref_kps,
						estimator->inliers, match_inlier_im);
					fs::path match_inlier_im_file = res_match_path
						/ (*test_iter + "_" + *tmpl_iter + "_" + to_string(*valid_iter) + "_inliers.jpg");
					cv::imwrite(match_inlier_im_file.string(), match_inlier_im);
				}

				// 变换测试图像
				// - 变换前, 测试图像原点移动到中心
				Mat T_mat_pre = Mat::zeros(3, 3, CV_64F);
				T_mat_pre.at<double>(0, 0) = T_mat_pre.at<double>(1, 1) = T_mat_pre.at<double>(2, 2) = 1;
				T_mat_pre.at<double>(0, 2) = -test_center.x;
				T_mat_pre.at<double>(1, 2) = -test_center.y;

				// - 优化后的仿射矩阵
				Mat A_mat_h = Mat::zeros(3, 3, A_mat.type());
				A_mat_h.at<double>(0, 0) = A_mat.at<double>(0, 0);
				A_mat_h.at<double>(0, 1) = A_mat.at<double>(0, 1);
				A_mat_h.at<double>(0, 2) = A_mat.at<double>(0, 2);
				A_mat_h.at<double>(1, 0) = A_mat.at<double>(1, 0);
				A_mat_h.at<double>(1, 1) = A_mat.at<double>(1, 1);
				A_mat_h.at<double>(1, 2) = A_mat.at<double>(1, 2);
				A_mat_h.at<double>(2, 2) = 1;

				// - 变换后, 测试图像原点移动到模板图像中心
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

				// 求模板图像有效区域变换到测试图像上的坐标
				// -- 模板图像有效区域四角坐标
				vector<Point2f> ref_corners(4);
				ref_corners[0] = ref_valid.tl(); // 左上
				ref_corners[1] = ref_valid.bl(); // 左下
				ref_corners[2] = ref_valid.tr(); // 右上
				ref_corners[3] = ref_valid.br(); // 右下
				// -- 模板图像变换到测试图像产生的候选包围盒
				Mat M_mat_inv_h = M_mat_h.inv();
				Mat M_mat_inv = M_mat_inv_h(Range(0, 2), Range(0, 3));
				transform(ref_corners, ref_corners, M_mat_inv);
				// -- 求包围盒
				RegionCoords this_trans_valid(
					min(min(min(ref_corners[0].x, ref_corners[1].x), ref_corners[2].x), ref_corners[3].x),
					max(max(max(ref_corners[0].x, ref_corners[1].x), ref_corners[2].x), ref_corners[3].x),
					min(min(min(ref_corners[0].y, ref_corners[1].y), ref_corners[2].y), ref_corners[3].y),
					max(max(max(ref_corners[0].y, ref_corners[1].y), ref_corners[2].y), ref_corners[3].y)
				);
				this_trans_valid.xmin = max(this_trans_valid.xmin, 0);
				this_trans_valid.ymin = max(this_trans_valid.ymin, 0);
				this_trans_valid.xmax = min(this_trans_valid.xmax, test_im_size.width - 1);
				this_trans_valid.ymax = min(this_trans_valid.ymax, test_im_size.height - 1);
				// -- 如果变换到测试图像的有效区域宽高比和原本的有效区域相反, 则跳过
				if ((this_trans_valid.h2w() < 1 && ref_valid.h2w() > 1)
					|| (this_trans_valid.h2w() > 1 && ref_valid.h2w() < 1))
				{
					continue;
				}
				// -- 根据与当前测试图像已有的变换后模板图像有效区域的重叠编号
				int ti;
				for (ti = 0; ti < trans_valid_regions.size(); ti++)
				{
					// 如果与已有的有效区域重叠大于0.5, 则用原来的编号
					if (trans_valid_regions[ti].overlap(this_trans_valid) > 0.5)
					{
						// 保存未裁剪结果
						if (save_align)
						{
							fs::path align_im_file = res_align_path /
								(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");
							cv::imwrite(align_im_file.string(), M_im);
						}

						// 保存裁剪结果
						if (save_crop)
						{
							int start_r = ref_valid.ymin;
							int start_c = ref_valid.xmin;
							int end_r = ref_valid.ymax;
							int end_c = ref_valid.xmax;

							Mat test_crop_im = M_im(Range(start_r, end_r), Range(start_c, end_c));
							fs::path test_crop_file = res_crop_path /
								(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");
							cv::imwrite(test_crop_file.string(), test_crop_im);
						}

						break;
					}
				}

				// 如果与所有已有有效区域重叠都小于0.5, 则添加新编号
				if (ti == trans_valid_regions.size())
				{
					trans_valid_regions.push_back(this_trans_valid);

					// 保存未裁剪结果
					if (save_align)
					{
						fs::path align_im_file = res_align_path /
							(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");;
						cv::imwrite(align_im_file.string(), M_im);
					}

					// 保存裁剪结果
					if (save_crop)
					{
						int start_r = ref_valid.ymin;
						int start_c = ref_valid.xmin;
						int end_r = ref_valid.ymax;
						int end_c = ref_valid.xmax;

						Mat test_crop_im = M_im(Range(start_r, end_r), Range(start_c, end_c));
						fs::path test_crop_file = res_crop_path /
							(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");
						cv::imwrite(test_crop_file.string(), test_crop_im);
					}
				}


				// 保存把模板图像变换到测试图像产生的候选包围盒
				// -- 在测试图像上画出来
				rectangle(trans_tmpl_valid_im,
					this_trans_valid.tl(), this_trans_valid.br(),
					Scalar(0, 0, 255));
				// -- 在txt文件中保存坐标
				trans_tmpl_valid_txt
					<< this_trans_valid.xmin << " "
					<< this_trans_valid.xmax << " "
					<< this_trans_valid.ymin << " "
					<< this_trans_valid.ymax << endl;
			}
		}
		
		//  保存变换后的模板有效区域画在测试图像上的结果
		fs::path trans_tmpl_valid_im_file = res_box_path /
			(*test_iter + ".jpg");
		cv::imwrite(trans_tmpl_valid_im_file.string(), trans_tmpl_valid_im);
		trans_tmpl_valid_txt.close();
	}

	res_log.close();

	return 0;
}
