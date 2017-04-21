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

int main()
{
	/*****************************************************************
	* 数据目录
	******************************************************************/
	// 图像等输入的根目录
	string root_dir = "D:/datasets/vobile_project/shelf/shampoo/test";
	

	// 测试图像
	// - 品牌
	string brand_name = "pr";

	//	- 图像位置
	string test_im_dir = root_dir + "/5_x_split_img_2852";
	//	- 图像名列表
	string test_name_file = root_dir + "/16_recurrent_pattern_brand_result" 
		+ "/" + "rp_" + brand_name + "_all.txt";
	//	- recurrent pattern粗分类结果
	//		- 测试图像名->label0, label1
	string test_rp_file = root_dir + "/16_recurrent_pattern_brand_result"
		+ "/classification_2_labels.txt";
	//		- 特征匹配结果
	string test_rp_match_dir = root_dir + "/26_match_result-multi_obj"
		+ "/result-" + brand_name + "/match_list";


	// 模板图像
	//	- 图像位置
	string ref_im_dir = root_dir + "/15_tmpl_new_center" 
		+ "/img_padding_black";
	// valid region文件(其中每一行依次是xmin, xmax, ymin, ymax)
	string ref_valid_dir = root_dir + "/15_tmpl_new_center"
		+ "/coords_valid_region";


	// 类名->模板图像名
	string cls_name2tmpl_file = root_dir + "/15_tmpl_new_center" 
		+ "/cls_name_2_tmpl.txt";

	// 粗分类label->类名
	string cls_id2name_file = root_dir + "/16_recurrent_pattern_brand_result"
		+ "/label_2_name.txt";


	// 结果目录
	string res_root_dir = root_dir + "/35_aligned_rp_rp_crop_valid_multi" + "/" + brand_name;
	string res_kp_dir = res_root_dir + "/keypoint"; // 保存特征提取结果
	string res_match_dir = res_root_dir + "/match"; // 保存匹配结果
	string res_align_dir = res_root_dir + "/result"; // 保存对齐结果
	string res_crop_dir = res_root_dir + "/crop"; // 保存裁剪结果
	string res_box_dir = res_root_dir + "/box_coord";


	/*****************************************************************
	* 选项
	******************************************************************/
	// 保存选项
	bool save_kp = false; // 保存特征点画在图像上的结果
	bool save_match = false; // 保存全部匹配和inlier
	bool save_align = false; // 如果是true, 则同时在_corner.txt中保存左上, 左下, 右上, 右下点的坐标
	bool save_crop = false; // 保存_test_crop.jpg
	bool save_box = true; // 

	/*******************************************************************
	*	初始化
	********************************************************************/
	// 求解仿射矩阵的方法
	MyAffineEstimator *estimator = new RansacAffineEstimator();

	/************************************************************************/
	/* 准备工作 : 检查目录是否存在, 读测试和模板图像名, 新建结果目录                                                                     */
	/************************************************************************/
	// 检查图像目录
	fs::path test_im_path(test_im_dir);
	if (!fs::exists(test_im_path))
	{
		std::cout << "test_im_dir not exist" << std::endl;
		return -1;
	}
	
	fs::path ref_im_path(ref_im_dir);
	if (!fs::exists(ref_im_path))
	{
		std::cout << "ref_im_dir not exist" << std::endl;
		return -1;
	}
	
	// 检查RP匹配目录
	fs::path test_rp_match_path(test_rp_match_dir);
	if (!fs::exists(test_rp_match_path))
	{
		std::cout << "test_rp_match_dir not exist" << std::endl;
		return -1;
	}

	// 检查模板图像valid region目录
	fs::path ref_valid_path(ref_valid_dir);
	if (!fs::exists(ref_valid_path))
	{
		std::cout << "ref_valid_dir not exist" << std::endl;
		return -1;
	}

	// 新建结果文件夹
	fs::path res_kp_path(res_kp_dir);
	if (!fs::exists(res_kp_path))
	{
		fs::create_directories(res_kp_path);
		std::cout << "res_kp_dir not exist, created" << std::endl;
	}

	fs::path res_match_path(res_match_dir);
	if (!fs::exists(res_match_path))
	{
		fs::create_directories(res_match_path);
		std::cout << "res_match_dir not exist, created" << std::endl;
	}

	fs::path res_align_path(res_align_dir);
	if (!fs::exists(res_align_path))
	{
		fs::create_directories(res_align_path);
		std::cout << "res_align_path not exist, created" << std::endl;
	}

	fs::path res_crop_path(res_crop_dir);
	if (!fs::exists(res_crop_path))
	{
		fs::create_directories(res_crop_path);
		std::cout << "res_crop_path not exist, created" << std::endl;
	}

	fs::path res_box_path(res_box_dir);
	if (!fs::exists(res_box_path))
	{
		fs::create_directories(res_box_path);
		std::cout << "res_box_path not exist, created" << std::endl;
	}

	// 读测试图像名列表
	vector<string> test_names;
	fs::path test_name_path(test_name_file);
	if (!fs::exists(test_name_path) || !fs::is_regular_file(test_name_path))
	{
		std::cout << "test_name_file not exist or is not regular file" << std::endl;
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

	// 读recurrent pattern粗分类结果
	unordered_map<string, vector<int>> test_rp_res;
	fs::path test_rp_path(test_rp_file);
	if (!fs::exists(test_rp_path) || !fs::is_regular_file(test_rp_path))
	{
		std::cout << "test_rp_file not exist or is not regular file" << std::endl;
		return -1;
	}
	else
	{
		ifstream txt(test_rp_file);
		string line;
		vector<string> parts;

		while (getline(txt, line))
		{
			boost::split(parts, line, boost::is_any_of(" "));
			test_rp_res[parts[0]].push_back(atoi(parts[1].c_str()));
			test_rp_res[parts[0]].push_back(atoi(parts[2].c_str()));
		}

		txt.close();
	}

	// 读label->类名映射
	unordered_map<int, string> cls_id2name;
	fs::path cls_id2name_path(cls_id2name_file);
	if (!fs::exists(cls_id2name_path) || !fs::is_regular_file(cls_id2name_path))
	{
		std::cout << "cls_id2name_file not exist or is not regular file" << std::endl;
		return -1;
	}
	else
	{
		ifstream txt(cls_id2name_file);
		string line;
		vector<string> parts;

		while (getline(txt, line))
		{
			boost::split(parts, line, boost::is_any_of(" "));
			cls_id2name[atoi(parts[0].c_str())] = parts[1];
		}

		txt.close();
	}

	//	读类名->模板图像名
	unordered_map<string, vector<string>> cls_name2tmpl;
	fs::path cls_name2tmpl_path(cls_name2tmpl_file);
	if (!fs::exists(cls_name2tmpl_path) || !fs::is_regular_file(cls_name2tmpl_path))
	{
		std::cout << "cls_name2tmpl_file not exist or is not regular file" << std::endl;
		return -1;
	}
	else
	{
		ifstream txt(cls_name2tmpl_file);
		string line;
		vector<string> parts;

		while (getline(txt, line))
		{
			boost::split(parts, line, boost::is_any_of(" "));
			cls_name2tmpl[parts[0]].push_back(parts[1]);
		}

		txt.close();
	}

	// log
	string now_str = boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time());
	ofstream res_log(res_root_dir + "/" + "log_" + now_str + ".txt");

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
		std::cout << "test_img " << *test_iter << " : "  << std::endl;
		res_log << "test_img " << *test_iter << " : " << std::endl;

		// 读图像
		Mat test_im_c = imread(test_im_dir + '/' + *test_iter + ".jpg");
		Mat test_im_g;
		cvtColor(test_im_c, test_im_g, CV_BGR2GRAY);

		// 图像尺寸
		Size test_im_size = test_im_c.size();

		// 变换后的模板图像有效区域文件
		fs::path trans_ref_valid_file = res_box_path / (*test_iter + ".txt");
		ofstream trans_ref_valid_txt(trans_ref_valid_file.string(), ios::out);

		// 画出变换后的模板有效区域
		Mat trans_ref_valid_im(test_im_c);
		namedWindow("transformed valid region");

		/******************************************************************
		*	根据粗分类的2个label, 与该类的所有模板图像对齐
		*******************************************************************/
		for (vector<int>::iterator lite = test_rp_res[*test_iter + ".jpg"].begin(); lite != test_rp_res[*test_iter + ".jpg"].end(); lite++)
		{
			std::cout << "\tlabel = " << *lite;
			res_log << "\tlabel = " << *lite;

			// 如果该label没有对应的类名 || 该类名没有模板图像, 则继续
			if (cls_id2name.find(*lite) == cls_id2name.end() 
				|| 
				cls_name2tmpl.find(cls_id2name[*lite]) == cls_name2tmpl.end())
			{
				std::cout << ",\tinvalid class id or no template images" << std::endl;
				res_log << ",\tinvalid class id or no template images" << std::endl;
				continue;
			}

			// 取该大类的模板图像
			vector<string> this_ref_names = cls_name2tmpl[cls_id2name[*lite]];

			std::cout << ",\tcls_name = " << cls_id2name[*lite] << ",\t" << this_ref_names.size() << " template images" << std::endl;
			res_log << ",\tcls_name = " << cls_id2name[*lite] << ",\t" << this_ref_names.size() << " template images" << std::endl;

			// 与每幅模板图像对齐
			for (vector<string>::iterator rite = this_ref_names.begin(); rite != this_ref_names.end(); rite++)
			{
				std::cout << "\t\t template image : " << *rite << " ";
				res_log << "\t\t template image : " << *rite << " ";

				// 读图像
				Mat ref_im_c, ref_im_g;
				if (ref_name2imc.find(*rite) == ref_name2imc.end())
				{
					ref_im_c = imread(ref_im_dir + '/' + *rite + ".jpg");
					cvtColor(ref_im_c, ref_im_g, CV_BGR2GRAY);

					ref_name2imc[*rite] = ref_im_c;
					ref_name2img[*rite] = ref_im_g;
				}
				else
				{
					ref_im_c = ref_name2imc[*rite];
					ref_im_g = ref_name2img[*rite];
				}

				// 图像尺寸
				Size ref_im_size = ref_im_c.size();

				// 读valid region坐标
				RegionCoords ref_valid;
				if (ref_name2valid.find(*rite) == ref_name2valid.end())
				{
					ref_valid = load_region_txt(ref_valid_dir + "/" + *rite + ".txt");
					ref_name2valid[*rite] = ref_valid;
				}
				else
				{
					ref_valid = ref_name2valid[*rite];
				}

				// 读recurrent pattern匹配结果
				// - 先读validID_模板图像名.txt
				vector<int> valid_ids;
				ifstream valid_txt(test_rp_match_dir + 
					"/" + *test_iter + 
					"/" + "validID_" + *rite + ".txt");
				if (!valid_txt)
				{
					std::cout << test_rp_match_dir +
						"/" + *test_iter +
						"/" + "validID_" + *rite + ".txt" << " not exist" << std::endl;
					res_log << test_rp_match_dir +
						"/" + *test_iter +
						"/" + "validID_" + *rite + ".txt" << " not exist" << std::endl;

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
					std::cout << "no match result" << std::endl;
					res_log << "no match result" << std::endl;
					continue;
				}

				// -- 否则, 列表中有非0数字k, 则再读模板图像名_k.txt
				std::cout << valid_ids.size() << " match results" << std::endl;
				res_log << valid_ids.size() << " match results" << std::endl;
				for (vector<int>::iterator vite = valid_ids.begin(); vite != valid_ids.end(); vite++)
				{
					std::cout << "\t\t\t validID " << *vite << " : ";
					res_log << "\t\t\t validID " << *vite << " : ";

					// 读模板图像名_k.txt
					vector<Point2f> test_pts, ref_pts;
					load_match_pts_txt(test_rp_match_dir +
						"/" + *test_iter +
						"/" + *rite + "_" + to_string(*vite) + ".txt", test_pts, ref_pts);
												
					// 求仿射矩阵
					Mat A_mat;
					if (!estimator->estimate_affine_matrix(test_pts, ref_pts, A_mat))
					{
						std::cout << "failed to estimate affine matrix" << std::endl;
						res_log << "failed to estimate affine matrix" << std::endl;
						continue;
					}
					std::cout << "estimating affine matrix succeed" << std::endl;
					res_log << "estimating affine matrix succeed" << std::endl;
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
							/ (*test_iter + "_" + *rite + "_" + to_string(*vite) + "_all.jpg");
						imwrite(match_all_im_file.string(), match_all_im);

						Mat match_inlier_im;
						drawMatches(test_im_c, test_kps,
							ref_im_c, ref_kps,
							estimator->inliers, match_inlier_im);
						fs::path match_inlier_im_file = res_match_path
							/ (*test_iter + "_" + *rite + "_" + to_string(*vite) + "_inliers.jpg");
						imwrite(match_inlier_im_file.string(), match_inlier_im);
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

					// 仿射变换并保存结果 (可选)
					Mat M_im;
					warpAffine(test_im_c, M_im, M_mat, ref_im_size);
					if (save_align)
					{
						fs::path align_im_file = res_align_path / (*test_iter + "_" + *rite + "_" + to_string(*vite) + ".jpg");
						imwrite(align_im_file.string(), M_im);
					}

					// 保存裁剪结果 (可选)
					if (save_crop)
					{
						int start_r = ref_valid.ymin;
						int start_c = ref_valid.xmin;
						int end_r = ref_valid.ymax;
						int end_c = ref_valid.xmax;

						Mat test_crop_im = M_im(Range(start_r, end_r), Range(start_c, end_c));
						fs::path test_crop_file = res_crop_path / 
							(*test_iter + "_" + *rite + "_" + to_string(*vite) + "_test_crop.jpg");
						imwrite(test_crop_file.string(), test_crop_im);
					}

					// 保存把模板图像变换到测试图像产生的候选包围盒
					if (save_box)
					{
						// 模板图像有效区域四角坐标
						vector<Point2f> ref_corners(4);
						ref_corners[0] = Point2f(ref_valid.xmin, ref_valid.ymin); // 左上
						ref_corners[1] = Point2f(ref_valid.xmin, ref_valid.ymax); // 左下
						ref_corners[2] = Point2f(ref_valid.xmax, ref_valid.ymin); // 右上
						ref_corners[3] = Point2f(ref_valid.xmax, ref_valid.ymax); // 右下

						// 求模板图像有效区域变换到测试图像上的坐标
						Mat M_mat_inv_h = M_mat_h.inv();
						Mat M_mat_inv = M_mat_inv_h(Range(0, 2), Range(0, 3));
						transform(ref_corners, ref_corners, M_mat_inv);

						// 求包围盒
						RegionCoords trans_ref_valid(
							min(min(min(ref_corners[0].x, ref_corners[1].x), ref_corners[2].x), ref_corners[3].x),
							max(max(max(ref_corners[0].x, ref_corners[1].x), ref_corners[2].x), ref_corners[3].x),
							min(min(min(ref_corners[0].y, ref_corners[1].y), ref_corners[2].y), ref_corners[3].y),
							max(max(max(ref_corners[0].y, ref_corners[1].y), ref_corners[2].y), ref_corners[3].y)
						);
						trans_ref_valid.xmin = max(trans_ref_valid.xmin, 0);
						trans_ref_valid.ymin = max(trans_ref_valid.ymin, 0);
						trans_ref_valid.xmax = min(trans_ref_valid.xmax, test_im_size.width-1);
						trans_ref_valid.ymax = min(trans_ref_valid.ymax, test_im_size.height-1);

						// 在测试图像上画出来
						rectangle(trans_ref_valid_im, trans_ref_valid.tl(), trans_ref_valid.br(), Scalar(0, 0, 255));
						imshow("transformed valid region", trans_ref_valid_im);

						// 保存
						trans_ref_valid_txt
							<< trans_ref_valid.xmin << " "
							<< trans_ref_valid.xmax << " "
							<< trans_ref_valid.ymin << " "
							<< trans_ref_valid.ymax << std::endl;
					}
				}

				// 拒绝不合理的变换, 用变换后的裁剪测试图像和模板图像
				/*
				std::cout << ",\t";
				res_log << ",\t";

				// 求测试图像4个角变换后的坐标
				vector<Point2f> test_corners(4);
				test_corners[0] = Point2f(0, 0); // 左上
				test_corners[1] = Point2f(0, test_im_size.height); // 左下
				test_corners[2] = Point2f(test_im_size.width, test_im_size.height); // 右下
				test_corners[3] = Point2f(test_im_size.width, 0); // 右上
				transform(test_corners, test_corners, M_mat);

				// 如果save_align是true, 则把变换后四个角的坐标保存到res_align_dir的_corner.txt里
				// 依次保存左上, 左下, 右上, 右下点的坐标(先y后x)
				if (save_align)
				{
					fs::path align_corner_file = res_align_path / (*test_iter + "_" + *rite + "_corner.txt");
					ofstream corner_txt(align_corner_file.string(), ios::out);

					// 左上角
					corner_txt << test_corners[0].y << " " << test_corners[0].x << std::endl;
					// 左下角
					corner_txt << test_corners[1].y << " " << test_corners[1].x << std::endl;
					// 右上角
					corner_txt << test_corners[3].y << " " << test_corners[3].x << std::endl;
					// 右下角
					corner_txt << test_corners[2].y << " " << test_corners[2].x << std::endl;

					corner_txt.close();
				}

				// 如果x方向顺序不对, 则拒绝
				if (!(test_corners[0].x < min(test_corners[2].x, test_corners[3].x))
					||
					!(test_corners[1].x < min(test_corners[2].x, test_corners[3].x)))
				{
					res_log << "rejected because wrong relative position in x direction" << std::endl;
					std::cout << "rejected because wrong relative position in x direction" << std::endl;
					continue;
				}

				// 如果y方向顺序不对, 则拒绝
				if (!(test_corners[0].y < min(test_corners[1].y, test_corners[2].y))
					||
					!(test_corners[3].y < min(test_corners[1].y, test_corners[2].y)))
				{
					res_log << "rejected because wrong relative position in y direction" << std::endl;
					std::cout << "rejected because wrong relative position in y direction" << std::endl;
					continue;
				}
				*/

				// 裁剪并保存 (可选)
				/*
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

				// 如果长宽比相反, 则拒绝
				if ((abs(end_r - start_r) > abs(end_c - start_c) && (test_im_g.rows < test_im_g.cols))
					||
					(abs(end_r - start_r) < abs(end_c - start_c) && (test_im_g.rows > test_im_g.cols)))
				{
					res_log << "rejected because wrong aspect ratio" << std::endl;
					std::cout << "rejected because wrong aspect ratio" << std::endl;
					continue;
				}

				// 如果长或宽为0, 则拒绝
				if (end_r - start_r == 0 || end_c - start_c == 0)
				{
					res_log << "rejected because zero size" << std::endl;
					std::cout << "rejected because zero size" << std::endl;
					continue;
				}
				*/

			}

		}

		//  保存变换后的模板有效区域画在测试图像上的结果
		fs::path trans_ref_valid_im_file = res_box_path /
			(*test_iter + ".jpg");
		imwrite(trans_ref_valid_im_file.string(), trans_ref_valid_im);

		trans_ref_valid_txt.close();

		waitKey();
	}

	res_log.close();

	return 0;
}

