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
	if(argc != 2)
	{
		cout << "Usage : opencv_alignment <yml_file>" << endl;
		return -1;
	}

	/*****************************************************************
	* 读配置文件
	******************************************************************/
	std::string config_file = argv[1];
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

	// 保存裁剪后的测试图像(有效区域)
	bool save_valid_crop;
	conf_fs["save_valid_crop"] >> save_valid_crop;

	// 保存裁剪后的测试图像(logo区域)
	bool save_logo_crop;
	conf_fs["save_logo_crop"] >> save_logo_crop;

	// 模板图像有效区域在测试图像中的包围盒坐标 & 测试图像中所有包围盒画在图像上的结果
	bool save_valid_box;
	conf_fs["save_valid_box"] >> save_valid_box;

	// 模板图像logo区域在测试图像中的包围盒坐标 & 测试图像中所有包围盒画在图像上的结果
	bool save_logo_box;
	conf_fs["save_logo_box"] >> save_logo_box;

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

	// 模板图像logo区域目录
	string tmpl_logo_dir;
	conf_fs["tmpl_logo_dir"] >> tmpl_logo_dir;

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

	// 保存裁剪后的测试图像对齐结果(有效区域)
	string res_valid_crop_dir;
	if (save_valid_crop)
	{
		conf_fs["res_valid_crop_dir"] >> res_valid_crop_dir;
	}

	// 保存裁剪后的测试图像对齐结果(logo区域)
	string res_logo_crop_dir;
	if (save_logo_crop) 
	{
		conf_fs["res_logo_crop_dir"] >> res_logo_crop_dir;
	}

	// 保存模板图像有效区域在测试图像中的包围盒坐标 & 测试图像中所有包围盒画在图像上的结果
	string res_valid_box_dir;
	if (save_valid_box)
	{
		conf_fs["res_valid_box_dir"] >> res_valid_box_dir;
	}
	
	// 保存模板图像logo区域在测试图像中的包围盒坐标 & 测试图像中所有包围盒画在图像上的结果
	string res_logo_box_dir;
	if (save_logo_box)
	{
		conf_fs["res_logo_box_dir"] >> res_logo_box_dir;
	}

	// -------------------- 编号与筛选选项 ----------------------------
	// 变换后的logo区域重叠大于阈值时, 认为对应同一个商品
	float logo_overlap_thres;
	conf_fs["logo_overlap_thres"] >> logo_overlap_thres;

	// 如果变换后有效区域的宽高比与1的关系和之前不一样则忽略
	bool filter_h2w;
	conf_fs["filter_h2w"] >> filter_h2w;

	// 一个商品与所有模板对齐产生的包围盒综合成一个所用方法
	string summarize_box;
	conf_fs["summarize_box"] >> summarize_box;

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
		std::cout << "test_im_dir " << test_im_dir << " not exist" << std::endl;
		return -1;
	}

	fs::path tmpl_im_path(tmpl_im_dir);
	if (!fs::exists(tmpl_im_path))
	{
		std::cout << "tmpl_im_dir " << tmpl_im_dir << " not exist" << std::endl;
		return -1;
	}

	// 检查RP匹配目录
	fs::path test_rp_match_path(test_rp_match_dir);
	if (!fs::exists(test_rp_match_path))
	{
		std::cout << "test_rp_match_dir " << test_rp_match_dir << " not exist" << std::endl;
		return -1;
	}

	// 检查模板图像valid region目录
	fs::path tmpl_valid_path(tmpl_valid_dir);
	if (!fs::exists(tmpl_valid_path))
	{
		std::cout << "tmpl_valid_dir " << tmpl_valid_dir << " not exist" << std::endl;
		return -1;
	}

	// 检查模板图像logo区域目录
	fs::path tmpl_logo_path(tmpl_logo_dir);
	if (!fs::exists(tmpl_logo_path))
	{
		std::cout << "tmpl_logo_dir " << tmpl_logo_dir << "not exist" << std::endl;
		return -1;
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

	// 读模板图像名列表
	vector<string> tmpl_names;
	fs::path tmpl_name_path(tmpl_name_file);
	if (!fs::exists(tmpl_name_path) || !fs::is_regular_file(tmpl_name_path))
	{
		std::cout << "tmpl_name_file not exist or is not regular file" << std::endl;
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
		std::cout << "res_match_dir not exist, created" << std::endl;
	}

	fs::path res_align_path(res_align_dir);
	if (save_align && !fs::exists(res_align_path))
	{
		fs::create_directories(res_align_path);
		std::cout << "res_align_path not exist, created" << std::endl;
	}

	fs::path res_valid_crop_path(res_valid_crop_dir);
	if (save_valid_crop && !fs::exists(res_valid_crop_path))
	{
		fs::create_directories(res_valid_crop_path);
		std::cout << "res_valid_crop_path not exist, created" << std::endl;

	}

	fs::path res_logo_crop_path(res_logo_crop_dir);
	if (save_logo_crop && !fs::exists(res_logo_crop_path))
	{
		fs::create_directories(res_logo_crop_path);
		std::cout << "res_logo_crop_path not exist, created" << std::endl;
	}

	fs::path res_valid_box_path(res_valid_box_dir);
	if (save_valid_box && !fs::exists(res_valid_box_path))
	{
		fs::create_directories(res_valid_box_path);
		std::cout << "res_valid_box_path not exist, created" << std::endl;
	}

	fs::path res_logo_box_path(res_logo_box_dir);
	if (save_logo_box && !fs::exists(res_logo_box_path))
	{
		fs::create_directories(res_logo_box_path);
		std::cout << "res_logo_box_path not exist, created" << std::endl;
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

	// 模板图像名->logo区域坐标映射
	unordered_map<string, RegionCoords> ref_name2logo;

	//namedWindow("ttv_on_im_all");

	/************************************************************************/
	/* 对齐                                                                     */
	/************************************************************************/
	// 对每幅测试图像
	for (vector<string>::const_iterator test_iter = test_names.begin(); test_iter != test_names.end(); test_iter++)
	{
		std::cout << "test_img " << *test_iter << " : " << std::endl;
		res_log << "test_img " << *test_iter << " : " << std::endl;

		// 读图像
		Mat test_im_c = imread(test_im_dir + '/' + *test_iter + ".jpg");
		Mat test_im_g;
		cvtColor(test_im_c, test_im_g, CV_BGR2GRAY);

		// 图像尺寸
		Size test_im_size = test_im_c.size();

		// 每个商品的logo区域
		vector<vector<RegionCoords>> ttl_regions;
		// 每个商品的所有包围盒 (只在summarize_box != "none"时使用)
		vector<vector<RegionCoords>> ttv_regions;
		// 每个商品编号对应的颜色
		vector<Scalar> obj_colors;

		// - 所有模板图像的变换后有效区域包围盒画在图像上
		Mat ttv_on_im_all = test_im_c.clone();
		// - 所有模板图像的变换后logo区域包围盒画在图像上
		Mat ttl_on_im_all = test_im_c.clone();

		// 与所有模板图像对齐
		for (vector<string>::const_iterator tmpl_iter = tmpl_names.begin(); tmpl_iter != tmpl_names.end(); tmpl_iter++)
		{
			std::cout << "\t\t template image : " << *tmpl_iter << " ";
			res_log << "\t\t template image : " << *tmpl_iter << " ";

			// 读图像
			Mat ref_im_c, ref_im_g;
			if (ref_name2imc.find(*tmpl_iter) == ref_name2imc.end())
			{
				ref_im_c = imread(tmpl_im_dir + '/' + *tmpl_iter + ".jpg");
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

			// 读logo区域坐标
			RegionCoords ref_logo;
			if (ref_name2logo.find(*tmpl_iter) == ref_name2logo.end())
			{
				ref_logo = load_region_txt(tmpl_logo_dir + "/" + *tmpl_iter + ".txt");
				ref_name2logo[*tmpl_iter] = ref_logo;
			}
			else
			{
				ref_logo = ref_name2logo[*tmpl_iter];
			}

			// 读recurrent pattern匹配结果
			// - 先读validID_模板图像名.txt
			vector<int> valid_ids;
			ifstream valid_txt(test_rp_match_dir +
				"/" + *test_iter +
				"/" + "validID_" + *tmpl_iter + ".txt");
			if (!valid_txt)
			{
				std::cout << test_rp_match_dir +
					"/" + *test_iter +
					"/" + "validID_" + *tmpl_iter + ".txt" << " not exist" << std::endl;
				res_log << test_rp_match_dir +
					"/" + *test_iter +
					"/" + "validID_" + *tmpl_iter + ".txt" << " not exist" << std::endl;

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

			// 画出与该模板图像对齐后变换到测试图像的有效区域
			//Mat ref_ttv_on_im = test_im_c.clone();
			// 该模板图像画图时颜色
			Scalar ref_color(rand()%255, rand()%255, rand()%255);

			for (vector<int>::iterator valid_iter = valid_ids.begin(); valid_iter != valid_ids.end(); valid_iter++)
			{
				std::cout << "\t\t\t validID " << *valid_iter << " : ";
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
						/ (*test_iter + "_" + *tmpl_iter + "_" + to_string(*valid_iter) + "_all.jpg");
					cv::imwrite(match_all_im_file.string(), match_all_im);

					Mat match_inlier_im;
					drawMatches(test_im_c, test_kps,
						ref_im_c, ref_kps,
						estimator->inliers, match_inlier_im);
					fs::path match_inlier_im_file = res_match_path
						/ (*test_iter + "_" + *tmpl_iter + "_" + to_string(*valid_iter) + "_inliers.jpg");
					if (!fs::exists(match_inlier_im_file.parent_path()))
					{
						fs::create_directories(match_inlier_im_file.parent_path());
					}
					cv::imwrite(match_inlier_im_file.string(), match_inlier_im);
				}

				// 变换测试图像
				// - 变换前, 测试图像原点移动到中心
				Mat T_mat_pre = Mat::zeros(3, 3, CV_64F);
				T_mat_pre.at<double>(0, 0) = T_mat_pre.at<double>(1, 1) = T_mat_pre.at<double>(2, 2) = 1;
				T_mat_pre.at<double>(0, 2) = -test_center.x;
				T_mat_pre.at<double>(1, 2) = -test_center.y;

				// - 仿射矩阵
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

				// 求模板图像区域变换到测试图像上的坐标
				// - 有效区域四角坐标
				vector<Point2f> ref_valid_corners(4);
				ref_valid_corners[0] = ref_valid.tl(); // 左上
				ref_valid_corners[1] = ref_valid.bl(); // 左下
				ref_valid_corners[2] = ref_valid.tr(); // 右上
				ref_valid_corners[3] = ref_valid.br(); // 右下
				
				// - logo区域四角坐标
				vector<Point2f> ref_logo_corners(4);
				ref_logo_corners[0] = ref_logo.tl(); // 左上
				ref_logo_corners[1] = ref_logo.bl(); // 左下
				ref_logo_corners[2] = ref_logo.tr(); // 右上
				ref_logo_corners[3] = ref_logo.br(); // 右下

				// - 变换到测试图像产生的包围盒
				Mat M_mat_inv_h = M_mat_h.inv();
				Mat M_mat_inv = M_mat_inv_h(Range(0, 2), Range(0, 3));
				transform(ref_valid_corners, ref_valid_corners, M_mat_inv);
				transform(ref_logo_corners, ref_logo_corners, M_mat_inv);

				// -- 有效区域
				RegionCoords this_ttv(
					min_of_four(ref_valid_corners[0].x, ref_valid_corners[1].x, ref_valid_corners[2].x, ref_valid_corners[3].x),
					max_of_four(ref_valid_corners[0].x, ref_valid_corners[1].x, ref_valid_corners[2].x, ref_valid_corners[3].x),
					min_of_four(ref_valid_corners[0].y, ref_valid_corners[1].y, ref_valid_corners[2].y, ref_valid_corners[3].y),
					max_of_four(ref_valid_corners[0].y, ref_valid_corners[1].y, ref_valid_corners[2].y, ref_valid_corners[3].y)
				);

				// 如果在图像外则忽略
				if (this_ttv.xmax <= 0 || this_ttv.xmin >= test_im_size.width || this_ttv.ymax <= 0 || this_ttv.ymin >= test_im_size.height)
				{
					continue;
				}

				// 如果变换后有效区域的宽高比与1的关系和之前不一样则忽略
				if (filter_h2w)
				{
					if ((ref_valid.h2w() >= 1 && this_ttv.h2w() < 1) || (ref_valid.h2w() <= 1 && this_ttv.h2w() > 1))
					{
						continue;
					}
				}

				// -- logo区域
				RegionCoords this_ttl(
					min_of_four(ref_logo_corners[0].x, ref_logo_corners[1].x, ref_logo_corners[2].x, ref_logo_corners[3].x),
					max_of_four(ref_logo_corners[0].x, ref_logo_corners[1].x, ref_logo_corners[2].x, ref_logo_corners[3].x),
					min_of_four(ref_logo_corners[0].y, ref_logo_corners[1].y, ref_logo_corners[2].y, ref_logo_corners[3].y),
					max_of_four(ref_logo_corners[0].y, ref_logo_corners[1].y, ref_logo_corners[2].y, ref_logo_corners[3].y)
				);

				// 如果在图像外则忽略
				if (this_ttl.xmax <= 0 || this_ttl.xmin >= test_im_size.width || this_ttl.ymax <= 0 || this_ttl.ymin >= test_im_size.height)
				{
					continue;
				}

				// -- 画在当前模板图像产生的有效区域包围盒图像上
				/*
				rectangle(ref_ttv_on_im,
				this_ttv.tl(), this_ttv.br(),
				ref_color, 2);
				*/

				// -- 根据与当前测试图像上已有的变换后模板图像logo区域的重叠编号
				int ti;
				for (ti = 0; ti < ttl_regions.size(); ti++)
				{
					for (int tci = 0; tci < ttl_regions[ti].size(); tci++) 
					{
						// 如果与已有的有效区域重叠大于阈值, 则用原来的编号
						if (ttl_regions[ti][tci].overlap(this_ttl) > logo_overlap_thres)
						{
							// 保存未裁剪结果
							if (save_align)
							{
								fs::path align_im_file = res_align_path /
									(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");
								if (!fs::exists(align_im_file.parent_path()))
								{
									fs::create_directories(align_im_file.parent_path());
								}
								cv::imwrite(align_im_file.string(), M_im);
							}

							ttl_regions[ti].push_back(this_ttl);
							ttv_regions[ti].push_back(this_ttv);

							// 如果不综合, 提前保存裁剪后的图像和坐标
							if (summarize_box == "none")
							{
								// 保存裁剪结果
								if (save_valid_crop)
								{
									int start_r = ref_valid.ymin;
									int start_c = ref_valid.xmin;
									int end_r = ref_valid.ymax;
									int end_c = ref_valid.xmax;

									Mat test_crop_im = M_im(Range(start_r, end_r + 1), Range(start_c, end_c + 1));
									fs::path test_crop_file = res_valid_crop_path /
										(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");
									if (!fs::exists(test_crop_file.parent_path()))
									{
										fs::create_directories(test_crop_file.parent_path());
									}
									cv::imwrite(test_crop_file.string(), test_crop_im);
								}

								if (save_logo_crop)
								{
									int start_r = ref_logo.ymin;
									int start_c = ref_logo.xmin;
									int end_r = ref_logo.ymax;
									int end_c = ref_logo.xmax;

									Mat test_crop_im = M_im(Range(start_r, end_r + 1), Range(start_c, end_c + 1));
									fs::path test_crop_file = res_logo_crop_path /
										(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");
									if (!fs::exists(test_crop_file.parent_path()))
									{
										fs::create_directories(test_crop_file.parent_path());
									}
									cv::imwrite(test_crop_file.string(), test_crop_im);
								}

								// 保存坐标
								if (save_valid_box)
								{
									// 如果不对同一个商品的包围盒综合, 则保存与一个模板图像对应的包围盒坐标, 画在图像上
									fs::path ttv_coord_file = res_valid_box_path /
										(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".txt");
									if (!fs::exists(ttv_coord_file.parent_path()))
									{
										fs::create_directories(ttv_coord_file.parent_path());
									}
									ofstream ttv_coord_txt(ttv_coord_file.string(), ios::out);
									ttv_coord_txt << this_ttv;
									ttv_coord_txt.close();

									// 同个物体同颜色画在测试图像上
									rectangle(ttv_on_im_all,
										this_ttv.tl(), this_ttv.br(),
										obj_colors[ti], 2);
								}

								if (save_logo_box)
								{
									// 如果不对同一个商品的包围盒综合, 则保存与一个模板图像对应的包围盒坐标, 画在图像上
									fs::path ttv_coord_file = res_logo_box_path /
										(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".txt");
									if (!fs::exists(ttv_coord_file.parent_path()))
									{
										fs::create_directories(ttv_coord_file.parent_path());
									}
									ofstream ttv_coord_txt(ttv_coord_file.string(), ios::out);
									ttv_coord_txt << this_ttl;
									ttv_coord_txt.close();

									// 同个物体同颜色画在测试图像上
									rectangle(ttl_on_im_all,
										this_ttl.tl(), this_ttl.br(),
										obj_colors[ti], 2);
								}
							}

							break;
						}
					}
				}

				// 如果与所有已有有效区域重叠都小于阈值, 则添加新编号
				if (ti == ttl_regions.size())
				{
					ttl_regions.push_back(vector<RegionCoords>());
					ttl_regions.back().push_back(this_ttl);
					ttv_regions.push_back(vector<RegionCoords>());
					ttv_regions.back().push_back(this_ttv);

					obj_colors.push_back(Scalar(rand() % 255, rand() % 255, rand() % 255));

					// 保存未裁剪结果
					if (save_align)
					{
						fs::path align_im_file = res_align_path /
							(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");
						if (!fs::exists(align_im_file.parent_path()))
						{
							fs::create_directories(align_im_file.parent_path());
						}
						cv::imwrite(align_im_file.string(), M_im);
					}

					// 如果不综合, 提前保存裁剪后的图像和坐标
					if (summarize_box == "none")
					{
						// 保存裁剪结果
						if (save_valid_crop)
						{
							int start_r = ref_valid.ymin;
							int start_c = ref_valid.xmin;
							int end_r = ref_valid.ymax;
							int end_c = ref_valid.xmax;

							Mat test_crop_im = M_im(Range(start_r, end_r + 1), Range(start_c, end_c + 1));
							fs::path test_crop_file = res_valid_crop_path /
								(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");
							if (!fs::exists(test_crop_file.parent_path()))
							{
								fs::create_directories(test_crop_file.parent_path());
							}
							cv::imwrite(test_crop_file.string(), test_crop_im);
						}

						if (save_logo_crop)
						{
							int start_r = ref_logo.ymin;
							int start_c = ref_logo.xmin;
							int end_r = ref_logo.ymax;
							int end_c = ref_logo.xmax;

							Mat test_crop_im = M_im(Range(start_r, end_r + 1), Range(start_c, end_c + 1));
							fs::path test_crop_file = res_logo_crop_path /
								(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");
							if (!fs::exists(test_crop_file.parent_path()))
							{
								fs::create_directories(test_crop_file.parent_path());
							}
							cv::imwrite(test_crop_file.string(), test_crop_im);
						}

						// 保存坐标
						if (save_valid_box)
						{
							// 如果不对同一个商品的包围盒综合, 则保存与一个模板图像对应的包围盒坐标, 画在图像上
							fs::path ttv_coord_file = res_valid_box_path /
								(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".txt");
							if (!fs::exists(ttv_coord_file.parent_path()))
							{
								fs::create_directories(ttv_coord_file.parent_path());
							}
							ofstream ttv_coord_txt(ttv_coord_file.string(), ios::out);
							ttv_coord_txt << this_ttv;
							ttv_coord_txt.close();

							// 同个物体同颜色画在测试图像上
							rectangle(ttv_on_im_all,
								this_ttv.tl(), this_ttv.br(),
								obj_colors[ti], 2);

							// 在包围盒中间标注物体id
							putText(ttv_on_im_all, to_string(ti),
								this_ttv.center(),
								FONT_HERSHEY_SIMPLEX, 2, obj_colors[ti],
								2);
						}

						if (save_logo_box)
						{
							// 如果不对同一个商品的包围盒综合, 则保存与一个模板图像对应的包围盒坐标, 画在图像上
							fs::path ttv_coord_file = res_logo_box_path /
								(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".txt");
							if (!fs::exists(ttv_coord_file.parent_path()))
							{
								fs::create_directories(ttv_coord_file.parent_path());
							}
							ofstream ttv_coord_txt(ttv_coord_file.string(), ios::out);
							ttv_coord_txt << this_ttl;
							ttv_coord_txt.close();

							// 同个物体同颜色画在测试图像上
							rectangle(ttl_on_im_all,
								this_ttl.tl(), this_ttl.br(),
								obj_colors[ti], 2);

							// 在包围盒中间标注物体id
							putText(ttl_on_im_all, to_string(ti),
								this_ttl.center(),
								FONT_HERSHEY_SIMPLEX, 2, obj_colors[ti],
								2);
						}

					}
				}
			}
		}

		// 保存变换后的模板区域画在测试图像上的结果
		fs::path ttv_on_im_all_file = res_valid_box_path / (*test_iter + ".jpg");
		cv::imwrite(ttv_on_im_all_file.string(), ttv_on_im_all);

		fs::path ttl_on_im_all_file = res_logo_box_path / (*test_iter + ".jpg");
		cv::imwrite(ttl_on_im_all_file.string(), ttl_on_im_all);

		/*
		//  保存变换后的模板有效区域画在测试图像上的结果
		if (summarize_box != "none")
		{
			for (int oi = 0; oi < ttv_regions.size(); oi++)
			{
				RegionCoords summ_bbx;

				if (summarize_box == "mean")
				{
					summ_bbx = mean_of_regions(ttv_regions[oi]);
				}
				else
				{
					summ_bbx = median_of_regions(ttv_regions[oi]);
				}

				// 保存裁剪结果
				if (save_valid_crop)
				{
					int start_r = clamp_between(summ_bbx.ymin, 0, test_im_c.rows); // clamp_between(int v, int lower, int upper);
					int start_c = clamp_between(summ_bbx.xmin, 0, test_im_c.cols);
					int end_r = clamp_between(summ_bbx.ymax, 0, test_im_c.rows);
					int end_c = clamp_between(summ_bbx.xmax, 0, test_im_c.cols);

					Mat test_crop_im = test_im_c(Range(start_r, end_r), Range(start_c, end_c));
					fs::path test_crop_file = res_crop_path /
						(*test_iter + "_" + to_string(oi) + ".jpg");
					cv::imwrite(test_crop_file.string(), test_crop_im);
				}

				fs::path ttv_coord_file = res_box_path /
					(*test_iter + "_" + to_string(oi) + ".txt");
				ofstream ttv_coord_txt(ttv_coord_file.string(), ios::out);
				ttv_coord_txt << summ_bbx;
				ttv_coord_txt.close();

				rectangle(ttv_on_im_all,
					summ_bbx.tl(), summ_bbx.br(),
					obj_colors[oi], 2);

				putText(ttv_on_im_all, to_string(oi),
					summ_bbx.center(),
					FONT_HERSHEY_SIMPLEX, 2, obj_colors[oi],
					2);
			}
		}
		*/

		/*
		Mat ttv_on_im_all_small;
		resize(ttv_on_im_all, ttv_on_im_all_small,
		Size(ttv_on_im_all.cols / 2, ttv_on_im_all.rows / 2),
		0, 0, INTER_LINEAR);
		imshow("ttv_on_im_all", ttv_on_im_all_small);
		waitKey();
		*/
	}

	res_log.close();

	//destroyWindow("ttv_on_im_all");

	return 0;
}

