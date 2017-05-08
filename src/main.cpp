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
	* �������ļ�
	******************************************************************/
	string config_file = argv[1];
	FileStorage conf_fs(config_file, FileStorage::READ);
	if (!conf_fs.isOpened())
	{
		cout << "failed to open file config.yml" << endl;
		return -1;
	}

	// ------------------ ����ѡ�� -----------------------------------
	// ȫ��ƥ���inlier����ͼ���ϵĽ��
	bool save_match;
	conf_fs["save_match"] >> save_match;
	// ����δ�ü��Ľ��
	bool save_align;
	conf_fs["save_align"] >> save_align;
	// ����ü���Ĳ���ͼ��
	bool save_crop;
	conf_fs["save_crop"] >> save_crop;

	// -------------- ����ͼ����� -----------------------------------
	// ����ͼ��Ŀ¼
	string test_im_dir;
	conf_fs["test_im_dir"] >> test_im_dir;
	// ����ͼ�����б��ļ�
	string test_name_file;
	conf_fs["test_name_file"] >> test_name_file;
	// ����ͼ��recurrent pattern�ַ�������ƥ����Ŀ¼
	string test_rp_match_dir;
	conf_fs["test_rp_match_dir"] >> test_rp_match_dir;

	// -------------- ģ��ͼ����� -----------------------------------
	// ģ��ͼ��Ŀ¼
	string tmpl_im_dir;
	conf_fs["tmpl_im_dir"] >> tmpl_im_dir;
	// ģ��ͼ�����б��ļ�
	string tmpl_name_file;
	conf_fs["tmpl_name_file"] >> tmpl_name_file;
	// ģ��ͼ��valid regionĿ¼
	string tmpl_valid_dir;
	conf_fs["tmpl_valid_dir"] >> tmpl_valid_dir;

	// ------------------- ���Ŀ¼ ----------------------------------
	// ������ͼ���ϻ�������ƥ��Ľ��
	string res_match_dir;
	if (save_match)
	{
		conf_fs["res_match_dir"] >> res_match_dir;
	}
	// ����δ�ü��Ĳ���ͼ�������
	string res_align_dir;
	if (save_align)
	{
		conf_fs["res_align_dir"] >> res_align_dir;
	}
	// ����ü���Ĳ���ͼ�������
	string res_crop_dir;
	if (save_crop)
	{
		conf_fs["res_crop_dir"] >> res_crop_dir;
	}
	// �����ģ��ͼ����Ч����任������ͼ������İ�Χ������
	string res_box_dir;
	conf_fs["res_box_dir"] >> res_box_dir;
	
	conf_fs.release();

	/*******************************************************************
	*	��ʼ��
	********************************************************************/
	// ���������ķ���
	MyAffineEstimator *estimator = new RansacAffineEstimator();

	// ���ͼ��Ŀ¼
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
	
	// ���RPƥ��Ŀ¼
	fs::path test_rp_match_path(test_rp_match_dir);
	if (!fs::exists(test_rp_match_path))
	{
		cout << "test_rp_match_dir " << test_rp_match_dir << " not exist" << endl;
		return -1;
	}

	// ���ģ��ͼ��valid regionĿ¼
	fs::path tmpl_valid_path(tmpl_valid_dir);
	if (!fs::exists(tmpl_valid_path))
	{
		cout << "tmpl_valid_dir " << tmpl_valid_dir << " not exist" << endl;
		return -1;
	}

	// ������ͼ�����б�
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

	// ��ģ��ͼ�����б�
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

	// �����Ŀ¼
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

	// ģ��ͼ����->��ɫͼ��ӳ��
	unordered_map<string, Mat> ref_name2imc;

	// ģ��ͼ����->�Ҷ�ͼ��ӳ��
	unordered_map<string, Mat> ref_name2img;
	
	// ģ��ͼ����->valid region����ӳ��
	unordered_map<string, RegionCoords> ref_name2valid;

	/************************************************************************/
	/* ����                                                                     */
	/************************************************************************/
	// ��ÿ������ͼ��
	for (vector<string>::const_iterator test_iter = test_names.begin(); test_iter != test_names.end(); test_iter++)
	{
		cout << "test_img " << *test_iter << " : "  << endl;
		res_log << "test_img " << *test_iter << " : " << endl;

		// ��ͼ��
		Mat test_im_c = imread(test_im_dir + '/' + *test_iter + ".jpg", IMREAD_COLOR);
		Mat test_im_g;
		cvtColor(test_im_c, test_im_g, CV_BGR2GRAY);

		// ͼ��ߴ�
		Size test_im_size = test_im_c.size();

		// �任���ģ��ͼ����Ч����
		vector<RegionCoords> trans_valid_regions;
		// - ���浽�ļ�
		fs::path trans_tmpl_valid_file = res_box_path / (*test_iter + ".txt");
		ofstream trans_tmpl_valid_txt(trans_tmpl_valid_file.string(), ios::out);
		// - ����ͼ����
		Mat trans_tmpl_valid_im = test_im_c.clone();

		// ������ģ��ͼ�����
		for (vector<string>::const_iterator tmpl_iter = tmpl_names.begin(); tmpl_iter != tmpl_names.end(); tmpl_iter++)
		{
			cout << "\t\t template image : " << *tmpl_iter << " ";
			res_log << "\t\t template image : " << *tmpl_iter << " ";

			// ��ͼ��
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

			// ͼ��ߴ�
			Size ref_im_size = ref_im_c.size();

			// ��valid region����
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

			// ��recurrent patternƥ����
			// - �ȶ�validID_ģ��ͼ����.txt
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

			// -- ���ֻ��һ��0, �����ͼ�����ģ��ͼ����ƥ��, ���������һ��ģ��ͼ��
			if (valid_ids.size() == 1 && valid_ids[0] == 0)
			{
				cout << "no match result" << endl;
				res_log << "no match result" << endl;
				continue;
			}

			// -- ����, �б����з�0����k, ���ٶ�ģ��ͼ����_k.txt
			cout << valid_ids.size() << " match results" << endl;
			res_log << valid_ids.size() << " match results" << endl;
			for (vector<int>::iterator valid_iter = valid_ids.begin(); valid_iter != valid_ids.end(); valid_iter++)
			{
				cout << "\t\t\t validID " << *valid_iter << " : ";
				res_log << "\t\t\t validID " << *valid_iter << " : ";

				// ��ģ��ͼ����_k.txt
				vector<Point2f> test_pts, ref_pts;
				load_match_pts_txt(test_rp_match_dir +
					"/" + *test_iter +
					"/" + *tmpl_iter + "_" + to_string(*valid_iter) + ".txt", test_pts, ref_pts);

				// ��������
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

				// ����ȫ��������ƥ���ϵinliers������ (��ѡ)
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

				// �任����ͼ��
				// - �任ǰ, ����ͼ��ԭ���ƶ�������
				Mat T_mat_pre = Mat::zeros(3, 3, CV_64F);
				T_mat_pre.at<double>(0, 0) = T_mat_pre.at<double>(1, 1) = T_mat_pre.at<double>(2, 2) = 1;
				T_mat_pre.at<double>(0, 2) = -test_center.x;
				T_mat_pre.at<double>(1, 2) = -test_center.y;

				// - �Ż���ķ������
				Mat A_mat_h = Mat::zeros(3, 3, A_mat.type());
				A_mat_h.at<double>(0, 0) = A_mat.at<double>(0, 0);
				A_mat_h.at<double>(0, 1) = A_mat.at<double>(0, 1);
				A_mat_h.at<double>(0, 2) = A_mat.at<double>(0, 2);
				A_mat_h.at<double>(1, 0) = A_mat.at<double>(1, 0);
				A_mat_h.at<double>(1, 1) = A_mat.at<double>(1, 1);
				A_mat_h.at<double>(1, 2) = A_mat.at<double>(1, 2);
				A_mat_h.at<double>(2, 2) = 1;

				// - �任��, ����ͼ��ԭ���ƶ���ģ��ͼ������
				Mat T_mat_post = Mat::zeros(3, 3, CV_64F);
				T_mat_post.at<double>(0, 0) = T_mat_post.at<double>(1, 1) = T_mat_post.at<double>(2, 2) = 1;
				T_mat_post.at<double>(0, 2) = ref_center.x;
				T_mat_post.at<double>(1, 2) = ref_center.y;

				// ��ϱ任����
				Mat M_mat_h = T_mat_post * (A_mat_h * T_mat_pre);
				Mat M_mat = M_mat_h(Range(0, 2), Range(0, 3));

				// ����任
				Mat M_im;
				warpAffine(test_im_c, M_im, M_mat, ref_im_size);

				// ��ģ��ͼ����Ч����任������ͼ���ϵ�����
				// -- ģ��ͼ����Ч�����Ľ�����
				vector<Point2f> ref_corners(4);
				ref_corners[0] = ref_valid.tl(); // ����
				ref_corners[1] = ref_valid.bl(); // ����
				ref_corners[2] = ref_valid.tr(); // ����
				ref_corners[3] = ref_valid.br(); // ����
				// -- ģ��ͼ��任������ͼ������ĺ�ѡ��Χ��
				Mat M_mat_inv_h = M_mat_h.inv();
				Mat M_mat_inv = M_mat_inv_h(Range(0, 2), Range(0, 3));
				transform(ref_corners, ref_corners, M_mat_inv);
				// -- ���Χ��
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
				// -- ����任������ͼ�����Ч�����߱Ⱥ�ԭ������Ч�����෴, ������
				if ((this_trans_valid.h2w() < 1 && ref_valid.h2w() > 1)
					|| (this_trans_valid.h2w() > 1 && ref_valid.h2w() < 1))
				{
					continue;
				}
				// -- �����뵱ǰ����ͼ�����еı任��ģ��ͼ����Ч������ص����
				int ti;
				for (ti = 0; ti < trans_valid_regions.size(); ti++)
				{
					// ��������е���Ч�����ص�����0.5, ����ԭ���ı��
					if (trans_valid_regions[ti].overlap(this_trans_valid) > 0.5)
					{
						// ����δ�ü����
						if (save_align)
						{
							fs::path align_im_file = res_align_path /
								(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");
							cv::imwrite(align_im_file.string(), M_im);
						}

						// ����ü����
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

				// ���������������Ч�����ص���С��0.5, ������±��
				if (ti == trans_valid_regions.size())
				{
					trans_valid_regions.push_back(this_trans_valid);

					// ����δ�ü����
					if (save_align)
					{
						fs::path align_im_file = res_align_path /
							(*test_iter + "_" + to_string(ti) + "_" + *tmpl_iter + ".jpg");;
						cv::imwrite(align_im_file.string(), M_im);
					}

					// ����ü����
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


				// �����ģ��ͼ��任������ͼ������ĺ�ѡ��Χ��
				// -- �ڲ���ͼ���ϻ�����
				rectangle(trans_tmpl_valid_im,
					this_trans_valid.tl(), this_trans_valid.br(),
					Scalar(0, 0, 255));
				// -- ��txt�ļ��б�������
				trans_tmpl_valid_txt
					<< this_trans_valid.xmin << " "
					<< this_trans_valid.xmax << " "
					<< this_trans_valid.ymin << " "
					<< this_trans_valid.ymax << endl;
			}
		}
		
		//  ����任���ģ����Ч�����ڲ���ͼ���ϵĽ��
		fs::path trans_tmpl_valid_im_file = res_box_path /
			(*test_iter + ".jpg");
		cv::imwrite(trans_tmpl_valid_im_file.string(), trans_tmpl_valid_im);
		trans_tmpl_valid_txt.close();
	}

	res_log.close();

	return 0;
}
