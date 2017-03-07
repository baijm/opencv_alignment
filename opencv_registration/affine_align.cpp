#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>

#include <nlopt.hpp>

#include <iostream>
#include <fstream>

#include "util.h"

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;


/*
��С��Ŀ�꺯�� \sum_{i=1}^N ||Mp_{i} - p_{i'}||_2
����NΪƥ����, p_{i}Ϊ����ͼ��������������, p_{i'}Ϊģ��ͼ��������������
*/

const int MATCHES_NUM = 4;
const float PORTION_DIFF_THRES = 0.15;

int main()
{
	// ������txtλ��
	string test_kp_dir = "D:/datasets/shelf/patch/query/easy/crop_brisk"; // ����
	string ref_kp_dir = "D:/datasets/shelf/patch/ref/all/crop_brisk"; // ģ��

	// ͼ��λ��
	string test_im_dir = "D:/datasets/shelf/patch/query/easy/crop";
	string ref_im_dir = "D:/datasets/shelf/patch/ref/all/crop";

	// word txtλ��
	string word_dir = "D:/datasets/shelf/patch/query/easy/word/crop2crop_3";

	// ͼ�����б�
	string test_name_file = "D:/datasets/shelf/patch/query/easy/img_list_bbx_crop.txt";
	string ref_name_file = "D:/datasets/shelf/patch/ref/all/img_list_3_per_class.txt";

	// ���Ŀ¼
	string res_dir = "D:/datasets/shelf/patch/query/easy/alignment/brisk/";
	res_dir += to_string(MATCHES_NUM);
	res_dir += ("/" + to_string(PORTION_DIFF_THRES));
	string res_match_dir = res_dir + "/match"; // ����ƥ����
	string res_align_dir = res_dir + "/result"; // ���������
	string res_crop_dir = res_dir + "/crop"; // ����ü����

	/************************************************************************/
	/* ׼������ : ���Ŀ¼�Ƿ����, �����Ժ�ģ��ͼ����, �½����Ŀ¼                                                                     */
	/************************************************************************/
	// ���������txtĿ¼
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
	
	// ���ͼ��Ŀ¼
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
	
	// ���word txtĿ¼�Ƿ����
	fs::path word_path(word_dir);
	if (!fs::exists(word_path))
	{
		cout << "word_dir not exist" << endl;
		return -1;
	}

	// ���ͼ�����б��ļ��Ƿ����, ����������ͼ�����б�
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

	// �½�����ļ���
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
	/* ����                                                                     */
	/************************************************************************/
	// ��ÿ������ͼ��
	for (vector<string>::const_iterator test_iter = test_names.begin(); test_iter != test_names.end(); test_iter++)
	{
		cout << "test image : " << *test_iter;

		// ��ͼ��
		Mat test_im = imread(test_im_dir + '/' + *test_iter + ".jpg");

		// ͼ��ߴ�
		Size test_im_size = test_im.size();

		// ������������
		vector<Point2f> test_kps;
		load_kp_pos_txt(test_kp_dir + '/' + *test_iter + "_kp.txt", test_kps);
		cout << ", " << test_kps.size() << " keypoints" << endl;

		// ��������λ�������ͼ��ߴ�ı���
		vector<Point2f> test_kps_portion(test_kps.size());
		for (int pi = 0; pi < test_kps.size(); pi++)
		{
			test_kps_portion[pi] = Point2f(
				test_kps[pi].x / test_im_size.width,
				test_kps[pi].y / test_im_size.height);
		}

		// ��ÿ��ģ��ͼ��
		for (vector<string>::const_iterator ref_iter = ref_names.begin(); ref_iter != ref_names.end(); ref_iter++)
		{
			cout << "\tref image : " << *ref_iter;

			// ��ͼ��
			Mat ref_im = imread(ref_im_dir + '/' + *ref_iter + ".jpg");

			// ͼ��ߴ�
			Size ref_im_size = ref_im.size();

			// ������������
			vector<Point2f> ref_kps;
			load_kp_pos_txt(ref_kp_dir + '/' + *ref_iter + "_kp.txt", ref_kps);
			cout << ", " << ref_kps.size() << " keypoints";

			// ��������λ�������ͼ��ߴ�ı���
			vector<Point2f> ref_kps_portion(ref_kps.size());
			for (int pi = 0; pi < ref_kps.size(); pi++)
			{
				ref_kps_portion[pi] = Point2f(
					ref_kps[pi].x / ref_im_size.width,
					ref_kps[pi].y / ref_im_size.height);
			}

			// ��������ƥ���ϵ
			vector<MatchKpsSim> matches;
			load_match_txt(word_dir + '/' + *test_iter + '_' + *ref_iter + ".txt", matches);
			cout << ", " << matches.size() << " matches";

			// ����ȫ��������ƥ���ϵ
			Mat match_im = draw_MatchKpsSim(
				test_kps, ref_kps,
				test_im, ref_im,
				matches);
			fs::path match_im_file = res_match_path
				/ (*test_iter + "_" + *ref_iter + "_all.jpg");
			imwrite(match_im_file.string(), match_im);
			namedWindow("all matches");
			imshow("all_matches", match_im);

			/************************************************************************/
			/* �������������ĸ�ƥ����������                                                                     */
			/************************************************************************/
			// ɸѡƥ��: ����������λ�������ͼ��ߴ�ı���, ������̫����ȥ��
			vector<MatchKpsSim> matches_filter;
			for (vector<MatchKpsSim>::iterator mit = matches.begin(); mit != matches.end(); )
			{
				if (abs(test_kps_portion[mit->test_pid].x - ref_kps_portion[mit->tmpl_pid].x) > PORTION_DIFF_THRES
					|| abs(test_kps_portion[mit->test_pid].y - ref_kps_portion[mit->tmpl_pid].y) > PORTION_DIFF_THRES)
				{
					mit = matches.erase(mit);
				}
				else
				{
					mit++;
				}
			}

			// ����ɸѡ���ƥ��
			Mat match_filter_im = draw_MatchKpsSim(
				test_kps, ref_kps,
				test_im, ref_im,
				matches);
			fs::path match_filter_im_file = res_match_path
				/ (*test_iter + "_" + *ref_iter + "_filter.jpg");
			imwrite(match_filter_im_file.string(), match_filter_im);
			namedWindow("filtered matches");
			imshow("filtered matches", match_filter_im);

			// ѡ����������MATCHES_NUM��ƥ��
			float left_test = test_im.cols, right_test = 0;
			float top_test = test_im.rows, bottom_test = 0;
			float left_ref = ref_im.cols, right_ref = 0;
			float top_ref = ref_im.rows, bottom_ref = 0;
			float left_match_idx = -1, right_match_idx = -1, top_match_idx = -1, bottom_match_idx = -1;
			for (int mi = 0; mi < matches.size(); mi++)
			{
				MatchKpsSim m = matches[mi];

				if (test_kps[m.test_pid].x < left_test)
				{
					left_test = test_kps[m.test_pid].x;
					left_ref = ref_kps[m.tmpl_pid].x;
					left_match_idx = mi;
				}
				if (test_kps[m.test_pid].x > right_test)
				{
					right_test = test_kps[m.test_pid].x;
					right_ref = ref_kps[m.tmpl_pid].x;
					right_match_idx = mi;
				}
				if (test_kps[m.test_pid].y < top_test)
				{
					top_test = test_kps[m.test_pid].y;
					top_ref = ref_kps[m.tmpl_pid].y;
					top_match_idx = mi;
				}
				if (test_kps[m.test_pid].y > bottom_test)
				{
					bottom_test = test_kps[m.test_pid].y;
					bottom_ref = ref_kps[m.tmpl_pid].y;
					bottom_match_idx = mi;
				}
			}

			// �������MATCHES_NUM��������
			if ((left_match_idx == right_match_idx)
				|| (left_match_idx == top_match_idx)
				|| (left_match_idx == bottom_match_idx)
				|| (right_match_idx == top_match_idx)
				|| (right_match_idx == bottom_match_idx)
				|| (top_match_idx == bottom_match_idx)
				)
			{
				cout << ", less than " << MATCHES_NUM << "matches, skpipped" << endl;
				continue;
			}

			// ���������Ż���MATCHES_NUM��ƥ��
			vector<MatchKpsSim> matches_n;
			matches_n.push_back(matches[left_match_idx]);
			matches_n.push_back(matches[right_match_idx]);
			matches_n.push_back(matches[top_match_idx]);
			matches_n.push_back(matches[bottom_match_idx]);
			Mat match_n_im = draw_MatchKpsSim(
				test_kps, ref_kps,
				test_im, ref_im,
				matches_n);
			fs::path match_n_im_file = res_match_path
				/ (*test_iter + "_" + *ref_iter + "_n.jpg");
			imwrite(match_n_im_file.string(), match_n_im);
			namedWindow("matches for optimization");
			imshow("matches for optimization", match_n_im);

			// ����������
			Point2f test_center(
				abs(right_test - left_test) / 2, 
				abs(bottom_test - top_test) / 2);
			Point2f ref_center(
				abs(right_ref - left_ref) / 2, 
				abs(bottom_ref - top_ref) / 2);

			// ���������� - ��������
			vector<Point2f> test_pts_3(3), ref_pts_3(3);
			vector<Point2f> test_kps_new(test_kps), ref_kps_new(ref_kps);
			for (int ni = 0; ni < matches_n.size(); ni++)
			{
				test_kps_new[matches_n[ni].test_pid].x -= test_center.x;
				test_kps_new[matches_n[ni].test_pid].y -= test_center.y;
				ref_kps_new[matches_n[ni].tmpl_pid].x -= ref_center.x;
				ref_kps_new[matches_n[ni].tmpl_pid].y -= ref_center.y;

				if (ni < 3)
				{
					test_pts_3[ni] = test_kps_new[matches_n[ni].test_pid];
					ref_pts_3[ni] = ref_kps_new[matches_n[ni].tmpl_pid];
				}
			}

			// ���������ֵ
			Mat A_mat = getAffineTransform(test_pts_3, ref_pts_3);

			// ��MATCHES_NUM��ƥ���Ż�
			vector<double> a(6);
			a[0] = A_mat.at<double>(0, 0);
			a[1] = A_mat.at<double>(0, 1);
			a[2] = A_mat.at<double>(0, 2);
			a[3] = A_mat.at<double>(1, 0);
			a[4] = A_mat.at<double>(1, 1);
			a[5] = A_mat.at<double>(1, 2);

			ObjectiveFunctionData* obj_func_data =
				new ObjectiveFunctionData(matches_n, test_kps, ref_kps);
			nlopt::opt opt(nlopt::LD_MMA, 6);
			opt.set_min_objective(obj_func, obj_func_data);
			opt.set_ftol_abs(0.5);
			opt.set_stopval(3.0 * matches_n.size());
			double min_f;
			nlopt::result result = opt.optimize(a, min_f);
			if (result < 0)
			{
				cout << ", optimization failed" << endl;
				continue;
			}

			/************************************************************************/
			/* �任����ͼ��                                                                     */
			/************************************************************************/
			// �任ǰ, ����ͼ��ԭ���ƶ�������
			Mat T_mat_pre = Mat::zeros(3, 3, CV_64F);
			T_mat_pre.at<double>(0, 0) = T_mat_pre.at<double>(1, 1) = T_mat_pre.at<double>(2, 2) = 1;
			T_mat_pre.at<double>(0, 2) = -test_center.x;
			T_mat_pre.at<double>(1, 2) = -test_center.y;

			// �Ż���ķ������
			Mat A_mat_h = Mat::zeros(3, 3, A_mat.type());
			A_mat_h.at<double>(0, 0) = a[0];
			A_mat_h.at<double>(0, 1) = a[1];
			A_mat_h.at<double>(0, 2) = a[2];
			A_mat_h.at<double>(1, 0) = a[3];
			A_mat_h.at<double>(1, 1) = a[4];
			A_mat_h.at<double>(1, 2) = a[5];
			A_mat_h.at<double>(2, 2) = 1;

			// �任��, ����ͼ��ԭ���ƶ���ģ��ͼ������
			Mat T_mat_post = Mat::zeros(3, 3, CV_64F);
			T_mat_post.at<double>(0, 0) = T_mat_post.at<double>(1, 1) = T_mat_post.at<double>(2, 2) = 1;
			T_mat_post.at<double>(0, 2) = ref_center.x;
			T_mat_post.at<double>(1, 2) = ref_center.y;

			// ��ϱ任����
			Mat M_mat_h = T_mat_post * (A_mat_h * T_mat_pre);
			Mat M_mat = M_mat_h(Range(0, 2), Range(0, 3));

			// ����任
			Mat M_im;
			warpAffine(test_im, M_im, M_mat, ref_im.size());
			namedWindow("after affine");
			imshow("after affine", M_im);
			fs::path align_im_file = res_align_path / (*test_iter + "_" + *ref_iter + ".jpg");
			imwrite(align_im_file.string(), M_im);
		
			// �ñ任��Ĳü�����ͼ���ģ��ͼ��
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
