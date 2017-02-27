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
��С��Ŀ�꺯�� \sum_{i=1}^N ||Mp_{i} - p_{i'}||_2
����NΪƥ����, p_{i}Ϊ����ͼ��������������, p_{i'}Ϊģ��ͼ��������������
*/

int opt_iters = 0;

// Ŀ�꺯��
// a = {a11, a12, a13, a21, a22, a23}
double obj_func(const vector<double> &a, vector<double> &grad, void *func_data)
{
	opt_iters++;

	ObjectiveFunctionData *data = reinterpret_cast<ObjectiveFunctionData*>(func_data);
	// TODO : �����ø����ø�ֵ����ô��??
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

	// ��Ŀ�꺯����ֵ
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

// TODO : Լ��

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
	string res_dir = "D:/datasets/shelf/patch/query/easy/alignment";
	string res_match_dir = res_dir + "/match"; // ����ƥ����
	string res_align_dir = res_dir + "/result"; // ���������

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

	/************************************************************************/
	/* ����                                                                     */
	/************************************************************************/
	// ��ÿ������ͼ��
	for (vector<string>::const_iterator test_iter = test_names.begin(); test_iter != test_names.end(); test_iter++)
	{
		cout << "test image : " << *test_iter << endl;

		// ��ͼ��
		Mat test_im = imread(test_im_dir + '/' + *test_iter + ".jpg");

		// ������������
		vector<Point2f> test_kps;
		load_kp_pos_txt(test_kp_dir + '/' + *test_iter + "_kp.txt", test_kps);

		// ������������
		float left = test_im.cols, right = 0, top = test_im.rows, bottom = 0;
		for (vector<Point2f>::const_iterator ite = test_kps.begin(); ite != test_kps.end(); ite++)
		{
			left = min(left, ite->x);
			right = max(right, ite->x);
			top = min(top, ite->y);
			bottom = max(bottom, ite->y);
		}
		Point2f test_center((right - left) / 2, (bottom - top) / 2);

		// ����������-��������
		for (vector<Point2f>::iterator ite = test_kps.begin(); ite != test_kps.end(); ite++)
		{
			ite->x -= test_center.x;
			ite->y -= test_center.y;
		}

		// ��ÿ��ģ��ͼ��
		for (vector<string>::const_iterator ref_iter = ref_names.begin(); ref_iter != ref_names.end(); ref_iter++)
		{
			cout << "\tref image : " << *ref_iter;

			// ��ͼ��
			Mat ref_im = imread(ref_im_dir + '/' + *ref_iter + ".jpg");

			// ������������
			vector<Point2f> ref_kps;
			load_kp_pos_txt(ref_kp_dir + '/' + *ref_iter + "_kp.txt", ref_kps);

			// ��������ƥ���ϵ
			vector<MatchKpsSim> matches;
			load_match_txt(word_dir + '/' + *test_iter + '_' + *ref_iter + ".txt", matches);

			// ������������
			float left = ref_im.cols, right = 0, top = ref_im.rows, bottom = 0;
			for (vector<Point2f>::const_iterator ite = ref_kps.begin(); ite != ref_kps.end(); ite++)
			{
				left = min(left, ite->x);
				right = max(right, ite->x);
				top = min(top, ite->y);
				bottom = max(bottom, ite->y);
			}
			Point2f ref_center((right - left) / 2, (bottom - top) / 2);

			// ����������-��������
			for (vector<Point2f>::iterator ite = ref_kps.begin(); ite != ref_kps.end(); ite++)
			{
				ite->x -= ref_center.x;
				ite->y -= ref_center.y;
			}

			// ���������ʼֵ, �����ƶ���ߵ�3��ƥ��
			vector<Point2f> test_top3;
			vector<Point2f> ref_top3;
			for (int i = 0; i < 3; i++)
			{
				test_top3.push_back(test_kps[matches[i].test_pid]);
				ref_top3.push_back(ref_kps[matches[i].tmpl_pid]);
			}
			Mat affine_mat = getAffineTransform(test_top3, ref_top3);

			// �Ż�
			vector<double> a(6);
			a[0] = affine_mat.at<double>(0, 0);
			a[1] = affine_mat.at<double>(0, 1);
			a[2] = affine_mat.at<double>(0, 2);
			a[3] = affine_mat.at<double>(1, 0);
			a[4] = affine_mat.at<double>(1, 1);
			a[5] = affine_mat.at<double>(1, 2);
			ObjectiveFunctionData* obj_func_data = new ObjectiveFunctionData(matches, test_kps, ref_kps);
			nlopt::opt opt(nlopt::LD_MMA, 6);
			opt.set_min_objective(obj_func, obj_func_data);
			opt.set_ftol_abs(0.5);
			opt.set_stopval(3.0 * matches.size());
			double min_f;
			opt_iters = 0;
			nlopt::result result = opt.optimize(a, min_f);
			if (result > 0)
			{
				cout << "optimization succeeded, return code = " << result 
					<< ", objective function value = " << min_f
					<< ", iters = " << opt_iters << endl;

				// �Ѳ���ͼ��������ģ��ͼ�����Ķ���, �任��ģ��ͼ��
				Mat T(2, 3, affine_mat.type());
				T.at<double>(0, 0) = T.at<double>(1, 1) = 1;
				T.at<double>(0, 1) = T.at<double>(1, 0) = 0;
				T.at<double>(0, 2) = -test_center.x + ref_center.x;
				T.at<double>(1, 2) = -test_center.y + ref_center.y;
				Mat T_im;
				warpAffine(test_im, T_im, T, ref_im.size());
				namedWindow("after aligning centers");
				imshow("after aligning centers", T_im);

				// ����任
				affine_mat.at<double>(0, 0) = a[0];
				affine_mat.at<double>(0, 1) = a[1];
				affine_mat.at<double>(0, 2) = a[2];
				affine_mat.at<double>(1, 0) = a[3];
				affine_mat.at<double>(1, 1) = a[4];
				affine_mat.at<double>(1, 2) = a[5];

				Mat A_im;
				warpAffine(T_im, A_im, affine_mat, ref_im.size());
				namedWindow("after affine");
				imshow("after affine", A_im);
				fs::path align_im_file = res_align_path / (*test_iter + "_" + *ref_iter + ".jpg");
				imwrite(align_im_file.string(), A_im);
			}
			else
			{
				cout << "optimization failed, return code = " << result << endl;
			}

			// ���Ż�����Բ���ͼ��ÿ������λ�ý��б任


			waitKey();

			cout << endl;
		}
	}

	destroyAllWindows();

	return 0;
}

/*
// ����������ƥ���ϵ
// (Point2fת����KeyPoint)
vector<KeyPoint> test_keypoints;
vector<KeyPoint> ref_keypoints;
KeyPoint::convert(test_kps, test_keypoints);
KeyPoint::convert(ref_kps, ref_keypoints);
// (MatchKpsSimת����DMatch)
vector<DMatch> dmatches;
for (vector<MatchKpsSim>::const_iterator ite = matches.begin(); ite != matches.end(); ite++)
{
dmatches.push_back(DMatch(ite->test_pid, ite->tmpl_pid, 0));
}
// ��drawMatches������
Mat match_im;
drawMatches(test_im, test_keypoints,
ref_im, ref_keypoints,
dmatches, match_im);
fs::path match_im_file = res_match_path / (*test_iter + "_" + *ref_iter + ".jpg");
imwrite(match_im_file.string(), match_im);
imshow("matches", match_im);
*/