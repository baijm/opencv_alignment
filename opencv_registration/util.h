#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <iostream>
#include <fstream> 
#include <string>
#include <limits>
#include <time.h>

using namespace cv;
using namespace std;

// ����matlab�����word�ļ���ÿһ�е�����
struct MatchKpsSim 
{
	int test_pid;
	int tmpl_pid;
	double sim;

	MatchKpsSim(int test_p, int tmpl_p, double s = 0)
	{
		test_pid = test_p;
		tmpl_pid = tmpl_p;
		sim = s;
	}
};

void save_kp_txt(string txt_path, const vector<KeyPoint>& kp);

void save_des_txt(string des_path, const Mat& des);

// ��ȡ_kp.txt�ļ�
void load_kp_pos_txt(string txt_path, vector<Point2f>& kp);

// ��ȡmatlab�����word�ļ�, ����ÿһ��Ϊһ��ƥ��Ĳ���ͼ������������, ģ��ͼ������������, ���ƶ�
// [��Ϊmatlab������1��ʼ, �˴���������-1]
void load_match_txt(string txt_path, vector<MatchKpsSim>& matches);

void load_des_txt(string des_path, Mat& des);

void save_brisk(string img_dir, string img_name, string save_dir);

// RANSAC���
// �����[min_idx, max_idx]���±���ѡn�����ظ��ļ��뵽idx
void get_n_idx(int n, int min_idx, int max_idx, vector<int>& res);

// ���3�Ե��Ƿ���
// [�����Ǽ��p��3���Ƿ���, q��3���Ƿ���]
// TODO : �����������?
bool is_colinear_3(vector<Point2f>& p, vector<Point2f>& q);

// ����������ƥ���ϵ
// û������������İ汾
Mat draw_MatchKpsSim(const vector<Point2f>& test_pts, const vector<Point2f>& ref_pts, 
	const Mat& test_im, const Mat& ref_im,
	const vector<MatchKpsSim>& matches);

// �Ż����
// �Ż�ʱ����Ŀ�꺯��������
struct ObjectiveFunctionData
{
	vector<DMatch> *matches;
	vector<Point2f> *test_pts;
	vector<Point2f> *tmpl_pts;

	ObjectiveFunctionData(vector<DMatch> &matches_in, vector<Point2f> &test_pts_in, vector<Point2f> &tmpl_pts_in)
	{
		matches = &matches_in;
		test_pts = &test_pts_in;
		tmpl_pts = &tmpl_pts_in;
	}
};

// Ŀ�꺯��
// a = {a11, a12, a13, a21, a22, a23}
double obj_func(const vector<double> &a, vector<double> &grad, void *func_data);

