#ifndef _UTIL_H
#define _UTIL_H

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


// ������������
struct RegionCoords {
	int xmin, xmax, ymin, ymax;

	RegionCoords():xmin(-1), xmax(-1), ymin(-1), ymax(-1){}

	RegionCoords(int xmin_in, int xmax_in, int ymin_in, int ymax_in)
		:xmin(xmin_in), xmax(xmax_in), ymin(ymin_in), ymax(ymax_in) {}

	// �������Ͻ�����
	Point2f tl() const {
		return Point2f(xmin, ymin);
	}

	// �������Ͻ�����
	Point2f tr() const {
		return Point2f(xmax, ymin);
	}

	// �������½�����
	Point2f bl() const {
		return Point2f(xmin, ymax);
	}

	// �������½�����
	Point2f br() const {
		return Point2f(xmax, ymax);
	}

	// ���ؿ�
	int width() const {
		return xmax - xmin;
	}

	// ���ظ�
	int height() const {
		return ymax - ymin;
	}

	// ���ظ��ڿ�ı�ֵ
	float h2w() const {
		return 1.0 * height() / width();
	}

	// �������
	int area() const {
		return width() * height();
	}

	// �����ص�
	float overlap(const RegionCoords &r) const {
		int r_xmin = r.xmin, r_ymin = r.ymin, r_width = r.width(), r_height = r.height();

		int start_x = min(xmin, r_xmin), start_y = min(ymin, r_ymin);
		int end_x = max(xmax, r_xmin + r_width), end_y = max(ymax, r_ymin + r_height);

		int overlap_width = r_width + width() - (end_x - start_x);
		int overlap_height = r_height + height() - (end_y - start_y);

		if (overlap_width <= 0 || overlap_height <= 0)
		{
			return 0;
		}
		else
		{
			float overlap_area = overlap_width * overlap_height;
			return overlap_area / (area() + r.area() - overlap_area);
		}
	}
};


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


// ������������Ϣ��txt�ļ�, ÿһ��������y x size angle
void save_kp_txt(string txt_path, const vector<KeyPoint>& kp);
// ��ȡ_kp.txt�ļ�, ����������������Ϣ
vector<KeyPoint> load_kp_txt(string txt_path);
// ��ȡ_kp.txt�ļ�, ֻ����������λ��
vector<Point2f> load_kp_pos_txt(string txt_path);


// ���������ӵ�txt�ļ�, ÿһ����һ����������
void save_des_txt(string des_path, const Mat& des);
// ��ȡ_des.txt�ļ�
Mat load_des_txt(string txt_path);


// ��ȡ���������ļ�, ÿһ��������xmin xmax ymin ymax
RegionCoords load_region_txt(string txt_path);


// ��ȡmatlab�����word�ļ�, ����ÿһ��Ϊһ��ƥ��Ĳ���ͼ������������, ģ��ͼ������������, ���ƶ�
// [��Ϊmatlab������1��ʼ, �˴���������-1]
void load_match_txt(string txt_path, vector<DMatch>& matches);
// ��ȡrecurrent patternƥ����
// ÿһ��������: test_p_x test_p_y tmpl_p_x tmpl_p_y
void load_match_pts_txt(string txt_path, vector<Point2f>& test_pts, vector<Point2f>& ref_pts);


// ����brisk����
void save_brisk(string img_dir, string img_name, string save_dir);
// ����SIFT����
//void compute_and_save_sift(string img_name, Mat& im_g, vector<KeyPoint>& kps, string save_dir);


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


#endif // !_UTIL_H
