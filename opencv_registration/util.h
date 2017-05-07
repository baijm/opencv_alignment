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


// 保存区域坐标
struct RegionCoords {
	int xmin, xmax, ymin, ymax;

	RegionCoords():xmin(-1), xmax(-1), ymin(-1), ymax(-1){}

	RegionCoords(int xmin_in, int xmax_in, int ymin_in, int ymax_in)
		:xmin(xmin_in), xmax(xmax_in), ymin(ymin_in), ymax(ymax_in) {}

	// 返回左上角坐标
	Point2f tl() const {
		return Point2f(xmin, ymin);
	}

	// 返回右上角坐标
	Point2f tr() const {
		return Point2f(xmax, ymin);
	}

	// 返回左下角坐标
	Point2f bl() const {
		return Point2f(xmin, ymax);
	}

	// 返回右下角坐标
	Point2f br() const {
		return Point2f(xmax, ymax);
	}

	// 返回宽
	int width() const {
		return xmax - xmin;
	}

	// 返回高
	int height() const {
		return ymax - ymin;
	}

	// 返回高于宽的比值
	float h2w() const {
		return 1.0 * height() / width();
	}

	// 返回面积
	int area() const {
		return width() * height();
	}

	// 计算重叠
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


// 保存matlab输出的word文件中每一行的内容
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


// 保存特征点信息到txt文件, 每一行依次是y x size angle
void save_kp_txt(string txt_path, const vector<KeyPoint>& kp);
// 读取_kp.txt文件, 保留特征点所有信息
vector<KeyPoint> load_kp_txt(string txt_path);
// 读取_kp.txt文件, 只保留特征点位置
vector<Point2f> load_kp_pos_txt(string txt_path);


// 保存描述子到txt文件, 每一行是一个特征向量
void save_des_txt(string des_path, const Mat& des);
// 读取_des.txt文件
Mat load_des_txt(string txt_path);


// 读取区域坐标文件, 每一行依次是xmin xmax ymin ymax
RegionCoords load_region_txt(string txt_path);


// 读取matlab输出的word文件, 其中每一行为一个匹配的测试图像特征点索引, 模板图像特征点索引, 相似度
// [因为matlab索引从1开始, 此处所有索引-1]
void load_match_txt(string txt_path, vector<DMatch>& matches);
// 读取recurrent pattern匹配点对
// 每一行依次是: test_p_x test_p_y tmpl_p_x tmpl_p_y
void load_match_pts_txt(string txt_path, vector<Point2f>& test_pts, vector<Point2f>& ref_pts);


// 保存brisk特征
void save_brisk(string img_dir, string img_name, string save_dir);
// 保存SIFT特征
//void compute_and_save_sift(string img_name, Mat& im_g, vector<KeyPoint>& kps, string save_dir);


// 优化相关
// 优化时传给目标函数的数据
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

// 目标函数
// a = {a11, a12, a13, a21, a22, a23}
double obj_func(const vector<double> &a, vector<double> &grad, void *func_data);


#endif // !_UTIL_H
