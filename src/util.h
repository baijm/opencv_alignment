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
#include <vector>
#include <limits>
#include <time.h>

using namespace cv;
using namespace std;

// 保存区域坐标
class RegionCoords {
public:
	int xmin, xmax, ymin, ymax;

	RegionCoords() :xmin(0), xmax(0), ymin(0), ymax(0) {}

	RegionCoords(int xmin_in, int xmax_in, int ymin_in, int ymax_in)
		:xmin(xmin_in), xmax(xmax_in), ymin(ymin_in), ymax(ymax_in) {}

	RegionCoords(const RegionCoords &a);

	RegionCoords& operator +=(const RegionCoords& a);

	RegionCoords& operator /=(const int a);

	friend ofstream& operator <<(ofstream &out, const RegionCoords& r);

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

	// 返回中心坐标
	Point2f center() const {
		return Point2f((xmin + xmax) / 2, (ymin + ymax) / 2);
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
	float overlap(const RegionCoords &r) const;

	// 返回当前区域是否包含在另一个区域内
	bool is_within(const RegionCoords &r) const;
};

// 求一组区域坐标的平均值
RegionCoords mean_of_regions(vector<RegionCoords> &rlist);

// 求一组区域坐标的中值
RegionCoords median_of_regions(vector<RegionCoords> &rlist);

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
// 保存匹配点对, 每一行依次是: test_p_x test_p_y tmpl_p_x tmpl_p_y
void save_match_pts_txt(string txt_path, vector<DMatch>& matches, vector<KeyPoint> &test_kps, vector<KeyPoint> &ref_kps);

// 求4个值中的最大值
int max_of_four(int v0, int v1, int v2, int v3);

// 求4个值中的最小值
int min_of_four(int v0, int v1, int v2, int v3);

// 将值限制在[lower, upper]之间
int clamp_between(int v, int lower, int upper);

#endif // !_UTIL_H
