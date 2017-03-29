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


// 读取matlab输出的word文件, 其中每一行为一个匹配的测试图像特征点索引, 模板图像特征点索引, 相似度
// [因为matlab索引从1开始, 此处所有索引-1]
void load_match_txt(string txt_path, vector<MatchKpsSim>& matches);


// 读取logo矩形, 按照最小y坐标, 最大y坐标, 最小x坐标, 最大x坐标返回
vector<int> load_logo_region(string txt_path);


// 保存brisk特征
void save_brisk(string img_dir, string img_name, string save_dir);
// 保存SIFT特征
//void compute_and_save_sift(string img_name, Mat& im_g, vector<KeyPoint>& kps, string save_dir);


// RANSAC相关
// 随机从[min_idx, max_idx]的下标中选n个不重复的加入到idx
void get_n_idx(int n, int min_idx, int max_idx, vector<int>& res);

// 检查3对点是否共线
// [现在是检查p中3点是否共线, q中3点是否共线]
// TODO : 是这样检查吗?
bool is_colinear_3(vector<Point2f>& p, vector<Point2f>& q);

// 画出特征点匹配关系
// 没减掉中心坐标的版本
Mat draw_MatchKpsSim(const vector<Point2f>& test_pts, const vector<Point2f>& ref_pts, 
	const Mat& test_im, const Mat& ref_im,
	const vector<MatchKpsSim>& matches);

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

