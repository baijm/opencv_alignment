#include "util.h"


// y, x, size, angle
void save_kp_txt(string txt_path, const vector<KeyPoint>& kp)
{
	ofstream kp_txt(txt_path, ios::out);
	if(!kp_txt)
	{
		cout << "cannot open " << txt_path << " for writing" << endl;
	}
	else
	{
		for (int i = 0; i < kp.size(); i++) {
			kp_txt << kp[i].pt.y << ' ' << kp[i].pt.x << ' ' << kp[i].size << ' ' << kp[i].angle << endl;
		}
		kp_txt.close();
	}
}

vector<KeyPoint> load_kp_txt(string txt_path)
{
	vector<KeyPoint> res;

	ifstream kp_txt(txt_path);
	if (!kp_txt)
	{
		cout << "cannot open " << txt_path << " for reading" << endl;
	}
	else
	{
		string line;
		vector<string> parts;
		while (getline(kp_txt, line))
		{
			boost::split(parts, line, boost::is_any_of(" "));
			res.push_back(KeyPoint(
				atof(parts[1].c_str()), // x
				atof(parts[0].c_str()), // y
				atof(parts[2].c_str()), // size
				atof(parts[3].c_str())// angle
			));
		}
	}

	return res;
}

vector<Point2f> load_kp_pos_txt(string txt_path)
{
	vector<Point2f> res;

	ifstream kp_txt(txt_path);
	if(!kp_txt)
	{
		cout << "cannot open " << txt_path << " for reading" << endl;
	}
	else
	{
		string line;
		vector<string> parts;
		while(getline(kp_txt, line))
		{
			boost::split(parts, line, boost::is_any_of(" "));
			// Point2f构造函数Point_(_Tp _x, _Tp _y);
			res.push_back(Point2f(
				atof(parts[1].c_str()), 
				atof(parts[0].c_str())
			));
		}
	}

	return res;
}


void save_des_txt(string des_path, const Mat& des)
{
	ofstream des_txt(des_path, ios::out);
	if (!des_txt)
	{
		cout << "cannot open " << des_path << " for writing" << endl;
	}
	else
	{
		for (int i = 0; i < des.rows; i++) {
			for (int j = 0; j < des.cols; j++) {
				//des_txt << int(des.at<uchar>(i, j)) << ' ';
				des_txt << *(float *)(des.data + des.step[0] * i + des.step[1] * j) << " ";
			}
			des_txt << endl;
		}
		des_txt.close();
	}
}

Mat load_des_txt(string des_path)
{
	ifstream des_txt(des_path);
	if (!des_txt)
	{
		cout << "cannot open " << des_path << " for reading" << endl;

		Mat res;
		return res;
	}
	else
	{
		string line;
		vector<string> parts;
		vector<vector<float>> buffer;
		while (getline(des_txt, line))
		{
			boost::split(parts, line, boost::is_any_of(" "));
			parts.pop_back(); // 去掉最后一个" "

			vector<float> tmp;
			for (int pi = 0; pi < parts.size(); pi++)
			{
				tmp.push_back(atof(parts[pi].c_str()));
			}
			buffer.push_back(tmp);
		}

		//CV_32F
		Mat res(buffer.size(), buffer[0].size(), CV_32F);
		for (int pi = 0; pi < buffer.size(); pi++)
		{
			for (int ii = 0; ii < buffer[pi].size(); ii++)
			{
				res.at<float>(pi, ii) = buffer[pi][ii];
			}
		}

		return res;
	}
}


vector<int> load_logo_region(string txt_path)
{
	vector<int> res;
	
	ifstream region_txt(txt_path);
	if (!region_txt)
	{
		cout << "cannot open " << txt_path << " for reading" << endl;
	}
	else
	{
		string line;
		vector<string> parts;

		getline(region_txt, line);
		boost::split(parts, line, boost::is_any_of(" "));
		for (int pi = 0; pi < 4; pi++)
		{
			res.push_back(atoi(parts[pi].c_str()));
		}
	}

	return res;
}


void load_match_txt(string txt_path, vector<MatchKpsSim>& matches)
{
	matches.clear();

	ifstream txt(txt_path);
	if (!txt)
	{
		cout << "cannot open " << txt_path << "for reading" << endl;
	}
	else
	{
		string line;
		vector<string> parts;
		while (getline(txt, line))
		{
			boost::split(parts, line, boost::is_any_of(" "));
			matches.push_back(MatchKpsSim(atoi(parts[0].c_str())-1, atoi(parts[1].c_str())-1, atof(parts[2].c_str())));
		}
	}
}

void save_brisk(string img_dir, string img_name, string save_dir)
{
	string imgpath = img_dir + img_name;
	Mat src = imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);
	if (!src.data)
	{
		std::cout << "Error reading images " << std::endl;
		return;
	}

	//feature detect  
	BRISK detector;
	vector<KeyPoint> kp;
	detector.detect(src, kp);		//keypoint

	Mat des;//descriptor  
	detector.compute(src, kp, des);

	string txtname = img_name.substr(0, img_name.length()-4);
	string kpname = save_dir + txtname + "_kp.txt";
	save_kp_txt(kpname, kp);
	string desname = save_dir + txtname + "_des.txt";
	save_des_txt(desname, des);
}

/*
void compute_and_save_sift(string img_name, Mat& im_g, vector<KeyPoint>& kps, string save_dir)
{
	// feature detect
	cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();

	Mat des;
	detector->compute(im_g, kps, des);

	string des_name = save_dir + "/" + img_name + "_des.txt";
	save_des_txt(des_name, des);
}
*/

// RANSAC相关
// 随机从[min_idx, max_idx]的下标中选n个不重复的加入到idx
void get_n_idx(int n, int min_idx, int max_idx, vector<int>& res)
{
	res.clear();

	while (res.size() != n)
	{
		srand((int)time(NULL));
		int idx = rand()%(max_idx-min_idx+1)+min_idx;
		if (find(res.begin(), res.end(), idx) != res.end())
		{
			continue;
		}
		else
		{
			res.push_back(idx);
		}
	}
}

// 检查3对点是否共线
bool is_colinear_3(vector<Point2f>& p, vector<Point2f>& q)
{
	// 如果不是3对点, 返回true
	if (p.size() != 3 || q.size() != 3)
	{
		return true;
	}

	// 如果有重复点, 返回true
	if (p[0] == p[1] || p[0] == p[2] || p[1] == p[2] || q[0] == q[1] || q[0] == q[2] || q[1] == q[2])
	{
		return true;
	}

	Point2f p01(p[1].x - p[0].x, p[1].y - p[0].y);
	Point2f p02(p[2].x - p[0].x, p[2].y - p[0].y);
	Point2f q01(q[1].x - q[0].x, q[1].y - q[0].y);
	Point2f q02(q[2].x - q[0].x, q[2].y - q[0].y);

	return p01.cross(p02) == 0 || q01.cross(q02) == 0;
}

// 画出特征点匹配关系
// 没减掉中心坐标的版本
Mat draw_MatchKpsSim(const vector<Point2f>& test_pts, const vector<Point2f>& ref_pts,
	const Mat& test_im, const Mat& ref_im,
	const vector<MatchKpsSim>& matches)
{
	// (Point2f转换成KeyPoint)
	vector<KeyPoint> test_kps, ref_kps;
	KeyPoint::convert(test_pts, test_kps);
	KeyPoint::convert(ref_pts, ref_kps);

	// (MatchKpsSim转换成DMatch)
	vector<DMatch> dmatches;
	for (vector<MatchKpsSim>::const_iterator ite = matches.begin(); ite != matches.end(); ite++)
	{
		dmatches.push_back(DMatch(ite->test_pid, ite->tmpl_pid, 0));
	}

	// 用drawMatches画出来
	Mat match_im;
	drawMatches(test_im, test_kps,
		ref_im, ref_kps,
		dmatches, match_im);
	
	return match_im;
}

// 优化相关
// 目标函数
// a = {a11, a12, a13, a21, a22, a23}
double obj_func(const vector<double> &a, vector<double> &grad, void *func_data)
{
	ObjectiveFunctionData *data = reinterpret_cast<ObjectiveFunctionData*>(func_data);
	// TODO : 用引用给引用赋值会怎么样??
	vector<DMatch> *matches = data->matches;
	vector<Point2f> *test_pts = data->test_pts;
	vector<Point2f> *ref_pts = data->tmpl_pts;

	if (!grad.empty())
	{
		for (int gi = 0; gi < grad.size(); gi++)
		{
			grad[gi] = 0;
		}
	}

	// 求目标函数的值
	double val = 0;
	for (int i = 0; i < matches->size(); i++)
	{
		Point2f p_test = (*test_pts)[(*matches)[i].queryIdx];
		Point2f p_ref = (*ref_pts)[(*matches)[i].trainIdx];

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

/*
int main()
{
	string root_dir = "D:/datasets/vobile_project/shelf/shampoo/test/12_classified_2852";
	string cls_name = "2-prjyqx";

	ifstream img_list(root_dir + "/" + cls_name + ".txt");
	string img_dir = root_dir + "/" + cls_name;
	string kp_dir = 

	string res_dir = "D:/datasets/vobile_project/shelf/shampoo/test/15_x_split_sift_txt";
	res_dir += ("/" + cls_name);

	if (!img_list){
		cout << "Unable to open imglist";
		exit(1);
	}

	string img_name;
	while (getline(img_list, img_name)){
		save_brisk(img_dir, img_name + ".jpg", briskfolder);
	}

	imglist.close();

	return 0;
}
*/