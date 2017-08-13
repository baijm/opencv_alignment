#include "util.h"

// y, x, size, angle
void save_kp_txt(string txt_path, const vector<KeyPoint>& kp)
{
	ofstream kp_txt(txt_path, ios::out);
	if (!kp_txt)
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
		kp_txt.close();
	}

	return res;
}

vector<Point2f> load_kp_pos_txt(string txt_path)
{
	vector<Point2f> res;
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
			// Point2f构造函数Point_(_Tp _x, _Tp _y);
			res.push_back(Point2f(
				atof(parts[1].c_str()),
				atof(parts[0].c_str())
			));
		}
		kp_txt.close();
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

		des_txt.close();

		return res;
	}
}

RegionCoords load_region_txt(string txt_path)
{
	ifstream region_txt(txt_path);
	if (!region_txt)
	{
		cout << "cannot open " << txt_path << " for reading" << endl;
		return RegionCoords();
	}
	else
	{
		string line;
		vector<string> parts;
		getline(region_txt, line);
		boost::split(parts, line, boost::is_any_of(" "));
		return RegionCoords(
			atoi(parts[0].c_str()), // xmin
			atoi(parts[1].c_str()), // xmax
			atoi(parts[2].c_str()), // ymin
			atoi(parts[3].c_str()) // ymax
		);
	}
}

void load_match_txt(string txt_path, vector<DMatch>& matches)
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
			matches.push_back(
				DMatch(atoi(parts[0].c_str()) - 1, // _queryIdx
					atoi(parts[1].c_str()) - 1, // _trainIdx
					0)); // _distance
		}
		txt.close();
	}
}

void load_match_pts_txt(string txt_path, vector<Point2f>& test_pts, vector<Point2f>& ref_pts)
{
	test_pts.clear();
	ref_pts.clear();

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
			test_pts.push_back(Point2f(atof(parts[0].c_str()), atof(parts[1].c_str())));
			ref_pts.push_back(Point2f(atof(parts[2].c_str()), atof(parts[3].c_str())));
		}
		txt.close();
	}
}

// 保存匹配点对, 每一行依次是: test_p_x test_p_y tmpl_p_x tmpl_p_y
void save_match_pts_txt(string txt_path, vector<DMatch>& matches, vector<KeyPoint> &test_kps, vector<KeyPoint> &ref_kps)
{
	ofstream match_txt(txt_path, ios::out);
	if (!match_txt)
	{
		cout << "cannot open " << txt_path << " for writing" << endl;
	}
	else
	{
		for (int mi = 0; mi < matches.size(); mi++)
		{
			Point2f test_p = test_kps[matches[mi].queryIdx].pt;
			Point2f ref_p = ref_kps[matches[mi].trainIdx].pt;
			match_txt << test_p.x << " " << test_p.y << " " << ref_p.x << " " << ref_p.y << endl;

		}
		match_txt.close();
	}
}