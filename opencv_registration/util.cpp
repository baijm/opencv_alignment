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

void load_kp_pos_txt(string txt_path, vector<Point2f>& kp)
{
	kp.clear();
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
			// Point2f¹¹Ôìº¯ÊýPoint_(_Tp _x, _Tp _y);
			kp.push_back(Point2f(atof(parts[1].c_str()), atof(parts[0].c_str())));
		}
	}
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

void load_des_txt(string des_path, Mat& des)
{
	
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
				des_txt << int(des.at<uchar>(i, j)) << ' ';
			}
			des_txt << endl;
		}
		des_txt.close();
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
int main()
{
	ifstream imglist("D:\\datasets\\shelf\\patch\\query\\easy\\img_list_bbx_crop.txt");
	string img_dir = "D:\\datasets\\shelf\\patch\\query\\easy\\crop\\";
	string briskfolder = "D:\\datasets\\shelf\\patch\\query\\easy\\crop_brisk\\";		//txt save folder
	string img_name;

	if (!imglist){
		cout << "Unable to open imglist";
		exit(1);
	}

	while (getline(imglist, img_name)){
		save_brisk(img_dir, img_name + ".jpg", briskfolder);
	}

	imglist.close();

	return 0;
}
*/