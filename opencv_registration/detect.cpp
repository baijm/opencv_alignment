#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <boost/filesystem.hpp> 

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

/************************************************************************/
/* ������ͼ����ÿ��ѵ��ͼ����׼, ������׼��Ĳ���ͼ����м�ƥ����
1. ��ȡͼ��, ת����ɫ�ռ�
2. ��ȡ����
3. ƥ��
4. ��Ӳ���ͼ��ѵ��ͼ���ͶӰ����
5. �Բ���ͼ����б任 */
/************************************************************************/

int main()
{
	string train_dir = "D:/datasets/shelf/patch/ref/small/raw"; // ģ��ͼ��Ŀ¼
	string test_dir = "D:/datasets/shelf/patch/query/new/raw/easy"; // ����ͼ��Ŀ¼
	string res_dir = "D:/datasets/shelf/patch/result/no_resize_easy_cross_bf";
	string res_reg_dir = res_dir+"/registrated_query"; // ��׼֮���µĲ���ͼ��Ŀ¼
	string res_mid_dir = res_dir+"/match_result"; // ƥ����
	string ext = ".jpg";

	/************************************************************************/
	/* ׼������: ��ȡ����ģ��ͼ��Ͳ���ͼ������, (�½����Ŀ¼)                                                                     */
	/************************************************************************/
	// ��ȡ����ģ��ͼ��Ͳ���ͼ������
	fs::path train_path(train_dir);
	if(!fs::exists(train_path))
	{
		std::cout << "train_dir not exist" << std::endl;
		return -1;
	}
	fs::path test_path(test_dir);
	if(!fs::exists(test_path))
	{
		std::cout << "test_dir not exist" << std::endl;
	}
	// ������չ����ͼ����
	vector<string> train_names;
	vector<string> test_names;

	fs::directory_iterator end_iter;
	for(fs::directory_iterator iter(train_path); iter != end_iter; ++iter)
	{
		if(fs::is_regular_file(iter->status()) && iter->path().extension().string() == ext)
		{
			train_names.push_back(iter->path().stem().string());
		}
	}
	for(fs::directory_iterator iter(test_path); iter != end_iter; ++iter)
	{
		if(fs::is_regular_file(iter->status()) && iter->path().extension().string() == ext)
		{
			test_names.push_back(iter->path().stem().string());
		}
	}

	// �������ļ��в��������½�
	fs::path res_reg_path(res_reg_dir);
	if(!fs::exists(res_reg_path))
	{
		fs::create_directory(res_reg_path);
		std::cout << "res_reg_dir not exist, created" << std::endl;
	}
	fs::path res_mid_path(res_mid_dir);
	if(!fs::exists(res_mid_path))
	{
		fs::create_directory(res_mid_path);
		std::cout << "res_mid_dir not exist, created" << std::endl;
	}

	// log�ļ�
	std::ofstream log_file;
	log_file.open(res_reg_dir + "/log.txt", std::ios::out);

	log_file << train_names.size() << " files in train_dir, " << test_names.size() << " files in test_dir\n";
	std::cout << train_names.size() << " files in train_dir, " << test_names.size() << " files in test_dir\n";

	// ��������, ��ʾ�м�������׼���
	namedWindow("match_result");
	namedWindow("registration_result");

	BRISK detector;
	BFMatcher matcher(NORM_HAMMING);

	for(vector<string>::const_iterator test_iter = test_names.begin(); test_iter != test_names.end(); ++test_iter)
	{
		log_file << "test image : " << *test_iter;
		std::cout << "test image : " << *test_iter;

		// ��ȡ����ͼ��
		Mat test_g = imread((test_path / (*test_iter + ext)).string(), CV_LOAD_IMAGE_GRAYSCALE);
		if(!test_g.data)
		{
			std::cout << "error when reading test image" << std::endl;
			return -1;
		}

		// ���keypoint
		vector<KeyPoint> test_kp;
		detector.detect(test_g, test_kp);
		log_file << ", " << test_kp.size() << " keypoints in test image\n";
		std::cout << ", " << test_kp.size() << " keypoints in test image\n";

		// ����descriptor
		Mat test_des;
		detector.compute(test_g, test_kp, test_des);

		// ��ÿ��ѵ��ͼ����׼
		for (vector<string>::const_iterator train_iter = train_names.begin(); train_iter != train_names.end(); ++train_iter)
		{
			log_file << "\ttrain image : " << *train_iter;
			std::cout << "\ttrain image : " << *train_iter;

			// ��ȡѵ��ͼ��
			Mat train_g = imread((train_path / (*train_iter + ext)).string(), CV_LOAD_IMAGE_GRAYSCALE);
			if (!train_g.data)
			{
				std::cout << "error when reading train image" << std::endl;
				return -1;
			}

			// ���keypoint
			vector<KeyPoint> train_kp;
			detector.detect(train_g, train_kp);
			log_file << ", " << train_kp.size() << " keypoints in train image";
			std::cout << ", " << train_kp.size() << " keypoints in train image";

			// ����descriptor
			Mat train_des;
			detector.compute(train_g, train_kp, train_des);

			// ---------- ƥ�� ----------
			/*
			// [��brute force matcherƥ��descriptor]
			vector<DMatch> matches;
			matcher.match(test_des, train_des, matches);
			*/
			// ��2NNƥ��: ������һ��ƥ����ڶ���ƥ��֮��ľ����㹻Сʱ, ����Ϊ����һ��ƥ��
			const int k = 2;
			const float min_ratio = 1.f / 1.5f;
			vector<vector<DMatch>> knn_matches;
			vector<DMatch> matches;
			matcher.knnMatch(test_des, train_des, knn_matches, k);
			for (int kmi = 0; kmi < knn_matches.size(); kmi++)
			{
				const DMatch& first_match = knn_matches[kmi][0];
				const DMatch& second_match = knn_matches[kmi][1];
				float dist_ratio = first_match.distance / second_match.distance;
				if (dist_ratio < min_ratio)
				{
					matches.push_back(first_match);
				}
			}
			log_file << ", " << matches.size() << " good matches";
			std::cout << ", " << matches.size() << " good matches";

			// ����ƥ����������, ���ڼ��
			Mat match_res;
			fs::path match_res_file = res_mid_path / (*test_iter + "_" + *train_iter + ext);
			drawMatches(test_g, test_kp, 
				train_g, train_kp, 
				matches, match_res);
			imshow("match_result", match_res);
			imwrite(match_res_file.string(), match_res);

			// ɸѡ������
			/*
			// ǰN������ƥ��
			int N = std::min(20, (int)matches.size());
			sort(matches.begin(), matches.end());
			matches.erase(matches.begin()+N, matches.end());

			log_file << ", keep " << matches.size() << " matches";
			std::cout << ", keep " << matches.size() << " matches";
			*/

			vector<Point2f> test_pts, train_pts;
			for (int mi = 0; mi < matches.size(); mi++)
			{
				test_pts.push_back(test_kp[matches[mi].queryIdx].pt);
				train_pts.push_back(train_kp[matches[mi].trainIdx].pt);
			}

			// ---------- ��Ӳ���ͼ��ѵ��ͼ���ͶӰ���� ----------
			vector<uchar> homo_stat;
			//Homography����һ���任(3*3����), ��һ��ͼ�еĵ�ӳ�䵽��һ��ͼ�ж�Ӧ�ĵ�; ��������֪��4����ͬ��Ӧλ�õĵ�
			// Split matches to inliers and outliers using homography estimation
			Mat homo = findHomography(test_pts, train_pts, CV_RANSAC, 3, homo_stat);
			log_file << ", " << countNonZero(Mat(homo_stat)) << " inliers\n";
			std::cout << ", " << countNonZero(Mat(homo_stat)) << " inliers\n";

			// TODO: ����inliers


			// ---------- �Բ���ͼ����б任������ ----------
			Mat test_r;
			warpPerspective(test_g, test_r, homo, Size(train_g.cols, train_g.rows));

			fs::path test_r_file = res_reg_path / (*test_iter + "_" + *train_iter + ext);
			imshow("registration_result", test_r);
			imwrite(test_r_file.string(), test_r);

			waitKey();
		}
	}

	destroyAllWindows();

	/*
	The following metrics are calculated:

	Percent of matches - quotient of dividing matches count on the minimum of keypoints count on two frames in percents.
	Percent of correct matches - quotient of dividing correct matches count on total matches count in percents.
	Matching ratio - percent of matches * percent of correct matches.
	In all charts i will use "Matching ratio" ( in percents) value for Y-axis.
	*/

	log_file.close();

	return 0;
}