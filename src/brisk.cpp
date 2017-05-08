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



const int min_matches_allowed = 4; // fixed

const int dist_thresh = 90; // fixed

const bool cross_check = true;  // fixed

const double reproj_thresh = 3.0; // fixed

const bool use_contour_filter = false; // fixed



int main()

{

	// ģ��ͼ��Ŀ¼

	string train_orig_dir = "F:/datasets/patch/ref/all/crop";  // ��ɫͼ

	string train_mask_dir = "F:/datasets/patch/ref/all/density"; // �����ܶ�ͼ

	// ����ͼ��Ŀ¼

	string test_orig_dir = "F:/datasets/patch/query/new/all/crop"; // ��ɫͼ

	string test_mask_dir = "F:/datasets/patch/query/new/all/bbx_density"; // �����ܶ�ͼ

	// ���Ŀ¼

	string res_dir = "F:/datasets/patch/result/brisk/bf_cross_all_proj3_90_crop";

	string res_reg_dir = res_dir+"/registrated_query"; // ��׼֮���µĲ���ͼ��Ŀ¼

	string res_mid_dir = res_dir+"/match_result"; // ƥ����

	string orig_ext = ".jpg";

	string mask_ext = ".png";



	/************************************************************************/

	/* ׼������: ��ȡ����ģ��ͼ��Ͳ���ͼ������, (�½����Ŀ¼)                                                                     */

	/************************************************************************/

	// ��ȡ����ģ��ͼ��Ͳ���ͼ������

	fs::path train_orig_path(train_orig_dir);

	if(!fs::exists(train_orig_path))

	{

		cout << "train_orig_dir not exist" << endl;

		return -1;

	}

	fs::path train_mask_path(train_mask_dir);

	if (!fs::exists(train_mask_path))

	{

		cout << "train_mask_dir not exist" << endl;

		return -1;

	}

	fs::path test_orig_path(test_orig_dir);

	if(!fs::exists(test_orig_path))

	{

		cout << "test_orig_dir not exist" << endl;

		return -1;

	}

	fs::path test_mask_path(test_mask_dir);

	if (!fs::exists(test_mask_path))

	{

		cout << "test_mask_dir not exist" << endl;

		return -1;

	}

	

	// ������չ����ͼ����

	vector<string> train_names;

	vector<string> test_names;



	fs::directory_iterator end_iter;

	for(fs::directory_iterator iter(train_orig_path); iter != end_iter; ++iter)

	{

		if(fs::is_regular_file(iter->status()) && iter->path().extension().string() == orig_ext)

		{

			train_names.push_back(iter->path().stem().string());

		}

	}

	for(fs::directory_iterator iter(test_orig_path); iter != end_iter; ++iter)

	{

		if(fs::is_regular_file(iter->status()) && iter->path().extension().string() == orig_ext)

		{

			test_names.push_back(iter->path().stem().string());

		}

	}



	// �������ļ��в��������½�

	fs::path res_reg_path(res_reg_dir);

	if(!fs::exists(res_reg_path))

	{

		fs::create_directories(res_reg_path);

		cout << "res_reg_dir not exist, created" << endl;

	}

	fs::path res_mid_path(res_mid_dir);

	if(!fs::exists(res_mid_path))

	{

		fs::create_directories(res_mid_path);

		cout << "res_mid_dir not exist, created" << endl;

	}



	// log�ļ�

	ofstream log_file;

	log_file.open(res_reg_dir + "/log.txt", ios::out);



	log_file << train_names.size() << " files in train_orig_dir, " << test_names.size() << " files in test_orig_dir\n";

	cout << train_names.size() << " files in train_orig_dir, " << test_names.size() << " files in test_orig_dir\n";



	// ��������, ��ʾ�м�������׼���

	//namedWindow("match before filter");

	//namedWindow("match after filter");

	//namedWindow("match after findHomography");

	//namedWindow("result");



	/************************************************************************/

	/* ��׼����                                                                     */

	/************************************************************************/

	BRISK detector;

	BFMatcher matcher(NORM_HAMMING, cross_check);



	for(vector<string>::const_iterator test_iter = test_names.begin(); test_iter != test_names.end(); ++test_iter)

	{

		log_file << "test image : " << *test_iter;

		cout << "test image : " << *test_iter;



		// ��ȡ����ͼ��

		Mat test_c = imread((test_orig_path / (*test_iter + orig_ext)).string()); // ��ɫͼ

		Mat test_g = imread((test_orig_path / (*test_iter + orig_ext)).string(), CV_LOAD_IMAGE_GRAYSCALE); // �Ҷ�ͼ

		Mat test_mask = imread((test_mask_path / (*test_iter + mask_ext)).string(), CV_LOAD_IMAGE_GRAYSCALE);

		if(!test_c.data || !test_g.data || !test_mask.data)

		{

			cout << "error when reading test image" << endl;

			return -1;

		}



		// ���keypoint

		vector<KeyPoint> test_kp;

		detector.detect(test_g, test_kp);



		// ��mask����������, ֻ������������������е�keypoint

		if (use_contour_filter)

		{

			vector<vector<Point>> contours;

			findContours(test_mask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);



			double area_max = 0;

			int contour_max_idx = 0;

			for (int ci = 0; ci < contours.size(); ci++)

			{

				if (contourArea(contours[ci]) > area_max)

				{

					area_max = contourArea(contours[ci]);

					contour_max_idx = ci;

				}

			}



			for (vector<KeyPoint>::iterator kp_ite = test_kp.begin(); kp_ite != test_kp.end(); )

			{

				if (pointPolygonTest(contours[contour_max_idx], kp_ite->pt, false) == 1)

				{

					kp_ite++;

				}

				else

				{

					kp_ite = test_kp.erase(kp_ite);

				}

			}

		}



		log_file << ", " << test_kp.size() << " keypoints in test image\n";

		cout << ", " << test_kp.size() << " keypoints in test image\n"; 



		// ���û������keypoint, ������

		if (test_kp.size() == 0)

		{

			continue;

		}



		// ����descriptor

		Mat test_des;

		detector.compute(test_g, test_kp, test_des);



		// ��ÿ��ѵ��ͼ����׼

		for (vector<string>::const_iterator train_iter = train_names.begin(); train_iter != train_names.end(); ++train_iter)

		{

			log_file << "\ttrain image : " << *train_iter;

			cout << "\ttrain image : " << *train_iter;



			// ��ȡѵ��ͼ��

			Mat train_g = imread((train_orig_path / (*train_iter + orig_ext)).string(), CV_LOAD_IMAGE_GRAYSCALE);

			Mat train_mask = imread((train_mask_path / (*test_iter + mask_ext)).string(), CV_LOAD_IMAGE_GRAYSCALE);

			if (!train_g.data || !train_mask.data)

			{

				cout << "error when reading train image" << endl;

				return -1;

			}



			// ���keypoint

			vector<KeyPoint> train_kp;

			detector.detect(train_g, train_kp);



			// ��mask����������, ֻ������������������е�keypoint

			if (use_contour_filter)

			{

				vector<vector<Point>> train_contours;

				findContours(train_mask, train_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);



				double area_max = 0;

				int contour_max_idx = 0;

				for (int ci = 0; ci < train_contours.size(); ci++)

				{

					if (contourArea(train_contours[ci]) > area_max)

					{

						area_max = contourArea(train_contours[ci]);

						contour_max_idx = ci;

					}

				}



				for (vector<KeyPoint>::iterator kp_ite = train_kp.begin(); kp_ite != train_kp.end(); )

				{

					if (pointPolygonTest(train_contours[contour_max_idx], kp_ite->pt, false) == 1)

					{

						kp_ite++;

					}

					else

					{

						kp_ite = train_kp.erase(kp_ite);

					}

				}

			}



			log_file << ", " << train_kp.size() << " keypoints in train image";

			cout << ", " << train_kp.size() << " keypoints in train image";



			// ���û������������, ������

			if (train_kp.size() == 0)

			{

				continue;

			}



			// ����descriptor

			Mat train_des;

			detector.compute(train_g, train_kp, train_des);



			// ---------- ƥ�� ----------

			// ��brute force matcherƥ��descriptor

			vector<DMatch> matches;

			matcher.match(test_des, train_des, matches);



			log_file << ", get " << matches.size() << " matches";

			cout << ", get " << matches.size() << " matches";

			Mat match_res;

			fs::path match_res_file = res_mid_path / (*test_iter + "_" + *train_iter + orig_ext);

			drawMatches(test_g, test_kp, 

				train_g, train_kp, 

				matches, match_res);

			//imshow("match before filter", match_res);

			//imwrite(match_res_file.string(), match_res);



			// ɸѡ���

			// distance threshold (�ο�����)

			vector<DMatch> good_matches;

			for(int mi = 0; mi < matches.size(); mi++)

			{

				if(matches[mi].distance < dist_thresh)

				{

					good_matches.push_back(matches[mi]);

				}

			}



			log_file << ", keep " << good_matches.size() << " good matches";

			cout << ", keep " << good_matches.size() << " good matches";

			Mat match_good_res;

			fs::path match_good_res_file = res_mid_path / (*test_iter + "_" + *train_iter + "_good" + orig_ext);

			drawMatches(test_g, test_kp,

				train_g, train_kp,

				good_matches, match_good_res);

			//imshow("match after filter", match_good_res);

			imwrite(match_good_res_file.string(), match_good_res);



			// ������µĵ�Բ������ڼ���homography, ����ǰ����

			if (good_matches.size() < min_matches_allowed)

			{

				log_file << ", not enough good matches\n";

				cout << ", not enough good matches" << endl;

				continue;

			}



			vector<Point2f> test_pts, train_pts;

			for (int gi = 0; gi < good_matches.size(); gi++)

			{

				test_pts.push_back(test_kp[good_matches[gi].queryIdx].pt);

				train_pts.push_back(train_kp[good_matches[gi].trainIdx].pt);

			}



			// ---------- ��Ӳ���ͼ��ѵ��ͼ���ͶӰ���� ----------

			vector<uchar> homo_stat;

			//Homography����һ���任(3*3����), ��һ��ͼ�еĵ�ӳ�䵽��һ��ͼ�ж�Ӧ�ĵ�; ��������֪��4����ͬ��Ӧλ�õĵ�

			Mat homo = findHomography(test_pts, train_pts, CV_RANSAC, reproj_thresh, homo_stat);

			vector<DMatch> inlier_matches;

			for (int gi = 0; gi < good_matches.size(); gi++)

			{

				if(homo_stat[gi] != 0)

				{

					inlier_matches.push_back(good_matches[gi]);

				}

			}



			log_file << ", " << countNonZero(Mat(homo_stat)) << " inliers\n";

			cout << ", " << countNonZero(Mat(homo_stat)) << " inliers\n";

			Mat match_inlier_res;

			fs::path match_inlier_res_file = res_mid_path / (*test_iter + "_" + *train_iter + "_inlier" + orig_ext);

			drawMatches(test_g, test_kp, 

				train_g, train_kp, 

				inlier_matches, match_inlier_res);

			//imshow("match after findHomography", match_inlier_res);

			imwrite(match_inlier_res_file.string(), match_inlier_res);



			// ---------- �Բ���ͼ����б任������ ----------

			Mat test_r;

			warpPerspective(test_c, test_r, homo, Size(train_g.cols, train_g.rows));



			fs::path test_r_file = res_reg_path / (*test_iter + "_" + *train_iter + orig_ext);

			//imshow("result", test_r);

			imwrite(test_r_file.string(), test_r);



			//waitKey();

		}

	}



	//destroyAllWindows();



	log_file.close();



	return 0;

}