#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>

#include "my_detector.h"

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;


/*
��С��Ŀ�꺯�� \sum_{i=1}^N ||Mp_{i} - p_{i'}||_2
����NΪƥ����, p_{i}Ϊ����ͼ��������������, p_{i'}Ϊģ��ͼ��������������
*/

const int MATCHES_NUM = 4;
const float PORTION_DIFF_THRES = 0.15;

int main()
{
	// ͼ��λ��
	string test_im_dir = "D:/datasets/shelf/patch/query/easy/crop";
	string ref_im_dir = "D:/datasets/shelf/patch/ref/all/crop";

	// ͼ�����б�
	string test_name_file = "D:/datasets/shelf/patch/query/easy/img_list_bbx_crop.txt";
	string ref_name_file = "D:/datasets/shelf/patch/ref/all/img_list_3_per_class.txt";

	// ���Ŀ¼
	string res_dir = "D:/datasets/shelf/patch/query/easy/alignment/sift/cross_no_filter_reject_strict";
	string res_kp_dir = res_dir + "/keypoint"; // ����������ȡ���
	string res_match_dir = res_dir + "/match"; // ����ƥ����
	string res_align_dir = res_dir + "/result"; // ���������
	string res_crop_dir = res_dir + "/crop"; // ����ü����

	// ��������
	string feature = "SIFT";
	// �������ⷽ��
	MyDetector detector(feature);

	// ƥ�䷽��
	Ptr<DescriptorMatcher> matcher;
	if (feature == "BRISK")
	{
		matcher = BFMatcher::create(NORM_HAMMING, true);
	}
	else
	{
		matcher = BFMatcher::create(NORM_L2, true);
	}

	// ƥ��ɸѡ����
	bool use_filter = false;
	MyMatchFilter *filter = NULL;
	if (use_filter)
	{
		if (feature == "BRISK")
		{
			filter = new DistThresFilter();
		}
		else
		{
			filter = new DistPortionFilter();
		}
	}

	// ���������ķ���
	string estimate = "RANSAC";
	MyAffineEstimator *estimator = NULL;
	if (estimate == "RANSAC")
	{
		estimator = new RansacAffineEstimator();
	}
	else
	{
		estimator = new Point4AffineEstimator();
	}

	/************************************************************************/
	/* ׼������ : ���Ŀ¼�Ƿ����, �����Ժ�ģ��ͼ����, �½����Ŀ¼                                                                     */
	/************************************************************************/
	// ���ͼ��Ŀ¼
	fs::path test_im_path(test_im_dir);
	if (!fs::exists(test_im_path))
	{
		cout << "test_im_dir not exist" << endl;
		return -1;
	}
	
	fs::path ref_im_path(ref_im_dir);
	if (!fs::exists(ref_im_path))
	{
		cout << "ref_im_dir not exist" << endl;
		return -1;
	}
	
	// ���ͼ�����б��ļ��Ƿ����, ����������ͼ�����б�
	vector<string> test_names;
	fs::path test_name_path(test_name_file);
	if (!fs::exists(test_name_path) || !fs::is_regular_file(test_name_path))
	{
		cout << "test_name_file not exist or is not regular file" << endl;
		return -1;
	}
	else
	{
		ifstream txt(test_name_file);
		string line;
		while (getline(txt, line))
		{
			test_names.push_back(line);
		}
	}

	vector<string> ref_names;
	fs::path ref_name_path(ref_name_file);
	if (!fs::exists(ref_name_path) || !fs::is_regular_file(ref_name_path))
	{
		cout << "ref_name_file not exist or is not regular file" << endl;
		return -1;
	}
	else
	{
		ifstream txt(ref_name_file);
		string line;
		while (getline(txt, line))
		{
			ref_names.push_back(line);
		}
	}

	// �½�����ļ���
	fs::path res_kp_path(res_kp_dir);
	if (!fs::exists(res_kp_path))
	{
		fs::create_directories(res_kp_path);
		cout << "res_kp_dir not exist, created" << endl;
	}

	fs::path res_match_path(res_match_dir);
	if (!fs::exists(res_match_path))
	{
		fs::create_directories(res_match_path);
		cout << "res_match_dir not exist, created" << endl;
	}

	fs::path res_align_path(res_align_dir);
	if (!fs::exists(res_align_path))
	{
		fs::create_directories(res_align_path);
		cout << "res_align_path not exist, created" << endl;
	}

	fs::path res_crop_path(res_crop_dir);
	if (!fs::exists(res_crop_path))
	{
		fs::create_directories(res_crop_path);
		cout << "res_crop_path not exist, created" << endl;
	}

	/************************************************************************/
	/* ����                                                                     */
	/************************************************************************/
	// ��ÿ������ͼ��
	for (vector<string>::const_iterator test_iter = test_names.begin(); test_iter != test_names.end(); test_iter++)
	{
		// ��ͼ��
		Mat test_im_c = imread(test_im_dir + '/' + *test_iter + ".jpg");
		Mat test_im_g;
		cvtColor(test_im_c, test_im_g, CV_BGR2GRAY);

		// ͼ��ߴ�
		Size test_im_size = test_im_c.size();

		// ���������
		vector<KeyPoint> test_kps;
		detector.detect(test_im_g, test_kps);

		// �������������㲢����
		Mat test_kps_im;
		drawKeypoints(test_im_c, test_kps, test_kps_im);
		fs::path test_kps_im_file = res_kp_path
			/ (*test_iter + ".jpg");
		imwrite(test_kps_im_file.string(), test_kps_im);

		// ������������
		Mat test_des;
		detector.compute(test_im_g, test_kps, test_des);

		// ��ÿ��ģ��ͼ��
		for (vector<string>::const_iterator ref_iter = ref_names.begin(); ref_iter != ref_names.end(); ref_iter++)
		{
			cout << "test image : " << *test_iter << "\tref image : " << *ref_iter << endl;

			// ��ͼ��
			Mat ref_im_c = imread(ref_im_dir + '/' + *ref_iter + ".jpg");
			Mat ref_im_g;
			cvtColor(ref_im_c, ref_im_g, CV_BGR2GRAY);

			// ͼ��ߴ�
			Size ref_im_size = ref_im_c.size();

			// ���������
			vector<KeyPoint> ref_kps;
			detector.detect(ref_im_g, ref_kps);

			cout << "\ttest image has " << test_kps.size() << " keypoints, ref image has " << ref_kps.size() << " keypoints" << endl;

			// �������������㲢����
			Mat ref_kps_im;
			drawKeypoints(ref_im_c, ref_kps, ref_kps_im);
			fs::path ref_kps_im_file = res_kp_path
				/ (*ref_iter + ".jpg");
			imwrite(ref_kps_im_file.string(), ref_kps_im);

			// ������������
			Mat ref_des;
			detector.compute(ref_im_g, ref_kps, ref_des);

			// ƥ��
			vector<DMatch> matches;
			matcher->match(test_des, ref_des, matches);
			cout << "\t" << matches.size() << " matches" << endl;

			// ����ȫ��������ƥ���ϵ������
			Mat match_all_im;
			drawMatches(test_im_c, test_kps,
				ref_im_c, ref_kps,
				matches, match_all_im);
			fs::path match_all_im_file = res_match_path
				/ (*test_iter + "_" + *ref_iter + "_all.jpg");
			imwrite(match_all_im_file.string(), match_all_im);

			// ɸѡƥ��
			if (use_filter)
			{
				filter->filter(matches);
			}
			
			// ����ɸѡ���������ƥ���ϵ������
			Mat match_filt_im;
			drawMatches(test_im_c, test_kps,
				ref_im_c, ref_kps,
				matches, match_filt_im);
			fs::path match_filt_im_file = res_match_path
				/ (*test_iter + "_" + *ref_iter + "_filter.jpg");
			imwrite(match_filt_im_file.string(), match_filt_im);

			/************************************************************************/
			/* ��������                                                                     */
			/************************************************************************/
			Mat A_mat;
			cout << "\t";
			if (!estimator->estimate_affine_matrix(test_kps, ref_kps, matches, A_mat))
			{
				cout << "failed to estimate affine matrix" << endl;
			}
			Point2f test_center = estimator->test_center;
			Point2f ref_center = estimator->ref_center;

			// ����inliers������
			Mat match_inlier_im;
			drawMatches(test_im_c, test_kps,
				ref_im_c, ref_kps,
				estimator->inliers, match_inlier_im);
			fs::path match_inlier_im_file = res_match_path
				/ (*test_iter + "_" + *ref_iter + "_inliers.jpg");
			imwrite(match_inlier_im_file.string(), match_inlier_im);

			/************************************************************************/
			/* �任����ͼ��                                                                     */
			/************************************************************************/
			// �任ǰ, ����ͼ��ԭ���ƶ�������
			Mat T_mat_pre = Mat::zeros(3, 3, CV_64F);
			T_mat_pre.at<double>(0, 0) = T_mat_pre.at<double>(1, 1) = T_mat_pre.at<double>(2, 2) = 1;
			T_mat_pre.at<double>(0, 2) = -test_center.x;
			T_mat_pre.at<double>(1, 2) = -test_center.y;

			// �Ż���ķ������
			Mat A_mat_h = Mat::zeros(3, 3, A_mat.type());
			A_mat_h.at<double>(0, 0) = A_mat.at<double>(0, 0);
			A_mat_h.at<double>(0, 1) = A_mat.at<double>(0, 1);
			A_mat_h.at<double>(0, 2) = A_mat.at<double>(0, 2);
			A_mat_h.at<double>(1, 0) = A_mat.at<double>(1, 0);
			A_mat_h.at<double>(1, 1) = A_mat.at<double>(1, 1);
			A_mat_h.at<double>(1, 2) = A_mat.at<double>(1, 2);
			A_mat_h.at<double>(2, 2) = 1;

			// �任��, ����ͼ��ԭ���ƶ���ģ��ͼ������
			Mat T_mat_post = Mat::zeros(3, 3, CV_64F);
			T_mat_post.at<double>(0, 0) = T_mat_post.at<double>(1, 1) = T_mat_post.at<double>(2, 2) = 1;
			T_mat_post.at<double>(0, 2) = ref_center.x;
			T_mat_post.at<double>(1, 2) = ref_center.y;

			// ��ϱ任����
			Mat M_mat_h = T_mat_post * (A_mat_h * T_mat_pre);
			Mat M_mat = M_mat_h(Range(0, 2), Range(0, 3));

			// ����任
			Mat M_im;
			warpAffine(test_im_c, M_im, M_mat, ref_im_size);
			namedWindow("after affine");
			imshow("after affine", M_im);
			fs::path align_im_file = res_align_path / (*test_iter + "_" + *ref_iter + ".jpg");
			imwrite(align_im_file.string(), M_im);
		
			/************************************************************************/
			/* �ܾ�������ı任, �ñ任��Ĳü�����ͼ���ģ��ͼ��                                                                     */
			/************************************************************************/
			cout << "\t";

			// �����ͼ��4���Ǳ任�������
			vector<Point2f> test_corners(4);
			test_corners[0] = Point2f(0, 0); // ����
			test_corners[1] = Point2f(0, test_im_size.height); // ����
			test_corners[2] = Point2f(test_im_size.width, test_im_size.height); // ����
			test_corners[3] = Point2f(test_im_size.width, 0); // ����
			transform(test_corners, test_corners, M_mat);

			// ���x����˳�򲻶���ܾ�
			if (!(test_corners[0].x < min(test_corners[2].x, test_corners[3].x))
				||
				!(test_corners[1].x < min(test_corners[2].x, test_corners[3].x)))
			{
				cout << "rejected because wrong relative position in x direction" << endl;
				continue;
			}

			// ���y����˳�򲻶���ܾ�
			if (!(test_corners[0].y < min(test_corners[1].y, test_corners[2].y))
				||
				!(test_corners[3].y < min(test_corners[1].y, test_corners[2].y)))
			{
				cout << "rejected because wrong relative position in y direction" << endl;
				continue;
			}

			// �ü�
			vector<int> xs(4), ys(4);
			for (int ci = 0; ci < test_corners.size(); ci++)
			{
				xs[ci] = test_corners[ci].x;
				ys[ci] = test_corners[ci].y;
			}
			sort(xs.begin(), xs.end());
			sort(ys.begin(), ys.end());

			int start_r = min(max(ys[1], 0), ref_im_size.height);
			int start_c = min(max(xs[1], 0), ref_im_size.width);
			int end_r = max(min(ys[2], ref_im_size.height), 0);
			int end_c = max(min(xs[2], ref_im_size.width), 0);

			Mat test_crop_im = M_im(Range(start_r, end_r), Range(start_c, end_c));
			fs::path test_crop_file = res_crop_path / (*test_iter + "_" + *ref_iter + "_test_crop.jpg");
			imwrite(test_crop_file.string(), test_crop_im);

			Mat ref_crop_im = ref_im_c(Range(start_r, end_r), Range(start_c, end_c));
			fs::path ref_crop_file = res_crop_path / (*test_iter + "_" + *ref_iter + "_ref_crop.jpg");
			imwrite(ref_crop_file.string(), ref_crop_im);

			waitKey();
			cout << endl;
		}
	}

	destroyAllWindows();

	return 0;
}
