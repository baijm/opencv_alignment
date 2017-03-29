#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>

#include <iostream>
#include <fstream>
#include <unordered_map>

#include "my_detector.h"

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;

int main()
{
	// ����
	string cls_name = "3-prlh";

	// ͼ�������ĸ�Ŀ¼
	string root_dir = "D:/datasets/vobile_project/shelf/shampoo/test";

	// ͼ��λ��
	// ����ͼ��
	string test_im_dir = root_dir + "/12_classified_2852" + "/" + cls_name;
	// ģ��ͼ��
	string ref_im_dir = root_dir + "/15_tmpl_new_center_logo/img_padding_black";

	// ͼ�����б�
	// ����ͼ��
	string test_name_file = root_dir + "/12_classified_2852" + "/" + cls_name + ".txt";
	// ģ��ͼ��
	string ref_name_file = root_dir + "/15_tmpl_new_center_logo/pr_2.txt";

	// �������sift�����ļ�
	// ����ͼ��
	string test_sift_dir = root_dir + "/10_x_split_sift_txt";
	// ģ��ͼ��
	string ref_sift_dir = root_dir + "/15_tmpl_new_center_logo" + "/sift_logo_region";


	// ���Ŀ¼
	string res_root_dir = root_dir + "/22_aligned_test";
	string res_dir = res_root_dir + "/" + cls_name;
	string res_kp_dir = res_dir + "/keypoint"; // ����������ȡ���
	string res_match_dir = res_dir + "/match"; // ����ƥ����
	string res_align_dir = res_dir + "/result"; // ���������
	string res_crop_dir = res_dir + "/crop"; // ����ü����

	// ����ѡ��
	bool save_kp = true;
	bool save_match = true;
	bool save_align = true; // �����true, ��ͬʱ��_corner.txt�б�������, ����, ����, ���µ������
	bool save_crop = true;

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
	MyAffineEstimator *estimator = new RansacAffineEstimator();

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

	// log
	ofstream res_log(res_dir + "/" + "log.txt");

	// ģ��ͼ����->��ɫͼ��ӳ��
	unordered_map<string, Mat> ref_name2imc;

	// ģ��ͼ����->�Ҷ�ͼ��ӳ��
	unordered_map<string, Mat> ref_name2img;
	
	// ģ��ͼ����->������ӳ��
	unordered_map<string, vector<KeyPoint>> ref_name2kps;

	// ģ��ͼ����->descriptorӳ��
	unordered_map<string, Mat> ref_name2des;

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
		vector<KeyPoint> test_kps = load_kp_txt(test_sift_dir + "/" + *test_iter + "_kp.txt");
		//detector.detect(test_im_g, test_kps);

		// �������������㲢���� (��ѡ)
		if (save_kp)
		{
			Mat test_kps_im;
			drawKeypoints(test_im_c, test_kps, test_kps_im);
			fs::path test_kps_im_file = res_kp_path
				/ (*test_iter + ".jpg");
			imwrite(test_kps_im_file.string(), test_kps_im);
		}

		// ������������(�������Ϊ����������, ����Ϊ128, ����ΪCV_32F)
		Mat test_des = load_des_txt(test_sift_dir + "/" + *test_iter + "_des.txt");

		// ��ÿ��ģ��ͼ��
		for (vector<string>::const_iterator ref_iter = ref_names.begin(); ref_iter != ref_names.end(); ref_iter++)
		{
			res_log << "test image : " << *test_iter << "\tref image : " << *ref_iter << endl;
			cout << "test image : " << *test_iter << "\tref image : " << *ref_iter << endl;

			// ��ͼ��
			Mat ref_im_c, ref_im_g;
			if (ref_name2imc.find(*ref_iter) == ref_name2imc.end())
			{
				ref_im_c = imread(ref_im_dir + '/' + *ref_iter + ".jpg");
				cvtColor(ref_im_c, ref_im_g, CV_BGR2GRAY);

				ref_name2imc[*ref_iter] = ref_im_c;
				ref_name2img[*ref_iter] = ref_im_g;
			}
			else
			{
				ref_im_c = ref_name2imc[*ref_iter];
				ref_im_g = ref_name2img[*ref_iter];
			}

			// ͼ��ߴ�
			Size ref_im_size = ref_im_c.size();

			// ���������
			vector<KeyPoint> ref_kps;
			if (ref_name2kps.find(*ref_iter) == ref_name2kps.end())
			{
				//detector.detect(ref_im_g, ref_kps);

				ref_kps = load_kp_txt(ref_sift_dir + "/" + *ref_iter + "_kp.txt");

				ref_name2kps[*ref_iter] = ref_kps;
			}
			else
			{
				ref_kps = ref_name2kps[*ref_iter];
			}

			res_log << "\ttest image has " << test_kps.size() << " keypoints, ref image has " << ref_kps.size() << " keypoints" << endl;
			cout << "\ttest image has " << test_kps.size() << " keypoints, ref image has " << ref_kps.size() << " keypoints" << endl;

			// �������������㲢���� (��ѡ)
			if (save_kp)
			{
				Mat ref_kps_im;
				drawKeypoints(ref_im_c, ref_kps, ref_kps_im);
				fs::path ref_kps_im_file = res_kp_path
					/ (*ref_iter + ".jpg");
				imwrite(ref_kps_im_file.string(), ref_kps_im);
			}

			// ������������
			Mat ref_des;
			if (ref_name2des.find(*ref_iter) == ref_name2des.end())
			{
				//detector.compute(ref_im_g, ref_kps, ref_des);

				ref_des = load_des_txt(ref_sift_dir + "/" + *ref_iter + "_des.txt");

				ref_name2des[*ref_iter] = ref_des;
			}
			else
			{
				ref_des = ref_name2des[*ref_iter];
			}

			// ƥ��
			vector<DMatch> matches;
			matcher->match(test_des, ref_des, matches);
			res_log << "\t" << matches.size() << " matches" << endl;
			cout << "\t" << matches.size() << " matches" << endl;

			// ����ȫ��������ƥ���ϵ������ (��ѡ)
			if (save_match)
			{
				Mat match_all_im;
				drawMatches(test_im_c, test_kps,
					ref_im_c, ref_kps,
					matches, match_all_im);
				fs::path match_all_im_file = res_match_path
					/ (*test_iter + "_" + *ref_iter + "_all.jpg");
				imwrite(match_all_im_file.string(), match_all_im);
			}

			// ɸѡƥ��
			if (use_filter)
			{
				filter->filter(matches);
			}
			
			// ����ɸѡ���������ƥ���ϵ������ (��ѡ)
			if (use_filter && save_match)
			{
				Mat match_filt_im;
				drawMatches(test_im_c, test_kps,
					ref_im_c, ref_kps,
					matches, match_filt_im);
				fs::path match_filt_im_file = res_match_path
					/ (*test_iter + "_" + *ref_iter + "_filter.jpg");
				imwrite(match_filt_im_file.string(), match_filt_im);
			}

			/************************************************************************/
			/* ��������                                                                     */
			/************************************************************************/
			Mat A_mat;
			res_log << "\t";
			cout << "\t";
			if (!estimator->estimate_affine_matrix(test_kps, ref_kps, matches, A_mat))
			{
				res_log << "failed to estimate affine matrix" << endl;
				cout << "failed to estimate affine matrix" << endl;
			}
			Point2f test_center = estimator->test_center;
			Point2f ref_center = estimator->ref_center;

			// ����inliers������ (��ѡ)
			if (save_match)
			{
				Mat match_inlier_im;
				drawMatches(test_im_c, test_kps,
					ref_im_c, ref_kps,
					estimator->inliers, match_inlier_im);
				fs::path match_inlier_im_file = res_match_path
					/ (*test_iter + "_" + *ref_iter + "_inliers.jpg");
				imwrite(match_inlier_im_file.string(), match_inlier_im);
			}

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

			// ����任�������� (��ѡ)
			Mat M_im;
			warpAffine(test_im_c, M_im, M_mat, ref_im_size);
			//namedWindow("after affine");
			//imshow("after affine", M_im);
			if (save_align)
			{
				fs::path align_im_file = res_align_path / (*test_iter + "_" + *ref_iter + ".jpg");
				imwrite(align_im_file.string(), M_im);
			}
		
			/************************************************************************/
			/* �ܾ�������ı任, �ñ任��Ĳü�����ͼ���ģ��ͼ��                                                                     */
			/************************************************************************/
			res_log << "\t";
			cout << "\t";

			// �����ͼ��4���Ǳ任�������
			vector<Point2f> test_corners(4);
			test_corners[0] = Point2f(0, 0); // ����
			test_corners[1] = Point2f(0, test_im_size.height); // ����
			test_corners[2] = Point2f(test_im_size.width, test_im_size.height); // ����
			test_corners[3] = Point2f(test_im_size.width, 0); // ����
			transform(test_corners, test_corners, M_mat);

			// ���save_align��true, ��ѱ任���ĸ��ǵ����걣�浽res_align_dir��_corner.txt��
			// ���α�������, ����, ����, ���µ������(��y��x)
			if (save_align)
			{
				fs::path align_corner_file = res_align_path / (*test_iter + "_" + *ref_iter + "_corner.txt");
				ofstream corner_txt(align_corner_file.string(), ios::out);

				// ���Ͻ�
				corner_txt << test_corners[0].y << " " << test_corners[0].x << endl;
				// ���½�
				corner_txt << test_corners[1].y << " " << test_corners[1].x << endl;
				// ���Ͻ�
				corner_txt << test_corners[3].y << " " << test_corners[3].x << endl;
				// ���½�
				corner_txt << test_corners[2].y << " " << test_corners[2].x << endl;

				corner_txt.close();
			}

			// ���x����˳�򲻶�, ��ܾ�
			if (!(test_corners[0].x < min(test_corners[2].x, test_corners[3].x))
				||
				!(test_corners[1].x < min(test_corners[2].x, test_corners[3].x)))
			{
				res_log << "rejected because wrong relative position in x direction" << endl;
				cout << "rejected because wrong relative position in x direction" << endl;
				continue;
			}

			// ���y����˳�򲻶�, ��ܾ�
			if (!(test_corners[0].y < min(test_corners[1].y, test_corners[2].y))
				||
				!(test_corners[3].y < min(test_corners[1].y, test_corners[2].y)))
			{
				res_log << "rejected because wrong relative position in y direction" << endl;
				cout << "rejected because wrong relative position in y direction" << endl;
				continue;
			}

			// �ü������� (��ѡ)
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

			// ���������෴, ��ܾ�
			if ((abs(end_r - start_r) > abs(end_c - start_c) && (test_im_g.rows < test_im_g.cols))
				||
				(abs(end_r - start_r) < abs(end_c - start_c) && (test_im_g.rows > test_im_g.cols)))
			{
				res_log << "rejected because wrong aspect ratio" << endl;
				cout << "rejected because wrong aspect ratio" << endl;
				continue;
			}

			// ��������Ϊ0, ��ܾ�
			if (end_r - start_r == 0 || end_c - start_c == 0)
			{
				res_log << "rejected because zero size" << endl;
				cout << "rejected because zero size" << endl;
				continue;
			}


			if (save_crop)
			{
				Mat test_crop_im = M_im(Range(start_r, end_r), Range(start_c, end_c));
				fs::path test_crop_file = res_crop_path / (*test_iter + "_" + *ref_iter + "_test_crop.jpg");
				imwrite(test_crop_file.string(), test_crop_im);

				Mat ref_crop_im = ref_im_c(Range(start_r, end_r), Range(start_c, end_c));
				fs::path ref_crop_file = res_crop_path / (*test_iter + "_" + *ref_iter + "_ref_crop.jpg");
				imwrite(ref_crop_file.string(), ref_crop_im);
			}

			//waitKey();
			res_log << endl;
			cout << endl;
		}
	}

	res_log.close();
	//destroyAllWindows();

	return 0;
}
