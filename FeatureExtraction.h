#pragma once

#include <armadillo>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "DataStruct.h"
#include "RuiLBP.h"
#include "cvgabor.h"

using namespace arma;
using namespace std;
using namespace cv;

class FeatureExtract
{
	public:
		FeatureExtract();
		~FeatureExtract();
		void SetInData(string OrgImgPath, string GtImgPath, int ImgNum);
		void RunExtract();
		vector<float> GetFeatureVec();
	private:
		/*assistant function*/
		cv::Mat LoadImageCheck(string ImgPath);
		cv::Mat ExtractBlueChannel(cv::Mat SourceImg);
		void InitialFeatueValue();
		//each feature extraction function
		void SaveFeatures();
		
		//nuclei binary image extract function
		void GetNucleiRegion();
		//cytoplasm binary image extract function
		void GetCytoplasmRegion();
		
		void GetN_Counter_feat();
		void GetC_Counter_feat();
		
		void GetN_Rect_feat();
		void GetC_Rect_feat();

		void GetN_Ellipse_feat();
		void GetC_Ellipse_feat();
		
		/*assistant variables*/
		//input images
		cv::Mat OrgImg, GtImg;
		//generated images
		cv::Mat GrayImg, BlueGrayImg;

		string ImgNumber;
		
		//nuclei counter
		vector<Point> N_contour;
		vector<vector<Point>> N_contours;
		//cytoplasm counter
		vector<Point> C_contour;
		vector<vector<Point>> C_contours;
		
		//cell feature vector
		vector<float> FeatureVec;
		
		//nuclei feature computing assistant variables
		cv::Mat N_binaryImg;
		//cytoplasm feature computing assistant variables
		cv::Mat C_binaryImg;
		
		//feature variables
		float N_area, C_area;
		float N_Peri, C_Peri;
		float N_brightness, C_brightness;
		float N_gray_sum, C_gray_sum;
		float NrC;//ratio of nuclei and cell
		float N_Rect_short, N_Rect_long;
		float C_Rect_short, C_Rect_long;
		float N_elongation, C_elongation;
		float N_roundness, C_roundness;
		float N_min, N_max;
		float C_min, C_max;
		Point2f N_center, C_center;
		float nuclei_pos;
		float N_long_Axis, N_short_Axis;
		float C_long_Axis, C_short_Axis;

		/*----------local binary patteren feature----------*/
		//destination variables
		cv::Mat norLBP, cirLBP;
		//LBP parameters
		int Lbp_radius, Lbp_neighbors;
		//function
		void normalLBP(cv::Mat& src, cv::Mat &dst);//normal LBP function
		void circleLBP(cv::Mat& src, cv::Mat &dst, int radius, int neighbors);//circle LBP function
		void LBPTest();

		void ruiLBPTest();
		vector<float> riuLBPVec;
		void ruiLBPTest2();
		vector<float> riuLBPVec2;
		void GaborTest();
		vector<float> GaborMu, GaborStd;

		//show rst image
		void showImg(cv::Mat& rst, const char* NameStr);
		void showcvImg(IplImage * img, const char* WinName);

		//assistant variables for nucleus textural feature
		cv::Mat ResizedN;

		//RGB to HSV, LAB
		void RGB2HSVLAB();
		cv::Mat OrgHsv, OrgLab;
		cv::Mat HsvH, HsvS, HsvV;
		cv::Mat LabL, LabA, LabB;
		cv::Mat RgbR, RgbG, RgbB;

		float N_muR, N_muG, N_muB_rgb;
		float N_muH, N_muS, N_muV;
		float N_muL, N_muA, N_muB_lab;

		float C_muR, C_muG, C_muB_rgb;
		float C_muH, C_muS, C_muV;
		float C_muL, C_muA, C_muB_lab;

		float N_sigmaR, N_sigmaG, N_sigmaB_rgb;
		float N_sigmaH, N_sigmaS, N_sigmaV;
		float N_sigmaL, N_sigmaA, N_sigmaB_lab;

		float C_sigmaR, C_sigmaG, C_sigmaB_rgb;
		float C_sigmaH, C_sigmaS, C_sigmaV;
		float C_sigmaL, C_sigmaA, C_sigmaB_lab;

		//calculate std of region
		void CalNucleusStd();
		float CalNRegionStd(cv::Mat& OneChanImg, float& mu);

		void CalCytoplasmStd();
		float CalCRegionStd(cv::Mat& OneChanImg,  float& mu);

		void SaveIntensityFeat();
};
