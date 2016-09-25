#include "FeatureExtraction.h"

FeatureExtract::FeatureExtract(){}

FeatureExtract::~FeatureExtract(){}

void FeatureExtract::SetInData(string OrgImgPath, string GtImgPath, int ImgNum)
{
	OrgImg = LoadImageCheck(OrgImgPath);
	GtImg = LoadImageCheck(GtImgPath);
	BlueGrayImg = ExtractBlueChannel(GtImg);
	cvtColor(OrgImg, GrayImg, CV_BGR2GRAY);

	InitialFeatueValue();

	ImgNumber = NumToString(ImgNum);
}
void FeatureExtract::RunExtract()
{
	//convert RGB to HSV and LAB
	RGB2HSVLAB();

	//1.compute nuclei area, sum_brightness, min, max
	GetNucleiRegion();
	
	//2.compute cytoplasm area, sum_brightness, min, max
	GetCytoplasmRegion();
	
	//3.compute nuclei counter features
	GetN_Counter_feat();
	
	//4.compute cytoplasm counter features
	GetC_Counter_feat();
	
	//5.compute nuclei short and long of Rect
	GetN_Ellipse_feat();
	
	//6.compute cytoplasm short and long of Rect
	GetC_Ellipse_feat();

	//7.save all above features
	//SaveFeatures();

	//8.extract normal LBP texture features
	//LBPTest();

	//9.extract uniform rotation LBP features
	ruiLBPTest();
	ruiLBPTest2();

	//10.extract Gabor wavelet transformation features
	GaborTest();

	//11.calculate region std 
	CalNucleusStd();
	CalCytoplasmStd();

	//change this function location on 
	SaveFeatures();
}
vector<float> FeatureExtract::GetFeatureVec()
{
	return FeatureVec;
}

void FeatureExtract::SaveFeatures()
{
	//1.nucleus area 
	FeatureVec.push_back(N_area);
	//2.cytoplasm area
	FeatureVec.push_back(C_area);
	//3.ratio between nucleus area and cytoplasm area
	NrC = N_area /(N_area + C_area);
	FeatureVec.push_back(NrC);
	//4.nucleus brightness
	N_brightness = N_gray_sum / N_area;
	FeatureVec.push_back(N_brightness);
	//5.cytoplasm brightness
	C_brightness = C_gray_sum / C_area;
	FeatureVec.push_back(C_brightness);
	//6.the short axis(the second axis) of nuclei
	FeatureVec.push_back(N_short_Axis);
	//7.the long axis(the first axis) of nuclei
	FeatureVec.push_back(N_long_Axis);
	//8.nuclei elongation
	N_elongation = N_short_Axis / N_long_Axis;
	FeatureVec.push_back(N_elongation);
	//9.nuclei roundness
	N_roundness = (4 * N_area) / sqrt(CV_PI*N_long_Axis);
	FeatureVec.push_back(N_roundness);
	//10.short axis(the second axis) of cytoplasm
	FeatureVec.push_back(C_short_Axis);
	//11.the longest axis(the first axis) of cytoplasm
	FeatureVec.push_back(C_long_Axis);
	//12. cytoplasm elongation
	C_elongation = C_short_Axis / C_long_Axis;
	FeatureVec.push_back(C_elongation);
	//13. cytoplasm roundness
	C_roundness = (4 * C_area) / sqrt(CV_PI*C_short_Axis);
	FeatureVec.push_back(C_roundness);
	//14. nucleus perimeter
	FeatureVec.push_back(N_Peri);
	//15.cytoplasm perimeter
	FeatureVec.push_back(C_Peri);
	//16.nuclei site******
	nuclei_pos = 2 * sqrt((N_center.x - C_center.x)*(N_center.x - C_center.x) +
		(N_center.y - C_center.y)*(N_center.y - C_center.y)) / C_long_Axis;
	FeatureVec.push_back(nuclei_pos);
	//17.maximum value of nucleus
	FeatureVec.push_back(N_max);
	//18.minimum value of nucleus
	FeatureVec.push_back(N_min);
	//19.maximum value of cytoplasm
	FeatureVec.push_back(C_max);
	//20.minimum value of cytoplasm
	FeatureVec.push_back(C_min);

	//riuLBP features  P=16, R=2
	for (int i = 0; i < riuLBPVec.size(); i++)
	{
		FeatureVec.push_back(riuLBPVec[i]);
	}
	
	//riuLBP features  P=8, R=1
	for (int i = 0; i < riuLBPVec2.size(); i++)
	{
		FeatureVec2.push_back(riuLBPVec[i]);
	}

	//Gabor mean and std features
	for (int i = 0; i < GaborMu.size(); i++)
	{
		FeatureVec.push_back(GaborMu[i]);
		FeatureVec.push_back(GaborStd[i]);
	}
	
	SaveIntensityFeat();
	
}

//assistant function
cv::Mat FeatureExtract::LoadImageCheck(string ImgPath)
{
	cv::Mat RstImg;
	RstImg = cv::imread(ImgPath, 1);
	if(!RstImg.data)
	{
		cout << "can not load image, check the image path." << endl;
		system("pause");
	}
	else
	{
		return RstImg;
	}
}

cv::Mat FeatureExtract::ExtractBlueChannel(cv::Mat SourceImg)
{
	if(!SourceImg.data)
	{
		cout << "can not load source image, check the image path." << endl;
		system("pause");
	}
	cv::Mat RstImg;
	int nl = SourceImg.rows;
	int nc = SourceImg.cols;
	RstImg.create(nl, nc, CV_8U);
	MatIterator_<Vec3b> itGT = SourceImg.begin<Vec3b>();
	Mat_<uchar>::iterator itout = RstImg.begin<uchar>();
	
	for (int j = 0; j < nl; j++)
	{
		for (int i = 0; i < nc; i++)
		{
			int indice = j*nc + i;
			*(itout + indice) = (*(itGT + indice))[0];//B channel
		}
	}
	return RstImg;
}

void FeatureExtract::InitialFeatueValue()
{
	N_area = 0;
	C_area = 0;
	N_Peri = 0; 
	C_Peri = 0;
	N_brightness = 0; 
	C_brightness = 0;
	N_gray_sum = 0; 
	C_gray_sum = 0;
	NrC = 0;
	N_Rect_short = 0; 
	N_Rect_long = 0;
	C_Rect_short = 0; 
	C_Rect_long = 0;
	N_elongation = 0;
	C_elongation = 0;
	N_roundness = 0;
	C_roundness = 0;
	N_min = 300;
	N_max = -1;
	C_min = 300;
	C_max = -1;
	nuclei_pos = 0;
	N_long_Axis = 0; 
	N_short_Axis = 0;

	N_muR = 0.0;
	N_muG = 0.0;
	N_muB_rgb =0.0;
	N_muH =0.0;
	N_muS =0.0; 
	N_muV =0.0;
	N_muL =0.0; 
	N_muA =0.0; 
	N_muB_lab = 0.0;

	C_muR = 0.0;
	C_muG = 0.0;
	C_muB_rgb =0.0;
	C_muH = 0.0;
	C_muS = 0.0;
	C_muV = 0.0;
	C_muL = 0.0;
	C_muA = 0.0;
	C_muB_lab = 0.0;

	N_sigmaR =0.0;
	N_sigmaG = 0.0;
	N_sigmaB_rgb = 0.0;
	N_sigmaH = 0.0;
	N_sigmaS = 0.0;
	N_sigmaV = 0.0;
	N_sigmaL = 0.0;
	N_sigmaA = 0.0;
	N_sigmaB_lab = 0.0;

	C_sigmaR = 0.0;
	C_sigmaG = 0.0;
	C_sigmaB_rgb = 0.0;
	C_sigmaH = 0.0;
	C_sigmaS = 0.0;
	C_sigmaV = 0.0;
	C_sigmaL = 0.0;
	C_sigmaA = 0.0;
	C_sigmaB_lab = 0.0;
}

//Get N_area, N_gray_sum, N_min, N_max
void FeatureExtract::GetNucleiRegion()
{
	int nl = BlueGrayImg.rows;
	int nc = BlueGrayImg.cols;

	N_binaryImg.create(nl, nc, CV_8U);
	MatIterator_<uchar> itGT = BlueGrayImg.begin<uchar>();
	Mat_<uchar>::iterator itout = N_binaryImg.begin<uchar>();
	MatIterator_<uchar> itGray = GrayImg.begin<uchar>();

	MatIterator_<uchar> itRGBr = RgbR.begin<uchar>();
	MatIterator_<uchar> itRGBg = RgbG.begin<uchar>();
	MatIterator_<uchar> itRGBb = RgbB.begin<uchar>();

	MatIterator_<uchar> itHSVh = HsvH.begin<uchar>();
	MatIterator_<uchar> itHSVs = HsvS.begin<uchar>();
	MatIterator_<uchar> itHSVv = HsvV.begin<uchar>();

	MatIterator_<uchar> itLABl = LabL.begin<uchar>();
	MatIterator_<uchar> itLABa = LabA.begin<uchar>();
	MatIterator_<uchar> itLABb = LabB.begin<uchar>();

	for (int j = 0; j < nl; j++)
	{
		for (int i = 0; i < nc; i++)
		{
			int indice = j*nc + i;
			if ( int((*(itGT + indice))) == 255 )//nucleus region operation
			{			
				*(itout + indice) = (*(itGT + indice));
				N_area++;//nuclei area computing
				float gray_pixel_value = (*(itGray + indice));
				//nuclei brightness sum
				N_gray_sum = N_gray_sum + gray_pixel_value;
				if (gray_pixel_value < N_min)//nuclei min pixel value
					N_min = gray_pixel_value;
				if (gray_pixel_value > N_max)//nuclei max pixel value
					N_max = gray_pixel_value;

				//nucleus sum from every channels of RGB, HSV, LAB 
				N_muR = N_muR + float(*(itRGBr + indice));
				N_muG = N_muG + float(*(itRGBg + indice));
				N_muB_rgb = N_muB_rgb + float(*(itRGBb + indice));

				N_muH = N_muH + float(*(itHSVh + indice));
				N_muS = N_muS + float(*(itHSVs + indice));
				N_muV = N_muV + float(*(itHSVv + indice));

				N_muL = N_muL + float(*(itLABl + indice));
				N_muA = N_muA + float(*(itLABa + indice));
				N_muB_lab = N_muB_lab + float(*(itLABb + indice));
			}
			else
			{
				*(itout + indice) = 0;
			}
		}
	}
	N_muR = N_muR / N_area;
	N_muG = N_muG / N_area;
	N_muB_rgb = N_muB_rgb / N_area;

	N_muH = N_muH / N_area;
	N_muS = N_muS / N_area;
	N_muV = N_muV / N_area;

	N_muL = N_muL / N_area;
	N_muA = N_muA / N_area;
	N_muB_lab = N_muB_lab / N_area;
}
//Get C_area, C_gray_sum, C_min, C_max	
void FeatureExtract::GetCytoplasmRegion()
{
	int nl = BlueGrayImg.rows;
	int nc = BlueGrayImg.cols;

	C_binaryImg.create(nl, nc, CV_8U);
	MatIterator_<uchar> itGT = BlueGrayImg.begin<uchar>();
	Mat_<uchar>::iterator itout = C_binaryImg.begin<uchar>();
	MatIterator_<uchar> itGray = GrayImg.begin<uchar>();

	MatIterator_<uchar> itRGBr = RgbR.begin<uchar>();
	MatIterator_<uchar> itRGBg = RgbG.begin<uchar>();
	MatIterator_<uchar> itRGBb = RgbB.begin<uchar>();

	MatIterator_<uchar> itHSVh = HsvH.begin<uchar>();
	MatIterator_<uchar> itHSVs = HsvS.begin<uchar>();
	MatIterator_<uchar> itHSVv = HsvV.begin<uchar>();

	MatIterator_<uchar> itLABl = LabL.begin<uchar>();
	MatIterator_<uchar> itLABa = LabA.begin<uchar>();
	MatIterator_<uchar> itLABb = LabB.begin<uchar>();

	for (int j = 0; j < nl; j++)
	{
		for (int i = 0; i < nc; i++)
		{
			int indice = j*nc + i;
			if ( int((*(itGT + indice))) == 128 )
			{			
				*(itout + indice) = (*(itGT + indice));
				C_area++;//cytoplasm area computing
				float gray_pixel_value = (*(itGray + indice));
				//cytoplasm brightness sum
				C_gray_sum = C_gray_sum + gray_pixel_value;
				if (gray_pixel_value < C_min)//cytoplasm min pixel value
					C_min = gray_pixel_value;
				if (gray_pixel_value > C_max)//cytoplasm max pixel value
					C_max = gray_pixel_value;

				//nucleus sum from every channels of RGB, HSV, LAB
				C_muR = C_muR + float(*(itRGBr + indice));
				C_muG = C_muG + float(*(itRGBg + indice));
				C_muB_rgb = C_muB_rgb + float(*(itRGBb + indice));

				C_muH = C_muH + float(*(itHSVh + indice));
				C_muS = C_muS + float(*(itHSVs + indice));
				C_muV = C_muV + float(*(itHSVv + indice));

				C_muL = C_muL + float(*(itLABl + indice));
				C_muA = C_muA + float(*(itLABa + indice));
				C_muB_lab = C_muB_lab + float(*(itLABb + indice));
			}
			else
			{
				*(itout + indice) = 0;
			}
		}
	}
	C_muR = C_muR / C_area;
	C_muG = C_muG / C_area;
	C_muB_rgb = C_muB_rgb / C_area;

	C_muH = C_muH / C_area;
	C_muS = C_muS / C_area;
	C_muV = C_muV / C_area;

	C_muL = C_muL / C_area;
	C_muA = C_muA / C_area;
	C_muB_lab = C_muB_lab / C_area;
}

//N_contour
void FeatureExtract::GetN_Counter_feat()
{
	findContours(N_binaryImg, N_contours, 
	CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	
	N_contour = N_contours[0];
	//find out the biggest countours as nuclei
	for (int i = 0; i < N_contours.size(); i++)
	{
		if ( N_contours[i].size() > N_contour.size() )
			N_contour = N_contours[i];
	}

	cv::Moments N_mom = cv::moments(cv::Mat(N_contour));
	N_center.x = N_mom.m10 / N_mom.m00;
	N_center.y = N_mom.m01 / N_mom.m00;

	float cv_ArcL = arcLength(N_contour, true);
	N_Peri = cv_ArcL;
	

	/*---------show rect fitting nucleus ----------*/
	Rect boundRect;
	boundRect = boundingRect(cv::Mat(N_contour));

	/*---------------------------------------------------*/
	//crop nucleus region
	cv::Mat CropNucleus(OrgImg, boundRect);
	resize(CropNucleus, ResizedN, cv::Size(48, 48), 0, 0, CV_INTER_LINEAR);
}

//C_contour
void FeatureExtract::GetC_Counter_feat()
{
	findContours(C_binaryImg, C_contours, 
	CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	
	C_contour = C_contours[0];
	//find out the biggest countours as cytoplasm
	for (int i = 0; i < C_contours.size(); i++)
	{
		if (C_contours[i].size() > C_contour.size())
			C_contour = C_contours[i];
	}
	
	cv::Moments C_mom = cv::moments(cv::Mat(C_contour));

	C_center.x = C_mom.m10 / C_mom.m00;
	C_center.y = C_mom.m01 / C_mom.m00;

	float cv_ArcL = arcLength(C_contour, true);
	C_Peri = cv_ArcL;
}

//nuclei rotated rectangle
void FeatureExtract::GetN_Rect_feat()
{
	RotatedRect minRect;
	minRect = minAreaRect(cv::Mat(N_contour));
	// rotated rectangle
	Point2f rect_points[4];
	minRect.points(rect_points);
	
	float rect_width = minRect.size.width;
	float rect_height = minRect.size.height;
	
	if(rect_height >= rect_width)
	{
		N_Rect_long = rect_height;
		N_Rect_short = rect_width;
	}
	else
	{
		N_Rect_long = rect_width;
		N_Rect_short = rect_height;
	}
}

void FeatureExtract::GetN_Ellipse_feat()
{
	RotatedRect minEllipse;
	minEllipse = fitEllipse(cv::Mat(N_contour));

	Size2f size;
	size = minEllipse.size;
	N_long_Axis = size.width, N_short_Axis = size.height;

}

void FeatureExtract::GetC_Ellipse_feat()
{
	RotatedRect minEllipse;
	minEllipse = fitEllipse(cv::Mat(C_contour));

	Size2f size;
	size = minEllipse.size;
	C_long_Axis = size.width, C_short_Axis = size.height;

}

//cytoplasm rotated rectangle
void FeatureExtract::GetC_Rect_feat()
{
	RotatedRect minRect;
	minRect = minAreaRect(cv::Mat(C_contour));
	// rotated rectangle
	Point2f rect_points[4];
	minRect.points(rect_points);
	
	float rect_width = minRect.size.width;
	float rect_height = minRect.size.height;
	
	if(rect_height >= rect_width)
	{
		C_Rect_long = rect_height;
		C_Rect_short = rect_width;
	}
	else
	{
		C_Rect_long = rect_width;
		C_Rect_short = rect_height;
	}
}

//normal LBP function
void FeatureExtract::normalLBP(cv::Mat& src, cv::Mat &dst)
{
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			uchar tt = 0;
			int tt1 = 0;
			uchar u = src.at<uchar>(i, j);
			if (src.at<uchar>(i - 1, j - 1)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i - 1, j)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i - 1, j + 1)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i, j + 1)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i + 1, j + 1)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i + 1, j)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i + 1, j - 1)>u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i - 1, j)>u) { tt += 1 << tt1; }
			tt1++;

			dst.at<uchar>(i - 1, j - 1) = tt;
			
		}
		
	}
}
//circle LBP function
void FeatureExtract::circleLBP(cv::Mat& src, cv::Mat &dst, int radius, int neighbors)
{
	for (int n = 0; n<neighbors; n++)
	{
		
		float x = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
		float y = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
		
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		
		float ty = y - fy;
		float tx = x - fx;
		
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		
		for (int i = radius; i < src.rows - radius; i++)
		{
			for (int j = radius; j < src.cols - radius; j++)
			{
				
				float t = static_cast<float>(w1*src.at<uchar>(i + fy, j + fx) +
					w2*src.at<uchar>(i + fy, j + cx) +
					w3*src.at<uchar>(i + cy, j + fx) + w4*src.at<uchar>(i + cy, j + cx));
				
				dst.at<uchar>(i - radius, j - radius) +=
					((t > src.at<uchar>(i, j)) || (std::abs(t - src.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
				
			}
			
		}
	}
}

void FeatureExtract::LBPTest()
{
	cv::Mat N_Gray = cv::Mat(ResizedN.rows , ResizedN.cols , CV_8UC1, Scalar(0));
	cvtColor(ResizedN, N_Gray, CV_BGR2GRAY);
	cv::Mat Nucleus_norLBP = cv::Mat(ResizedN.rows - 2 * Lbp_radius, 
		ResizedN.cols - 2 * Lbp_radius, CV_8UC1, Scalar(0));
	normalLBP(N_Gray, Nucleus_norLBP);

	Lbp_radius = 1;
	Lbp_neighbors = 8;
	cv::Mat Nucleus_cirLBP = cv::Mat(ResizedN.rows, ResizedN.cols, CV_8UC1, Scalar(0));
	circleLBP(N_Gray, Nucleus_cirLBP, Lbp_radius, Lbp_neighbors);
	
}

void FeatureExtract::ruiLBPTest()
{

	RuiLbp lbpObj;
	lbpObj.SetInDataParam(ResizedN, 16, 2);//P=16, R=2
	lbpObj.RunLbpExtraction();

	cv::Mat LbpRstImg = lbpObj.GetLBPRstImg();

	cv::Mat LbpHistImg = lbpObj.GetLBPRstHistImg();

	riuLBPVec = lbpObj.GetLbpHistVec();
	cout << "The size of hist vector : " << riuLBPVec.size() << endl;
	
}

void FeatureExtract::ruiLBPTest2()
{

	RuiLbp lbpObj2;
	lbpObj.SetInDataParam(ResizedN, 8, 1);//P=8, R=1
	lbpObj.RunLbpExtraction();

	cv::Mat LbpRstImg2 = lbpObj2.GetLBPRstImg();

	cv::Mat LbpHistImg2 = lbpObj2.GetLBPRstHistImg();

	riuLBPVec2 = lbpObj2.GetLbpHistVec();
	cout << "The size of hist vector : " << riuLBPVec2.size() << endl;
	
}

void FeatureExtract::GaborTest()
{
	set<double> OrientSet = { 0, CV_PI / 8, CV_PI * 2 / 8, CV_PI * 3 / 8, CV_PI * 4 / 8, 
		CV_PI * 5 / 8, CV_PI * 6 / 8, CV_PI * 7 / 8 };
	set<int> ScaleSet = { 0, 2, 4, 6, 8 };
	double F = sqrt(2.0);
	double Sigma = 2 * CV_PI;

	IplImage tmpResizedN = ResizedN;
	IplImage *img = &(tmpResizedN);
	

	for (std::set<double>::iterator it1 = OrientSet.begin(); it1 != OrientSet.end(); ++it1)
	{
		for (std::set<int>::iterator it2 = ScaleSet.begin(); it2 != ScaleSet.end(); ++it2)
		{
			//create gabor filter
			CvGabor *gabor1 = new CvGabor;
			gabor1->Init(*it1, *it2, Sigma, F);

			IplImage *reimg3 = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
			gabor1->conv_img(img, reimg3, CV_GABOR_MAG);

			//mean and std computing
			cv::Mat GaborRst = reimg3;
			Scalar mu,stddev;
			cv::meanStdDev(GaborRst, mu, stddev);
			uchar mean_pxl = mu.val[0];
			uchar stddev_pxl = stddev.val[0];
			
			GaborMu.push_back( float(mean_pxl) );
			GaborStd.push_back( float(stddev_pxl) );
		}
	}
}

void FeatureExtract::showImg(cv::Mat& rst, const char* NameStr)
{
	namedWindow(NameStr);
	imshow(NameStr, rst);
	cv::waitKey(0);
	cvDestroyWindow(NameStr);
}

void FeatureExtract::showcvImg(IplImage * img, const char* WinName)
{
	cvNamedWindow(WinName, 1);
	cvShowImage(WinName, img);
	cvWaitKey(0);
	cvDestroyWindow(WinName);
}

void FeatureExtract::RGB2HSVLAB()
{
	vector<cv::Mat> RGBmv;
	split(OrgImg, RGBmv);
	RgbR = RGBmv[0];
	RgbG = RGBmv[1];
	RgbB = RGBmv[2];

	cvtColor(OrgImg, OrgHsv, CV_BGR2HSV);
	vector<cv::Mat> HSVmv;
	split(OrgHsv, HSVmv);

	HsvH = HSVmv[0];
	HsvS = HSVmv[1];
	HsvV = HSVmv[2];

	cvtColor(OrgImg, OrgLab, CV_RGB2Lab);
	vector<cv::Mat> LABmv;
	split(OrgLab, LABmv);

	LabL = LABmv[0];
	LabA = LABmv[1];
	LabB = LABmv[2];
}

float FeatureExtract::CalNRegionStd(cv::Mat& OneChanImg, float& mu)
{

	float std_rst = 0.0;

	int nl = BlueGrayImg.rows;
	int nc = BlueGrayImg.cols;

	MatIterator_<uchar> itOrgImg = OneChanImg.begin<uchar>();
	MatIterator_<uchar> itBinImg = BlueGrayImg.begin<uchar>();

	for (int j = 0; j < nl; j++)
	{
		for (int i = 0; i < nc; i++)
		{
			int indice = j*nc + i;
			if (int((*(itBinImg + indice))) == 255)//nucleus region
			{
				float val = float((*(itOrgImg + indice)));
				std_rst = std_rst + (val-mu)*(val-mu);
			}
		}
	}
	return std_rst;
}

void FeatureExtract::CalNucleusStd()
{
	N_sigmaR = sqrt(CalNRegionStd(RgbR,  N_muR) / (N_area - 1));
	N_sigmaG = sqrt(CalNRegionStd(RgbG,  N_muG) / (N_area - 1));
	N_sigmaB_rgb = sqrt(CalNRegionStd(RgbB, N_muB_rgb) / (N_area - 1));

	N_sigmaH = sqrt(CalNRegionStd(HsvH, N_muH) / (N_area - 1));
	N_sigmaS = sqrt(CalNRegionStd(HsvS, N_muS) / (N_area - 1));
	N_sigmaV = sqrt(CalNRegionStd(HsvV, N_muV) / (N_area - 1));

	N_sigmaL = sqrt(CalNRegionStd(LabL, N_muL) / (N_area - 1));
	N_sigmaA = sqrt(CalNRegionStd(LabA, N_muA) / (N_area - 1));
	N_sigmaB_lab = sqrt(CalNRegionStd(LabB, N_muB_lab) / (N_area - 1));
}

void FeatureExtract::CalCytoplasmStd()
{
	C_sigmaR = sqrt(CalNRegionStd(RgbR, C_muR) / (C_area - 1));
	C_sigmaG = sqrt(CalNRegionStd(RgbG, C_muG) / (C_area - 1));
	C_sigmaB_rgb = sqrt(CalNRegionStd(RgbB, C_muB_rgb) / (C_area - 1));

	C_sigmaH = sqrt(CalNRegionStd(HsvH, C_muH) / (C_area - 1));
	C_sigmaS = sqrt(CalNRegionStd(HsvS, C_muS) / (C_area - 1));
	C_sigmaV = sqrt(CalNRegionStd(HsvV, C_muV) / (C_area - 1));

	C_sigmaL = sqrt(CalNRegionStd(LabL, C_muL) / (C_area - 1));
	C_sigmaA = sqrt(CalNRegionStd(LabA, C_muA) / (C_area - 1));
	C_sigmaB_lab = sqrt(CalNRegionStd(LabB, C_muB_lab) / (C_area - 1));
}

float FeatureExtract::CalCRegionStd(cv::Mat& OneChanImg, float& mu)
{
	float std_rst = 0.0;

	int nl = BlueGrayImg.rows;
	int nc = BlueGrayImg.cols;

	MatIterator_<uchar> itOrgImg = OneChanImg.begin<uchar>();
	MatIterator_<uchar> itBinImg = BlueGrayImg.begin<uchar>();

	for (int j = 0; j < nl; j++)
	{
		for (int i = 0; i < nc; i++)
		{
			int indice = j*nc + i;
			if (int((*(itBinImg + indice))) == 128)//cytoplasm region
			{
				float val = float((*(itOrgImg + indice)));
				std_rst = std_rst + (val - mu)*(val - mu);
			}
		}
	}
	return std_rst;
}

void FeatureExtract::SaveIntensityFeat()
{
	FeatureVec.push_back(N_muR);
	FeatureVec.push_back(N_muG);
	FeatureVec.push_back(N_muB_rgb);

	FeatureVec.push_back(N_muH);
	FeatureVec.push_back(N_muS);
	FeatureVec.push_back(N_muV);

	FeatureVec.push_back(N_muL);
	FeatureVec.push_back(N_muA);
	FeatureVec.push_back(N_muB_lab);


	FeatureVec.push_back(C_muR);
	FeatureVec.push_back(C_muG);
	FeatureVec.push_back(C_muB_rgb);

	FeatureVec.push_back(C_muH);
	FeatureVec.push_back(C_muS);
	FeatureVec.push_back(C_muV);

	FeatureVec.push_back(C_muL);
	FeatureVec.push_back(C_muA);
	FeatureVec.push_back(C_muB_lab);

	//std of nucleus
	FeatureVec.push_back(N_sigmaR);
	FeatureVec.push_back(N_sigmaG);
	FeatureVec.push_back(N_sigmaB_rgb);

	FeatureVec.push_back(N_sigmaH);
	FeatureVec.push_back(N_sigmaS);
	FeatureVec.push_back(N_sigmaV);

	FeatureVec.push_back(N_sigmaL);
	FeatureVec.push_back(N_sigmaA);
	FeatureVec.push_back(N_sigmaB_lab);

	FeatureVec.push_back(C_sigmaR);
	FeatureVec.push_back(C_sigmaG);
	FeatureVec.push_back(C_sigmaB_rgb);

	FeatureVec.push_back(C_sigmaH);
	FeatureVec.push_back(C_sigmaS);
	FeatureVec.push_back(C_sigmaV);

	FeatureVec.push_back(C_sigmaL);
	FeatureVec.push_back(C_sigmaA);
	FeatureVec.push_back(C_sigmaB_lab);

}
