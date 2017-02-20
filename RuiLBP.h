#ifndef RUILBP_H_
#define RUILBP_H_

#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;

#define  lbpPI 3.1415926
#define lbpMAX(x,y) (x)>(y)?(x):(y)
#define lbpMIN(x,y) (x)<(y)?(x):(y)

typedef struct MyPoint
{
	double x;
	double y;
}MyPoint;

class RuiLbp
{
public:
	RuiLbp();
	~RuiLbp();
	void SetInDataParam(cv::Mat OrgImg, int NumOfSamples, int Radius);
	void RunLbpExtraction();
	//output result
	cv::Mat GetLBPRstImg();
	cv::Mat GetLBPRstHistImg();
	vector<float> GetLbpHistVec();

private:
	void rotation_uniform_invariant_lbp(IplImage *src,
		int height, int width, int num_sp, MyPoint *spoint, int *Mapping);
	void rotation_uniform_invariant_mapping(int range, int num_sp, int *Mapping);

	//assistant variables
	IplImage src_ipl_img;
	IplImage *src_in, *gray, *result;
	int P, R, range, *mapping;
	MyPoint *spoint;
	//rst variables
	cv::Mat RstImg, HistImg;
	vector<float> HistVec;

	//assistant function
	int calc_sum(int r);
	void calc_position(int radius, int num_sp, MyPoint *spoint);
};

#endif
