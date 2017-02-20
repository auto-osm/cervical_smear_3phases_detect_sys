#include "RuiLBP.h"

RuiLbp::RuiLbp(){}

RuiLbp::~RuiLbp(){}

void RuiLbp::SetInDataParam(cv::Mat OrgImg, int NumOfSamples, int Radius)
{
	src_ipl_img = OrgImg;
	src_in = &(src_ipl_img);//need to check

	P = NumOfSamples;
	R = Radius;

	float Mi = 2.0;
	range = pow(Mi, P);
	
}

void RuiLbp::RunLbpExtraction()
{

	gray = cvCreateImage(cvSize(src_in->width, src_in->height), IPL_DEPTH_8U, 1);
	cvCvtColor(src_in, gray, CV_BGR2GRAY);

	mapping = (int *)malloc(sizeof(int)*range);
	memset(mapping, 0, sizeof(int)*range);

	//compute the relative coordinate of sampling points
	spoint = (MyPoint *)malloc(sizeof(MyPoint)*P);
	calc_position(R, P, spoint);

	//compute rotation uniform invariant features
	rotation_uniform_invariant_mapping(range, P, mapping);
	rotation_uniform_invariant_lbp(gray, src_in->height, src_in->width, P, spoint, mapping);

}

void RuiLbp::rotation_uniform_invariant_lbp(IplImage *src,
	int height, int width, int num_sp, MyPoint *spoint, int *Mapping)
{
	IplImage *target, *hist;
	int i, j, k, box_x, box_y, orign_x, orign_y, dx, dy, tx, ty, fy, fx, cy, cx, v;
	double min_x, max_x, min_y, max_y, w1, w2, w3, w4, N, x, y;
	int *result;
	float dishu;

	dishu = 2.0;
	max_x = 0; max_y = 0; min_x = 0; min_y = 0;
	for (k = 0; k<num_sp; k++)
	{
		if (max_x<spoint[k].x)
		{
			max_x = spoint[k].x;
		}
		if (max_y<spoint[k].y)
		{
			max_y = spoint[k].y;
		}
		if (min_x>spoint[k].x)
		{
			min_x = spoint[k].x;
		}
		if (min_y>spoint[k].y)
		{
			min_y = spoint[k].y;
		}
	}

	box_x = ceil(lbpMAX(max_x, 0)) - floor(lbpMIN(min_x, 0)) + 1;
	box_y = ceil(lbpMAX(max_y, 0)) - floor(lbpMIN(min_y, 0)) + 1;

	if (width<box_x || height<box_y)
	{
		printf("Too small input image. Should be at least (2*radius+1) x (2*radius+1)");
		return;
	}

	orign_x = 0 - floor(lbpMIN(min_x, 0));
	orign_y = 0 - floor(lbpMIN(min_x, 0));

	dx = width - box_x + 1;
	dy = height - box_y + 1;

	target = cvCreateImage(cvSize(dx, dy), IPL_DEPTH_8U, 1);
	result = (int *)malloc(sizeof(int)*dx*dy);

	memset(result, 0, sizeof(int)*dx*dy);
	CvRect roi = cvRect(orign_x, orign_y, dx, dy);
	cvSetImageROI(src, roi);
	cvCopy(src, target);
	cvResetImageROI(src);

	for (k = 0; k<num_sp; k++)
	{
		x = spoint[k].x + orign_x;
		y = spoint[k].y + orign_y;

		fy = floor(y);	
		fx = floor(x);
		cy = ceil(y);	
		cx = ceil(x);
		ty = y - fy;
		tx = x - fx;
		w1 = (1 - tx) * (1 - ty);
		w2 = tx  * (1 - ty);
		w3 = (1 - tx) * ty;
		w4 = tx * ty;
		v = pow(dishu, (float)k);

		for (i = 0; i<dy; i++)
		{
			for (j = 0; j<dx; j++)
			{

				N = w1 * (double)(unsigned char)src->imageData[(i + fy)*src->width + j + fx] +
					w2 * (double)(unsigned char)src->imageData[(i + fy)*src->width + j + cx] +
					w3 * (double)(unsigned char)src->imageData[(i + cy)*src->width + j + fx] +
					w4 * (double)(unsigned char)src->imageData[(i + cy)*src->width + j + cx];

				if (N >= (double)(unsigned char)target->imageData[i*dx + j])
				{
					result[i*dx + j] = result[i*dx + j] + v * 1;
				}
				else{
					result[i*dx + j] = result[i*dx + j] + v * 0;
				}
			}
		}
	}

	for (i = 0; i < dy; i++)
	{
		for (j = 0; j < dx; j++)
		{
			result[i*dx + j] = Mapping[result[i*dx + j]];
		}
	}

	int cols = 0;
	int mapping_size = pow(dishu, (float)num_sp);
	for (i = 0; i < mapping_size; i++)
	{
		if (cols < Mapping[i])
		{
			cols = Mapping[i];
		}
	}

	if (cols < 255)
	{
		
		for (i = 0; i<dy; i++)
		{
			for (j = 0; j<dx; j++)
			{
				target->imageData[i*dx + j] = (unsigned char)result[i*dx + j];
				
			}
		}
	
		cv::Mat tmpImg =target;
		RstImg = tmpImg;

	}

	hist = cvCreateImage(cvSize(300, 200), IPL_DEPTH_8U, 3);

	vector<double> hist_val;
	for (i = 0; i<cols; i++)
	{

		hist_val.push_back(0.0);
	}
	for (i = 0; i<dy*dx; i++)
	{

		hist_val[result[i]]++;
	}

	double temp_max = 0.0;

	for (i = 0; i<cols; i++)			
	{
	
		if (temp_max<hist_val[i])
		{
			
			temp_max = hist_val[i];
		}

	}

	CvPoint p1, p2;
	double bin_width = (double)hist->width / cols;
	double bin_unith = (double)hist->height / temp_max;

	for (i = 0; i<cols; i++)
	{
		p1.x = i*bin_width; p1.y = hist->height;
		p2.x = (i + 1)*bin_width; p2.y = hist->height - /*val_hist[i]*/hist_val[i] * bin_unith;
		cvRectangle(hist, p1, p2, cvScalar(0, 255), -1, 8, 0);
		HistVec.push_back(hist_val[i]);
	}
	
	cv::Mat tmphist = hist;
	HistImg = tmphist;

	
}

void RuiLbp::rotation_uniform_invariant_mapping(int range, int num_sp, int *Mapping)
{
	int numt, i, j, tem_xor;

	numt = 0;
	tem_xor = 0;
	for (i = 0; i< range; i++)
	{
		j = i << 1;
		if (j > range - 1)
		{
			j = j - (range - 1);
		}

		tem_xor = i ^ j;	
		numt = calc_sum(tem_xor);

		if (numt <= 2)
		{
			Mapping[i] = calc_sum(i);
		}
		else{
			Mapping[i] = num_sp + 1;
		}
	}
}

cv::Mat RuiLbp::GetLBPRstImg()
{
	return RstImg;
}
cv::Mat RuiLbp::GetLBPRstHistImg()
{
	return HistImg;
}
vector<float> RuiLbp::GetLbpHistVec()
{
	return HistVec;
}

//assistant function
void RuiLbp::calc_position(int radius, int num_sp, MyPoint *spoint)
{
	double theta;

	theta = 2 * CV_PI / num_sp;

	for (int i = 0; i < num_sp; i++)
	{
		spoint[i].y = -radius * sin(i * theta);
		spoint[i].x = radius * cos(i * theta);
	}
}
//assistant function
int RuiLbp::calc_sum(int r)
{
	int res_sum;

	res_sum = 0;
	while (r)
	{
		res_sum = res_sum + r % 2;
		r /= 2;
	}
	return res_sum;
}
