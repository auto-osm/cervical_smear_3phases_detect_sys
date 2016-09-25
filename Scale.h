#ifndef SCALEDATA_H_
#define SCALEDATA_H_

#include "DataStruct.h"

#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))

class Scale
{
public:
	Scale();
	~Scale();
	void SetInData(vector<CellInstance> Train,
		vector<CellInstance> Test, int dimNum, int ScaleMethod);
	void Scale::RunScale();
	vector<CellInstance> GetScaledTrain();
	vector<CellInstance> GetScaledTest();
	void SaveDividedDataTolibsvm(string TrainFile, string TestFile);

	void SaveScaledData(string TrainName, string TestName);
private:
	vector<CellInstance> OrgTrain, OrgTest;
	vector<CellInstance> ScaledTrain, ScaledTest;

	int feat_dim, scaling_mehtod;//scale flag

	double lower, upper;
	double y_lower, y_upper;
	double y_max = -DBL_MAX;
	double y_min = DBL_MAX;

	//assistant variable
	vector<float> feature_max;
	vector<float> feature_min;
	vector<float> feature_mean;
	vector<float> feature_std;

	//assistant function
	void ComputeMinMax();
	void ComputeM_S();
	void RunTrainScale();
	void RunTestScale();

	void SaveDataTolibsvmFile(string libFileName, vector<CellInstance> data);
	void SaveToFile(string FineName, vector<CellInstance> data);
};

#endif
