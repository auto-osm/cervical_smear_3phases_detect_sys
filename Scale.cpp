#include "Scale.h"

Scale::Scale(){}
Scale::~Scale(){}

void Scale::SetInData(vector<CellInstance> Train,
	vector<CellInstance> Test, int dimNum, int ScaleMethod)
{
	OrgTrain = Train;
	OrgTest = Test;
	feat_dim = dimNum;
	scaling_mehtod = ScaleMethod;
	//initial assistant variables for 0-1 scaling
	lower = -1.0, upper = 1.0;
	for (int i = 0; i < dimNum; i++)
	{
		feature_max.push_back(0);
		feature_min.push_back(50000);
		//mean std scaling
		feature_mean.push_back(0);
		feature_std.push_back(0);
	}
	
}
void Scale::RunScale()
{
	if (scaling_mehtod == 0)
	{
		//0-1 scaling
		for (int i = 0; i < feat_dim; i++)
		{
			feature_max[i] = -DBL_MAX;
			feature_min[i] = DBL_MAX;
		}
		ComputeMinMax();
		RunTrainScale();
		RunTestScale();
	}
	else{
		//mean std scaling
		for (int i = 0; i < feat_dim; i++)
		{
			feature_mean[i] = 0.0;
			feature_std[i] = 0.0;
		}
		ComputeM_S();
		RunTrainScale();
		RunTestScale();
	}
}
vector<CellInstance> Scale::GetScaledTrain()
{
	return ScaledTrain;
}
vector<CellInstance> Scale::GetScaledTest()
{
	return ScaledTest;
}

//compute min and max of each feature
void Scale::ComputeMinMax()
{
	//find out max/min
	for (int i = 0; i < OrgTrain.size(); i++)
	{
		CellInstance temp = OrgTrain.at(i);
		for (int j = 0; j < feat_dim; j++)
		{
			feature_max[j] = max(temp.feature[j], feature_max[j]);
			feature_min[j] = min(temp.feature[j], feature_min[j]);
		}
	}
}

//compute mean and std of each feature
void Scale::ComputeM_S()
{
	int instance_num = OrgTrain.size();
	//mean
	for (int i = 0; i < instance_num; i++)
	{
		CellInstance temp = OrgTrain.at(i);
		for (int j = 0; j < feat_dim; j++)
		{
			feature_mean[j] = feature_mean[j] + temp.feature[j];
		}
	}

	for (int j = 0; j < feat_dim; j++)
	{
		feature_mean[j] = feature_mean[j] / instance_num;
	}

	//std
	for (int i = 0; i < OrgTrain.size(); i++)
	{
		CellInstance tmp = OrgTrain.at(i);
		for (int j = 0; j < feat_dim; j++)
		{
			feature_std[j] = feature_std[j] +
				((tmp.feature[j] - feature_mean[j])*(tmp.feature[j] - feature_mean[j]));
		}
	}
	for (int j = 0; j < feat_dim; j++)
	{
		feature_std[j] = sqrt(feature_std[j] / instance_num);
		
	}
}

void Scale::RunTrainScale()
{
	if (scaling_mehtod == 0)
	{
		for (int i = 0; i < OrgTrain.size(); i++)
		{
			CellInstance temp = OrgTrain.at(i);
			CellInstance new_instance = temp;
			for (int j = 0; j < feat_dim; j++)
			{
				double value = temp.feature[j];
				double tmp1 = value - feature_min[j];
				double tmp2 = feature_max[j] - feature_min[j];
				double tmp3 = tmp1 / tmp2;
				double new_value = lower + (upper - lower) * tmp3;
				new_instance.feature[j] = new_value;
			}
			ScaledTrain.push_back(new_instance);
		}
	}
	else
	{
		for (int i = 0; i < OrgTrain.size(); i++)
		{
			CellInstance temp = OrgTrain.at(i);
			CellInstance new_instance = temp;
			for (int j = 0; j < feat_dim; j++)
			{
				double value = temp.feature[j];
				double new_value = /*lower + (upper - lower) **/ (value - feature_mean[j]) / feature_std[j];
				new_instance.feature[j] = new_value;
			}
			ScaledTrain.push_back(new_instance);
		}
	}
}

void Scale::RunTestScale()
{
	if (scaling_mehtod == 0)
	{
		for (int i = 0; i < OrgTest.size(); i++)
		{
			CellInstance temp = OrgTest.at(i);
			CellInstance new_instance = temp;
			for (int j = 0; j < feat_dim; j++)
			{
				double value = temp.feature[j];
				double tmp1 = value - feature_min[j];
				double tmp2 = feature_max[j] - feature_min[j];
				double tmp3 = tmp1 / tmp2;
				double new_value = lower + (upper - lower) * tmp3;
				new_instance.feature[j] = new_value;
			}
			ScaledTest.push_back(new_instance);
		}
	}
	else
	{
		for (int i = 0; i < OrgTest.size(); i++)
		{
			CellInstance temp = OrgTest.at(i);
			CellInstance new_instance = temp;
			for (int j = 0; j < feat_dim; j++)
			{
				double value = temp.feature[j];
				double new_value =/* lower + (upper - lower) **/ (value - feature_mean[j]) / feature_std[j];
				new_instance.feature[j] = new_value;
			}
			ScaledTest.push_back(new_instance);
		}
	}
}

void Scale::SaveDividedDataTolibsvm(string TrainFile, string TestFile)
{
	SaveDataTolibsvmFile(TrainFile, ScaledTrain);
	SaveDataTolibsvmFile(TestFile, ScaledTest);
}

void Scale::SaveDataTolibsvmFile(string libFileName, vector<CellInstance> data)
{
	ofstream File(libFileName);
	for (int i = 0; i < data.size(); i++)
	{
		CellInstance tmp = data.at(i);
		File << tmp.label << " ";
		for (int j = 0; j < tmp.feature.size(); j++)
		{
			File << j + 1 << ":" << tmp.feature[j] << " ";
		}
		File << endl;
	}
	File.close();
}

void Scale::SaveScaledData(string TrainName, string TestName)
{
	SaveToFile(TrainName, ScaledTrain);
	SaveToFile(TestName, ScaledTest);
}

void Scale::SaveToFile(string FineName, vector<CellInstance> data)
{
	ofstream File(FineName);
	for (int i = 0; i < data.size(); i++)
	{
		CellInstance tmp = data.at(i);
		File << tmp.ImageName << " ";
		for (int j = 0; j < tmp.feature.size(); j++)
		{
			File << tmp.feature[j] << " ";
		}
		File << tmp.label << endl;
	}
	File.close();
}
