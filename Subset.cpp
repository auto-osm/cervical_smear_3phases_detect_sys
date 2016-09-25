#include "Subset.h"

Subset::Subset(){};

Subset::~Subset(){};

void Subset::SetInMultiClassData(vector<CellInstance> OrgData,
	float proportion, int BeginLabel, int EndLabel)
{
	Org_Data = OrgData;
	BeginL = BeginLabel;
	EndL = EndLabel;
	p = proportion;
}

void Subset::RunMultiClassSubset()
{
	for (int i = BeginL; i <= EndL; i++)
	{
		vector<CellInstance> OneClassInstance;
		//selected out each class instances
		for (int j = 0; j < Org_Data.size(); j++)
		{
			CellInstance tmp_instance = Org_Data.at(j);
			if (tmp_instance.label == i)
			{
				OneClassInstance.push_back(tmp_instance);
			}
		}
		
		//divided each class into train and test sets
		DividedToP(OneClassInstance);
		
	}
}

void Subset::SetInBinaryClassData(vector<CellInstance> OrgData,
	float proportion)
{
	Org_Data = OrgData;
	p = proportion;
}
void Subset::RunBinaryClassSubset()
{
	//selected out positive and negitive instance
	vector<CellInstance> P_instance, N_instance;
	for (int i = 0; i < Org_Data.size(); i++)
	{
		CellInstance tmp_instance = Org_Data.at(i);
		if (tmp_instance.label == +1)
		{
			P_instance.push_back(tmp_instance);
		}
		if (tmp_instance.label == -1)
		{
			N_instance.push_back(tmp_instance);
		}
	}
	DividedToP(P_instance);
	DividedToP(N_instance);
}

void Subset::DividedToP(vector<CellInstance> OneClassInstance)
{
	int SelectedNum = OneClassInstance.size()*p;
	for (int j = 0; j < SelectedNum; j++)
	{
		int SelectedIndex = GenRandomNum(0, OneClassInstance.size()-1);
	
		CellInstance selected_instance;
		selected_instance = OneClassInstance[SelectedIndex];
		train.push_back(selected_instance);
		OneClassInstance.erase(OneClassInstance.begin() + SelectedIndex);
	}
	for (int j = 0; j < OneClassInstance.size(); j++)
	{
		test.push_back(OneClassInstance.at(j));
	}
}

int Subset::GenRandomNum(int lower, int upper)
{
	int interval = upper-lower ;
	srand((unsigned)time(NULL));
	int RandomNum = rand() % interval + 1;
	return RandomNum;
}

vector<CellInstance> Subset::GetTrainSet()
{
	return train;
}
vector<CellInstance> Subset::GetTestSet()
{
	return test;
}

void Subset::SaveDividedData(string TrainFile, string TestFile)
{
	SaveDataToFile(TrainFile, train);
	SaveDataToFile(TestFile, test);
}

void Subset::SaveDividedDataTolibsvm(string TrainFile, string TestFile)
{
	SaveDataTolibsvmFile(TrainFile, train);
	SaveDataTolibsvmFile(TestFile, test);
}

//assistant function
void Subset::SaveDataToFile(string FileName, vector<CellInstance> data)
{
	ofstream File(FileName);
	for (int i = 0; i < data.size(); i++)
	{
		CellInstance tmp = data.at(i);
		if (tmp.ImageName.empty() == 0)
		{
			File << tmp.ImageName << " ";
		}
		for (int j = 0; j < tmp.feature.size(); j++)
		{
			File << tmp.feature[j] << " ";
		}
		File << tmp.label << endl;
	}
	File.close();
}

void Subset::SaveDataTolibsvmFile(string libFileName, vector<CellInstance> data)
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
