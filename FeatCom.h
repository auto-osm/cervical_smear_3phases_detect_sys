#ifndef FEATCOM_H_
#define FEATCOM_H_

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <strstream>
#include <sstream>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "DataStruct.h"

class FeatCom
{
public:
	FeatCom();
	~FeatCom();
	void SetInData(vector<CellInstance> train_data,
		vector<CellInstance> test_data, int RandInstanceNum);
	void RunRelief_F();
	void RunFeatCom();
	vector<float> GetWeightVec();
	vector<CellInstance> GetW_train();
	vector<CellInstance> GetW_test();

	void SaveDividedData(string TrainFile, string TestFile);
	void SavelibsvmData(string w_trainfile, string w_testfile);

private:
	//input variables
	vector<CellInstance> train, test;
	int SelectedInstanceNum;
	//result variable
	vector<float> WeightVec;
	vector<CellInstance> weight_train, weight_test;
	
	//assistent variable
	vector<vector<float>> SampleMatrix;//train matrix
	vector<vector<float>> SelectedSamples;

	//assistent funciton
	void initWeight();
	void InstanceToMatrix();
	void RandomSelectedISamples();
	void FindSameClassNearest(vector<float> X, vector<float>& NH);
	void FindDiffClassNearest(vector<float> X, vector<float>& NM);

	void ComputeWeights(vector<float>, vector<float>, vector<float>);
	void ComputeDist(vector<float> x, vector<float> curSample, float& dist);
	
	int GenRandomNum(int lower, int upper);

	void FindSameClassKNearest(vector<float> X, vector<float>& NH);
	void  FindDiffClassKNearest(vector<float> X, vector<float>& NM);
	vector<float> ComputeCenter(vector<vector<float>> top_k);
	void SelectTopK(vector<vector<float>>& top_k,
		vector<float> curInstance, int InstanceNum, int IsSame);
	vector<float> findMiniInstance(vector<float>curInstance,
		vector<vector<float>>CurrentClass, int& index);

	void SaveDataTolibsvmFile(string libFileName, vector<CellInstance> data);
	void SaveDataToFile(string FileName, vector<CellInstance> data);
};

#endif
