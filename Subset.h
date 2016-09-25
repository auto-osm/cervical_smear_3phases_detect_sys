#ifndef SUBSET_H_
#define SUBSET_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

//#include <boost/random.hpp>
//#include <boost/random/random_device.hpp>

#include "DataStruct.h"

using namespace std;

class Subset
{
public:
	Subset();
	~Subset();
	void SetInMultiClassData(vector<CellInstance> OrgData, 
		float proportion, int BeginLabel, int EndLabel);
	void RunMultiClassSubset();

	void SetInBinaryClassData(vector<CellInstance> OrgData,
		float proportion);
	void RunBinaryClassSubset();

	vector<CellInstance> GetTrainSet();
	vector<CellInstance> GetTestSet();

	void SaveDividedData(string TrainFile, string TestFile);
	void SaveDividedDataTolibsvm(string TrainFile, string TestFile);
private:
	//assistant variables
	vector<CellInstance> Org_Data;
	vector<CellInstance> train, test;
	
	int BeginL, EndL;
	float p;
	//assistant functions
	void DividedToP(vector<CellInstance> OneClassInstance);
	int GenRandomNum(int lower, int upper);
	void SaveDataToFile(string FileName, vector<CellInstance> data);
	void SaveDataTolibsvmFile(string libFileName, vector<CellInstance> data);
};

#endif
