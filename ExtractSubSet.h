#ifndef EXTRACTSUBSET_H_
#define EXTRACTSUBSET_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "DataStruct.h"

using namespace std;

class ExtractSubClass
{
public:
	ExtractSubClass();
	~ExtractSubClass();
	void SetInData(string OrgMultiFile, int dim);
	void SetInInstanceData(vector<CellInstance> Multi_data);
	vector<CellInstance> GetSevenToBinary();
	
private:
	vector<CellInstance> LoadFeatureFile(string featurefile, int feature_dim);
	vector<CellInstance> multi_org_data;
};

#endif EXTRACTSUBFROMORGMULTI_H_
