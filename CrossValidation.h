#ifndef CROSSVALIDATION_H_
#define CROSSVALIDATION_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "svm.h"
#include "SVMPort.h"
#include "Subset.h"
#include "DataStruct.h"

using namespace std;

struct CVRst
{
	float C;
	float g;
	float Acc;
};

class CrossValid
{
public:
	CrossValid();
	~CrossValid();
	void SetInData(float C_step, float C_upper, float C_lower,
		float g_step, float g_upper, float g_lower, vector<CellInstance> train,
		 int feat_dim, int problem_f, int cv_folder);
	void RunK_folderCV();
	void RunLeaveOneOut();
	float GetBestC();
	float GetBestg();
	float GetK_CVAcc();
private:
	float C_s, C_u, C_l, g_s, g_u, g_l;
	vector<CellInstance> train_data, validation_data;
	vector<vector<CellInstance>> k_train, k_validation;
	int dim;
	int prob_flag;
	float best_C, best_g, best_acc;
	int k_folder;
	vector<CVRst> ParamAcc;
	vector<CVRst> k_BestParamAcc;

	void Split_k_TrainValidation();
};

#endif 
