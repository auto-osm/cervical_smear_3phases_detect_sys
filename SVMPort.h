#ifndef SVMPORT_H_
#define SVMPORT_H_

#include "DataStruct.h"
#include "svm.h"

class libSVM
{
public:
	libSVM();
	~libSVM();
	void SetInData(vector<CellInstance> train_data,
		vector<CellInstance> test_data, int dim_num, const char *ModelFileName, int Problem_flag,
		double set_C, double set_g);
	void SetInExistingModel(const char * modelfile, vector<CellInstance> test_data,
		int dim_num,  int Problem_flag);

	void SetInOneClassData(vector<CellInstance> train_data,
		vector<CellInstance> test_data, int dim_num, const char *ModelFileName, int Problem_flag,
		double set_C, double set_g, double nu);

	void RunTraining();
	void RunTesting();
	vector<double> GetTestPredictRst();
	vector<double> GetTrainPredictRst();
	svm_model* GetTrainedModel();
	void ShowTestRstCompare();
	void ShowTrainRstCompare();
	
	vector<CellInstance> Get_test_TruePositive();
	vector<CellInstance> Get_test_TrueNegitive();
	vector<CellInstance> Get_test_FalsePositive();
	vector<CellInstance> Get_test_FalseNegitive();

	double GetTrainAcc();
	double GetTestAcc();

	void ShowSaveFalseInstance(string FPSaveName, string FNSaveName);

	void SaveTestTP(string SaveName);
	void SaveTestTN(string SaveName);
	void SaveTestFP(string SaveName);
	void SaveTestFN(string SaveName);

	vector<CellInstance> Get_train_TruePositive();
	vector<CellInstance> Get_train_TrueNegitive();
	vector<CellInstance> Get_train_FalsePositive();
	vector<CellInstance> Get_train_FalseNegitive();

	void SaveTrainTP(string SaveName);
	void SaveTrainTN(string SaveName);
	void SaveTrainFP(string SaveName);
	void SaveTrainFN(string SaveName);

	void SaveTrainTPwithImg(string SaveName);
	void SaveTrainTNwithImg(string SaveName);

	vector<CellInstance> Get_train_Positive();
	vector<CellInstance> Get_train_Negitive();

	vector<ClassifiedInstance> Get_multi_RstData();

	void ShowMultiErrorInstance(string SaveName);

	double Get_2_classTestPrec();
	double Get_2_classTestRec();
	double Get_2_classTestF();

private:
	void InitialParam();
	void LoadDataTosvm_node();

	svm_model* model;
	svm_parameter para;
	double C, g;
	double one_class_nu;//one-class SVM parameter
	svm_problem prob;
	vector<CellInstance> train, test;
	int feat_num;
	const char* ModelFile;
	vector<double> TestPredictRst;
	vector<double> TrainPredictRst;

	double train_acc, test_acc;
	void ComputeTrainAcc();

	vector<CellInstance> _test_TruePositive, _test_TrueNegitive;
	vector<CellInstance> _test_FalsePositive, _test_FalseNegitive;
	int prob_flag;

	vector<CellInstance> _train_TruePositive, _train_TrueNegitive;
	vector<CellInstance> _train_FalsePositive, _train_FalseNegitive;

	//multi-class error 
	vector<ClassifiedInstance> multi_Result;

	double test_Prec, test_Rec, test_F;
};

#endif
