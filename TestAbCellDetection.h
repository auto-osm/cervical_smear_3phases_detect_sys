#include "DataStruct.h"
#include "FeatureExtraction.h"
#include "ExtractSubSet.h"
#include "FileIO.h"
#include "Scale.h"
#include "Subset.h"
#include "CrossValidation.h"
#include "FeatCom.h" 
#include "svm.h"
#include "SVMPort.h"

//1.Feature Extraction
vector<CellInstance> FeatureExtraction(string ImgList, string DictPath)
{
	vector<CellInstance> InstanceVec;
	FileInTranOut InOutObj;
	vector<string> NameList = InOutObj.LoadImgList(ImgList);
	map<string, double> NameLabelDict = InOutObj.LoadImgDict(DictPath);
	clock_t startTime, finishTime;
	startTime = clock();

	for (int i = 0; i < NameList.size(); i++)
	{
		CellInstance OneCellInstance;
		string ImgName = NameList[i];
		OneCellInstance.ImageName = ImgName;
		cout << "Feature Extraction ImageName and No.: " << endl;
		cout << ImgName << " and " << i +1<< endl;
		//identify the image label
		string ImgfullName = ImgName + ".BMP";
		int CellLabel = InOutObj.DictIdentifyLabel(ImgfullName, NameLabelDict);
		OneCellInstance.label = CellLabel;
		
		string OriginalImage = ".//original_data//";
		string GtImage = ".//original_data//";
		switch (CellLabel)
		{
		case 1:
			OriginalImage = OriginalImage + "//normal_superficiel//" + ImgName + ".BMP";
			GtImage = GtImage + "//normal_superficiel//" + ImgName + "-d.bmp";
			break;
		case 2:
			OriginalImage = OriginalImage + "//normal_intermediate//" + ImgName + ".BMP";
			GtImage = GtImage + "//normal_intermediate//" + ImgName + "-d.bmp";
			break;
		case 3:
			OriginalImage = OriginalImage + "//normal_columnar//" + ImgName + ".BMP";
			GtImage = GtImage + "//normal_columnar//" + ImgName + "-d.bmp";
			break;
		case 4:
			OriginalImage = OriginalImage + "//light_dysplastic//" + ImgName + ".BMP";
			GtImage = GtImage + "//light_dysplastic//" + ImgName + "-d.bmp";
			break;
		case 5:
			OriginalImage = OriginalImage + "//moderate_dysplastic//" + ImgName + ".BMP";
			GtImage = GtImage + "//moderate_dysplastic//" + ImgName + "-d.bmp";
			break;
		case 6:
			OriginalImage = OriginalImage + "//severe_dysplastic//" + ImgName + ".BMP";
			GtImage = GtImage + "//severe_dysplastic//" + ImgName + "-d.bmp";
			break;
		case 7:
			OriginalImage = OriginalImage + "//carcinoma_in_situ//" + ImgName + ".BMP";
			GtImage = GtImage + "//carcinoma_in_situ//" + ImgName + "-d.bmp";
			break;
			
		}
		FeatureExtract feat_ext_obj;
		feat_ext_obj.SetInData(OriginalImage, GtImage, i);
		feat_ext_obj.RunExtract();
		vector<float> FeatureVec = feat_ext_obj.GetFeatureVec();
		OneCellInstance.feature = FeatureVec;
		InstanceVec.push_back(OneCellInstance);
		InOutObj.SaveVecToTxt(OneCellInstance, ".//generated_data//FeatExtraction.txt");
	}

	finishTime = clock();
	double time_length = double((finishTime - startTime) / NameList.size()) / CLOCKS_PER_SEC;
	cout << "The average feature extraction time per image  : " << time_length << endl;
	return InstanceVec;
}

//2.Feature Combination
vector<vector<CellInstance>> FeatCom(vector<CellInstance>& CellFeats)
{
	vector<vector<CellInstance>> TrainTestRst;
	vector<float> OneFeatVec=CellFeats[0].feature;
	int feature_d = OneFeatVec.size();
	cout << "Convert labels ..." << endl;
	ExtractSubClass subclass_obj;
	subclass_obj.SetInInstanceData(CellFeats);
	vector<CellInstance> BinaryData = subclass_obj.GetSevenToBinary();
	cout << "Data size : " << BinaryData.size() << endl;
	
	cout << "Subset data ..." << endl;
	Subset subset_obj;
	subset_obj.SetInBinaryClassData(BinaryData, 0.7);
	subset_obj.RunBinaryClassSubset();
	vector<CellInstance> train_data = subset_obj.GetTrainSet();
	vector<CellInstance> test_data = subset_obj.GetTestSet();

	//scaling train and test
	Scale scaling_obj;
	scaling_obj.SetInData(train_data, test_data, feature_d, 0);
	scaling_obj.RunScale();
	vector<CellInstance> scaled_train = scaling_obj.GetScaledTrain();
	vector<CellInstance> scaled_test = scaling_obj.GetScaledTest();

	//Feature Combination
	FeatCom f_obj;
	f_obj.SetInData(scaled_train, scaled_test, 100);
	clock_t startTime, finishTime;
	startTime = clock();
	f_obj.RunFeatCom();
	finishTime = clock();
	double time_length = double(finishTime - startTime) / CLOCKS_PER_SEC;
	cout << "The feature combination Time : " << time_length << endl;
	vector<float> Weights_rff = f_obj.GetWeightVec();
	vector<CellInstance> w_train_rff = f_obj.GetW_train();
	vector<CellInstance> w_test_rff = f_obj.GetW_test();

	TrainTestRst.push_back(w_train_rff);
	TrainTestRst.push_back(w_test_rff);

	return TrainTestRst;
}

//3.Classification Strategy
vector<ClassifiedInstance> Abnormal_detection(vector<CellInstance> wtrain, vector<CellInstance> wtest)
{
	vector<ClassifiedInstance> svm_rst;
	int feature_d = 160;
	//scaling train and test for libsvm
	Scale scaling_obj_svm;
	scaling_obj_svm.SetInData(wtrain, wtest, feature_d, 0);//0 for 0-1 normalization
	scaling_obj_svm.RunScale();
	vector<CellInstance> scaled_wtrain_svm = scaling_obj_svm.GetScaledTrain();
	vector<CellInstance> scaled_wtest_svm = scaling_obj_svm.GetScaledTest();
	
	int problem_flag = 2;// problem flag

	//cross validation
	CrossValid cv_obj;
	cv_obj.SetInData(2, 8, 0.5, 0.5, 4, -2, scaled_wtrain_svm, feature_d, problem_flag, 5);
	cv_obj.RunK_folderCV();
	cv_obj.GetK_CVAcc();
	double bestC = cv_obj.GetBestC();
	double bestg = cv_obj.GetBestg();
	cout << "Cross validation best parameters: C = " << bestC << ", g = " << bestg << endl;

	clock_t startTime, finishTime;
	startTime = clock();
	libSVM svm_obj;
	const char *ModelFileName = ".//generated_data//TrainedModel_2class.txt";
	svm_obj.SetInData(scaled_wtrain_svm, scaled_wtest_svm, feature_d, ModelFileName, problem_flag, bestC, bestg);
	svm_obj.RunTraining();
	svm_obj.RunTesting();
	cout << "Training Accuracy : " << svm_obj.GetTrainAcc() <<
		" " << "Testing Accuracy : " << svm_obj.GetTestAcc() << endl;
	cout << "Testing Precision : " << svm_obj.Get_2_classTestPrec() << endl;
	cout << "Testing Recall : " << svm_obj.Get_2_classTestRec() << endl;
	cout << "Testing F-score : " << svm_obj.Get_2_classTestF() << endl;
	finishTime = clock();
	double time_length = double(finishTime - startTime) / CLOCKS_PER_SEC;
	cout << "SVM Testing Time : " << time_length << endl;
	svm_obj.ShowTestRstCompare();
	svm_rst = svm_obj.Get_multi_RstData();//Rst
	return svm_rst;
}

//main test function
void ComDetectAbCell()
{
	string ImageList = ".//original_data//Image917.txt";
	string DictPath = ".//original_data//dict.txt";
	//1.Cell Feature Extraction
	vector<CellInstance> CellFeatVec = FeatureExtraction(ImageList, DictPath);
	cout << "The number of cervical cell: " << CellFeatVec.size() << endl;
	
	//2.Feature Combination
	vector<vector<CellInstance>> W_Data = FeatCom(CellFeatVec);
	vector<CellInstance> w_train = W_Data[0];
	vector<CellInstance> w_test = W_Data[1];

	//3.Abnormal Cell Detection using classification strategy
	vector<ClassifiedInstance> TestRst = Abnormal_detection(w_train, w_test);

	system("pause");
}
