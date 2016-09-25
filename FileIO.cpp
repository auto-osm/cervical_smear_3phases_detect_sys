#include "FileIO.h"

FileInTranOut::FileInTranOut(){}

FileInTranOut::~FileInTranOut(){}

vector<string> FileInTranOut::LoadImgList(string ImgFile)
{
	ifstream ImgListFile(ImgFile);
	vector<string> ImgList;
	string ImgName;
	while (ImgListFile >> ImgName)
	{
		ImgName = ImgName.substr(0, ImgName.size() - 4);
		ImgList.push_back(ImgName);
	}
	ImgListFile.close();
	return ImgList;
}
map<string, double> FileInTranOut::LoadImgDict(string pathfile)
{
	map<string, double> Dict;
	ifstream dictFile;
	string ImgName;
	double Label;
	dictFile.open(pathfile, ios::in);
	while (!dictFile.eof())
	{
		dictFile >> ImgName;
		dictFile >> Label;
		Dict.insert(std::pair<string, double>(ImgName, Label));
	}
	dictFile.close();
	return Dict;
}

int FileInTranOut::DictIdentifyLabel(string ImgName, 
	map<string, double> ImgLabeldict)
{
	std::map<string, double>::iterator it;
	if (ImgName.empty() == 0)
	{
		it = ImgLabeldict.find(ImgName);
	}
	else
	{
		cout << "The image name of identify label function is empty. " << endl;
		system("pause");
	}

	int label = it->second;
	return label;
}

void FileInTranOut::SaveVecToTxt(CellInstance& OneCell, string SaveFilePath)
{
		ofstream cell_featfile(SaveFilePath, ios::app);
		string ImageName = OneCell.ImageName;
		cell_featfile << ImageName << " ";
		vector<float> Feat = OneCell.feature;
		for (int i = 0; i < Feat.size(); i++)
		{
			cell_featfile << Feat[i] << " ";
		}
		double label =OneCell.label;
		cell_featfile << label << endl;
		cell_featfile.close();
}
vector<CellInstance> FileInTranOut::LoadFeatFile()
{
	vector<CellInstance> tmp;
	return tmp;
}
