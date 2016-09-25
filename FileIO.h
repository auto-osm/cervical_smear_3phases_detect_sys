#ifndef FILEIO_H_
#define FILEIO_H_

#include <armadillo>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>

#include "DataStruct.h"

class FileInTranOut
{
public:
	FileInTranOut();
	~FileInTranOut();

	vector<string> LoadImgList(string ImgFile);
	map<string, double> LoadImgDict(string pathfile);
	int DictIdentifyLabel(string ImgName, map<string, double> ImgLabeldict);
	
	void SaveVecToTxt(CellInstance& OneCell, string SaveFilePath);
	vector<CellInstance> LoadFeatFile();

private:

};

#endif FILEIO_H_
