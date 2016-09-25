#ifndef DATASTRUCT_H_
#define DATASTRUCT_H_

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <strstream>
#include <sstream>

using namespace std;

// convert string into any numerical type
template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}

//convert any numerical  type into string
template <class Type>
string NumToString(const Type& num)
{
	strstream ss;
	string s;
	ss << num;
	ss >> s;
	return s;
}

//cell instance struct 
//with features, gound truth label, and image name
struct CellInstance
{
	vector<float> feature;
	int label;
	string ImageName;
};

//classified result of each cell instance
//adding classified label
struct ClassifiedInstance
{
	int RstLabel;
	CellInstance OneInstance;
};

#endif
