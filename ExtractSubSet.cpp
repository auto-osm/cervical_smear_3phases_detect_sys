#include "ExtractSubSet.h"

ExtractSubClass::ExtractSubClass(){}
ExtractSubClass::~ExtractSubClass(){}

void ExtractSubClass::SetInData(string OrgMultiFile, int dim)
{
	multi_org_data = LoadFeatureFile(OrgMultiFile, dim);
}

void ExtractSubClass::SetInInstanceData(vector<CellInstance> Multi_data)
{
	multi_org_data = Multi_data;
}
vector<CellInstance> ExtractSubClass::GetSevenToBinary()
{
	vector<CellInstance> BinaryClass;
	for (int j = 0; j < multi_org_data.size(); j++)
	{
		CellInstance tmp = multi_org_data.at(j);
		if (tmp.label <= 3)
		{
			tmp.label = +1;
			BinaryClass.push_back(tmp);
		}
		else
		{
			tmp.label = -1;
			BinaryClass.push_back(tmp);
		}
	}
	return BinaryClass;
}
