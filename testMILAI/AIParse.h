#pragma once

#ifdef AIPARSE_EXPORTS
#define AIPARSE_DECLSPEC __declspec(dllexport)
#else
#define AIPARSE_DECLSPEC __declspec(dllimport)
#endif

#include "MachineLearningExp.h"

#include <mil.h>
#include <boost/smart_ptr.hpp>
#include <windows.h>
#include <map>
#include <io.h>
#include <fstream>
#include <iostream>


using namespace std;

class CAIParse;
typedef boost::shared_ptr<CAIParse>CAIParsePtr;


struct Point {
	int px, py;};

struct Box {

	int x1, y1, x2, y2;			//for record trainning box
	MIL_DOUBLE CX, CY, H, W;	//for record predicted box
};


class MACHINELEARNING_DECLSPEC CAIParse {

public:
	CAIParse(MIL_ID MilSystem);
	~CAIParse() {};

	void MIL_STRING2string(MIL_STRING milstrX, string& strX);
	wchar_t* string2MIL_STRING(string strX);

	void CreateFolder(const MIL_STRING& FolderPath);
	void Split(const string& str, vector<string>& tokens, const string& delimiters);
	void getFoldersInFolder(string Path, vector<string>& Folders);
	void getFilesInFolder(string Path, string FileType, vector<MIL_STRING>& Files);

	void getLabelFilesInFolder(string Path,
		string Label, 
		string FileType, 
		map<string, vector<MIL_STRING>>& mapLabelFiles);

	void getFilesInFolder(string Path, string FileType, vector<string>& Files);

	void readBlob2Vector(string file, vector<int>& blob_px, vector<int>& blob_py);
	void readClasses2Vector(string file, vector<MIL_STRING>& vecImgPaths);
	void readDataSet2Vector(string file, vector<MIL_STRING>& vecImgPaths, vector<vector<Box>>& vec2Boxes, vector<vector<int>>& veclabels);

	void blobClip(string ImgDir, string ImgName, vector<int> blob_px, vector<int> blob_py, vector<MIL_ID>& ClipImgs, string CropedImgDir);
	void cropBox(vector<int>blob_px, vector<int>blob_py, int CropSizeX, int CropSizeY, int ImgSizeX, int ImgSizeY, vector<Box>& vecCropedBoxes);

	void ImgCenterCrop(MIL_ID ImgIn,int CropWH,MIL_ID& ImgOut);


public:
	MIL_ID m_MilSystem;

};