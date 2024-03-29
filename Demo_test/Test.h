#pragma once
#include <direct.h>
#include<map>
#include<string>
#include<Python.h>
#include<opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <mil.h>
#include <boost/smart_ptr.hpp>

//#include "../testMILAI/MLClassCNN.h"
#include "../testMILAI/MLDetCNN.h"
#include "../testMILAI/DBSCAN.h"
#include <thread>
//#include "kttreedbscan.cpp"

using namespace cv;
using namespace std;
class Test;
typedef boost::shared_ptr<Test>TestPtr;
class Test {
public:
	Test(MIL_ID MilSystem, MIL_ID MilDisplay, MIL_STRING strProject);
	~Test();

	void getIcon(vector<MIL_STRING> OriginalDataPath,
		vector<vector<MIL_STRING>> ClassName,
		vector<vector<MIL_STRING>>& ClassIcon);

	void getModelInfo(MIL_UNIQUE_CLASS_ID& Model);

	void savePredictedImg();

	//void predictBegin();

	void InitClassWeights();

	//void ReadTxt2Vector(vector<vector<double>>& AllImgsData);

	void CropImgs();

	void FillImgs();

	//void MILTestGenDataset();

	//void MILTestTrain();

	//void MILTestPredict();

	void MILTestPredictAP();

	void MILTestPredictWithBlob();

	void MILTestPredictEngine();

	void MILTestGenDetDataset();

	void MILTestDetTrain();

	void MILTestDetPredict();

	void MILTestValDetModel();
	void MILTestValTxtAP50();
	//多进程测试使用
	void MILTestDetPredictMutiProcessSingle();

	void MILTestDetPredictMutiProcess(string strShareMame,
		size_t ShareMameSize,
		string Index,
		string ImgType,
		string strProject);
	void MILTestDetPredictCore(MIL_UNIQUE_CLASS_ID& TestCtx,
		string SrcDir,
		string DstRst,
		double& calc_time);

	void MILTestPredictShareMem(string strShareMame, string index, size_t filesize);

	//多线程测试使用
	void mutiThreadPrepare();
	void MILTestDetPredictMutiThreadCore();
	void MILTestDetPredictMutiThread();

	//onnx例子
	void MILTestONNXPredict();
	void OpencvONNXPredict();
	void MILTestKTtreedbscan();
	void Pytest();


public:
	MIL_ID m_MilSystem;
	MIL_ID m_MilDisplay;

	//CMLClassCNNPtr  m_MLClassCNN;
	CMLDetCNNPtr	m_MLDetCNN;
	vector<MIL_DOUBLE>m_ClassWeights;
	vector<MIL_STRING> m_ClassNames;
	MIL_INT m_ClassesNum = 0;
	MIL_INT m_InputSizeX = 0;
	MIL_INT m_InputSizeY = 0;

	//测试参数
	MIL_STRING m_ClassifierSrcDataDir = L"G:/DefectDataCenter/原始_现场分类数据/LJX/SpTrainData/";
	/*MIL_STRING m_ClassifierSrcDataDir = L"G:/DefectDataCenter/原始_现场分类数据/LJX/TrainData/";*/
	/*MIL_STRING m_ClassifierSrcDataDir = L"G:/DefectDataCenter/ParseData/Classifier/";*/
	MIL_STRING m_DetectionSrcDataDir = L"G:/DefectDataCenter/ParseData/Detection/";

	MIL_STRING m_ClassifierWorkSpace = L"G:/DefectDataCenter/WorkSpace/Classifier/";
	MIL_STRING m_DetectionWorkSpace = L"G:/DefectDataCenter/WorkSpace/Detection/";

	MIL_STRING m_strProject;
	vector<MIL_STRING> m_FilesInFolder;
	vector < ClassificationResultStruct> m_vecResults;
	bool m_SavePredictedImg = 1;
	MIL_STRING m_DstImgDir;
	MIL_UNIQUE_CLASS_ID m_TrainedCtx;
	map<string, MIL_ID >m_PathRawImageMap;
	vector<DET_RESULT_STRUCT> m_vecDetResults;
};