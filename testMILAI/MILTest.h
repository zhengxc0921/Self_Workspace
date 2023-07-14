#pragma once
#include<iostream>
#include<Python.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <direct.h>
#include<map>
#include "MLClassCNN.h"
#include "MLDetCNN.h"
#include <thread>
#include"dbscan.h"


using namespace cv;

class MILTest;
typedef boost::shared_ptr<MILTest>MILTestPtr;
class MILTest {
public:
	MILTest(MIL_ID MilSystem, MIL_ID MilDisplay, MIL_STRING strProject);
	~MILTest();

	void getIcon(vector<MIL_STRING> OriginalDataPath,
		vector<vector<MIL_STRING>> ClassName,
		vector<vector<MIL_STRING>>& ClassIcon);

	void getModelInfo(MIL_UNIQUE_CLASS_ID& Model);

	void savePredictedImg();

	void predictBegin();

	void InitClassWeights();

	void CropImgs();

	void OpencvTest(MIL_ID& ImageReshape);
	
	void FillImgs();

	void ReduceSimilarityImg();

	void Pytest();
	//void isTagSameClass(
	//	vector<MIL_STRING>BaseClassIcons,
	//	const vector<MIL_STRING>& TagClassIcons,
	//	map<MIL_STRING, int>& TagClassIconsIndex);

	void MILTestWKSPDataset(MIL_STRING TagFolder);

	void MILTestWKSPRCDataset(MIL_STRING TagFolder);

	void MILTestWKSPTrain();

	void MILTestWKSPUpdate();

	void MILTestGenDataset();

	void MILTestTrain();

	void MILTestPredict(MIL_STRING TagFolder);

	void MILTestPredictWithBlob();

	void MILTestPredictEngine();

	//void MILTestGenDetDataset();

	//void MILTestDetTrain();

	//void MILTestDetPredict();

	//多进程测试使用
	//void MILTestDetPredictMutiProcessSingle();

	//void MILTestDetPredictMutiProcess(string strShareMame,
	//	size_t ShareMameSize,
	//	string Index,
	//	string ImgType,
	//	string strProject);
	//void MILTestDetPredictCore(MIL_UNIQUE_CLASS_ID& TestCtx,
	//	string SrcDir,
	//	string DstRst,
	//	double& calc_time);


	void MILTestPredictShareMem(string strShareMame, string index, size_t filesize);

	//多线程测试使用
	//void mutiThreadPrepare();
	//void MILTestDetPredictMutiThreadCore();
	//void MILTestDetPredictMutiThread();



	////onnx例子
	//void MILTestONNXPredict();
	//void MILTestCNNONNXPredict();




public:
	MIL_ID m_MilSystem;
	MIL_ID m_MilDisplay;

	CMLClassCNNPtr  m_MLClassCNN;
	//CMLDetCNNPtr	m_MLDetCNN;
	vector<MIL_DOUBLE>m_ClassWeights;
	vector<MIL_STRING> m_ClassNames;
	MIL_INT m_ClassesNum = 0;
	MIL_INT m_InputSizeX = 0;
	MIL_INT m_InputSizeY = 0;


	//测试参数

	MIL_STRING m_TagDataDir = L"G:/DefectDataCenter/WorkSpace/Tag/";
	
	MIL_STRING m_ClassifierSrcDataDir = L"G:/DefectDataCenter/ParseData/Classifier/";
	//MIL_STRING m_DetectionSrcDataDir = L"G:/DefectDataCenter/ParseData/Detection/";


	MIL_STRING m_ClassifierWorkSpace = L"G:/DefectDataCenter/WorkSpace/Classifier/";
	//MIL_STRING m_DetectionWorkSpace = L"G:/DefectDataCenter/WorkSpace/Detection/";


	MIL_STRING m_strProject;  //
	MIL_STRING m_TagFolder;

	vector<MIL_STRING> m_FilesInFolder;
	vector < ClassificationResultStruct> m_vecResults;
	bool m_SavePredictedImg = TRUE;
	MIL_STRING m_DstImgDir;
	MIL_UNIQUE_CLASS_ID m_TrainedCtx;
	map<string,MIL_ID >m_PathRawImageMap;
	//vector<DetResult> m_vecDetResults;


};