#pragma once
#include <direct.h>

#include "MLClassCNN.h"
#include "MLDetCNN.h"


class MILTest;
typedef boost::shared_ptr<MILTest>MILTestPtr;
class MILTest {
public:
	MILTest(MIL_ID MilSystem, MIL_ID MilDisplay);
	~MILTest();

	void getIcon(vector<MIL_STRING> OriginalDataPath,
		vector<vector<MIL_STRING>> ClassName,
		vector<vector<MIL_STRING>>& ClassIcon);

	void getModelInfo(MIL_UNIQUE_CLASS_ID& Model);

	void savePredictedImg();

	void InitClassWeights();

	void CropImgs();

	void FillImgs();

	void MILTestGenDataset();

	void MILTestTrain();

	void MILTestPredict();

	void MILTestPredictWithBlob();

	void MILTestPredictEngine();

	void MILTestGenDetDataset();

	void MILTestDetTrain();

	void MILTestDetPredict();

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

	void MILTestONNXPredict();




public:
	MIL_ID m_MilSystem;
	MIL_ID m_MilDisplay;

	CMLClassCNNPtr  m_MLClassCNN;
	CMLDetCNNPtr	m_MLDetCNN;
	vector<MIL_DOUBLE>m_ClassWeights;
	vector<MIL_STRING> m_ClassNames;
	MIL_INT m_ClassesNum = 0;
	MIL_INT m_InputSizeX = 0;
	MIL_INT m_InputSizeY = 0;

	//测试参数
	vector<MIL_STRING> m_FilesInFolder;
	vector < ClassificationResultStruct> m_vecResults;
	bool m_SavePredictedImg = TRUE;
	MIL_STRING m_DstImgDir;
};