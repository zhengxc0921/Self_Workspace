#pragma once
#include <iostream>

#if 1
#include "MILTest.h"
int main(int argc, char* argv[]) {
	MIL_STRING strProject = L"FZ";

	//MIL_STRING strProject = L"DSW_random";
	MIL_UNIQUE_APP_ID MilApplication = MappAlloc(M_NULL, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_SYS_ID MilSystem = MsysAlloc(M_DEFAULT, M_SYSTEM_HOST, M_DEFAULT, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_DISP_ID MilDisplay = MdispAlloc(MilSystem, M_DEFAULT, MIL_TEXT("M_DEFAULT"), M_DEFAULT, M_UNIQUE_ID);
	MILTestPtr m_MILTestPtr = MILTestPtr(new MILTest(MilSystem, MilDisplay, strProject));
	//m_MILTestPtr->MILTestDetPredictMutiThread();

	//Classifier CNN test
	//m_MILTestPtr->FillImgs();
	//m_MILTestPtr->CropImgs();
	

	//将数据集和
	MIL_STRING TagFolder = L"Resize_Crop/";  // L"Resize_Crop/",Resize_Crop_1;

	m_MILTestPtr->ReduceSimilarityImg();
	//m_MILTestPtr->MILTestWKSPRCDataset(TagFolder);


	//m_MILTestPtr->MILTestWKSPTrain();
	//m_MILTestPtr->MILTestPredict(TagFolder);
	//m_MILTestPtr->MILTestWKSPUpdate();
	//m_MILTestPtr->MILTestPredict(TagFolder);

	//m_MILTestPtr->MILTestGenDataset();
	//m_MILTestPtr->MILTestTrain(); 
	//m_MILTestPtr->MILTestPredict();
	//m_MILTestPtr->MILTestPredictWithBlob();
	//m_MILTestPtr->MILTestPredictEngine();
	
	//Detection CNN test
	//m_MILTestPtr->MILTestGenDetDataset();
	//m_MILTestPtr->MILTestDetTrain();
	//m_MILTestPtr->MILTestDetPredict();
	//ONNX test
	//m_MILTestPtr->OpencvTest();
	//m_MILTestPtr->MILTestONNXPredict();
	//m_MILTestPtr->MILTestCNNONNXPredict();
	//多进程测试前的预备测试
	//m_MILTestPtr->MILTestDetPredictMutiProcessSingle();



#if 0
	//多进程测试
	string strShareMame = argv[0];
	string strfilesize = argv[1];
	string Index = argv[2];
	string ImgType = argv[3];
	string strProject = argv[4];
	auto m_hMap = ::OpenFileMappingA(FILE_MAP_READ, FALSE, strShareMame.c_str());
	int ShareMameSize = 0;
	std::istringstream ss(strfilesize);
	ss >> ShareMameSize;
	
	m_MILTestPtr->MILTestDetPredictMutiProcess(strShareMame, ShareMameSize, Index, ImgType, strProject);

#endif

	return 1;
}

#endif
