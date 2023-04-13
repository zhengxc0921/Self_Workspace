#pragma once
#include "MILTest.h"


int main() {
	MIL_UNIQUE_APP_ID MilApplication = MappAlloc(M_NULL, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_SYS_ID MilSystem = MsysAlloc(M_DEFAULT, M_SYSTEM_HOST, M_DEFAULT, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_DISP_ID MilDisplay = MdispAlloc(MilSystem, M_DEFAULT, MIL_TEXT("M_DEFAULT"), M_DEFAULT, M_UNIQUE_ID);
	MILTestPtr m_MILTestPtr = MILTestPtr(new MILTest(MilSystem, MilDisplay));

	//Classifier CNN test
	//m_MILTestPtr->CropImgs();

	//m_MILTestPtr->MILTestGenDataset();
	//m_MILTestPtr->MILTestTrain();
	//m_MILTestPtr->MILTestPredict();
	//m_MILTestPtr->MILTestPredictWithBlob();
	//m_MILTestPtr->MILTestPredictEngine();
	
	//Detection CNN test
	m_MILTestPtr->MILTestGenDetDataset();
	m_MILTestPtr->MILTestDetTrain();
	m_MILTestPtr->MILTestDetPredict();

	//ONNX test
	m_MILTestPtr->MILTestONNXPredict();

#if 0
	//¶à½ø³Ì²âÊÔ
	string strShareMame = argv[0];
	string strfilesize = argv[1];
	string index = argv[2];
	auto m_hMap = ::OpenFileMappingA(FILE_MAP_READ, FALSE, strShareMame.c_str());
	int filesize = 0;
	std::istringstream ss(strfilesize);
	ss >> filesize;

	m_MILTestPtr->MILTestPredictShareMem(strShareMame, index, filesize);
#endif

	return 1;
}

