#pragma once
#include <iostream>




#if 1
#include "MILTest.h"

#include <iostream>
#include <list>
#include <chrono>
void test() {

		std::list<int> myList;

		// 使用 push_back() 向 list 中插入一百万个元素
		auto start_push_back = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1000000; ++i) {
			myList.push_back(i);
		}
		auto end_push_back = std::chrono::high_resolution_clock::now();
		auto duration_push_back = std::chrono::duration_cast<std::chrono::milliseconds>(end_push_back - start_push_back);
		std::cout << "push_back() duration: " << duration_push_back.count() << " ms" << std::endl;

		// 使用 emplace_back() 向 list 中插入一百万个元素
		std::list<int> myList2;
		auto start_emplace_back = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 1000000; ++i) {
			myList2.emplace_back(i);
		}
		auto end_emplace_back = std::chrono::high_resolution_clock::now();
		auto duration_emplace_back = std::chrono::duration_cast<std::chrono::milliseconds>(end_emplace_back - start_emplace_back);
		std::cout << "emplace_back() duration: " << duration_emplace_back.count() << " ms" << std::endl;



}

int main(int argc, char* argv[]) {
	
	//int temp = 100000000;
	//const char* temp_char = std::to_string(temp).c_str();
	//int a = int(*temp_char);
	//test();
	



	//MIL_STRING strProject = L"Sp_SPA_ASI_Reclass_DataSet";
	MIL_STRING strProject = L"COT_Raw";
	MIL_UNIQUE_APP_ID MilApplication = MappAlloc(M_NULL, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_SYS_ID MilSystem = MsysAlloc(M_DEFAULT, M_SYSTEM_HOST, M_DEFAULT, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_DISP_ID MilDisplay = MdispAlloc(MilSystem, M_DEFAULT, MIL_TEXT("M_DEFAULT"), M_DEFAULT, M_UNIQUE_ID);
	MILTestPtr m_MILTestPtr = MILTestPtr(new MILTest(MilSystem, MilDisplay, strProject));
	
	//m_MILTestPtr->MILTestKTtreedbscan();

	//vector<vector<double>> AllImgsData;
	//m_MILTestPtr->ReadTxt2Vector(AllImgsData);
	//m_MILTestPtr->MILTestDetPredictMutiThread();
	//Classifier CNN test
	//m_MILTestPtr->FillImgs();
	//m_MILTestPtr->CropImgs();

	//m_MILTestPtr->MILTestGenDataset();
	//m_MILTestPtr->MILTestTrain();
	//m_MILTestPtr->MILTestPredictAP();
	//m_MILTestPtr->MILTestPredictWithBlob();
	//m_MILTestPtr->MILTestPredictEngine();
	
	//Detection CNN test
	
	//m_MILTestPtr->MILTestGenDetDataset();
	//m_MILTestPtr->MILTestDetTrain();
	m_MILTestPtr->MILTestDetPredict();
	//m_MILTestPtr->MILTestValDetModel();
	//ONNX test
	//m_MILTestPtr->MILTestONNXPredict();
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
