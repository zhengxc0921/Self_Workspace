#pragma once
#include <iostream>




#if 1
#include "MILTest.h"
int main(int argc, char* argv[]) {


	string strProject = "lslm_bmp";
	MIL_UNIQUE_APP_ID MilApplication = MappAlloc(M_NULL, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_SYS_ID MilSystem = MsysAlloc(M_DEFAULT, M_SYSTEM_HOST, M_DEFAULT, M_DEFAULT, M_UNIQUE_ID);
	MIL_UNIQUE_DISP_ID MilDisplay = MdispAlloc(MilSystem, M_DEFAULT, MIL_TEXT("M_DEFAULT"), M_DEFAULT, M_UNIQUE_ID);
	MILTestPtr m_MILTestPtr = MILTestPtr(new MILTest(MilSystem, MilDisplay, strProject));
	m_MILTestPtr->MILTestDetPredictMutiThread();
	////多线程测试
	//MILTest ppt(MilSystem, MilDisplay, strProject);
	//thread tt1(&MILTest::MILTestDetPredictMutiThreadCore, &ppt);
	//tt1.join();



	//MILTest* pt = new MILTest(MilSystem, MilDisplay, strProject);
	//thread t2(&MILTest::MILTestDetPredictMutiThreadCore, *pt);
	//t2.join();


	//MILTest* pt = new MILTest(MilSystem, MilDisplay, strProject);
	//thread t1(&MILTest::MILTestDetPredictMutiThreadCore, *pt);
	//t1.join();
	//thread t2(&MILTest::MILTestDetPredictMutiThreadCore, *pt);
	//t2.join();


	//std::thread tmp_t(&MILTestPtr::MILTestDetPredictMutiThread ,&m_MILTestPtr);
	//std::thread myobj(function_1);//创建了线程，一旦创建，线程就开始执行
	//myobj.join();//主线程阻塞到这里等待function_1执行完毕，当子线程执行完毕，主线程就继续执行

	//Classifier CNN test
	//m_MILTestPtr->FillImgs();
	//m_MILTestPtr->CropImgs();

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
