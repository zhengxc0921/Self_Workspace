#pragma once
#include <iostream>

#if 1
#include "MILTest.h"
#include "yolo.h"
#include <iostream>
#include <list>
#include <chrono>


//void Pytest()
//{
//		Py_Initialize(); //��ʼ��python������
//	if (!Py_IsInitialized()) {
//		std::system("pause");
//		//return -99;
//	} //�鿴python�������Ƿ�ɹ���ʼ��
//
//
//	PyRun_SimpleString("import sys");
//	PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");
//
//	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
//	if (!pModule) { 
//		cout << "Can't find  Img_Cluster" << endl;
//		std::system("pause");
//	}
//
//	////������ͨ���� xml2txt
//	PyObject* Start_train = PyObject_GetAttrString(pModule, "Start_train");//������Ҫ���õĺ�����
//	PyObject_CallObject(Start_train, NULL);//���ú���
//	PyObject* Start_predict = PyObject_GetAttrString(pModule, "Start_predict");//������Ҫ���õĺ�����
//	PyObject_CallObject(Start_predict, NULL);//���ú���
//
//
//
//	//	////������ͨ���� xml2txt
//	//PyObject* pFunc = PyObject_GetAttrString(pModule, "xml2txt");//������Ҫ���õĺ�����
//	//PyObject* pyParams = PyTuple_New(4); //������������
//	//string	xml_src = "G:/DefectDataCenter/ParseData/Detection/COT_LabelImged/XML";
//	//string src_img_dir = "G:/DefectDataCenter/ParseData/Detection/COT_LabelImged/COT_RAW";
//	//string dst_xml_path = "G:/DefectDataCenter/ParseData/Detection/COT_LabelImged/cot_xml2txt.txt";
//	//string dst_cls_path = "G:/DefectDataCenter/ParseData/Detection/COT_LabelImged/cot_classes.txt";
//	//PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", xml_src.c_str()));// ������ʽת����python��ʽ
//	//PyTuple_SetItem(pyParams, 1, Py_BuildValue("s", src_img_dir.c_str()));// ������ʽת����python��ʽ
//	//PyTuple_SetItem(pyParams, 2, Py_BuildValue("s", dst_xml_path.c_str()));// ������ʽת����python��ʽ
//	//PyTuple_SetItem(pyParams, 3, Py_BuildValue("s", dst_cls_path.c_str()));// ������ʽת����python��ʽ
//	//PyObject_CallObject(pFunc, pyParams);//���ú���
//
//
//	//////����pt1����
//	//PyObject* pFunc = PyObject_GetAttrString(pModule, "pt1");//������Ҫ���õĺ�����
//	//PyObject* pyParams = PyTuple_New(4); //������������
//	//string Csrc_dir = "G:/DefectDataCenter/Test/Src/90";
//	//string dst_dir = "G:/DefectDataCenter/Test/ImgCluster";
//	//float	Eeps = 1.2;
//	////const char* pn = "SPA90";
//	//string pn = "SPA90";
//	//PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", Csrc_dir.c_str()));// ������ʽת����python��ʽ
//	//PyTuple_SetItem(pyParams, 1, Py_BuildValue("s", dst_dir.c_str()));// ������ʽת����python��ʽ
//	//PyTuple_SetItem(pyParams, 2, Py_BuildValue("f", Eeps));// ������ʽת����python��ʽ
//	//PyTuple_SetItem(pyParams, 3, Py_BuildValue("s", pn.c_str()));// ������ʽת����python��ʽ
//	//PyObject_CallObject(pFunc, pyParams);//���ú���
//
//	//PyObject* pFunc = PyObject_GetAttrString(pModule, "pt2");//������Ҫ���õĺ�����
//	//PyEval_CallObject(pFunc, NULL);//���ú���
//	//����python���
//	//Py_DECREF(pyParams);
//	//Py_DECREF(pFunc);
//	Py_DECREF(pModule);
//	Py_Finalize();
//}

void Python_Cpp() {
	//ParaseXML_Direct();
	//ParseXML_Resize();
	//Check_DataSet();
	//Yolo_Train();
	Yolo_Predict();
}

int main(int argc, char* argv[]) {
	//Py_yolo_val();
	Python_Cpp();
	//Pytest();
	//MIL_STRING strProject = L"Sp_SPA_ASI_Reclass_DataSet";
	////MIL_STRING strProject = L"COT_Raw";
	//MIL_UNIQUE_APP_ID MilApplication = MappAlloc(M_NULL, M_DEFAULT, M_UNIQUE_ID);
	//MIL_UNIQUE_SYS_ID MilSystem = MsysAlloc(M_DEFAULT, M_SYSTEM_HOST, M_DEFAULT, M_DEFAULT, M_UNIQUE_ID);
	//MIL_UNIQUE_DISP_ID MilDisplay = MdispAlloc(MilSystem, M_DEFAULT, MIL_TEXT("M_DEFAULT"), M_DEFAULT, M_UNIQUE_ID);
	//MILTestPtr m_MILTestPtr = MILTestPtr(new MILTest(MilSystem, MilDisplay, strProject));
	
	//m_MILTestPtr->MILTestKTtreedbscan();
	////Classifier CNN test
	//m_MILTestPtr->MILTestGenDataset();
	//m_MILTestPtr->MILTestTrain();
	//m_MILTestPtr->MILTestPredictAP();
	//m_MILTestPtr->MILTestPredictWithBlob();
	//m_MILTestPtr->MILTestPredictEngine();
	
	//Detection CNN test
	//m_MILTestPtr->MILTestGenDetDataset();
	//m_MILTestPtr->MILTestDetTrain();
	//m_MILTestPtr->MILTestDetPredict();
	//m_MILTestPtr->MILTestValDetModel();
	//ONNX test
	 //m_MILTestPtr->OpencvONNXPredict();//�ݱ���
	//m_MILTestPtr->MILTestONNXPredict();
	//����̲���ǰ��Ԥ������
	//m_MILTestPtr->MILTestDetPredictMutiProcessSingle();
	 //Pytest 
	 //m_MILTestPtr->Pytest();


#if 0
	//����̲���
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

	return 0;
}

#endif
