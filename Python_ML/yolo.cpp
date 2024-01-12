#pragma once
#include "yolo.h"


void  ParaseXML_Direct()
{
	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_ParseXML_Direct = PyObject_GetAttrString(pModule, "Start_ParseXML_Direct");//������Ҫ���õĺ�����
	PyObject* pyParams = PyTuple_New(2); //������������
	string detection_root = "G:/DefectDataCenter/ParseData/Detection";
	string project = "COT_Raw";
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", detection_root.c_str()));// ������ʽת����python��ʽ
	PyTuple_SetItem(pyParams, 1, Py_BuildValue("s", project.c_str()));// ������ʽת����python��ʽ
	PyObject_CallObject(Start_ParseXML_Direct, pyParams);//���ú���

	Py_DECREF(pModule);
}

void  ParseXML_Resize()
{
	//Py_Initialize(); //��ʼ��python������
	//if (!Py_IsInitialized()) {
	//	std::system("pause");
	//	//return -99;
	//} //�鿴python�������Ƿ�ɹ���ʼ��
	//PyRun_SimpleString("import sys");
	//PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");
	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_ParseXML_Direct = PyObject_GetAttrString(pModule, "Start_ParseXML_Resize");//������Ҫ���õĺ�����
	PyObject* pyParams = PyTuple_New(6); //������������
	string detection_root = "G:/DefectDataCenter/ParseData/Detection";
	string project = "COT_Raw";
	float ratio = 2.5;
	int ch = 424;
	int cw = 2688;
	int bd = 300;
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", detection_root.c_str()));// ������ʽת����python��ʽ
	PyTuple_SetItem(pyParams, 1, Py_BuildValue("s", project.c_str()));// ������ʽת����python��ʽ
	PyTuple_SetItem(pyParams, 2, Py_BuildValue("f", ratio));// ������ʽת����python��ʽ
	PyTuple_SetItem(pyParams, 3, Py_BuildValue("n", ch));// ������ʽת����python��ʽ
	PyTuple_SetItem(pyParams, 4, Py_BuildValue("n", cw));// ������ʽת����python��ʽ
	PyTuple_SetItem(pyParams, 5, Py_BuildValue("n", bd));// ������ʽת����python��ʽ
	PyObject* Start_predict = PyObject_GetAttrString(pModule, "Start_ParseXML_Resize");//������Ҫ���õĺ�����
	PyObject_CallObject(Start_predict, pyParams);//���ú���
	Py_DECREF(pModule);
	//Py_Finalize();
}

void  Check_DataSet()
{
	PyObject* pCheckModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pCheckModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_predict = PyObject_GetAttrString(pCheckModule, "Check_TrainDataSet");//������Ҫ���õĺ�����
	PyObject* pyParams = PyTuple_New(2); //������������
	string project = "DSW_random";
	int vis_num = 10;
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", project.c_str()));// ������ʽת����python��ʽ
	PyTuple_SetItem(pyParams, 1, Py_BuildValue("n", vis_num));// ������ʽת����python��ʽ
	PyObject_CallObject(Start_predict, pyParams);//���ú���
	Py_DECREF(pCheckModule);

	//	//////����pt1����

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
//	//PyObject* pFunc = PyObject_GetAttrString(pModule, "pt2");//������Ҫ���õĺ�����
//	//PyEval_CallObject(pFunc, NULL);//���ú���

	//Py_Finalize();
}

void  Yolo_Train()
{

	PyObject* pTrainModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pTrainModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_train = PyObject_GetAttrString(pTrainModule, "Start_train");//������Ҫ���õĺ�����
	PyObject* pyParams = PyTuple_New(1); //������������
	string project = "COT_Raw";
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", project.c_str()));// ������ʽת����python��ʽ

	PyObject* result = PyObject_CallObject(Start_train, pyParams);//���ú���
	if (result && PyIter_Check(result)) {
		PyObject* item;
		while ((item = PyIter_Next(result))) {
			if (PyLong_Check(item)) {
				int value = PyLong_AsLong(item);
				cout << "value: " << value << endl;
				// ���м������д���
			}
			Py_DECREF(item);
		}

		Py_DECREF(result);
	}

	Py_DECREF(pTrainModule);

}

void  Yolo_Predict()
{
	//result Ϊ�յ�ʱ�򣬽�������result
	PyObject* pPredictModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pPredictModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_predict = PyObject_GetAttrString(pPredictModule, "Start_predictFolder");//������Ҫ���õĺ�����

	PyObject* pyParams = PyTuple_New(1); //������������
	string project = "G:/DefectDataCenter/ParseData/Detection/COT_Raw/raw_data/TImg";
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", project.c_str()));// ������ʽת����python��ʽ

	PyObject* result = PyObject_CallObject(Start_predict, pyParams);//���ú���
	if (result && PyIter_Check(result)) {
		PyObject* item;
		while ((item = PyIter_Next(result))) {
			if (PyLong_Check(item)) {
				int value = PyLong_AsLong(item);
				cout << "value: " << value << endl;
				// ���м������д���
			}
			Py_DECREF(item);
		}

		Py_DECREF(result);
	}
	Py_DECREF(pPredictModule);
	//Py_Finalize();
}

void  Pth2ONNX()
{
	PyObject* pPth2ONNXModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pPth2ONNXModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
 	PyObject* Start_Pth2ONNX = PyObject_GetAttrString(pPth2ONNXModule, "TOO_ONNX");//������Ҫ���õĺ�����

	PyObject* pyParams = PyTuple_New(1); //������������
	string project = "COT_Raw";
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", project.c_str()));// ������ʽת����python��ʽ

	PyObject_CallObject(Start_Pth2ONNX, pyParams);//���ú���
	Py_DECREF(pPth2ONNXModule);
	Py_Finalize();
}