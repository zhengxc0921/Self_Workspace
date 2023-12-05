#pragma once
#include "yolo.h"
void  Yolo_Train()
{
	Py_Initialize(); //初始化python解释器
	if (!Py_IsInitialized()) {
		std::system("pause");
		//return -99;
	} //查看python解释器是否成功初始化
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");
	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_train = PyObject_GetAttrString(pModule, "Start_train");//这里是要调用的函数名
	PyObject_CallObject(Start_train, NULL);//调用函数
	Py_DECREF(pModule);
	Py_Finalize();
}

void  Yolo_Predict()
{
	Py_Initialize(); //初始化python解释器
	if (!Py_IsInitialized()) {
		std::system("pause");
		//return -99;
	} //查看python解释器是否成功初始化
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");

	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_predict = PyObject_GetAttrString(pModule, "Start_predict");//这里是要调用的函数名
	PyObject_CallObject(Start_predict, NULL);//调用函数
	Py_DECREF(pModule);
	Py_Finalize();
}

void  ParaseXML_Direct()
{
	Py_Initialize(); //初始化python解释器
	if (!Py_IsInitialized()) {
		std::system("pause");
		//return -99;
	} //查看python解释器是否成功初始化
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");

	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_predict = PyObject_GetAttrString(pModule, "Start_ParseXML_Direct");//这里是要调用的函数名
	PyObject_CallObject(Start_predict, NULL);//调用函数
	Py_DECREF(pModule);
	Py_Finalize();
}

void  ParseXML_Resize()
{
	Py_Initialize(); //初始化python解释器
	if (!Py_IsInitialized()) {
		std::system("pause");
		//return -99;
	} //查看python解释器是否成功初始化
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");

	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_predict = PyObject_GetAttrString(pModule, "Start_ParseXML_Resize");//这里是要调用的函数名
	PyObject_CallObject(Start_predict, NULL);//调用函数
	Py_DECREF(pModule);
	Py_Finalize();
}

void  Check_DataSet()
{
	Py_Initialize(); //初始化python解释器
	if (!Py_IsInitialized()) {
		std::system("pause");
		//return -99;
	} //查看python解释器是否成功初始化
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");

	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_predict = PyObject_GetAttrString(pModule, "Check_TrainDataSet");//这里是要调用的函数名
	PyObject_CallObject(Start_predict, NULL);//调用函数
	Py_DECREF(pModule);
	Py_Finalize();
}