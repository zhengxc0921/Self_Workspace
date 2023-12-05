#pragma once
#include "yolo.h"
void  Yolo_Train()
{
	Py_Initialize(); //��ʼ��python������
	if (!Py_IsInitialized()) {
		std::system("pause");
		//return -99;
	} //�鿴python�������Ƿ�ɹ���ʼ��
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");
	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_train = PyObject_GetAttrString(pModule, "Start_train");//������Ҫ���õĺ�����
	PyObject_CallObject(Start_train, NULL);//���ú���
	Py_DECREF(pModule);
	Py_Finalize();
}

void  Yolo_Predict()
{
	Py_Initialize(); //��ʼ��python������
	if (!Py_IsInitialized()) {
		std::system("pause");
		//return -99;
	} //�鿴python�������Ƿ�ɹ���ʼ��
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");

	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_predict = PyObject_GetAttrString(pModule, "Start_predict");//������Ҫ���õĺ�����
	PyObject_CallObject(Start_predict, NULL);//���ú���
	Py_DECREF(pModule);
	Py_Finalize();
}

void  ParaseXML_Direct()
{
	Py_Initialize(); //��ʼ��python������
	if (!Py_IsInitialized()) {
		std::system("pause");
		//return -99;
	} //�鿴python�������Ƿ�ɹ���ʼ��
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");

	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_predict = PyObject_GetAttrString(pModule, "Start_ParseXML_Direct");//������Ҫ���õĺ�����
	PyObject_CallObject(Start_predict, NULL);//���ú���
	Py_DECREF(pModule);
	Py_Finalize();
}

void  ParseXML_Resize()
{
	Py_Initialize(); //��ʼ��python������
	if (!Py_IsInitialized()) {
		std::system("pause");
		//return -99;
	} //�鿴python�������Ƿ�ɹ���ʼ��
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");

	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_predict = PyObject_GetAttrString(pModule, "Start_ParseXML_Resize");//������Ҫ���õĺ�����
	PyObject_CallObject(Start_predict, NULL);//���ú���
	Py_DECREF(pModule);
	Py_Finalize();
}

void  Check_DataSet()
{
	Py_Initialize(); //��ʼ��python������
	if (!Py_IsInitialized()) {
		std::system("pause");
		//return -99;
	} //�鿴python�������Ƿ�ɹ���ʼ��
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");

	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_predict = PyObject_GetAttrString(pModule, "Check_TrainDataSet");//������Ҫ���õĺ�����
	PyObject_CallObject(Start_predict, NULL);//���ú���
	Py_DECREF(pModule);
	Py_Finalize();
}