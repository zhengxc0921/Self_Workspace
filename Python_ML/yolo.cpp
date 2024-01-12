#pragma once
#include "yolo.h"


void  ParaseXML_Direct()
{
	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_ParseXML_Direct = PyObject_GetAttrString(pModule, "Start_ParseXML_Direct");//这里是要调用的函数名
	PyObject* pyParams = PyTuple_New(2); //定义两个变量
	string detection_root = "G:/DefectDataCenter/ParseData/Detection";
	string project = "COT_Raw";
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", detection_root.c_str()));// 变量格式转换成python格式
	PyTuple_SetItem(pyParams, 1, Py_BuildValue("s", project.c_str()));// 变量格式转换成python格式
	PyObject_CallObject(Start_ParseXML_Direct, pyParams);//调用函数

	Py_DECREF(pModule);
}

void  ParseXML_Resize()
{
	//Py_Initialize(); //初始化python解释器
	//if (!Py_IsInitialized()) {
	//	std::system("pause");
	//	//return -99;
	//} //查看python解释器是否成功初始化
	//PyRun_SimpleString("import sys");
	//PyRun_SimpleString("sys.path.append('I:/MIL_AI/Python/ML_detector')");
	PyObject* pModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_ParseXML_Direct = PyObject_GetAttrString(pModule, "Start_ParseXML_Resize");//这里是要调用的函数名
	PyObject* pyParams = PyTuple_New(6); //定义两个变量
	string detection_root = "G:/DefectDataCenter/ParseData/Detection";
	string project = "COT_Raw";
	float ratio = 2.5;
	int ch = 424;
	int cw = 2688;
	int bd = 300;
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", detection_root.c_str()));// 变量格式转换成python格式
	PyTuple_SetItem(pyParams, 1, Py_BuildValue("s", project.c_str()));// 变量格式转换成python格式
	PyTuple_SetItem(pyParams, 2, Py_BuildValue("f", ratio));// 变量格式转换成python格式
	PyTuple_SetItem(pyParams, 3, Py_BuildValue("n", ch));// 变量格式转换成python格式
	PyTuple_SetItem(pyParams, 4, Py_BuildValue("n", cw));// 变量格式转换成python格式
	PyTuple_SetItem(pyParams, 5, Py_BuildValue("n", bd));// 变量格式转换成python格式
	PyObject* Start_predict = PyObject_GetAttrString(pModule, "Start_ParseXML_Resize");//这里是要调用的函数名
	PyObject_CallObject(Start_predict, pyParams);//调用函数
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
	PyObject* Start_predict = PyObject_GetAttrString(pCheckModule, "Check_TrainDataSet");//这里是要调用的函数名
	PyObject* pyParams = PyTuple_New(2); //定义两个变量
	string project = "DSW_random";
	int vis_num = 10;
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", project.c_str()));// 变量格式转换成python格式
	PyTuple_SetItem(pyParams, 1, Py_BuildValue("n", vis_num));// 变量格式转换成python格式
	PyObject_CallObject(Start_predict, pyParams);//调用函数
	Py_DECREF(pCheckModule);

	//	//////调用pt1函数

//	//PyObject* pyParams = PyTuple_New(4); //定义两个变量
//	//string Csrc_dir = "G:/DefectDataCenter/Test/Src/90";
//	//string dst_dir = "G:/DefectDataCenter/Test/ImgCluster";
//	//float	Eeps = 1.2;
//	////const char* pn = "SPA90";
//	//string pn = "SPA90";
//	//PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", Csrc_dir.c_str()));// 变量格式转换成python格式
//	//PyTuple_SetItem(pyParams, 1, Py_BuildValue("s", dst_dir.c_str()));// 变量格式转换成python格式
//	//PyTuple_SetItem(pyParams, 2, Py_BuildValue("f", Eeps));// 变量格式转换成python格式
//	//PyTuple_SetItem(pyParams, 3, Py_BuildValue("s", pn.c_str()));// 变量格式转换成python格式
//	//PyObject_CallObject(pFunc, pyParams);//调用函数
//	//PyObject* pFunc = PyObject_GetAttrString(pModule, "pt2");//这里是要调用的函数名
//	//PyEval_CallObject(pFunc, NULL);//调用函数

	//Py_Finalize();
}

void  Yolo_Train()
{

	PyObject* pTrainModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pTrainModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_train = PyObject_GetAttrString(pTrainModule, "Start_train");//这里是要调用的函数名
	PyObject* pyParams = PyTuple_New(1); //定义两个变量
	string project = "COT_Raw";
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", project.c_str()));// 变量格式转换成python格式

	PyObject* result = PyObject_CallObject(Start_train, pyParams);//调用函数
	if (result && PyIter_Check(result)) {
		PyObject* item;
		while ((item = PyIter_Next(result))) {
			if (PyLong_Check(item)) {
				int value = PyLong_AsLong(item);
				cout << "value: " << value << endl;
				// 对中间结果进行处理
			}
			Py_DECREF(item);
		}

		Py_DECREF(result);
	}

	Py_DECREF(pTrainModule);

}

void  Yolo_Predict()
{
	//result 为空的时候，解析不出result
	PyObject* pPredictModule = PyImport_Import(PyUnicode_FromString("T_main")); //train  A  train.cpython-39
	if (!pPredictModule) {
		cout << "Can't find  Img_Cluster" << endl;
		std::system("pause");
	}
	PyObject* Start_predict = PyObject_GetAttrString(pPredictModule, "Start_predictFolder");//这里是要调用的函数名

	PyObject* pyParams = PyTuple_New(1); //定义两个变量
	string project = "G:/DefectDataCenter/ParseData/Detection/COT_Raw/raw_data/TImg";
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", project.c_str()));// 变量格式转换成python格式

	PyObject* result = PyObject_CallObject(Start_predict, pyParams);//调用函数
	if (result && PyIter_Check(result)) {
		PyObject* item;
		while ((item = PyIter_Next(result))) {
			if (PyLong_Check(item)) {
				int value = PyLong_AsLong(item);
				cout << "value: " << value << endl;
				// 对中间结果进行处理
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
 	PyObject* Start_Pth2ONNX = PyObject_GetAttrString(pPth2ONNXModule, "TOO_ONNX");//这里是要调用的函数名

	PyObject* pyParams = PyTuple_New(1); //定义两个变量
	string project = "COT_Raw";
	PyTuple_SetItem(pyParams, 0, Py_BuildValue("s", project.c_str()));// 变量格式转换成python格式

	PyObject_CallObject(Start_Pth2ONNX, pyParams);//调用函数
	Py_DECREF(pPth2ONNXModule);
	Py_Finalize();
}