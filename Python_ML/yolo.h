#pragma once
#ifdef DEFECTINSPECTOR_DECLSPEC
#define DEFECTINSPECTOR_DECLSPEC __declspec(dllexport)
#else
#define DEFECTINSPECTOR_DECLSPEC __declspec(dllimport)
#endif

#include<Python.h>
#include<string>
#include <iostream>
using namespace std;

DEFECTINSPECTOR_DECLSPEC void  Yolo_Train();
DEFECTINSPECTOR_DECLSPEC void  Yolo_Predict();
DEFECTINSPECTOR_DECLSPEC void  ParaseXML_Direct();
DEFECTINSPECTOR_DECLSPEC void  ParseXML_Resize();
DEFECTINSPECTOR_DECLSPEC void  Check_DataSet();

