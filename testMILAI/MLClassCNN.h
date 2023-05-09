#pragma once
#include "MLBase.h"
//#include "AIParse.h"
#include "Dashboard.h"
using namespace std;

//struct HookDataCNNStruct
//{
//    MIL_ID MilSystem;               //MIL system id
//    MIL_ID MilDisplay;              //MIL display id
//    MIL_INT DumpTmpRst;             //0: don't dump temp result, 1: dump temp result
//    MIL_STRING ClassifierDumpFile;  //dump file name
//    MIL_INT ControlFlag = 0;
//    char ControlType = 'N';
//    CDashboardPtr DashboardPtr;     //dashboard
//};


class CMLClassCNN;
typedef boost::shared_ptr<CMLClassCNN>CMLClassCNNPtr;
class CMLClassCNN :public CMLBase {

public:
    CMLClassCNN(MIL_ID MilSystem, MIL_ID MilDisplay);
    ~CMLClassCNN() {};

    virtual MIL_INT CnnTrainEngineDLLInstalled(MIL_ID MilSystem);

    void InitClassWeights();
    //单个文件-->Dataset
    void ConstructDataset(
        std::vector<MIL_STRING> ClassName,
        std::vector<MIL_STRING> ClassIcon,
        MIL_STRING AuthorName,
        MIL_STRING OriginalDataPath,
        const MIL_STRING& WorkingDataPath,
        MIL_UNIQUE_CLASS_ID& Dataset);
    //多个文件Dataset 汇总
    void GeneralDataset(
        std::vector<vector<MIL_STRING>> ClassName,
        std::vector<vector<MIL_STRING>> ClassIcon,
        std::vector<MIL_STRING>& AuthorName,
        std::vector<MIL_STRING>& OriginalDataPath,
        const MIL_STRING& WorkingDataPath);


    void ConstructDataContext(
        DataContextParasStruct DataCtxParas,
        MIL_UNIQUE_CLASS_ID& DataContext);

    void PrepareDataset(
        MIL_UNIQUE_CLASS_ID& DatasetContext,
        MIL_UNIQUE_CLASS_ID& PrepareDataset,
        MIL_UNIQUE_CLASS_ID& PreparedDataset,
        MIL_STRING PreparedDatasetPath);

    void ConstructTrainCtx(
        ClassifierParasStruct ClassifierParas,
        MIL_UNIQUE_CLASS_ID& TrainCtx);

    void TrainClassifier(
        MIL_UNIQUE_CLASS_ID& Dataset,
        MIL_UNIQUE_CLASS_ID& DatasetContext,
        MIL_UNIQUE_CLASS_ID& TrainCtx,
        MIL_UNIQUE_CLASS_ID& PrevClassifierCtx,
        MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx,
        MIL_STRING& ClassifierDumpFile);

    void PredictBegin(MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx, MIL_ID Image, vector<MIL_DOUBLE>Class_Weights);

    void FolderImgsPredict(
        vector<MIL_STRING> FilesInFolder,
        vector<MIL_DOUBLE>Class_Weights,
        MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx,
        vector<ClassificationResultStruct>& Result);

    void Predict(
        MIL_ID Image,
        vector<MIL_DOUBLE>Class_Weights,
        MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx,
        ClassificationResultStruct& Result);
    //大图通过blob的clip进行预测
    void Predict(
        MIL_ID Image,
        vector<int>blob_px,
        vector<int>blob_py,
        std::vector<MIL_DOUBLE>Class_Weights,
        MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx,
        ClassificationResultStruct& Result);


public:
    void AddClassDescription(
        MIL_ID         Dataset,
        const MIL_STRING& AuthorName,
        std::vector<MIL_STRING>  ClassName,
        std::vector<MIL_STRING>  ClassIcon,
        MIL_INT                  NumberOfClasses);

    void AddClassToDataset(
        MIL_INT           ClassIndex,
        const MIL_STRING& DataPath,
        const MIL_STRING& ClassName,
        const MIL_STRING& AuthorName,
        MIL_ID            Dataset);


public:
    MIL_ID m_MilSystem;
    MIL_ID m_MilDisplay;
    CAIParsePtr m_AIParse ;

    MIL_INT m_ImageSizeX;
    MIL_INT m_ImageSizeY;

    MIL_INT m_InputSizeX;
    MIL_INT m_InputSizeY;
    MIL_INT m_ClassesNum;
    vector<MIL_DOUBLE> m_ClassWeights;
};