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

    bool isTagSameClass(
        vector<MIL_STRING>BaseClsNames,
        vector<MIL_STRING> TagClsNames);

    void InitClassWeights();
    //单个文件-->Dataset
    void ConstructDataset(
        std::vector<MIL_STRING> ClassName,
        std::vector<MIL_STRING> ClassIcon,
        MIL_STRING AuthorName,
        MIL_STRING OriginalDataPath,
        const MIL_STRING& WorkingDataPath,
        MIL_UNIQUE_CLASS_ID& Dataset);

    void ConstructMergeDataset(
        MIL_STRING AuthorName,
        
        MIL_STRING BaseDataDir,
        MIL_STRING TagDataDir,
        vector<MIL_DOUBLE> vecSampleRatio,
        MIL_UNIQUE_CLASS_ID& BaseDataSet,
        MIL_UNIQUE_CLASS_ID& TagDataSet
 );

    void MergeTagData2BaseSet(MIL_STRING BaseDataDir,
        MIL_STRING BaseSetName,
        vector<MIL_STRING> BaseClsNames,
        MIL_STRING TagDataDir,
        vector<MIL_DOUBLE>vecSampleRatio,
        MIL_UNIQUE_CLASS_ID& MergeSet,
        MIL_UNIQUE_CLASS_ID& BaseSet);

    void Merge2Set(
        MIL_UNIQUE_CLASS_ID& BaseSet1,
        MIL_UNIQUE_CLASS_ID& BaseSet2,
        MIL_STRING BaseDataDir,
        MIL_STRING MergeSetName);

    void ConstructMergeRCDataset(
        MIL_STRING AuthorName,
        MIL_STRING strProject,
        MIL_STRING BaseDataDir,
        MIL_STRING TagDataDir,
        vector<MIL_DOUBLE> vecSampleRatio,
        MIL_UNIQUE_CLASS_ID& BaseDataSet,
        MIL_UNIQUE_CLASS_ID& TagDataSet
    );

    void InitializeMergeRCDataset(
        MIL_STRING AuthorName,
        MIL_STRING BaseDataDir,
        MIL_STRING TagDataDir,
        vector<MIL_STRING>TagClassNames,
        MIL_UNIQUE_CLASS_ID& BaseDataSet,
        MIL_UNIQUE_CLASS_ID& TagDataSet
    );

    void ConstructPartialDataset(
        vector<MIL_STRING> ClassName,
        vector<MIL_STRING> ClassIcon,
        MIL_STRING AuthorName,
        MIL_STRING OriginalDataPath,
        const MIL_STRING& WorkingDataPath,
        MIL_UNIQUE_CLASS_ID& Dataset,
        vector<MIL_DOUBLE> vecDSampleRatio);




    //多个文件Dataset 汇总
    void GeneralDataset(
        vector<MIL_STRING> ClassName,
        vector<MIL_STRING> ClassIcon,
        MIL_STRING AuthorName,
        MIL_STRING OriginalDataPath,
        MIL_STRING WorkingDataPath);


    void ConstructDataContext(
        DataContextParasStruct DataCtxParas,
        MIL_UNIQUE_CLASS_ID& DataContext);

    void PrepareDataset(
        MIL_UNIQUE_CLASS_ID& DatasetContext,
        MIL_UNIQUE_CLASS_ID& PrepareDataset,
        MIL_UNIQUE_CLASS_ID& PreparedDataset,
        MIL_STRING WorkingDataDir,
        MIL_STRING DatasetName);

    void ConstructTrainCtx(
        ClassifierParasStruct ClassifierParas,
        MIL_UNIQUE_CLASS_ID& TrainCtx);

    void TrainClassifier(
        MIL_UNIQUE_CLASS_ID& Dataset,
        //MIL_UNIQUE_CLASS_ID& DatasetContext,
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

    void AddClassToDataset(
        MIL_INT           ClassIndex,
        const MIL_STRING& DataPath,
        const MIL_STRING& ClassName,
        const MIL_STRING& AuthorName,
        MIL_UNIQUE_CLASS_ID& PartialDataset,
        MIL_UNIQUE_CLASS_ID& Dataset,
        MIL_DOUBLE dSampleRatio);



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