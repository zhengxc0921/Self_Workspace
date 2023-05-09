#pragma once
#include "MLBase.h"

#include "Dashboard.h"
using namespace std;




class CMLDetCNN;
typedef boost::shared_ptr<CMLDetCNN>CMLDetCNNPtr;
class CMLDetCNN :public CMLBase {

public:
    CMLDetCNN(MIL_ID MilSystem, MIL_ID MilDisplay);
    ~CMLDetCNN();

    virtual MIL_INT CnnTrainEngineDLLInstalled(MIL_ID MilSystem);

    void ConstructDataset(
        string IconInfo,
        //MIL_STRING IconDir,
        string ImgDataInfo,
        const MIL_STRING& WorkingDataPath);
    
    void ConstructDataset(string ClassesInfo,
        string IconDir,
        string ImgDataInfo,
        string WorkingDataPath,
        string DataSetName);


    void ConstructDataContext(
        DataContextParasStruct DataCtxParas,
        MIL_UNIQUE_CLASS_ID& DataContext);

    void PrepareDataset(
        MIL_UNIQUE_CLASS_ID& DatasetContext,
        MIL_UNIQUE_CLASS_ID& PrepareDataset,
        MIL_UNIQUE_CLASS_ID& PreparedDataset);

    void ConstructTrainCtx(
        DetParas DetParas,
        MIL_UNIQUE_CLASS_ID& TrainCtx);

    void TrainClassifier(
        MIL_UNIQUE_CLASS_ID& Dataset,
        MIL_UNIQUE_CLASS_ID& DatasetContext,
        MIL_UNIQUE_CLASS_ID& TrainCtx,
       
        MIL_UNIQUE_CLASS_ID& TrainedDetCtx,
        MIL_STRING& DetDumpFile);

    void PredictBegin(MIL_UNIQUE_CLASS_ID& TrainedDetCtx, MIL_ID Image);

    void Predict(
        MIL_ID Image,
        MIL_UNIQUE_CLASS_ID& TrainedDetCtx,
        DetResult& Result);

    void FolderImgsPredict(
        vector<MIL_STRING> FilesInFolder,
        MIL_UNIQUE_CLASS_ID& TrainedDetCtx,
        vector<DetResult>& Result);

    void PrintControls();

    void CDatasetViewer(MIL_ID Dataset);
public:

    MIL_ID m_MilSystem;
    MIL_ID m_MilDisplay;
    MIL_ID m_Dataset;
    
    CAIParsePtr m_AIParse;

    MIL_INT m_ImageSizeX;
    MIL_INT m_ImageSizeY;

    MIL_INT m_InputSizeX;
    MIL_INT m_InputSizeY;
    MIL_INT m_ClassesNum;

    const MIL_INT Y_MARGIN{ 15 };
    const MIL_INT TEXT_HEIGHT{ 20 };
    const MIL_INT TEXT_MARGIN{ 20 };
};