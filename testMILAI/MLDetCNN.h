#pragma once
#include "MLBase.h"
#include "Dashboard.h"
#include "MachineLearningExp.h"
using namespace std;

typedef struct DET_DATASET_PARAS_STRUCT {

    //Src Data info
    string          ClassesPath;
    string IconDir ;                //ClassesIcon :include Classes.bmp
    string TrainDataInfoPath;       //ImgBoxes_train.txt :include img_path box gt_label...
    string ValDataInfoPath;         //ImgBoxes_val.txt :include img_path box gt_label...
    
    //MIL WorkSpaceInfo                                 
    string WorkingDataDir;    //Dataset.mclassd saved Dir
    string PreparedDataDir;     //PreparedData  saved Dir

    //Related to Model Training
    int ImageSizeX;			//进入模型训练的图片的尺寸宽
    int ImageSizeY;			//进入模型训练的图片的尺寸高
    int AugFreq ;	            //进入模型训练的图片的扩充倍数
    MIL_DOUBLE TestDataRatio;       //

}DET_DATASET_PARAS_STRUCT;

typedef struct DET_TRAIN_STRUCT {
    MIL_INT    TrainMode;               //0:complete train, 1:fine tuning, 2:transfer learning
    MIL_INT    TrainEngineUsed;         //0:CPU, 1:GPU
    MIL_INT    MaxNumberOfEpoch;        //max epoch number
    MIL_INT    MiniBatchSize;           //mini batch size
    MIL_INT    SchedulerType;           //0:Cyclical Decay, 1:Decay
    MIL_DOUBLE LearningRate;            //learning rate
    MIL_DOUBLE LearningRateDecay;       //learning rate decay
    MIL_DOUBLE SplitPercent;            //split percent for train dataset and development dataset
    MIL_DOUBLE ClassWeight;             //class weight strength when training with an inverse class frequency weight mode, default 50.0
    
    MIL_STRING  WorkSpaceDir;            //
    MIL_STRING  DataSetName;            //the name of training data

    MIL_STRING TrainDstFolder;          //train destination folder
}DET_TRAIN_STRUCT;

typedef struct DET_RESULT_STRUCT {

    MIL_INT InstanceNum;
    string ImgPath;                         //src Img path for predict
    vector<MIL_STRING> ClassName;
    //vector<MIL_INT> ClassIndex;             //predict class
    vector<int> ClassIndex;             //predict class
    vector<MIL_DOUBLE> Score;               //predict score
    vector<Box>Boxes;

}DET_RESULT_STRUCT;

class CMLDetCNN;
typedef boost::shared_ptr<CMLDetCNN>CMLDetCNNPtr;
class MACHINELEARNING_DECLSPEC CMLDetCNN {

public:
    CMLDetCNN(MIL_ID MilSystem, MIL_ID MilDisplay);
    ~CMLDetCNN();
    //virtual MIL_INT CnnTrainEngineDLLInstalled(MIL_ID MilSystem);
    void GenDataSet(string DetDataSetConfigPath,string proj_n);
    void GenDataSet(DET_DATASET_PARAS_STRUCT DetDataSetPara);
    void CheckDataSet(string DetDataSetConfigPath, string proj_n);
    void ConstructDataContext(
        DataContextParasStruct DataCtxParas,
        MIL_UNIQUE_CLASS_ID& DataContext);

    void PrepareDataset(
        MIL_UNIQUE_CLASS_ID& DatasetContext,
        MIL_UNIQUE_CLASS_ID& PrepareDataset,
        MIL_UNIQUE_CLASS_ID& PreparedDataset,
        MIL_STRING WorkingDataPath,
        MIL_DOUBLE TestDatasetPercentage);

    void ConstructTrainCtx(
        DET_TRAIN_STRUCT DetParas,
        MIL_UNIQUE_CLASS_ID& TrainCtx);

    void TrainClassifier(
        MIL_UNIQUE_CLASS_ID& Dataset,
        MIL_UNIQUE_CLASS_ID& TrainCtx,
        MIL_UNIQUE_CLASS_ID& TrainedDetCtx,
        MIL_STRING& DetDumpFile);

    int TrainModel(DET_TRAIN_STRUCT DtParas);
    //针对离线测试：预测一个文件夹中的bmp图片
    int PredictFolderImgs(string SrcImgDir,
        MIL_STRING TdDetCtxPath,
        vector<DET_RESULT_STRUCT>&vecDetResults,
        bool SaveRst2file);    
    //针对在线测试：预测一个文件夹中的bmp图片 (暂定)
    void Predict(MIL_ID Image, MIL_UNIQUE_CLASS_ID& TdDetCtxPath,DET_RESULT_STRUCT& Result);
    //void PredictBegin(MIL_UNIQUE_CLASS_ID& TrainedDetCtx, MIL_ID Image);
    void ValModel_AP_50(string ValDataInfoPath, MIL_STRING TdDetCtxPath, string strPRResultPath, string strODNetResultPath);
    
    void Val_Txt_AP_50(string ValDataInfoPath, string PreDataInfoPath, string strPRResultPath);
    
    void PrintControls();
    void CDatasetViewer(MIL_ID Dataset);

    void saveResult2File(string strFilePath, vector<MIL_STRING>FilesInFolder, vector<DET_RESULT_STRUCT> vecDetResults);
private:
    void CreateFolder(const MIL_STRING& FolderPath);
    bool isfileNotExist(string fileNmae);
    bool isfileNotExist(MIL_STRING fileNmae);
    //void string2LPCTSTR(string inString, LPTSTR outString);
    LPTSTR string2LPCTSTR(string inString);
    void readDetDataSetConfig(string DetDataSetConfigPath, string proj_n);
    void addInfo2Dataset(MIL_UNIQUE_CLASS_ID& Dataset);
    int predictPrepare(MIL_STRING TdDetCtxPath);
    int predictPrepare(MIL_UNIQUE_CLASS_ID& TrainedDetCtx);
    void predict(MIL_ID Image, DET_RESULT_STRUCT& Result);

    void saveAP2File(string strFilePath, vector<float>Precisions, vector<float> Recalls);
    void calcAP4Vector(vector<vector<Box>>vecGTBoxes,
        vector<vector<int>>vecGTlabels,
        vector<vector<Box>>vecPdBoxes,
        vector<vector<int>>vecPdlabels,
        vector<float>& Precisions,
        vector<float>& Recalls);
    void findLabelIndexs(int label,vector<int>vecLabels,vector<int>&ids);
    bool matchTwoBoxes(Box bx1,Box bx2);


public:

    MIL_ID m_MilSystem;
    MIL_ID m_MilDisplay;
    MIL_ID m_Dataset;
    
    MIL_UNIQUE_CLASS_ID m_TrainedDetCtx;
    DET_DATASET_PARAS_STRUCT m_DetDataSetPara;
    CAIParsePtr m_AIParse;
    bool m_ModelNotPrePared =TRUE;
    MIL_INT m_ImageSizeX;
    MIL_INT m_ImageSizeY;

    MIL_INT m_InputSizeX;
    MIL_INT m_InputSizeY;
    MIL_INT m_InputSizeBand;
    MIL_INT m_ClassesNum;

    const MIL_INT Y_MARGIN{ 15 };
    const MIL_INT TEXT_HEIGHT{ 20 };
    const MIL_INT TEXT_MARGIN{ 20 };
    
    //控制中间测试时间的输出次数
    int m_ON = 0;
    int m_IOU_threshold = 0.5;
};