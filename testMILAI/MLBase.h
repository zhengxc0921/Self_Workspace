#pragma once
//#ifdef MLBASE_EXPORTS
//#define MLBASE_DECLSPEC __declspec(dllexport)
//#else
//#define MLBASE_DECLSPEC __declspec(dllimport)
//#endif

#include <mil.h>
#include <string.h>
#include <vector>
#include <boost/smart_ptr.hpp>
#include "AIParse.h"

struct AugmentationParasStruct
{
    MIL_INT    AugmentationNumPerImage; //augmentation number per raw image
    MIL_INT    SeedValue;               //random seed value
    MIL_INT    GammaEnable;             //gamma correction, 0 disable, 1 enable
    MIL_INT    CropEnable;              //crop operation, 0 disable, 1 enable
    MIL_INT    TranslationXMax;         //max translate x, default:10
    MIL_INT    TranslationYMax;         //max translate y, default:10
    MIL_DOUBLE ScaleFactorMin;          //min scale factor, default:0.5
    MIL_DOUBLE ScaleFactorMax;          //max scale factor, default:2.0
    MIL_DOUBLE RotateAngleDelta;        //rotate angle, default:360.0
    MIL_DOUBLE SmoothnessMin;           //min smoothness, default:0.0
    MIL_DOUBLE SmoothnessMax;           //max smoothness, default:100.0



    MIL_DOUBLE DirIntyMax;
    MIL_DOUBLE DirIntyMin;
    MIL_DOUBLE IntyDeltaAdd;     //add a value;value = random{[Inty-IntyDeltaAdd,Inty+IntyDeltaAdd]}

    MIL_DOUBLE GaussNoiseDelta;         //gauss noise mean, default:25.0
    MIL_DOUBLE GaussNoiseStdev;         //gauss noise stdev, default:25.0

    MIL_DOUBLE GammaValue;
    MIL_DOUBLE GammaDelta;
    MIL_DOUBLE InAddValue;
    MIL_DOUBLE InAddDelta;
    MIL_DOUBLE InMulValue;
    MIL_DOUBLE InMulDelta;

};

struct DataContextParasStruct
{
    int ImageSizeX;                     //image size x
    int ImageSizeY;                     //image size y
    int DstFolderMode=1;                  //0:write, 1:overwrite
    int ResizeModel=1;
    MIL_STRING PreparedDataFolder;      //prepared data folder
    AugmentationParasStruct AugParas;   //augmentation paraments
};

struct ClassifierParasStruct
{
    MIL_INT    TrainMode;               //0:complete train, 1:fine tuning, 2:transfer learning
    MIL_INT    TrainEngineUsed;         //0:CPU, 1:GPU
    MIL_INT    MaxNumberOfEpoch;        //max epoch number
    MIL_INT    MiniBatchSize;           //mini batch size
    MIL_INT    SchedulerType;           //0:Cyclical Decay, 1:Decay
    MIL_DOUBLE LearningRate;            //learning rate
    MIL_DOUBLE LearningRateDecay;       //learning rate decay
    MIL_DOUBLE SplitPercent;            //split percent for train dataset and development dataset
    MIL_DOUBLE ClassWeight;             //class weight strength when training with an inverse class frequency weight mode, default 50.0
    MIL_STRING TrainDstFolder;          //train destination folder
};

struct ClassificationResultStruct
{
    MIL_INT GroundTruth;                //ground truth
    MIL_STRING RootPath;              //Root Path of the img
    MIL_STRING PredictClassName;
    MIL_INT PredictClass;               //predict class
    MIL_DOUBLE PredictScore;            //predict score
    std::vector<MIL_DOUBLE> AllScores;        //predict all class scores
    //std::vector<double>AS;
};

struct DetParas{
    MIL_INT    TrainMode;               //0:complete train, 1:fine tuning, 2:transfer learning
    MIL_INT    TrainEngineUsed;         //0:CPU, 1:GPU
    MIL_INT    MaxNumberOfEpoch;        //max epoch number
    MIL_INT    MiniBatchSize;           //mini batch size
    MIL_INT    SchedulerType;           //0:Cyclical Decay, 1:Decay
    MIL_DOUBLE LearningRate;            //learning rate
    MIL_DOUBLE LearningRateDecay;       //learning rate decay
    MIL_DOUBLE SplitPercent;            //split percent for train dataset and development dataset
    MIL_DOUBLE ClassWeight;             //class weight strength when training with an inverse class frequency weight mode, default 50.0
    MIL_STRING TrainDstFolder;          //train destination folder
};

struct DetResult {

    MIL_INT InstanceNum;
    vector<MIL_STRING> ClassName;
    vector<MIL_INT> ClassIndex;               //predict class
    vector<MIL_DOUBLE> Score;            //predict score
    vector<Box>Boxes;

    //MIL_INT GroundTruth;                //ground truth
    //MIL_STRING RootPath;              //Root Path of the img

    //std::vector<MIL_DOUBLE> AllScores;        //predict all class scores

};


class CMLBase;
typedef boost::shared_ptr<CMLBase>CMLBasePtr;
class CMLBase {

public:
	CMLBase(MIL_ID MilSystem);
    CMLBase() {};
	virtual ~CMLBase();

	MIL_INT IsTrainingSupportedOnPlatform(MIL_ID MilSystem);

    virtual MIL_INT CnnTrainEngineDLLInstalled(MIL_ID MilSystem)=0;


    void CreateFolder(const MIL_STRING& FolderPath);

	//void DatasetSave(
	//	MIL_UNIQUE_CLASS_ID& Dataset,
	//	const MIL_STRING& DatasetFilePath);

	//void DatasetLoad(
	//	const MIL_STRING& DatasetFilePath,
	//	MIL_UNIQUE_CLASS_ID& Dataset);

	void ClassifierSave(
		MIL_UNIQUE_CLASS_ID& ClassifierCtx,
		const MIL_STRING& ClassifierFileName);

	void ClassifierLoad(
		const MIL_STRING& ClassifierFileName,
		MIL_UNIQUE_CLASS_ID& ClassifierCtx);

private:
	MIL_ID m_MilSystem;




};