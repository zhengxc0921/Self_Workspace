#pragma once


#include <mil.h>
#include <boost/smart_ptr.hpp>

// Dashboard for CNN
class CDashboard;
typedef boost::shared_ptr<CDashboard> CDashboardPtr;
class CDashboard
{
public:
    CDashboard(
        MIL_ID      MilSystem,
        MIL_INT     Mode,
        MIL_INT     MaxEpoch,
        MIL_INT     MinibatchSize,
        MIL_DOUBLE  LearningRate,
        MIL_INT     TrainImageSizeX,
        MIL_INT     TrainImageSizeY,
        MIL_INT     TrainDatasetSize,
        MIL_INT     DevDatasetSize,
        MIL_INT     TrainEngineUsed,
        MIL_STRING& TrainEngineDescription);

    ~CDashboard();

    void AddEpochData(
        MIL_DOUBLE TrainErrorRate,
        MIL_DOUBLE DevErrorRate,
        MIL_INT    CurEpoch,
        bool       TheEpochIsTheBestUpToNow,
        MIL_DOUBLE EpochBenchMean);
    void AddMiniBatchData(
        MIL_DOUBLE LossError,
        MIL_INT    MinibatchIdx,
        MIL_INT    EpochIdx,
        MIL_INT    NbBatchPerEpoch);
    void AddDatasetsPreparedData(MIL_INT TrainDatasetSize, MIL_INT DevDatasetSize);

    MIL_ID GetDashboardBufId() const { return m_DashboardBufId; }

protected:
    void UpdateEpochInfo(
        MIL_DOUBLE TrainErrorRate,
        MIL_DOUBLE DevErrorRate,
        MIL_INT    CurEpoch,
        bool       TheEpochIsTheBestUpToNow);

    void UpdateLoss(MIL_DOUBLE LossError);

    void UpdateEpochGraph(
        MIL_DOUBLE TrainErrorRate,
        MIL_DOUBLE DevErrorRate,
        MIL_INT    CurEpoch);
    void UpdateLossGraph(
        MIL_DOUBLE LossError,
        MIL_INT    MiniBatchIdx,
        MIL_INT    EpochIdx,
        MIL_INT    NbBatchPerEpoch);

    void UpdateProgression(
        MIL_INT MinibatchIdx,
        MIL_INT EpochIdx,
        MIL_INT NbBatchPerEpoch);

    void UpdateDatasetsSize(MIL_INT TrainDatasetSize, MIL_INT DevDatasetSize);

    void DrawSectionSeparators();

    void DrawBufferFrame(MIL_ID BufId, MIL_INT FrameThickness);

    void InitializeEpochGraph();

    void InitializeLossGraph();

    void WriteGeneralTrainInfo(
        MIL_INT     MinibatchSize,
        MIL_INT     TrainImageSizeX,
        MIL_INT     TrainImageSizeY,
        MIL_INT     TrainDatasetSize,
        MIL_INT     DevDatasetSize,
        MIL_DOUBLE  LearningRate,
        MIL_INT     TrainEngineUsed,
        MIL_STRING& TrainEngineDescription);

    MIL_UNIQUE_BUF_ID m_DashboardBufId;
    MIL_UNIQUE_GRA_ID m_TheGraContext;

    MIL_UNIQUE_BUF_ID m_EpochInfoBufId;
    MIL_UNIQUE_BUF_ID m_EpochGraphBufId;
    MIL_UNIQUE_BUF_ID m_LossInfoBufId;
    MIL_UNIQUE_BUF_ID m_LossGraphBufId;
    MIL_UNIQUE_BUF_ID m_ProgressionInfoBufId;

    MIL_INT m_MaxEpoch;
    MIL_INT m_DashboardWidth;
    MIL_INT m_LastTrainPosX;
    MIL_INT m_LastTrainPosY;
    MIL_INT m_LastDevPosX;
    MIL_INT m_LastDevPosY;
    MIL_INT m_LastTrainMinibatchPosX;
    MIL_INT m_LastTrainMinibatchPosY;

    MIL_INT m_YPositionForLossText;

    MIL_DOUBLE m_EpochBenchMean;

    // Constants useful for the graph.
    MIL_INT GRAPH_SIZE_X;
    MIL_INT GRAPH_SIZE_Y;
    MIL_INT GRAPH_TOP_MARGIN;
    MIL_INT MARGIN;
    MIL_INT EPOCH_AND_MINIBATCH_REGION_HEIGHT;
    MIL_INT PROGRESSION_INFO_REGION_HEIGHT;

    MIL_INT LOSS_EXPONENT_MAX;
    MIL_INT LOSS_EXPONENT_MIN;

    MIL_DOUBLE COLOR_GENERAL_INFO;
    MIL_DOUBLE COLOR_DEV_SET_INFO;
    MIL_DOUBLE COLOR_TRAIN_SET_INFO;
    MIL_DOUBLE COLOR_PROGRESS_BAR;
};

struct HookDataStruct
{
    MIL_ID MilSystem;               //MIL system id
    MIL_ID MilDisplay;              //MIL display id
    MIL_INT DumpTmpRst;             //0: don't dump temp result, 1: dump temp result
    MIL_STRING ClassifierDumpFile;  //dump file name
    MIL_INT TrainModel = 0;            //TrainModel:0 CNN ; 1:DET
    char ControlType = 'N';
    CDashboardPtr DashboardPtr;     //dashboard
};

MIL_INT MFTYPE HookFuncDatasetsPrepared(MIL_INT /*HookType*/, MIL_ID  EventId, void* UserData);

MIL_INT MFTYPE HookFuncEpoch(MIL_INT /*HookType*/, MIL_ID  EventId, void* UserData);

MIL_INT MFTYPE HookFuncMiniBatch(MIL_INT HookType, MIL_ID  EventId, void* UserData);

// Dashboard for Detection
class DetDashboard;
typedef boost::shared_ptr<DetDashboard> DetDashboardPtr;
class DetDashboard
{
public:
    DetDashboard(
        MIL_ID MilSystem,
        MIL_ID TrainCtx,
        MIL_INT TrainImageSizeX,
        MIL_INT TrainImageSizeY,
        MIL_INT TrainEngineUsed,
        const MIL_STRING& TrainEngineDescription);

    ~DetDashboard();

    void AddEpochData(MIL_DOUBLE Loss, MIL_INT CurEpoch, MIL_DOUBLE EpochBenchMean);

    void AddMiniBatchData(
        MIL_DOUBLE Loss,
        MIL_INT MinibatchIdx,
        MIL_INT EpochIdx,
        MIL_INT NbBatchPerEpoch);

    MIL_ID GetDashboardBufId() const { return m_DashboardBufId; }

private:
    void UpdateTrainLoss(MIL_DOUBLE Loss);

    void UpdateDevLoss(MIL_DOUBLE Loss);

    void UpdateTrainLossGraph(
        MIL_DOUBLE Loss,
        MIL_INT MiniBatchIdx,
        MIL_INT EpochIdx,
        MIL_INT NbBatchPerEpoch);

    void UpdateDevLossGraph(MIL_DOUBLE Loss, MIL_INT EpochIdx);

    void UpdateProgression(MIL_INT MinibatchIdx, MIL_INT EpochIdx, MIL_INT NbBatchPerEpoch);

    void DrawSectionSeparators();

    void DrawBufferFrame(MIL_ID BufId, MIL_INT FrameThickness);

    void InitializeLossGraph();

    void WriteGeneralTrainInfo(
        MIL_INT MinibatchSize,
        MIL_INT TrainImageSizeX,
        MIL_INT TrainImageSizeY,
        MIL_DOUBLE LearningRate,
        MIL_INT TrainEngineUsed,
        const MIL_STRING& TrainEngineDescription);

    MIL_UNIQUE_BUF_ID m_DashboardBufId{ M_NULL };
    MIL_UNIQUE_GRA_ID m_TheGraContext{ M_NULL };

    MIL_UNIQUE_BUF_ID m_LossInfoBufId{ M_NULL };
    MIL_UNIQUE_BUF_ID m_LossGraphBufId{ M_NULL };
    MIL_UNIQUE_BUF_ID m_ProgressionInfoBufId{ M_NULL };

    MIL_INT m_MaxEpoch{ 0 };
    MIL_INT m_DashboardWidth{ 0 };
    MIL_INT m_LastTrainMinibatchPosX{ 0 };
    MIL_INT m_LastTrainMinibatchPosY{ 0 };
    MIL_INT m_LastDevEpochLossPosX{ 0 };
    MIL_INT m_LastDevEpochLossPosY{ 0 };

    MIL_INT m_YPositionForTrainLossText{ 0 };
    MIL_INT m_YPositionForDevLossText{ 0 };

    MIL_DOUBLE m_EpochBenchMean{ -1.0 };

    // Constants useful for the graph.
    const MIL_INT GRAPH_SIZE_X{ 600 };
    const MIL_INT GRAPH_SIZE_Y{ 400 };
    const MIL_INT GRAPH_TOP_MARGIN{ 30 };
    const MIL_INT MARGIN{ 50 };
    const MIL_INT EPOCH_AND_MINIBATCH_REGION_HEIGHT{ 190 };
    const MIL_INT PROGRESSION_INFO_REGION_HEIGHT{ 100 };

    const MIL_INT LOSS_EXPONENT_MAX{ 0 };
    const MIL_INT LOSS_EXPONENT_MIN{ -5 };

    const MIL_DOUBLE COLOR_GENERAL_INFO{ M_RGB888(0, 176, 255) };
    const MIL_DOUBLE COLOR_DEV_SET_INFO{ M_COLOR_MAGENTA };
    const MIL_DOUBLE COLOR_TRAIN_SET_INFO{ M_COLOR_GREEN };
    const MIL_DOUBLE COLOR_PROGRESS_BAR{ M_COLOR_DARK_GREEN };
};

struct DetHookDataStruct
{
    MIL_ID MilSystem;               //MIL system id
    MIL_ID MilDisplay;              //MIL display id
    MIL_INT DumpTmpRst;             //0: don't dump temp result, 1: dump temp result
    MIL_STRING ClassifierDumpFile;  //dump file name
    MIL_INT TrainModel = 0;            //TrainModel:0 CNN ; 1:DET
    char ControlType = 'N';
    DetDashboardPtr DetDashboardPtr;     //dashboard
   int SaveModelPEpoch = 8;
};

MIL_INT MFTYPE DetHookFuncDatasetsPrepared(MIL_INT /*HookType*/, MIL_ID  EventId, void* UserData);

MIL_INT MFTYPE DetHookFuncEpoch(MIL_INT /*HookType*/, MIL_ID  EventId, void* UserData);

MIL_INT MFTYPE DetHookFuncMiniBatch(MIL_INT HookType, MIL_ID  EventId, void* UserData);

MIL_INT MFTYPE DetHookNumPreparedEntriesFunc(MIL_INT /*HookType*/, MIL_ID EventId, void* pUserData);





