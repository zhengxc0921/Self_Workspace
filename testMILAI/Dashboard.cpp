//*************************************************************************************
//
// File name: ClassCNNCompleteTrain.cpp
//
// Synopsis:  This program uses the classification module to train
//            a context able to classify 3 different types of fabrics.
//
// Note:      GPU training can be enabled via a MIL update for 64-bit.
//            This can dramatically increase the training speed.
//
// Copyright © Matrox Electronic Systems Ltd., 1992-2021.
// All Rights Reserved

#include "Dashboard.h"

// Dashboard for CNN
CDashboard::CDashboard(
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
    MIL_STRING& TrainEngineDescription)
    : m_DashboardBufId(M_NULL)
    , m_TheGraContext(M_NULL)
    , m_EpochInfoBufId(M_NULL)
    , m_EpochGraphBufId(M_NULL)
    , m_LossInfoBufId(M_NULL)
    , m_LossGraphBufId(M_NULL)
    , m_ProgressionInfoBufId(M_NULL)
    , m_MaxEpoch(MaxEpoch)
    , m_DashboardWidth(0)
    , m_LastTrainPosX(0)
    , m_LastTrainPosY(0)
    , m_LastDevPosX(0)
    , m_LastDevPosY(0)
    , m_LastTrainMinibatchPosX(0)
    , m_LastTrainMinibatchPosY(0)
    , m_YPositionForLossText(0)
    , m_EpochBenchMean(-1.0)
    , GRAPH_SIZE_X(400)
    , GRAPH_SIZE_Y(400)
    , GRAPH_TOP_MARGIN(30)
    , MARGIN(50)
    , EPOCH_AND_MINIBATCH_REGION_HEIGHT(190)
    , PROGRESSION_INFO_REGION_HEIGHT(100)
    , LOSS_EXPONENT_MAX(0)
    , LOSS_EXPONENT_MIN(-5)
    , COLOR_GENERAL_INFO(M_RGB888(0, 176, 255))
    , COLOR_DEV_SET_INFO(M_COLOR_MAGENTA)
    , COLOR_TRAIN_SET_INFO(M_COLOR_GREEN)
    , COLOR_PROGRESS_BAR(M_COLOR_DARK_GREEN)
{
    // One graph width.
    const MIL_INT GraphBoxWidth = GRAPH_SIZE_X + 2 * MARGIN;
    const MIL_INT GraphBoxHeight = GRAPH_SIZE_Y + MARGIN + GRAPH_TOP_MARGIN;
    // There are 2 graphs side by side.
    m_DashboardWidth = 2 * GraphBoxWidth;

    const MIL_INT DashboardHeight = EPOCH_AND_MINIBATCH_REGION_HEIGHT + GraphBoxHeight + PROGRESSION_INFO_REGION_HEIGHT;

    // Allocate the full dashboard buffer.
    m_DashboardBufId = MbufAllocColor(MilSystem, 3, m_DashboardWidth, DashboardHeight,
        8 + M_UNSIGNED, M_IMAGE + M_PROC + M_DISP, M_UNIQUE_ID);
    MbufClear(m_DashboardBufId, M_COLOR_BLACK);

    m_TheGraContext = MgraAlloc(MilSystem, M_UNIQUE_ID);

    // Allocate child buffers for each different dashboard sections.
    const MIL_INT GraphYPosition = EPOCH_AND_MINIBATCH_REGION_HEIGHT;
    const MIL_INT ProgressionInfoYPosition = GraphYPosition + GraphBoxHeight;

    m_EpochInfoBufId = MbufChild2d(m_DashboardBufId, 0, 0, GraphBoxWidth, EPOCH_AND_MINIBATCH_REGION_HEIGHT, M_UNIQUE_ID);
    m_LossInfoBufId = MbufChild2d(m_DashboardBufId, GraphBoxWidth, 0, GraphBoxWidth, EPOCH_AND_MINIBATCH_REGION_HEIGHT, M_UNIQUE_ID);
    m_EpochGraphBufId = MbufChild2d(m_DashboardBufId, 0, GraphYPosition, GraphBoxWidth, GraphBoxHeight, M_UNIQUE_ID);
    m_LossGraphBufId = MbufChild2d(m_DashboardBufId, GraphBoxWidth, GraphYPosition, GraphBoxWidth, GraphBoxHeight, M_UNIQUE_ID);
    m_ProgressionInfoBufId = MbufChild2d(m_DashboardBufId, 0, ProgressionInfoYPosition, m_DashboardWidth, PROGRESSION_INFO_REGION_HEIGHT, M_UNIQUE_ID);

    // Initialize the different dashboard sections.
    DrawSectionSeparators();

    InitializeEpochGraph();
    InitializeLossGraph();

    WriteGeneralTrainInfo(
        MinibatchSize,
        TrainImageSizeX,
        TrainImageSizeY,
        TrainDatasetSize,
        DevDatasetSize,
        LearningRate,
        TrainEngineUsed,
        TrainEngineDescription);
}

//............................................................................
CDashboard::~CDashboard()
{
    m_TheGraContext = M_NULL;
    m_EpochInfoBufId = M_NULL;
    m_LossInfoBufId = M_NULL;
    m_EpochGraphBufId = M_NULL;
    m_LossGraphBufId = M_NULL;
    m_ProgressionInfoBufId = M_NULL;
    m_DashboardBufId = M_NULL;
}

//............................................................................
void CDashboard::DrawBufferFrame(MIL_ID BufId, MIL_INT FrameThickness)
{
    MIL_ID SizeX = MbufInquire(BufId, M_SIZE_X, M_NULL);
    MIL_ID SizeY = MbufInquire(BufId, M_SIZE_Y, M_NULL);

    MgraColor(m_TheGraContext, COLOR_GENERAL_INFO);
    MgraRectFill(m_TheGraContext, BufId, 0, 0, SizeX - 1, FrameThickness - 1);
    MgraRectFill(m_TheGraContext, BufId, SizeX - FrameThickness, 0, SizeX - 1, SizeY - 1);
    MgraRectFill(m_TheGraContext, BufId, 0, SizeY - FrameThickness, SizeX - 1, SizeY - 1);
    MgraRectFill(m_TheGraContext, BufId, 0, 0, FrameThickness - 1, SizeY - 1);
}

//............................................................................
void CDashboard::DrawSectionSeparators()
{
    // Draw a frame for the whole dashboard.
    DrawBufferFrame(m_DashboardBufId, 4);
    // Draw a frame for each section.
    DrawBufferFrame(m_EpochInfoBufId, 2);
    DrawBufferFrame(m_EpochGraphBufId, 2);
    DrawBufferFrame(m_LossInfoBufId, 2);
    DrawBufferFrame(m_LossGraphBufId, 2);
    DrawBufferFrame(m_ProgressionInfoBufId, 2);
}

//............................................................................
void CDashboard::InitializeEpochGraph()
{
    // Draw axis.
    MgraColor(m_TheGraContext, M_COLOR_WHITE);
    MgraRect(m_TheGraContext, m_EpochGraphBufId, MARGIN, GRAPH_TOP_MARGIN, MARGIN + GRAPH_SIZE_X, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y);

    MgraControl(m_TheGraContext, M_TEXT_ALIGN_HORIZONTAL, M_RIGHT);
    MgraText(m_TheGraContext, m_EpochGraphBufId, MARGIN - 5, GRAPH_TOP_MARGIN, MIL_TEXT("100"));
    MgraText(m_TheGraContext, m_EpochGraphBufId, MARGIN - 5, GRAPH_TOP_MARGIN + ((MIL_INT)(0.25 * GRAPH_SIZE_Y)), MIL_TEXT("75"));
    MgraText(m_TheGraContext, m_EpochGraphBufId, MARGIN - 5, GRAPH_TOP_MARGIN + ((MIL_INT)(0.50 * GRAPH_SIZE_Y)), MIL_TEXT("50"));
    MgraText(m_TheGraContext, m_EpochGraphBufId, MARGIN - 5, GRAPH_TOP_MARGIN + ((MIL_INT)(0.75 * GRAPH_SIZE_Y)), MIL_TEXT("25"));
    MgraText(m_TheGraContext, m_EpochGraphBufId, MARGIN - 5, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y, MIL_TEXT("0"));

    MgraLine(m_TheGraContext, m_EpochGraphBufId, MARGIN, GRAPH_TOP_MARGIN + ((MIL_INT)(0.25 * GRAPH_SIZE_Y)), MARGIN + 5, GRAPH_TOP_MARGIN + ((MIL_INT)(0.25 * GRAPH_SIZE_Y)));
    MgraLine(m_TheGraContext, m_EpochGraphBufId, MARGIN, GRAPH_TOP_MARGIN + ((MIL_INT)(0.50 * GRAPH_SIZE_Y)), MARGIN + 5, GRAPH_TOP_MARGIN + ((MIL_INT)(0.50 * GRAPH_SIZE_Y)));
    MgraLine(m_TheGraContext, m_EpochGraphBufId, MARGIN, GRAPH_TOP_MARGIN + ((MIL_INT)(0.75 * GRAPH_SIZE_Y)), MARGIN + 5, GRAPH_TOP_MARGIN + ((MIL_INT)(0.75 * GRAPH_SIZE_Y)));

    MgraControl(m_TheGraContext, M_TEXT_ALIGN_HORIZONTAL, M_LEFT);

    MIL_INT NbTick = std::min<MIL_INT>(m_MaxEpoch, 10);
    const MIL_INT EpochTickValue = m_MaxEpoch / NbTick;

    for (MIL_INT CurTick = 1; CurTick <= m_MaxEpoch; CurTick += EpochTickValue)
    {
        MIL_DOUBLE Percentage = (MIL_DOUBLE)CurTick / (MIL_DOUBLE)m_MaxEpoch;
        MIL_INT XOffset = (MIL_INT)(Percentage * GRAPH_SIZE_X);
        MgraText(m_TheGraContext, m_EpochGraphBufId, MARGIN + XOffset, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y + 5, M_TO_STRING(CurTick - 1));
        MgraLine(m_TheGraContext, m_EpochGraphBufId, MARGIN + XOffset, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y - 5, MARGIN + XOffset, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y);
    }
}

//............................................................................
void CDashboard::InitializeLossGraph()
{
    // Draw axis.
    MgraColor(m_TheGraContext, M_COLOR_WHITE);
    MgraRect(m_TheGraContext, m_LossGraphBufId, MARGIN, GRAPH_TOP_MARGIN, MARGIN + GRAPH_SIZE_X, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y);

    MgraControl(m_TheGraContext, M_TEXT_ALIGN_HORIZONTAL, M_RIGHT);

    const MIL_INT NbLossValueTick = LOSS_EXPONENT_MAX - LOSS_EXPONENT_MIN;
    const MIL_DOUBLE TickRatio = 1.0 / (MIL_DOUBLE)NbLossValueTick;

    MIL_DOUBLE TickNum = 0.0;
    for (MIL_INT i = LOSS_EXPONENT_MAX; i >= LOSS_EXPONENT_MIN; i--)
    {
        MIL_TEXT_CHAR CurTickText[128];
        MosSprintf(CurTickText, 128, MIL_TEXT("1e%d"), i);

        MIL_INT TickYPos = (MIL_INT)(TickNum * TickRatio * GRAPH_SIZE_Y);
        MgraText(m_TheGraContext, m_LossGraphBufId, MARGIN - 5, GRAPH_TOP_MARGIN + TickYPos, CurTickText);
        if ((i != LOSS_EXPONENT_MAX) && (i != LOSS_EXPONENT_MIN))
        {
            MgraLine(m_TheGraContext, m_LossGraphBufId, MARGIN, GRAPH_TOP_MARGIN + TickYPos, MARGIN + 5, GRAPH_TOP_MARGIN + TickYPos);
        }
        TickNum = TickNum + 1.0;
    }

    MgraControl(m_TheGraContext, M_TEXT_ALIGN_HORIZONTAL, M_LEFT);

    const MIL_INT NbEpochTick = std::min<MIL_INT>(m_MaxEpoch, 10);
    const MIL_INT EpochTickValue = m_MaxEpoch / NbEpochTick;

    for (MIL_INT CurTick = 1; CurTick <= m_MaxEpoch; CurTick += EpochTickValue)
    {
        MIL_DOUBLE Percentage = (MIL_DOUBLE)CurTick / (MIL_DOUBLE)m_MaxEpoch;
        MIL_INT XOffset = (MIL_INT)(Percentage * GRAPH_SIZE_X);
        MgraText(m_TheGraContext, m_LossGraphBufId, MARGIN + XOffset, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y + 5, M_TO_STRING(CurTick - 1));
        MgraLine(m_TheGraContext, m_LossGraphBufId, MARGIN + XOffset, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y - 5, MARGIN + XOffset, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y);
    }
}

//............................................................................
void CDashboard::WriteGeneralTrainInfo(
    MIL_INT     MinibatchSize,
    MIL_INT     TrainImageSizeX,
    MIL_INT     TrainImageSizeY,
    MIL_INT     TrainDatasetSize,
    MIL_INT     DevDatasetSize,
    MIL_DOUBLE  LearningRate,
    MIL_INT     TrainEngineUsed,
    MIL_STRING& TrainEngineDescription)
{
    MgraControl(m_TheGraContext, M_BACKGROUND_MODE, M_OPAQUE);
    MgraControl(m_TheGraContext, M_BACKCOLOR, M_COLOR_BLACK);

    MgraControl(m_TheGraContext, M_TEXT_ALIGN_HORIZONTAL, M_LEFT);

    const MIL_INT YMargin = 15;
    const MIL_INT TextHeight = 20;
    const MIL_INT TextMargin = MARGIN - 10;

    MIL_INT TextYPos = YMargin;

    MgraColor(m_TheGraContext, COLOR_GENERAL_INFO);

    MIL_TEXT_CHAR TheString[512];
    if (TrainEngineUsed == M_CPU)
        MosSprintf(TheString, 512, MIL_TEXT("Training is being performed on the CPU"));
    else
        MosSprintf(TheString, 512, MIL_TEXT("Training is being performed on the GPU"));
    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
    TextYPos += TextHeight;

    MosSprintf(TheString, 512, MIL_TEXT("Training engine: %s"), TrainEngineDescription.c_str());
    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
    TextYPos += TextHeight;

    MosSprintf(TheString, 512, MIL_TEXT("Train image size: %dx%d"), TrainImageSizeX, TrainImageSizeY);
    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
    TextYPos += TextHeight;

    MosSprintf(TheString, 512, MIL_TEXT("Train and Dev dataset size: %d and %d images"), TrainDatasetSize, DevDatasetSize);
    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
    TextYPos += TextHeight;

    MosSprintf(TheString, 512, MIL_TEXT("Max number of epochs: %d"), m_MaxEpoch);
    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
    TextYPos += TextHeight;

    MosSprintf(TheString, 512, MIL_TEXT("Minibatch size: %d"), MinibatchSize);
    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
    TextYPos += TextHeight;

    MosSprintf(TheString, 512, MIL_TEXT("Learning rate: %.2e"), LearningRate);
    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
    TextYPos += TextHeight;

    // The loss will be drawn under later on, so we retain is position.
    m_YPositionForLossText = TextYPos;
}

//............................................................................
void CDashboard::AddEpochData(
    MIL_DOUBLE TrainErrorRate,
    MIL_DOUBLE DevErrorRate,
    MIL_INT    CurEpoch,
    bool       TheEpochIsTheBestUpToNow,
    MIL_DOUBLE EpochBenchMean)
{
    m_EpochBenchMean = EpochBenchMean;
    UpdateEpochInfo(TrainErrorRate, DevErrorRate, CurEpoch, TheEpochIsTheBestUpToNow);
    UpdateEpochGraph(TrainErrorRate, DevErrorRate, CurEpoch);
}

//............................................................................
void CDashboard::AddMiniBatchData(
    MIL_DOUBLE LossError,
    MIL_INT    MinibatchIdx,
    MIL_INT    EpochIdx,
    MIL_INT    NbBatchPerEpoch)
{
    UpdateLoss(LossError);
    UpdateLossGraph(LossError, MinibatchIdx, EpochIdx, NbBatchPerEpoch);
    UpdateProgression(MinibatchIdx, EpochIdx, NbBatchPerEpoch);
}

//............................................................................
void CDashboard::AddDatasetsPreparedData(MIL_INT TrainDatasetSize, MIL_INT DevDatasetSize)
{
    UpdateDatasetsSize(TrainDatasetSize, DevDatasetSize);
}

//............................................................................
void CDashboard::UpdateEpochInfo(
    MIL_DOUBLE TrainErrorRate,
    MIL_DOUBLE DevErrorRate,
    MIL_INT    CurEpoch,
    bool       TheEpochIsTheBestUpToNow)
{
    const MIL_INT YMargin = 15;
    const MIL_INT TextHeight = 20;
    const MIL_INT TextMargin = MARGIN - 10;

    MgraColor(m_TheGraContext, COLOR_DEV_SET_INFO);
    MIL_TEXT_CHAR DevError[512];
    MosSprintf(DevError, 512, MIL_TEXT("Current Dev error rate: %7.4lf %%"), DevErrorRate);
    MgraText(m_TheGraContext, m_EpochInfoBufId, TextMargin, YMargin, DevError);

    MgraColor(m_TheGraContext, COLOR_TRAIN_SET_INFO);
    MIL_TEXT_CHAR TrainError[512];
    MosSprintf(TrainError, 512, MIL_TEXT("Current Train error rate: %7.4lf %%"), TrainErrorRate);
    MgraText(m_TheGraContext, m_EpochInfoBufId, TextMargin, YMargin + TextHeight, TrainError);

    if (TheEpochIsTheBestUpToNow)
    {
        MgraColor(m_TheGraContext, COLOR_DEV_SET_INFO);
        MIL_TEXT_CHAR BestDevError[512];
        MosSprintf(BestDevError, 512, MIL_TEXT("Best epoch Dev error rate: %7.4lf %%   (Epoch %d)"), DevErrorRate, CurEpoch);
        MgraText(m_TheGraContext, m_EpochInfoBufId, TextMargin, YMargin + 2 * TextHeight, BestDevError);

        MgraColor(m_TheGraContext, COLOR_TRAIN_SET_INFO);
        MIL_TEXT_CHAR TrainErrorBest[512];
        MosSprintf(TrainErrorBest, 512, MIL_TEXT("Train error rate for the best epoch: %7.4lf %%"), TrainErrorRate);
        MgraText(m_TheGraContext, m_EpochInfoBufId, TextMargin, YMargin + 3 * TextHeight, TrainErrorBest);
    }
}

//............................................................................
void CDashboard::UpdateLoss(MIL_DOUBLE LossError)
{
    const MIL_INT TextMargin = MARGIN - 10;

    MgraColor(m_TheGraContext, COLOR_TRAIN_SET_INFO);
    MIL_TEXT_CHAR LossText[512];
    MosSprintf(LossText, 512, MIL_TEXT("Current loss value: %11.7lf"), LossError);

    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, m_YPositionForLossText, LossText);
}

//............................................................................
void CDashboard::UpdateEpochGraph(
    MIL_DOUBLE TrainErrorRate,
    MIL_DOUBLE DevErrorRate,
    MIL_INT    CurEpoch)
{
    MIL_INT EpochIndex = CurEpoch + 1;
    MIL_INT CurTrainPosX = MARGIN + (MIL_INT)((MIL_DOUBLE)(EpochIndex) / (MIL_DOUBLE)(m_MaxEpoch) * (MIL_DOUBLE)GRAPH_SIZE_X);
    MIL_INT CurTrainPosY = GRAPH_TOP_MARGIN + (MIL_INT)((MIL_DOUBLE)GRAPH_SIZE_Y * (1.0 - TrainErrorRate * 0.01));

    MIL_INT CurDevPosX = CurTrainPosX;
    MIL_INT CurDevPosY = GRAPH_TOP_MARGIN + (MIL_INT)((MIL_DOUBLE)GRAPH_SIZE_Y * (1.0 - DevErrorRate * 0.01));

    if (CurEpoch == 0)
    {
        MgraColor(m_TheGraContext, COLOR_TRAIN_SET_INFO);
        MgraArcFill(m_TheGraContext, m_EpochGraphBufId, CurTrainPosX, CurTrainPosY, 2, 2, 0, 360);
        MgraColor(m_TheGraContext, COLOR_DEV_SET_INFO);
        MgraArcFill(m_TheGraContext, m_EpochGraphBufId, CurDevPosX, CurDevPosY, 2, 2, 0, 360);
    }
    else
    {
        MgraColor(m_TheGraContext, COLOR_TRAIN_SET_INFO);
        MgraLine(m_TheGraContext, m_EpochGraphBufId, m_LastTrainPosX, m_LastTrainPosY, CurTrainPosX, CurTrainPosY);
        MgraColor(m_TheGraContext, COLOR_DEV_SET_INFO);
        MgraLine(m_TheGraContext, m_EpochGraphBufId, m_LastDevPosX, m_LastDevPosY, CurDevPosX, CurDevPosY);
    }

    m_LastTrainPosX = CurTrainPosX;
    m_LastTrainPosY = CurTrainPosY;
    m_LastDevPosX = CurDevPosX;
    m_LastDevPosY = CurDevPosY;

    MgraColor(m_TheGraContext, COLOR_GENERAL_INFO);
    MIL_TEXT_CHAR EpochText[128];
    MosSprintf(EpochText, 128, MIL_TEXT("Epoch %d completed"), CurEpoch);
    MgraText(m_TheGraContext, m_EpochGraphBufId, MARGIN, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y + 25, EpochText);
}

//............................................................................
void CDashboard::UpdateLossGraph(
    MIL_DOUBLE LossError,
    MIL_INT    MiniBatchIdx,
    MIL_INT    EpochIdx,
    MIL_INT    NbBatchPerEpoch)
{
    MIL_INT NBMiniBatch = m_MaxEpoch * NbBatchPerEpoch;
    MIL_INT CurMiniBatch = EpochIdx * NbBatchPerEpoch + MiniBatchIdx;

    MIL_DOUBLE XRatio = ((MIL_DOUBLE)CurMiniBatch / (MIL_DOUBLE)(NBMiniBatch));

    MIL_INT CurTrainMBPosX = MARGIN + (MIL_INT)(XRatio * (MIL_DOUBLE)GRAPH_SIZE_X);

    const MIL_DOUBLE MaxVal = std::pow(10.0, LOSS_EXPONENT_MAX);
    const MIL_INT    NbTick = LOSS_EXPONENT_MAX - LOSS_EXPONENT_MIN;

    // Saturate to the highest value of the graph.
    LossError = std::min<MIL_DOUBLE>(LossError, MaxVal);
    MIL_DOUBLE Log10RemapPos = std::max<MIL_DOUBLE>(log10(LossError) + (-LOSS_EXPONENT_MIN), 0.0);
    MIL_DOUBLE YRatio = Log10RemapPos / (MIL_DOUBLE)NbTick;

    MIL_INT CurTrainMBPosY = GRAPH_TOP_MARGIN + (MIL_INT)((MIL_DOUBLE)GRAPH_SIZE_Y * (1.0 - YRatio));

    if (EpochIdx == 0 && MiniBatchIdx == 0)
    {
        MgraColor(m_TheGraContext, COLOR_TRAIN_SET_INFO);
        MgraDot(m_TheGraContext, m_LossGraphBufId, CurTrainMBPosX, CurTrainMBPosY);
    }
    else
    {
        MgraColor(m_TheGraContext, COLOR_TRAIN_SET_INFO);
        MgraLine(m_TheGraContext, m_LossGraphBufId, m_LastTrainMinibatchPosX, m_LastTrainMinibatchPosY, CurTrainMBPosX, CurTrainMBPosY);
    }

    m_LastTrainMinibatchPosX = CurTrainMBPosX;
    m_LastTrainMinibatchPosY = CurTrainMBPosY;

    MgraColor(m_TheGraContext, COLOR_GENERAL_INFO);
    // To clear the previous information.
    MgraText(m_TheGraContext, m_LossGraphBufId, MARGIN, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y + 25, MIL_TEXT("                                                    "));
    MIL_TEXT_CHAR EpochText[512];
    MosSprintf(EpochText, 512, MIL_TEXT("Epoch %d :: Minibatch %d"), EpochIdx, MiniBatchIdx);
    MgraText(m_TheGraContext, m_LossGraphBufId, MARGIN, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y + 25, EpochText);
}

//............................................................................
void CDashboard::UpdateProgression(
    MIL_INT MinibatchIdx,
    MIL_INT EpochIdx,
    MIL_INT NbBatchPerEpoch)
{
    const MIL_INT YMargin = 20;
    const MIL_INT TextHeight = 30;

    const MIL_INT NbMinibatch = m_MaxEpoch * NbBatchPerEpoch;
    const MIL_INT NbMinibatchDone = EpochIdx * NbBatchPerEpoch + MinibatchIdx + 1;
    const MIL_INT NbMinibatchRemaining = NbMinibatch - NbMinibatchDone - 1;

    // Update estimated remaining time.
    MgraColor(m_TheGraContext, COLOR_GENERAL_INFO);

    // The first epoch implied data loading and cannot be used to estimate the
    // remaining time accurately.
    if (EpochIdx == 0)
    {
        MgraText(m_TheGraContext, m_ProgressionInfoBufId, MARGIN, YMargin, MIL_TEXT("Estimated remaining time: N/A"));
    }
    else
    {
        MIL_DOUBLE MinibatchBenchMean = m_EpochBenchMean / (MIL_DOUBLE)NbBatchPerEpoch;
        MIL_DOUBLE RemainingTime = MinibatchBenchMean * (MIL_DOUBLE)NbMinibatchRemaining;
        MIL_TEXT_CHAR RemainingTimeText[512];
        MosSprintf(RemainingTimeText, 512, MIL_TEXT("Estimated remaining time: %8.0lf seconds"), RemainingTime);

        if (NbMinibatchDone == NbMinibatch)
            MgraText(m_TheGraContext, m_ProgressionInfoBufId, MARGIN, YMargin, MIL_TEXT("Training completed!                         "));
        else
            MgraText(m_TheGraContext, m_ProgressionInfoBufId, MARGIN, YMargin, RemainingTimeText);
    }

    // Update the progression bar.
    const MIL_INT ProgressionBarWidth = m_DashboardWidth - 2 * MARGIN;
    const MIL_INT ProgressionBarHeight = 30;
    MgraColor(m_TheGraContext, COLOR_GENERAL_INFO);
    MgraRectFill(m_TheGraContext, m_ProgressionInfoBufId, MARGIN, YMargin + TextHeight, MARGIN + ProgressionBarWidth, YMargin + TextHeight + ProgressionBarHeight);

    MIL_DOUBLE PercentageComplete = (MIL_DOUBLE)(NbMinibatchDone) / (MIL_DOUBLE)(NbMinibatch);
    MIL_INT PercentageCompleteWidth = (MIL_INT)(PercentageComplete * ProgressionBarWidth);
    MgraColor(m_TheGraContext, COLOR_PROGRESS_BAR);
    MgraRectFill(m_TheGraContext, m_ProgressionInfoBufId, MARGIN, YMargin + TextHeight, MARGIN + PercentageCompleteWidth, YMargin + TextHeight + ProgressionBarHeight);
}

//............................................................................
void CDashboard::UpdateDatasetsSize(MIL_INT TrainDatasetSize, MIL_INT DevDatasetSize)
{
    const MIL_INT DatasetSizeOffset = 5;
    const MIL_INT YMargin = 15;
    const MIL_INT TextHeight = 20;
    const MIL_INT TextMargin = MARGIN - 10;

    MIL_INT TextYPos = DatasetSizeOffset * YMargin;

    MIL_TEXT_CHAR TheString[512];
    MosSprintf(TheString, 512, MIL_TEXT("Train and Dev dataset size: %d and %d images"), TrainDatasetSize, DevDatasetSize);
    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
}

// HookFunc for CNN
MIL_INT MFTYPE HookFuncDatasetsPrepared(
    MIL_INT /*HookType*/,
    MIL_ID  EventId,
    void* UserData)
{
    auto HookData = (HookDataStruct*)UserData;

    MIL_ID TrainResult;
    MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainResult);

    MIL_UNIQUE_CLASS_ID TrainPreparedDataset = MclassAlloc(HookData->MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    MclassCopyResult(TrainResult, M_DEFAULT, TrainPreparedDataset, M_DEFAULT, M_PREPARED_TRAIN_DATASET, M_DEFAULT);
    const MIL_INT TrainDatasetNbImages = MclassInquire(TrainPreparedDataset, M_DEFAULT, M_NUMBER_OF_ENTRIES, M_NULL);

    MIL_UNIQUE_CLASS_ID DevPreparedDataset = MclassAlloc(HookData->MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    MclassCopyResult(TrainResult, M_DEFAULT, DevPreparedDataset, M_DEFAULT, M_PREPARED_DEV_DATASET, M_DEFAULT);
    const MIL_INT DevDatasetNbImages = MclassInquire(DevPreparedDataset, M_DEFAULT, M_NUMBER_OF_ENTRIES, M_NULL);

    HookData->DashboardPtr->AddDatasetsPreparedData(TrainDatasetNbImages, DevDatasetNbImages);

    MdispSelect(HookData->MilDisplay, HookData->DashboardPtr->GetDashboardBufId());

    return M_NULL;
}

//.........................................................................
MIL_INT MFTYPE HookFuncEpoch(
    MIL_INT /*HookType*/,
    MIL_ID  EventId,
    void* UserData)
{
    auto HookData = (HookDataStruct*)UserData;

    MIL_DOUBLE CurBench = 0.0;
    MIL_DOUBLE CurBenchMean = -1.0;

    MIL_INT CurEpochIndex = 0;
    MclassGetHookInfo(EventId, M_EPOCH_INDEX + M_TYPE_MIL_INT, &CurEpochIndex);

    MappTimer(M_DEFAULT, M_TIMER_READ, &CurBench);
    MIL_DOUBLE EpochBenchMean = CurBench / (CurEpochIndex + 1);

    MIL_DOUBLE TrainErrorRate = 0;
    MclassGetHookInfo(EventId, M_TRAIN_DATASET_ERROR_RATE, &TrainErrorRate);
    MIL_DOUBLE DevErrorRate = 0;
    MclassGetHookInfo(EventId, M_DEV_DATASET_ERROR_RATE, &DevErrorRate);

    MIL_INT AreTrainedCNNParametersUpdated = M_FALSE;
    MclassGetHookInfo(EventId,
        M_TRAINED_PARAMETERS_UPDATED + M_TYPE_MIL_INT,
        &AreTrainedCNNParametersUpdated);

    // By default trained parameters are updated when the dev error rate
    // is the best up to now.
    bool TheEpochIsTheBestUpToNow = (AreTrainedCNNParametersUpdated == M_TRUE);

    HookData->DashboardPtr->AddEpochData(
        TrainErrorRate,
        DevErrorRate,
        CurEpochIndex,
        TheEpochIsTheBestUpToNow,
        EpochBenchMean);

    return M_NULL;
}

//............................................................................
MIL_INT MFTYPE HookFuncMiniBatch(
    MIL_INT HookType,
    MIL_ID  EventId,
    void* UserData)
{
    auto HookData = (HookDataStruct*)UserData;

    MIL_DOUBLE LossError = 0;
    MclassGetHookInfo(EventId, M_MINI_BATCH_LOSS, &LossError);

    MIL_INT MiniBatchIdx = 0;
    MclassGetHookInfo(EventId, M_MINI_BATCH_INDEX + M_TYPE_MIL_INT, &MiniBatchIdx);

    MIL_INT EpochIdx = 0;
    MclassGetHookInfo(EventId, M_EPOCH_INDEX + M_TYPE_MIL_INT, &EpochIdx);

    MIL_INT NbMiniBatchPerEpoch = 0;
    MclassGetHookInfo(EventId, M_MINI_BATCH_PER_EPOCH + M_TYPE_MIL_INT, &NbMiniBatchPerEpoch);

    if (EpochIdx == 0 && MiniBatchIdx == 0)
    {
        MappTimer(M_DEFAULT, M_TIMER_RESET, M_NULL);
    }

    if (MiniBatchIdx == NbMiniBatchPerEpoch - 1 && HookData->DumpTmpRst == 1)
    {
        MIL_ID TrainRes;
        MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainRes);
        MIL_UNIQUE_CLASS_ID TrainedClassifierCtx = MclassAlloc(HookData->MilSystem, M_CLASSIFIER_CNN_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
        MclassCopyResult(TrainRes, M_DEFAULT, TrainedClassifierCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);
        MclassSave(HookData->ClassifierDumpFile, TrainedClassifierCtx, M_DEFAULT);
    }

    HookData->DashboardPtr->AddMiniBatchData(LossError, MiniBatchIdx, EpochIdx, NbMiniBatchPerEpoch);

    //MosPrintf(MIL_TEXT("\nBatch[%d] loss error = %f.\n"), MiniBatchIdx, LossError);

    //EpochIdx > 1的时候再来判断

    if (HookData->ControlType == 's') {
        //在Stop后先保存当前模型
        MIL_ID TrainRes;
        MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainRes);

        MIL_UNIQUE_CLASS_ID TrainedClassifierCtx = MclassAlloc(HookData->MilSystem, M_CLASSIFIER_CNN_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
        MclassCopyResult(TrainRes, M_DEFAULT, TrainedClassifierCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);
        MclassSave(HookData->ClassifierDumpFile, TrainedClassifierCtx, M_DEFAULT);

        MIL_ID HookInfoTrainResId = M_NULL;
        MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &HookInfoTrainResId);
        MclassControl(HookInfoTrainResId, M_DEFAULT, M_STOP_TRAIN, M_DEFAULT);
        MosPrintf(MIL_TEXT("The training has been stopped.\n"));
    }
    else if (HookData->ControlType == 'p') {
        MosPrintf(MIL_TEXT("\nPress 's' to stop the training or any other key to continue.\n"));
        while (1) {
            //Sleep(1000);
            if (HookData->ControlType == 's')
            {
                //在Stop后先保存当前模型
                MIL_ID TrainRes;
                MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainRes);
                MIL_UNIQUE_CLASS_ID TrainedClassifierCtx = MclassAlloc(HookData->MilSystem, M_CLASSIFIER_CNN_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
                MclassCopyResult(TrainRes, M_DEFAULT, TrainedClassifierCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);
                MclassSave(HookData->ClassifierDumpFile, TrainedClassifierCtx, M_DEFAULT);


                MIL_ID HookInfoTrainResId = M_NULL;
                MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &HookInfoTrainResId);
                MclassControl(HookInfoTrainResId, M_DEFAULT, M_STOP_TRAIN, M_DEFAULT);
                MosPrintf(MIL_TEXT("The training has been stopped.\n"));
                break;
            }
            else if (HookData->ControlType == 'r')
            {
                MosPrintf(MIL_TEXT("The training will continue.\n"));
                break;
            }

        }

    }
    return M_NULL;
}


// HookFunc for Detection

//MIL_INT MFTYPE DetHookFuncDatasetsPrepared(MIL_INT, MIL_ID EventId, void* UserData)
//{
//    auto pHookData = (DetHookDataStruct*)UserData;
//    MIL_ID TrainRslt{ M_NULL };
//    MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainRslt);
//
//    MIL_ID MilSystem{ M_NULL };
//    MclassInquire(TrainRslt, M_DEFAULT, M_OWNER_SYSTEM + M_TYPE_MIL_ID, &MilSystem);
//
//    MIL_UNIQUE_CLASS_ID PrpTrainDataset = MclassAlloc(MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
//    MclassCopyResult(TrainRslt, M_DEFAULT, PrpTrainDataset, M_DEFAULT, M_PREPARED_TRAIN_DATASET, M_DEFAULT);
//
//    //MosPrintf(MIL_TEXT("Press <v> to view the augmented train dataset.\nPress <Enter> to continue...\n"));
//
//    //char KeyVal = (char)MosGetch();
//    //if (KeyVal == 'v' || KeyVal == 'V')
//    //{
//    //    MosPrintf(MIL_TEXT("\n\n*******************************************************\n"));
//    //    MosPrintf(MIL_TEXT("VIEWING THE AUGMENTED TRAIN DATASET..."));
//    //    MosPrintf(MIL_TEXT("\n*******************************************************\n\n"));
//    //    CDatasetViewer DatasetViewer(MilSystem, PrpTrainDataset, true);
//    //}
//
//    MosPrintf(MIL_TEXT("\nThe training has started.\n"));
//    MosPrintf(MIL_TEXT("It can be paused at any time by pressing 'p'.\n"));
//    MosPrintf(MIL_TEXT("It can then be stopped or continued.\n"));
//
//    MosPrintf(MIL_TEXT("\nDuring training, you can observe the evolution of the losses\n"));
//    MosPrintf(MIL_TEXT("of the train and dev datasets together.\n"));
//    MosPrintf(MIL_TEXT("The best epoch is determined by the epoch with the smallest dev loss.\n"));
//
//    MdispSelect(pHookData->MilDisplay, pHookData->DetDashboardPtr->GetDashboardBufId());
//
//    //MdispSelect(HookData->MilDisplay, HookData->DashboardPtr->GetDashboardBufId());
//
//    return M_NULL;
//}
//
//MIL_INT MFTYPE DetHookFuncEpoch(MIL_INT, MIL_ID EventId, void* UserData)
//{
//
//    auto pHookData = (DetHookDataStruct*)UserData;
//
//    MIL_DOUBLE CurBench = 0.0;
//    MIL_DOUBLE CurBenchMean = -1.0;
//
//    MIL_INT CurEpochIndex = 0;
//    MclassGetHookInfo(EventId, M_EPOCH_INDEX + M_TYPE_MIL_INT, &CurEpochIndex);
//
//    MappTimer(M_DEFAULT, M_TIMER_READ, &CurBench);
//    const MIL_DOUBLE EpochBenchMean = CurBench / (CurEpochIndex + 1);
//
//    MIL_DOUBLE DevLoss = 0;
//    MclassGetHookInfo(EventId, M_DEV_DATASET_LOSS, &DevLoss);
//
//    pHookData->DetDashboardPtr->AddEpochData(DevLoss, CurEpochIndex, EpochBenchMean);
//
//    
//    if (CurEpochIndex % pHookData->SaveModelPEpoch == 0) {
//        MIL_ID TrainRes;
//        MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainRes);
//        MIL_UNIQUE_CLASS_ID TrainedDetCtx = MclassAlloc(pHookData->MilSystem, M_CLASSIFIER_DET_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
//        MclassCopyResult(TrainRes, M_DEFAULT, TrainedDetCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);
//        //MIL_TEXT_CHAR tmpCfDumpFile[512];
//        //MosSprintf(tmpCfDumpFile, 512, MIL_TEXT("%s%d"), pHookData->ClassifierDumpFile, CurEpochIndex);
//        MclassSave(pHookData->ClassifierDumpFile, TrainedDetCtx, M_DEFAULT);
//    }
//
//
//
//    return M_NULL;
//}
//
////==============================================================================
//MIL_INT MFTYPE DetHookFuncMiniBatch(MIL_INT HookType, MIL_ID EventId, void* UserData)
//{
//    auto pHookData = (DetHookDataStruct*)UserData;
//
//    MIL_DOUBLE Loss = 0;
//    MclassGetHookInfo(EventId, M_MINI_BATCH_LOSS, &Loss);
//
//    MIL_INT MiniBatchIdx = 0;
//    MclassGetHookInfo(EventId, M_MINI_BATCH_INDEX + M_TYPE_MIL_INT, &MiniBatchIdx);
//
//    MIL_INT EpochIdx = 0;
//    MclassGetHookInfo(EventId, M_EPOCH_INDEX + M_TYPE_MIL_INT, &EpochIdx);
//
//    MIL_INT NbMiniBatchPerEpoch = 0;
//    MclassGetHookInfo(EventId, M_MINI_BATCH_PER_EPOCH + M_TYPE_MIL_INT, &NbMiniBatchPerEpoch);
//
//    if (EpochIdx == 0 && MiniBatchIdx == 0)
//    {
//        MappTimer(M_DEFAULT, M_TIMER_RESET, M_NULL);
//    }
//
//    pHookData->DetDashboardPtr->AddMiniBatchData(Loss, MiniBatchIdx, EpochIdx, NbMiniBatchPerEpoch);
//
//    if (MosKbhit() != 0)
//    {
//        char KeyVal = (char)MosGetch();
//        if (KeyVal == 'p' || KeyVal == 'P')
//        {
//            MosPrintf(MIL_TEXT("\nPress 's' to stop the training or any other key to continue.\n"));
//            while (MosKbhit() == 0)
//            {
//                KeyVal = (char)MosGetch();
//                if (KeyVal == 's' || KeyVal == 'S')
//                {
//                    MIL_ID HookInfoTrainResId = M_NULL;
//                    MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &HookInfoTrainResId);
//                    MclassControl(HookInfoTrainResId, M_DEFAULT, M_STOP_TRAIN, M_DEFAULT);
//                    MosPrintf(MIL_TEXT("The training has been stopped.\n"));
//                    break;
//                }
//                else
//                {
//                    MosPrintf(MIL_TEXT("The training will continue.\n"));
//                    break;
//                }
//            }
//        }
//    }
//
//    return(M_NULL);
//}
//
////==============================================================================
//MIL_STRING ConvertPrepareDataStatusToStr(MIL_INT Status)
//{
//    switch (Status)
//    {
//    case M_COMPLETE:
//        return MIL_TEXT("M_COMPLETE");
//    case M_INVALID_AUG_OP_FOR_1_BAND_BUFFER:
//        return MIL_TEXT("M_INVALID_AUG_OP_FOR_1_BAND_BUFFER");
//    case M_INVALID_AUG_OP_FOR_1_BIT_BUFFER:
//        return MIL_TEXT("M_INVALID_AUG_OP_FOR_1_BIT_BUFFER");
//    case M_SOURCE_TOO_SMALL_FOR_DERICHE_OP:
//        return MIL_TEXT("M_SOURCE_TOO_SMALL_FOR_DERICHE_OP");
//    case M_FLOAT_IMAGE_NOT_NORMALIZED:
//        return MIL_TEXT("M_FLOAT_IMAGE_NOT_NORMALIZED");
//    case M_FAILED_TO_SAVE_IMAGE:
//        return MIL_TEXT("M_FAILED_TO_SAVE_IMAGE");
//    case M_IMAGE_FILE_NOT_FOUND:
//        return MIL_TEXT("M_IMAGE_FILE_NOT_FOUND");
//    case M_INVALID_BUFFER_SIGN_FOR_AUG:
//        return MIL_TEXT("M_INVALID_BUFFER_SIGN_FOR_AUG");
//    case M_INVALID_CENTER:
//        return MIL_TEXT("M_INVALID_CENTER");
//    case M_MASK_FILE_NOT_FOUND:
//        return MIL_TEXT("M_MASK_FILE_NOT_FOUND");
//    case M_RESIZED_IMAGE_TOO_SMALL:
//        return MIL_TEXT("M_RESIZED_IMAGE_TOO_SMALL");
//    default:
//    case M_INTERNAL_ERROR:
//        return MIL_TEXT("M_INTERNAL_ERROR");
//    }
//}
//
////==============================================================================
//MIL_INT MFTYPE DetHookNumPreparedEntriesFunc(MIL_INT, MIL_ID EventId, void* pUserData)
//{
//    bool* pIsDevDataset = reinterpret_cast<bool*>(pUserData);
//
//    MIL_ID SrcDataset{ M_NULL };
//    MclassGetHookInfo(EventId, M_SRC_DATASET_ID + M_TYPE_MIL_ID, &SrcDataset);
//
//    MIL_INT NumPrpEntries{ 0 };
//    MclassGetHookInfo(EventId, M_NUMBER_OF_PREPARED_SRC_ENTRIES + M_TYPE_MIL_INT, &NumPrpEntries);
//
//    const MIL_INT NumEntries = MclassInquire(SrcDataset, M_DEFAULT, M_NUMBER_OF_ENTRIES, M_NULL);
//
//    if (NumPrpEntries == 1)
//    {
//        *pIsDevDataset ? MosPrintf(MIL_TEXT("Preparing the dev dataset...\n")) :
//            MosPrintf(MIL_TEXT("Augmenting the train dataset...\n"));
//    }
//
//    MIL_INT Status{ -1 };
//    MclassGetHookInfo(EventId, M_STATUS + M_TYPE_MIL_INT, &Status);
//
//    const MIL_STRING StatusStr = ConvertPrepareDataStatusToStr(Status);
//
//    MIL_TEXT_CHAR EndOfLine = '\r';
//    if (Status != M_COMPLETE)
//    {
//        EndOfLine = '\n';
//    }
//
//    MosPrintf(MIL_TEXT("Entry %d of %d completed with status: %s.%c"), NumPrpEntries, NumEntries, StatusStr.c_str(), EndOfLine);
//
//    if (NumPrpEntries == NumEntries)
//    {
//        EndOfLine == '\r' ? MosPrintf(MIL_TEXT("\n\n")) : MosPrintf(MIL_TEXT("\n"));
//        *pIsDevDataset = true;
//    }
//
//    return M_NULL;
//}


// Dashboard for Detection

//DetDashboard::DetDashboard(MIL_ID MilSystem, MIL_ID TrainCtx, MIL_INT TrainImageSizeX, MIL_INT TrainImageSizeY, MIL_INT TrainEngineUsed, const MIL_STRING& TrainEngineDescription)
//{
//    MclassInquire(TrainCtx, M_DEFAULT, M_MAX_EPOCH + M_TYPE_MIL_INT, &m_MaxEpoch);
//
//    MIL_DOUBLE InitLearningRate{ 0.0 };
//    MclassInquire(TrainCtx, M_DEFAULT, M_INITIAL_LEARNING_RATE + M_TYPE_MIL_DOUBLE, &InitLearningRate);
//    MIL_INT MiniBatchSize{ 0 };
//    MclassInquire(TrainCtx, M_DEFAULT, M_MINI_BATCH_SIZE + M_TYPE_MIL_INT, &MiniBatchSize);
//
//    const MIL_INT GraphBoxWidth = GRAPH_SIZE_X + 2 * MARGIN;
//    const MIL_INT GraphBoxHeight = GRAPH_SIZE_Y + GRAPH_TOP_MARGIN + MARGIN;
//
//    m_DashboardWidth = GraphBoxWidth;
//    const MIL_INT DashboardHeight = GraphBoxHeight + EPOCH_AND_MINIBATCH_REGION_HEIGHT + PROGRESSION_INFO_REGION_HEIGHT;
//
//    m_DashboardBufId = MbufAllocColor(MilSystem, 3, m_DashboardWidth, DashboardHeight, 8 + M_UNSIGNED, M_IMAGE + M_PROC + M_DISP, M_UNIQUE_ID);
//    MbufClear(m_DashboardBufId, M_COLOR_BLACK);
//
//    m_TheGraContext = MgraAlloc(MilSystem, M_UNIQUE_ID);
//
//    const MIL_INT GraphYPosition = EPOCH_AND_MINIBATCH_REGION_HEIGHT;
//    const MIL_INT ProgressionInfoYPosition = GraphYPosition + GraphBoxHeight;
//
//    m_LossInfoBufId = MbufChild2d(m_DashboardBufId, 0, 0, GraphBoxWidth, EPOCH_AND_MINIBATCH_REGION_HEIGHT, M_UNIQUE_ID);
//    m_LossGraphBufId = MbufChild2d(m_DashboardBufId, 0, GraphYPosition, GraphBoxWidth, GraphBoxHeight, M_UNIQUE_ID);
//    m_ProgressionInfoBufId = MbufChild2d(m_DashboardBufId, 0, ProgressionInfoYPosition, m_DashboardWidth, PROGRESSION_INFO_REGION_HEIGHT, M_UNIQUE_ID);
//
//    DrawSectionSeparators();
//
//    InitializeLossGraph();
//
//    WriteGeneralTrainInfo(MiniBatchSize, TrainImageSizeX, TrainImageSizeY, InitLearningRate, TrainEngineUsed, TrainEngineDescription);
//
//
//}
//
//DetDashboard::~DetDashboard()
//{
//    m_TheGraContext = M_NULL;
//    m_LossInfoBufId = M_NULL;
//    m_LossGraphBufId = M_NULL;
//    m_ProgressionInfoBufId = M_NULL;
//    m_DashboardBufId = M_NULL;
//
//}
//
//void DetDashboard::AddEpochData(MIL_DOUBLE Loss, MIL_INT CurEpoch, MIL_DOUBLE EpochBenchMean)
//{
//    m_EpochBenchMean = EpochBenchMean;
//    UpdateDevLoss(Loss);
//    UpdateDevLossGraph(Loss, CurEpoch);
//}
//
//void DetDashboard::AddMiniBatchData(MIL_DOUBLE Loss, MIL_INT MinibatchIdx, MIL_INT EpochIdx, MIL_INT NbBatchPerEpoch)
//{
//    UpdateTrainLoss(Loss);
//    UpdateTrainLossGraph(Loss, MinibatchIdx, EpochIdx, NbBatchPerEpoch);
//    UpdateProgression(MinibatchIdx, EpochIdx, NbBatchPerEpoch);
//
//}
//
//void DetDashboard::UpdateTrainLoss(MIL_DOUBLE Loss)
//{
//    const MIL_INT TextMargin = MARGIN - 10;
//
//    MgraColor(m_TheGraContext, COLOR_TRAIN_SET_INFO);
//    MIL_TEXT_CHAR LossText[512];
//    MosSprintf(LossText, 512, MIL_TEXT("Current train loss value: %11.7lf"), Loss);
//
//    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, m_YPositionForTrainLossText, LossText);
//}
//
//void DetDashboard::UpdateDevLoss(MIL_DOUBLE Loss)
//{
//    const MIL_INT TextMargin = MARGIN - 10;
//
//    MgraColor(m_TheGraContext, COLOR_DEV_SET_INFO);
//    MIL_TEXT_CHAR LossText[512];
//    MosSprintf(LossText, 512, MIL_TEXT("Current dev loss value: %11.7lf"), Loss);
//
//    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, m_YPositionForDevLossText, LossText);
//
//}
//
//void DetDashboard::UpdateTrainLossGraph(MIL_DOUBLE Loss, MIL_INT MiniBatchIdx, MIL_INT EpochIdx, MIL_INT NbBatchPerEpoch)
//{
//    const MIL_INT NbMiniBatch = m_MaxEpoch * NbBatchPerEpoch;
//    const MIL_INT CurMiniBatch = EpochIdx * NbBatchPerEpoch + MiniBatchIdx;
//
//    const MIL_DOUBLE XRatio = static_cast<MIL_DOUBLE>(CurMiniBatch) / static_cast<MIL_DOUBLE>(NbMiniBatch);
//
//    const MIL_INT CurTrainMBPosX = MARGIN + static_cast<MIL_INT>(XRatio * static_cast<MIL_DOUBLE>(GRAPH_SIZE_X));
//
//    const MIL_DOUBLE MaxVal = std::pow(10.0, LOSS_EXPONENT_MAX);
//    const MIL_INT NbTick = LOSS_EXPONENT_MAX - LOSS_EXPONENT_MIN;
//
//    // Saturate to the highest value of the graph.
//    Loss = std::min<MIL_DOUBLE>(Loss, MaxVal);
//    const MIL_DOUBLE Log10RemapPos = std::max<MIL_DOUBLE>(log10(Loss) + (-LOSS_EXPONENT_MIN), 0.0);
//    const MIL_DOUBLE YRatio = Log10RemapPos / static_cast<MIL_DOUBLE>(NbTick);
//
//    const MIL_INT CurTrainMBPosY = GRAPH_TOP_MARGIN + static_cast<MIL_INT>((1.0 - YRatio) * static_cast<MIL_DOUBLE>(GRAPH_SIZE_Y));
//
//    if (EpochIdx == 0 && MiniBatchIdx == 0)
//    {
//        MgraColor(m_TheGraContext, COLOR_TRAIN_SET_INFO);
//        MgraDot(m_TheGraContext, m_LossGraphBufId, CurTrainMBPosX, CurTrainMBPosY);
//    }
//    else
//    {
//        MgraColor(m_TheGraContext, COLOR_TRAIN_SET_INFO);
//        MgraLine(m_TheGraContext, m_LossGraphBufId, m_LastTrainMinibatchPosX, m_LastTrainMinibatchPosY, CurTrainMBPosX, CurTrainMBPosY);
//    }
//
//    m_LastTrainMinibatchPosX = CurTrainMBPosX;
//    m_LastTrainMinibatchPosY = CurTrainMBPosY;
//
//    MgraColor(m_TheGraContext, COLOR_GENERAL_INFO);
//    // To clear the previous information.
//    MgraText(m_TheGraContext, m_LossGraphBufId, MARGIN, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y + 25, MIL_TEXT("                                                    "));
//    MIL_TEXT_CHAR EpochText[512];
//    MosSprintf(EpochText, 512, MIL_TEXT("Epoch %d :: Minibatch %d"), EpochIdx, MiniBatchIdx);
//    MgraText(m_TheGraContext, m_LossGraphBufId, MARGIN, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y + 25, EpochText);
//
//}
//
//void DetDashboard::UpdateDevLossGraph(MIL_DOUBLE Loss, MIL_INT EpochIdx)
//{
//    const MIL_DOUBLE XRatio = static_cast<MIL_DOUBLE>(EpochIdx + 1) / static_cast<MIL_DOUBLE>(m_MaxEpoch);
//
//    const MIL_INT CurTrainMBPosX = MARGIN + static_cast<MIL_INT>(XRatio * static_cast<MIL_DOUBLE>(GRAPH_SIZE_X));
//
//    const MIL_DOUBLE MaxVal = std::pow(10.0, LOSS_EXPONENT_MAX);
//    const MIL_INT NbTick = LOSS_EXPONENT_MAX - LOSS_EXPONENT_MIN;
//
//    // Saturate to the highest value of the graph.
//    Loss = std::min<MIL_DOUBLE>(Loss, MaxVal);
//    const MIL_DOUBLE Log10RemapPos = std::max<MIL_DOUBLE>(log10(Loss) + (-LOSS_EXPONENT_MIN), 0.0);
//    const MIL_DOUBLE YRatio = Log10RemapPos / static_cast<MIL_DOUBLE>(NbTick);
//
//    const MIL_INT CurTrainMBPosY = GRAPH_TOP_MARGIN + static_cast<MIL_INT>((1.0 - YRatio) * static_cast<MIL_DOUBLE>(GRAPH_SIZE_Y));
//
//    if (EpochIdx == 0)
//    {
//        MgraColor(m_TheGraContext, COLOR_DEV_SET_INFO);
//        MgraDot(m_TheGraContext, m_LossGraphBufId, CurTrainMBPosX, CurTrainMBPosY);
//    }
//    else
//    {
//        MgraColor(m_TheGraContext, COLOR_DEV_SET_INFO);
//        MgraLine(m_TheGraContext, m_LossGraphBufId, m_LastDevEpochLossPosX, m_LastDevEpochLossPosY, CurTrainMBPosX, CurTrainMBPosY);
//    }
//
//    m_LastDevEpochLossPosX = CurTrainMBPosX;
//    m_LastDevEpochLossPosY = CurTrainMBPosY;
//
//}
//
//void DetDashboard::UpdateProgression(MIL_INT MinibatchIdx, MIL_INT EpochIdx, MIL_INT NbBatchPerEpoch)
//{
//    const MIL_INT YMargin = 20;
//    const MIL_INT TextHeight = 30;
//
//    const MIL_INT NbMinibatch = m_MaxEpoch * NbBatchPerEpoch;
//    const MIL_INT NbMinibatchDone = EpochIdx * NbBatchPerEpoch + MinibatchIdx + 1;
//    const MIL_INT NbMinibatchRemaining = NbMinibatch - NbMinibatchDone - 1;
//
//    // Update estimated remaining time.
//    MgraColor(m_TheGraContext, COLOR_GENERAL_INFO);
//
//    // The first epoch implied data loading and cannot be used to estimate the
//    // remaining time accurately.
//    if (EpochIdx == 0)
//    {
//        MgraText(m_TheGraContext, m_ProgressionInfoBufId, MARGIN, YMargin, MIL_TEXT("Estimated remaining time: N/A"));
//    }
//    else
//    {
//        const MIL_DOUBLE MinibatchBenchMean = m_EpochBenchMean / static_cast<MIL_DOUBLE>(NbBatchPerEpoch);
//        const MIL_DOUBLE RemainingTime = MinibatchBenchMean * static_cast<MIL_DOUBLE>(NbMinibatchRemaining);
//        MIL_TEXT_CHAR RemainingTimeText[512];
//        MosSprintf(RemainingTimeText, 512, MIL_TEXT("Estimated remaining time: %8.0lf seconds"), RemainingTime);
//
//        if (NbMinibatchDone == NbMinibatch)
//        {
//            MgraText(m_TheGraContext, m_ProgressionInfoBufId, MARGIN, YMargin, MIL_TEXT("Training completed!                         "));
//        }
//        else
//        {
//            MgraText(m_TheGraContext, m_ProgressionInfoBufId, MARGIN, YMargin, RemainingTimeText);
//        }
//    }
//
//    // Update the progression bar.
//    const MIL_INT ProgressionBarWidth = m_DashboardWidth - 2 * MARGIN;
//    const MIL_INT ProgressionBarHeight = 30;
//    MgraColor(m_TheGraContext, COLOR_GENERAL_INFO);
//    MgraRectFill(m_TheGraContext, m_ProgressionInfoBufId, MARGIN, YMargin + TextHeight, MARGIN + ProgressionBarWidth, YMargin + TextHeight + ProgressionBarHeight);
//
//    const MIL_DOUBLE PercentageComplete = static_cast<MIL_DOUBLE>(NbMinibatchDone) / static_cast<MIL_DOUBLE>(NbMinibatch);
//    const MIL_INT PercentageCompleteWidth = static_cast<MIL_INT>(PercentageComplete * ProgressionBarWidth);
//    MgraColor(m_TheGraContext, COLOR_PROGRESS_BAR);
//    MgraRectFill(m_TheGraContext, m_ProgressionInfoBufId, MARGIN, YMargin + TextHeight, MARGIN + PercentageCompleteWidth, YMargin + TextHeight + ProgressionBarHeight);
//
//}
//
//void DetDashboard::DrawSectionSeparators()
//{
//    // Draw a frame for the whole dashboard.
//    DrawBufferFrame(m_DashboardBufId, 4);
//    // Draw a frame for each section.
//    DrawBufferFrame(m_LossInfoBufId, 2);
//    DrawBufferFrame(m_LossGraphBufId, 2);
//    DrawBufferFrame(m_ProgressionInfoBufId, 2);
//
//}
//
//void DetDashboard::DrawBufferFrame(MIL_ID BufId, MIL_INT FrameThickness)
//{
//    const MIL_ID SizeX = MbufInquire(BufId, M_SIZE_X, M_NULL);
//    const MIL_ID SizeY = MbufInquire(BufId, M_SIZE_Y, M_NULL);
//
//    MgraColor(m_TheGraContext, COLOR_GENERAL_INFO);
//    MgraRectFill(m_TheGraContext, BufId, 0, 0, SizeX - 1, FrameThickness - 1);
//    MgraRectFill(m_TheGraContext, BufId, SizeX - FrameThickness, 0, SizeX - 1, SizeY - 1);
//    MgraRectFill(m_TheGraContext, BufId, 0, SizeY - FrameThickness, SizeX - 1, SizeY - 1);
//    MgraRectFill(m_TheGraContext, BufId, 0, 0, FrameThickness - 1, SizeY - 1);
//}
//
//void DetDashboard::InitializeLossGraph()
//{
//    // Draw axis.
//    MgraColor(m_TheGraContext, M_COLOR_WHITE);
//    MgraRect(m_TheGraContext, m_LossGraphBufId, MARGIN, GRAPH_TOP_MARGIN, MARGIN + GRAPH_SIZE_X, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y);
//
//    MgraControl(m_TheGraContext, M_TEXT_ALIGN_HORIZONTAL, M_RIGHT);
//
//    const MIL_INT NbLossValueTick = LOSS_EXPONENT_MAX - LOSS_EXPONENT_MIN;
//    const MIL_DOUBLE TickRatio = 1.0 / static_cast<MIL_DOUBLE>(NbLossValueTick);
//
//    MIL_DOUBLE TickNum = 0.0;
//    for (MIL_INT i = LOSS_EXPONENT_MAX; i >= LOSS_EXPONENT_MIN; i--)
//    {
//        MIL_TEXT_CHAR CurTickText[128];
//        MosSprintf(CurTickText, 128, MIL_TEXT("1e%d"), i);
//
//        const MIL_INT TickYPos = static_cast<MIL_INT>(TickNum * TickRatio * GRAPH_SIZE_Y);
//        MgraText(m_TheGraContext, m_LossGraphBufId, MARGIN - 5, GRAPH_TOP_MARGIN + TickYPos, CurTickText);
//        if ((i != LOSS_EXPONENT_MAX) && (i != LOSS_EXPONENT_MIN))
//        {
//            MgraLine(m_TheGraContext, m_LossGraphBufId, MARGIN, GRAPH_TOP_MARGIN + TickYPos, MARGIN + 5, GRAPH_TOP_MARGIN + TickYPos);
//        }
//        TickNum = TickNum + 1.0;
//    }
//
//    MgraControl(m_TheGraContext, M_TEXT_ALIGN_HORIZONTAL, M_LEFT);
//
//    const MIL_INT NbEpochTick = std::min<MIL_INT>(m_MaxEpoch, 10);
//    const MIL_INT EpochTickValue = m_MaxEpoch / NbEpochTick;
//
//    for (MIL_INT CurTick = 1; CurTick <= m_MaxEpoch; CurTick += EpochTickValue)
//    {
//        const MIL_DOUBLE Percentage = static_cast<MIL_DOUBLE>(CurTick) / static_cast<MIL_DOUBLE>(m_MaxEpoch);
//        const MIL_INT XOffset = static_cast<MIL_INT>(Percentage * GRAPH_SIZE_X);
//        MgraText(m_TheGraContext, m_LossGraphBufId, MARGIN + XOffset, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y + 5, M_TO_STRING(CurTick - 1));
//        MgraLine(m_TheGraContext, m_LossGraphBufId, MARGIN + XOffset, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y - 5, MARGIN + XOffset, GRAPH_TOP_MARGIN + GRAPH_SIZE_Y);
//    }
//}
//
//void DetDashboard::WriteGeneralTrainInfo(MIL_INT MinibatchSize, MIL_INT TrainImageSizeX, MIL_INT TrainImageSizeY, MIL_DOUBLE LearningRate, MIL_INT TrainEngineUsed, const MIL_STRING& TrainEngineDescription)
//{
//    MgraControl(m_TheGraContext, M_BACKGROUND_MODE, M_OPAQUE);
//    MgraControl(m_TheGraContext, M_BACKCOLOR, M_COLOR_BLACK);
//
//    MgraControl(m_TheGraContext, M_TEXT_ALIGN_HORIZONTAL, M_LEFT);
//
//    const MIL_INT YMargin = 15;
//    const MIL_INT TextHeight = 20;
//    const MIL_INT TextMargin = MARGIN - 10;
//
//    MIL_INT TextYPos = YMargin;
//
//    MgraColor(m_TheGraContext, COLOR_GENERAL_INFO);
//
//    MIL_TEXT_CHAR TheString[512];
//    if (TrainEngineUsed == M_CPU)
//    {
//        MosSprintf(TheString, 512, MIL_TEXT("Training is being performed on the CPU"));
//    }
//    else
//    {
//        MosSprintf(TheString, 512, MIL_TEXT("Training is being performed on the GPU"));
//    }
//
//    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
//    TextYPos += TextHeight;
//
//    MosSprintf(TheString, 512, MIL_TEXT("Engine: %s"), TrainEngineDescription.c_str());
//    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
//    TextYPos += TextHeight;
//
//    MosSprintf(TheString, 512, MIL_TEXT("Train image size: %dx%d"), TrainImageSizeX, TrainImageSizeY);
//    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
//    TextYPos += TextHeight;
//
//    MosSprintf(TheString, 512, MIL_TEXT("Max number of epochs: %d"), m_MaxEpoch);
//    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
//    TextYPos += TextHeight;
//
//    MosSprintf(TheString, 512, MIL_TEXT("Minibatch size: %d"), MinibatchSize);
//    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
//    TextYPos += TextHeight;
//
//    MosSprintf(TheString, 512, MIL_TEXT("Learning rate: %.2e"), LearningRate);
//    MgraText(m_TheGraContext, m_LossInfoBufId, TextMargin, TextYPos, TheString);
//    TextYPos += TextHeight;
//
//    // The loss will be drawn under later on, so we retain its position.
//    m_YPositionForTrainLossText = TextYPos;
//    TextYPos += TextHeight;
//    m_YPositionForDevLossText = TextYPos;
//
//}
