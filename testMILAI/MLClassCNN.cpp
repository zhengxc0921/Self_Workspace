#include"MLClassCNN.h"


//MIL_INT MFTYPE HookFuncDatasetsPrepared(
//    MIL_INT /*HookType*/,
//    MIL_ID  EventId,
//    void* UserData)
//{
//    auto HookData = (HookDataCNNStruct*)UserData;
//
//    MIL_ID TrainResult;
//    MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainResult);
//
//    MIL_UNIQUE_CLASS_ID TrainPreparedDataset = MclassAlloc(HookData->MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
//    MclassCopyResult(TrainResult, M_DEFAULT, TrainPreparedDataset, M_DEFAULT, M_PREPARED_TRAIN_DATASET, M_DEFAULT);
//    const MIL_INT TrainDatasetNbImages = MclassInquire(TrainPreparedDataset, M_DEFAULT, M_NUMBER_OF_ENTRIES, M_NULL);
//
//    MIL_UNIQUE_CLASS_ID DevPreparedDataset = MclassAlloc(HookData->MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
//    MclassCopyResult(TrainResult, M_DEFAULT, DevPreparedDataset, M_DEFAULT, M_PREPARED_DEV_DATASET, M_DEFAULT);
//    const MIL_INT DevDatasetNbImages = MclassInquire(DevPreparedDataset, M_DEFAULT, M_NUMBER_OF_ENTRIES, M_NULL);
//
//    HookData->DashboardPtr->AddDatasetsPreparedData(TrainDatasetNbImages, DevDatasetNbImages);
//
//    MdispSelect(HookData->MilDisplay, HookData->DashboardPtr->GetDashboardBufId());
//
//    return M_NULL;
//}
//
////.........................................................................
//MIL_INT MFTYPE HookFuncEpoch(
//    MIL_INT /*HookType*/,
//    MIL_ID  EventId,
//    void* UserData)
//{
//    auto HookData = (HookDataCNNStruct*)UserData;
//
//    MIL_DOUBLE CurBench = 0.0;
//    MIL_DOUBLE CurBenchMean = -1.0;
//
//    MIL_INT CurEpochIndex = 0;
//    MclassGetHookInfo(EventId, M_EPOCH_INDEX + M_TYPE_MIL_INT, &CurEpochIndex);
//
//    MappTimer(M_DEFAULT, M_TIMER_READ, &CurBench);
//    MIL_DOUBLE EpochBenchMean = CurBench / (CurEpochIndex + 1);
//
//    MIL_DOUBLE TrainErrorRate = 0;
//    MclassGetHookInfo(EventId, M_TRAIN_DATASET_ERROR_RATE, &TrainErrorRate);
//    MIL_DOUBLE DevErrorRate = 0;
//    MclassGetHookInfo(EventId, M_DEV_DATASET_ERROR_RATE, &DevErrorRate);
//
//    MIL_INT AreTrainedCNNParametersUpdated = M_FALSE;
//    MclassGetHookInfo(EventId,
//        M_TRAINED_PARAMETERS_UPDATED + M_TYPE_MIL_INT,
//        &AreTrainedCNNParametersUpdated);
//
//    // By default trained parameters are updated when the dev error rate
//    // is the best up to now.
//    bool TheEpochIsTheBestUpToNow = (AreTrainedCNNParametersUpdated == M_TRUE);
//
//    HookData->DashboardPtr->AddEpochData(
//        TrainErrorRate,
//        DevErrorRate,
//        CurEpochIndex,
//        TheEpochIsTheBestUpToNow,
//        EpochBenchMean);
//
//    return M_NULL;
//}
//
////............................................................................
//MIL_INT MFTYPE HookFuncMiniBatch(
//    MIL_INT HookType,
//    MIL_ID  EventId,
//    void* UserData)
//{
//    auto HookData = (HookDataCNNStruct*)UserData;
//
//    MIL_DOUBLE LossError = 0;
//    MclassGetHookInfo(EventId, M_MINI_BATCH_LOSS, &LossError);
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
//    if (MiniBatchIdx == NbMiniBatchPerEpoch - 1 && HookData->DumpTmpRst == 1)
//    {
//        MIL_ID TrainRes;
//        MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainRes);
//        MIL_UNIQUE_CLASS_ID TrainedClassifierCtx = MclassAlloc(HookData->MilSystem, M_CLASSIFIER_CNN_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
//        MclassCopyResult(TrainRes, M_DEFAULT, TrainedClassifierCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);
//        MclassSave(HookData->ClassifierDumpFile, TrainedClassifierCtx, M_DEFAULT);
//    }
//
//    HookData->DashboardPtr->AddMiniBatchData(LossError, MiniBatchIdx, EpochIdx, NbMiniBatchPerEpoch);
//
//    //MosPrintf(MIL_TEXT("\nBatch[%d] loss error = %f.\n"), MiniBatchIdx, LossError);
//
//    //EpochIdx > 1的时候再来判断
//
//    if (HookData->ControlType == 's') {
//        //在Stop后先保存当前模型
//        MIL_ID TrainRes;
//        MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainRes);
//        MIL_UNIQUE_CLASS_ID TrainedClassifierCtx = MclassAlloc(HookData->MilSystem, M_CLASSIFIER_CNN_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
//        MclassCopyResult(TrainRes, M_DEFAULT, TrainedClassifierCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);
//        //查询当前模型
//        //MIL_INT Status = -1;
//        //MclassGetResult(TrainRes, M_DEFAULT, M_STATUS + M_TYPE_MIL_INT, &Status);
//        //if (Status == M_FALSE) {
//        //    printf("FALSE");
//        //}
//        //if (Status == M_COMPLETE) {
//        //    printf("M_COMPLETE");
//        //}
//        //MclassPreprocess(TrainedClassifierCtx, M_DEFAULT);
//        MclassSave(HookData->ClassifierDumpFile, TrainedClassifierCtx, M_DEFAULT);
//
//        MIL_ID HookInfoTrainResId = M_NULL;
//        MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &HookInfoTrainResId);
//        MclassControl(HookInfoTrainResId, M_DEFAULT, M_STOP_TRAIN, M_DEFAULT);
//        MosPrintf(MIL_TEXT("The training has been stopped.\n"));
//    }
//    else if (HookData->ControlType == 'p') {
//        MosPrintf(MIL_TEXT("\nPress 's' to stop the training or any other key to continue.\n"));
//        while (1) {
//            //Sleep(1000);
//            if (HookData->ControlType == 's')
//            {
//                //在Stop后先保存当前模型
//                MIL_ID TrainRes;
//                MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &TrainRes);
//                MIL_UNIQUE_CLASS_ID TrainedClassifierCtx = MclassAlloc(HookData->MilSystem, M_CLASSIFIER_CNN_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
//                MclassCopyResult(TrainRes, M_DEFAULT, TrainedClassifierCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);
//                MclassSave(HookData->ClassifierDumpFile, TrainedClassifierCtx, M_DEFAULT);
//
//
//                MIL_ID HookInfoTrainResId = M_NULL;
//                MclassGetHookInfo(EventId, M_RESULT_ID + M_TYPE_MIL_ID, &HookInfoTrainResId);
//                MclassControl(HookInfoTrainResId, M_DEFAULT, M_STOP_TRAIN, M_DEFAULT);
//                MosPrintf(MIL_TEXT("The training has been stopped.\n"));
//                break;
//            }
//            else if (HookData->ControlType == 'r')
//            {
//                MosPrintf(MIL_TEXT("The training will continue.\n"));
//                break;
//            }
//
//        }
//
//    }
//    return M_NULL;
//}



CMLClassCNN::CMLClassCNN(MIL_ID MilSystem, MIL_ID MilDisplay) :
	m_MilSystem(MilSystem),
	m_MilDisplay(MilDisplay)
{

    m_AIParse = CAIParsePtr(new CAIParse(MilSystem));
}

MIL_INT CMLClassCNN::CnnTrainEngineDLLInstalled(MIL_ID MilSystem)
{

    MIL_INT IsInstalled = M_FALSE;
    MIL_UNIQUE_CLASS_ID TrainCtx = MclassAlloc(MilSystem, M_TRAIN_CNN, M_DEFAULT, M_UNIQUE_ID);
    MclassInquire(TrainCtx, M_DEFAULT, M_TRAIN_ENGINE_IS_INSTALLED + M_TYPE_MIL_INT, &IsInstalled);
    return IsInstalled;

}

void CMLClassCNN::InitClassWeights()
{
    
    m_ClassWeights.resize(m_ClassesNum, 1.0);
}

void CMLClassCNN::ConstructDataset(std::vector<MIL_STRING> ClassName, 
    std::vector<MIL_STRING> ClassIcon, 
    MIL_STRING AuthorName, 
    MIL_STRING OriginalDataPath, 
    const MIL_STRING& WorkingDataPath,
    MIL_UNIQUE_CLASS_ID& Dataset)
{
    MIL_INT  NumberOfClasses = ClassName.size();

    if (M_NULL == Dataset)
    {
        Dataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);
    }
    AddClassDescription(Dataset, AuthorName, ClassName, ClassIcon, NumberOfClasses);

    for (MIL_INT ClassIdx = 0; ClassIdx < NumberOfClasses; ClassIdx++)
    {
        AddClassToDataset(ClassIdx, OriginalDataPath, ClassName[ClassIdx], AuthorName, Dataset);
    }

}

void CMLClassCNN::GeneralDataset(vector<vector<MIL_STRING>> ClassName, 
    vector<vector<MIL_STRING>> ClassIcon, 
    vector<MIL_STRING>& AuthorName,
    vector<MIL_STRING>& OriginalDataPath,
    const MIL_STRING& WorkingDataPath)
{
    MIL_UNIQUE_CLASS_ID  Dataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);

    for (int p_i = 0; p_i < OriginalDataPath.size(); p_i++) {

        ConstructDataset(
            ClassName[p_i],
            ClassIcon[p_i],
            AuthorName[p_i],
            OriginalDataPath[p_i],
            WorkingDataPath,
            Dataset);
    }

    CreateFolder(WorkingDataPath);
    MclassControl(Dataset, M_CONTEXT, M_CONSOLIDATE_ENTRIES_INTO_FOLDER, WorkingDataPath);

    //汇总数据，然后再保存
    MIL_STRING OriginalAllDataPath = WorkingDataPath + MIL_TEXT("Images\\");
    MIL_STRING OriginalAllIconsPath = WorkingDataPath + MIL_TEXT("Icons\\");
    std::vector<MIL_STRING>ClassName_A;
    std::vector<MIL_STRING>ClassIcon_A;
    MIL_UNIQUE_CLASS_ID  AllDataset = MclassAlloc(m_MilSystem, M_DATASET_IMAGES, M_DEFAULT, M_UNIQUE_ID);

    string OrgAIconsPath;
    m_AIParse->MIL_STRING2string(OriginalAllIconsPath, OrgAIconsPath);
    m_AIParse->getFilesInFolder(OrgAIconsPath,"mim", ClassIcon_A);
    for (int i = 0; i < ClassIcon_A.size(); i++) {
        MIL_STRING ClassIconPath = ClassIcon_A[i];
        MIL_STRING::size_type iPos = ClassIconPath.find_last_of('\\') + 1;
        MIL_STRING filename = ClassIconPath.substr(iPos, ClassIconPath.length() - iPos);

        MIL_STRING::size_type iPosP = filename.find_last_of('.') + 1;
        MIL_STRING ClassName = filename.substr(0, iPosP - 1);

        ClassName_A.emplace_back(ClassName);
    }
    ConstructDataset(ClassName_A, ClassIcon_A, AuthorName[0], OriginalAllDataPath, WorkingDataPath, AllDataset);
    MIL_STRING WorkDatasetPath = WorkingDataPath + MIL_TEXT("DataSet.mclassd");
    MclassSave(WorkDatasetPath, AllDataset, M_DEFAULT);

}

void CMLClassCNN::ConstructDataContext(DataContextParasStruct DataCtxParas, MIL_UNIQUE_CLASS_ID& DataContext)
{
    if (M_NULL == DataContext)
    {
        DataContext = MclassAlloc(m_MilSystem, M_PREPARE_IMAGES_CNN, M_DEFAULT, M_UNIQUE_ID);
    }

    MIL_ID AugmentContext;
    MclassInquire(DataContext, M_CONTEXT, M_AUGMENT_CONTEXT_ID + M_TYPE_MIL_ID, &AugmentContext);

    if (DataCtxParas.ImageSizeX > 0 && DataCtxParas.ImageSizeY > 0)
    {
        MclassControl(DataContext, M_CONTEXT, M_SIZE_MODE, M_USER_DEFINED);
        MclassControl(DataContext, M_CONTEXT, M_SIZE_X, DataCtxParas.ImageSizeX);
        MclassControl(DataContext, M_CONTEXT, M_SIZE_Y, DataCtxParas.ImageSizeY);
        m_ImageSizeX = DataCtxParas.ImageSizeX;
        m_ImageSizeY = DataCtxParas.ImageSizeY;
    }

    MclassControl(DataContext, M_CONTEXT, M_RESIZE_SCALE_FACTOR, M_FILL_DESTINATION);

    if (DataCtxParas.DstFolderMode == 1)
    {
        MclassControl(DataContext, M_CONTEXT, M_DESTINATION_FOLDER_MODE, M_OVERWRITE);
    }

    //数据保存
    CreateFolder(DataCtxParas.PreparedDataFolder);
    MclassControl(DataContext, M_CONTEXT, M_PREPARED_DATA_FOLDER, DataCtxParas.PreparedDataFolder);
    // On average, we do two augmentations per image + the original images.
    MclassControl(DataContext, M_CONTEXT, M_AUGMENT_NUMBER_FACTOR, DataCtxParas.AugParas.AugmentationNumPerImage);
    // Ensure repeatability with a fixed seed.
    if (DataCtxParas.AugParas.SeedValue > 0)
    {
        MclassControl(DataContext, M_CONTEXT, M_SEED_MODE, M_USER_DEFINED);
        MclassControl(DataContext, M_CONTEXT, M_SEED_VALUE, DataCtxParas.AugParas.SeedValue);
    }
    // Translation augmentation and presets in the prepare data context.
    // MclassControl(TrainPrepareDataCtx, M_CONTEXT, M_PRESET_TRANSLATION, M_ENABLE);
    if (DataCtxParas.AugParas.TranslationXMax > 0)
    {
        MimControl(AugmentContext, M_AUG_TRANSLATION_X_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_TRANSLATION_X_OP_MAX, DataCtxParas.AugParas.TranslationXMax);
    }
    if (DataCtxParas.AugParas.TranslationYMax > 0)
    {
        MimControl(AugmentContext, M_AUG_TRANSLATION_Y_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_TRANSLATION_Y_OP_MAX, DataCtxParas.AugParas.TranslationYMax);
    }

    // Scale augmentation and presets in the prepare data context.
    // MclassControl(TrainPrepareDataCtx, M_CONTEXT, M_PRESET_SCALE, M_ENABLE);
    if ((DataCtxParas.AugParas.ScaleFactorMin > 0 && DataCtxParas.AugParas.ScaleFactorMin != 1.0)
        || (DataCtxParas.AugParas.ScaleFactorMax > 0 && DataCtxParas.AugParas.ScaleFactorMax != 1.0))
    {
        MimControl(AugmentContext, M_AUG_SCALE_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_SCALE_OP_FACTOR_MIN, DataCtxParas.AugParas.ScaleFactorMin);
        MimControl(AugmentContext, M_AUG_SCALE_OP_FACTOR_MAX, DataCtxParas.AugParas.ScaleFactorMax);
    }
    // Rotation augmentation and presets in the prepare data context.
    // MclassControl(TrainPrepareDataCtx, M_CONTEXT, M_PRESET_ROTATION, M_ENABLE);
    if (DataCtxParas.AugParas.RotateAngleDelta > 0)
    {
        MimControl(AugmentContext, M_AUG_ROTATION_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_ROTATION_OP_ANGLE_DELTA, DataCtxParas.AugParas.RotateAngleDelta);
    }
    // Smoothness augmentation and presets in the prepare data context.
    if (DataCtxParas.AugParas.SmoothnessMin > 0.0 && DataCtxParas.AugParas.SmoothnessMax >= DataCtxParas.AugParas.SmoothnessMin)
    {
        MimControl(AugmentContext, M_AUG_SMOOTH_DERICHE_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_SMOOTH_DERICHE_OP_FACTOR_MIN, DataCtxParas.AugParas.SmoothnessMin);
        MimControl(AugmentContext, M_AUG_SMOOTH_DERICHE_OP_FACTOR_MAX, DataCtxParas.AugParas.SmoothnessMax);
    }
    // Noise augmentation and presets in the prepare data context.
    if (DataCtxParas.AugParas.GaussNoiseDelta > 0.0 || DataCtxParas.AugParas.GaussNoiseStdev > 0.0)
    {
        MimControl(AugmentContext, M_AUG_NOISE_GAUSSIAN_ADDITIVE_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_NOISE_GAUSSIAN_ADDITIVE_OP_STDDEV, DataCtxParas.AugParas.GaussNoiseStdev);
        MimControl(AugmentContext, M_AUG_NOISE_GAUSSIAN_ADDITIVE_OP_STDDEV_DELTA, DataCtxParas.AugParas.GaussNoiseDelta);
    }
    if (DataCtxParas.AugParas.CropEnable == 1)
    {
        MimControl(AugmentContext, M_AUG_CROP_OP, M_ENABLE);
    }
    if (DataCtxParas.AugParas.GammaValue > 0)
    {
        MimControl(AugmentContext, M_AUG_GAMMA_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_GAMMA_OP_VALUE, DataCtxParas.AugParas.GammaValue);
        MimControl(AugmentContext, M_AUG_GAMMA_OP_DELTA, DataCtxParas.AugParas.GammaDelta);
    }

    if (DataCtxParas.AugParas.InAddValue > 0) {
        MimControl(AugmentContext, M_AUG_INTENSITY_ADD_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_INTENSITY_ADD_OP_VALUE, DataCtxParas.AugParas.InAddValue);
        MimControl(AugmentContext, M_AUG_INTENSITY_ADD_OP_DELTA, DataCtxParas.AugParas.InAddDelta);
    }

    if (DataCtxParas.AugParas.InMulValue > 0) {
        MimControl(AugmentContext, M_AUG_INTENSITY_MULTIPLY_OP, M_ENABLE);
        MimControl(AugmentContext, M_AUG_INTENSITY_MULTIPLY_OP_VALUE, DataCtxParas.AugParas.InMulValue);
       
        MimControl(AugmentContext, M_AUG_INTENSITY_MULTIPLY_OP_DELTA, DataCtxParas.AugParas.InMulDelta);
    }



}

void CMLClassCNN::PrepareDataset(MIL_UNIQUE_CLASS_ID& DatasetContext, MIL_UNIQUE_CLASS_ID& PrepareDataset, MIL_UNIQUE_CLASS_ID& PreparedDataset)
{
    MclassPreprocess(DatasetContext, M_DEFAULT);
    MclassPrepareData(DatasetContext, PrepareDataset, PreparedDataset, M_NULL, M_DEFAULT);

}

void CMLClassCNN::ConstructTrainCtx(ClassifierParasStruct ClassifierParas, MIL_UNIQUE_CLASS_ID& TrainCtx)
{
    if (M_NULL == TrainCtx)
    {
        TrainCtx = MclassAlloc(m_MilSystem, M_TRAIN_CNN, M_DEFAULT, M_UNIQUE_ID);
    }

    MclassControl(TrainCtx, M_CONTEXT, M_TRAIN_DESTINATION_FOLDER, ClassifierParas.TrainDstFolder);

    //if (ClassifierParas.TrainMode == 1)
    //{
    //    MclassControl(TrainCtx, M_CONTEXT, M_RESET_TRAINING_VALUES, M_FINE_TUNING);
    //}
    //else if (ClassifierParas.TrainMode == 2)
    //{
    //    MclassControl(TrainCtx, M_CONTEXT, M_RESET_TRAINING_VALUES, M_TRANSFER_LEARNING);
    //}

    if (ClassifierParas.SplitPercent > 0)
    {
        //MclassControl(TrainCtx, M_DEFAULT, M_SPLIT_SEED_MODE, M_FIXED);
        // Since we are performing a single dataset train, 
        //the dataset will be split into train/dev by the following percentage.
        MclassControl(TrainCtx, M_DEFAULT, M_SPLIT_PERCENTAGE, ClassifierParas.SplitPercent);
    }

    if (ClassifierParas.MaxNumberOfEpoch > 0)
    {
        MclassControl(TrainCtx, M_DEFAULT, M_MAX_EPOCH, ClassifierParas.MaxNumberOfEpoch);
    }

    if (ClassifierParas.MiniBatchSize > 0)
    {
        MclassControl(TrainCtx, M_DEFAULT, M_MINI_BATCH_SIZE, ClassifierParas.MiniBatchSize);
    }

    if (ClassifierParas.SchedulerType == 1)
    {
        MclassControl(TrainCtx, M_DEFAULT, M_SCHEDULER_TYPE, M_DECAY);
    }

    if (ClassifierParas.LearningRate > 0)
    {
        MclassControl(TrainCtx, M_DEFAULT, M_INITIAL_LEARNING_RATE, ClassifierParas.LearningRate);
    }

    if (ClassifierParas.LearningRateDecay > 0)
    {
        MclassControl(TrainCtx, M_DEFAULT, M_LEARNING_RATE_DECAY, ClassifierParas.LearningRate);
    }

    if (ClassifierParas.TrainEngineUsed == 1)
    {
        MclassControl(TrainCtx, M_CONTEXT, M_TRAIN_ENGINE, M_CPU);
    }
    MclassPreprocess(TrainCtx, M_DEFAULT);

}

void CMLClassCNN::TrainClassifier(MIL_UNIQUE_CLASS_ID& Dataset, MIL_UNIQUE_CLASS_ID& DatasetContext, MIL_UNIQUE_CLASS_ID& TrainCtx, MIL_UNIQUE_CLASS_ID& PrevClassifierCtx, MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx, MIL_STRING& ClassifierDumpFile)
{
    MIL_INT MaxEpoch = 0;
    MIL_INT MinibatchSize = 0;
    MIL_INT TrainEngineUsed = 0;
    MIL_INT TrainDatasetNbImages = 0;
    MIL_INT DevDatasetNbImages = 0;
    MIL_DOUBLE LearningRate = 0.0;
    MIL_STRING TrainEngineDescription;
    MclassInquire(TrainCtx, M_DEFAULT, M_MAX_EPOCH + M_TYPE_MIL_INT, &MaxEpoch);
    MclassInquire(TrainCtx, M_DEFAULT, M_MINI_BATCH_SIZE + M_TYPE_MIL_INT, &MinibatchSize);
    MclassInquire(TrainCtx, M_CONTEXT, M_INITIAL_LEARNING_RATE + M_TYPE_MIL_DOUBLE, &LearningRate);
    MclassInquire(TrainCtx, M_CONTEXT, M_TRAIN_ENGINE_USED + M_TYPE_MIL_INT, &TrainEngineUsed);
    MclassInquire(TrainCtx, M_CONTEXT, M_TRAIN_ENGINE_USED_DESCRIPTION, TrainEngineDescription);

    MIL_UNIQUE_CLASS_ID TrainRes = MclassAllocResult(m_MilSystem, M_TRAIN_CNN_RESULT, M_DEFAULT, M_UNIQUE_ID);
    HookDataStruct TheHookData;
    TheHookData.MilSystem = m_MilSystem;
    TheHookData.MilDisplay = m_MilDisplay;
    TheHookData.DumpTmpRst = 1;
    TheHookData.ClassifierDumpFile = ClassifierDumpFile;

    TheHookData.DashboardPtr = CDashboardPtr(new CDashboard(
        m_MilSystem,
        0,
        MaxEpoch,
        MinibatchSize,
        LearningRate,
        m_ImageSizeX,
        m_ImageSizeY,
        TrainDatasetNbImages,
        DevDatasetNbImages,
        TrainEngineUsed,
        TrainEngineDescription));

    // Initialize the hook associated to the epoch trained event.
    MclassHookFunction(TrainCtx, M_EPOCH_TRAINED, HookFuncEpoch, &TheHookData);

    // Initialize the hook associated to the mini batch trained event.
    MclassHookFunction(TrainCtx, M_MINI_BATCH_TRAINED, HookFuncMiniBatch, &TheHookData);

    // Initialize the hook associated to the datasets prepared event.
    MclassHookFunction(TrainCtx, M_DATASETS_PREPARED, HookFuncDatasetsPrepared, &TheHookData);
    // Start the training process.
    double time = 0;
    MIL_TEXT_CHAR TheString[512];
    //timeStart();
    MclassTrain(TrainCtx, PrevClassifierCtx, Dataset, M_NULL, TrainRes, M_DEFAULT);
    //timeEnd(time);
    if (TrainEngineUsed == M_CPU)

        MosPrintf(MIL_TEXT("Training is performed on the CPU used %f second\n"), time);
    else
        MosPrintf(MIL_TEXT("Training is performed on the GPU used %f second\n"), time);

    // Check the training status to ensure the training has completed properly.
    MIL_INT Status = -1;
    MclassGetResult(TrainRes, M_DEFAULT, M_STATUS + M_TYPE_MIL_INT, &Status);
    if (Status == M_COMPLETE)
    {

        //MIL_STRING Respath = MIL_TEXT("G:/DefectDataCenter/zhjuzhiqiang_2023/2023/spa/PreparedData/res.mclass");
        //MclassExport(Respath, TrainRes, M_DEFAULT);
        //MIL_UNIQUE_CLASS_ID TrainRes_dsk = MclassRestore(Respath, m_MilSystem, M_DEFAULT, M_UNIQUE_ID);

        TrainedClassifierCtx = MclassAlloc(m_MilSystem, M_CLASSIFIER_CNN_PREDEFINED, M_DEFAULT, M_UNIQUE_ID);
        MclassCopyResult(TrainRes, M_DEFAULT, TrainedClassifierCtx, M_DEFAULT, M_TRAINED_CLASSIFIER, M_DEFAULT);

    }

}

void CMLClassCNN::PredictBegin(MIL_UNIQUE_CLASS_ID& TrainedCCtx,MIL_ID Image,vector<MIL_DOUBLE>Class_Weights)
{  
    //获取模型输入尺寸
    MclassInquire(TrainedCCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_X + M_TYPE_MIL_INT, &m_InputSizeX);
    MclassInquire(TrainedCCtx, M_DEFAULT_SOURCE_LAYER, M_SIZE_Y + M_TYPE_MIL_INT, &m_InputSizeY);
    MclassInquire(TrainedCCtx, M_CONTEXT, M_NUMBER_OF_CLASSES + M_TYPE_MIL_INT, &m_ClassesNum);
    MIL_INT m_ImageSizeX = MbufInquire(Image, M_SIZE_X, M_NULL);
    MIL_INT m_ImageSizeY = MbufInquire(Image, M_SIZE_Y, M_NULL);
    if (Class_Weights.size()!= m_ClassesNum) {
        m_ClassWeights.resize(m_ClassesNum, 1.0);
    }
    else {
        m_ClassWeights.resize(m_ClassesNum, 1.0);
        for (int i = 0; i < m_ClassesNum; i++) {
            m_ClassWeights[i] = Class_Weights[i];
        }
    }
}

void CMLClassCNN::FolderImgsPredict(vector<MIL_STRING> FilesInFolder,
    vector<MIL_DOUBLE> Class_Weights, 
    MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx, 
    vector<ClassificationResultStruct>& Result)
{
    int nFileNum = FilesInFolder.size();
    ClassificationResultStruct Result_i; 
    for (int i = 0; i < nFileNum; i++) {
        memset(&Result_i, 0, sizeof(Result_i));
        MIL_ID RawImage = MbufRestore(FilesInFolder[i], m_MilSystem, M_NULL);
        Predict(RawImage, Class_Weights,TrainedClassifierCtx, Result_i);
        Result.emplace_back(Result_i);
        MbufFree(RawImage);
    }

}

void CMLClassCNN::Predict(MIL_ID Image, 
    vector<MIL_DOUBLE> Class_Weights,
    MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx, 
    ClassificationResultStruct& Result)
{
    PredictBegin(TrainedClassifierCtx, Image, Class_Weights);
    MIL_ID ImageReduce = MbufAlloc2d(m_MilSystem, m_InputSizeX, m_InputSizeY, 8 + M_UNSIGNED, M_IMAGE + M_PROC, M_NULL);
    MimResize(Image, ImageReduce, M_FILL_DESTINATION, M_FILL_DESTINATION, M_DEFAULT);

    MIL_INT Status = M_FALSE;
    MclassInquire(TrainedClassifierCtx, M_DEFAULT, M_PREPROCESSED + M_TYPE_MIL_INT, &Status);
    if (M_FALSE == Status)
    {
        //LARGE_INTEGER t1, t2, tc;
        //QueryPerformanceFrequency(&tc);
        //QueryPerformanceCounter(&t1);
        MclassPreprocess(TrainedClassifierCtx, M_DEFAULT);
        //QueryPerformanceCounter(&t2);
        //double time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
        //cout << "\nPreprocess_" << m_index << "time is = " << time << "\n" << endl;
    }
    MIL_UNIQUE_CLASS_ID ClassRes = MclassAllocResult(m_MilSystem, M_PREDICT_CNN_RESULT, M_DEFAULT, M_UNIQUE_ID);
    MclassPredict(TrainedClassifierCtx, ImageReduce, ClassRes, M_DEFAULT);
    MbufFree(ImageReduce);

    MclassGetResult(ClassRes, M_GENERAL, M_BEST_CLASS_SCORE + M_TYPE_MIL_DOUBLE, &Result.PredictScore);
    MclassGetResult(ClassRes, M_GENERAL, M_BEST_CLASS_INDEX + M_TYPE_MIL_INT, &Result.PredictClass);
    vector<MIL_DOUBLE>WeightScores;
    for (int i = 0; i < m_ClassWeights.size(); i++) {
        MIL_DOUBLE Score;
        MclassGetResult(ClassRes, M_CLASS_INDEX(i), M_CLASS_SCORES + M_TYPE_MIL_DOUBLE, &Score);
        MIL_DOUBLE WeightScore = Score * m_ClassWeights[i];
        WeightScores.emplace_back(WeightScore);
        Result.AllScores.emplace_back(Score);
    }
    vector<MIL_DOUBLE>::iterator biggest = max_element(begin(WeightScores), end(WeightScores));
    Result.PredictClass = distance(begin(WeightScores), biggest);
    MclassInquire(TrainedClassifierCtx, M_CLASS_INDEX(Result.PredictClass), M_CLASS_NAME, Result.PredictClassName);
}

void CMLClassCNN::Predict(MIL_ID Image, 
    vector<int> blob_px, 
    vector<int> blob_py,
    std::vector<MIL_DOUBLE> Class_Weights,
    MIL_UNIQUE_CLASS_ID& TrainedClassifierCtx,
    ClassificationResultStruct& Result)
{
    vector<Box> vecCropedBoxes;
    m_AIParse->cropBox(blob_px,blob_py,
        m_InputSizeX,
        m_InputSizeY,
        m_ImageSizeX,
        m_ImageSizeY,
        vecCropedBoxes);
    int CropedBN = vecCropedBoxes.size();
    for (int i = 0; i < CropedBN; i++) {
        ClassificationResultStruct tmpRst;

        MIL_ID InImg = MbufAlloc2d(m_MilSystem,m_InputSizeX,m_InputSizeY,
            8 + M_UNSIGNED,M_IMAGE + M_PROC, M_NULL);
        MbufClear(InImg, 0);
        auto CropBoxes = vecCropedBoxes[i];
        MbufCopyColor2d(Image,InImg,M_ALL_BANDS,CropBoxes.x1, CropBoxes.y1,
            M_ALL_BANDS, 0, 0,m_InputSizeX, m_InputSizeY);
        Predict(InImg, Class_Weights, TrainedClassifierCtx, tmpRst);
        for (int nCNum = 0; nCNum < m_ClassesNum; nCNum++) {
            Result.AllScores[nCNum] += tmpRst.AllScores[nCNum];
        }    
    }

    vector<MIL_DOUBLE>::iterator biggest = max_element(std::begin(Result.AllScores), std::end(Result.AllScores));
    Result.PredictClass = distance(begin(Result.AllScores), biggest);
    Result.PredictScore = Result.AllScores[Result.PredictClass]/(double)m_ClassesNum;
    MclassInquire(TrainedClassifierCtx, M_CLASS_INDEX(Result.PredictClass), M_CLASS_NAME, Result.PredictClassName);
}

void CMLClassCNN::AddClassDescription(MIL_ID Dataset, const MIL_STRING& AuthorName, std::vector<MIL_STRING> ClassName, std::vector<MIL_STRING> ClassIcon, MIL_INT NumberOfClasses)
{
    MclassControl(Dataset, M_DEFAULT, M_AUTHOR_ADD, AuthorName);

    for (MIL_INT i = 0; i < NumberOfClasses; i++)
    {
        MclassControl(Dataset, M_DEFAULT, M_CLASS_ADD, ClassName[i]);
        MIL_UNIQUE_BUF_ID IconImageId = MbufRestore(ClassIcon[i], m_MilSystem, M_UNIQUE_ID);
        MclassControl(Dataset, M_CLASS_INDEX(i), M_CLASS_ICON_ID, IconImageId);
    }

}

void CMLClassCNN::AddClassToDataset(MIL_INT ClassIndex, const MIL_STRING& DataPath, const MIL_STRING& ClassName, const MIL_STRING& AuthorName, MIL_ID Dataset)
{
    MclassControl(Dataset, M_DEFAULT, M_AUTHOR_ADD, AuthorName);
    MIL_INT NbEntries;
    MclassInquire(Dataset, M_DEFAULT, M_NUMBER_OF_ENTRIES + M_TYPE_MIL_INT, &NbEntries);
    MIL_STRING FolderName = DataPath + ClassName + MIL_TEXT("\\");
    std::vector<MIL_STRING> FilesInFolder;

    string strFolderName;
    m_AIParse->MIL_STRING2string(FolderName, strFolderName);
    m_AIParse->getFilesInFolder(strFolderName, "bmp", FilesInFolder);

    MIL_INT CurImageIndex = 0;
    for (const auto& File : FilesInFolder)
    {
        MclassControl(Dataset, M_DEFAULT, M_ENTRY_ADD, M_DEFAULT);
        MclassControlEntry(Dataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_REGION_INDEX(0), M_CLASS_INDEX_GROUND_TRUTH, ClassIndex, M_NULL, M_DEFAULT);
        MclassControlEntry(Dataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_DEFAULT, M_ENTRY_IMAGE_PATH, M_DEFAULT, File, M_DEFAULT);
        MclassControlEntry(Dataset, NbEntries + CurImageIndex, M_DEFAULT_KEY, M_REGION_INDEX(0), M_AUTHOR_NAME, M_DEFAULT, AuthorName, M_DEFAULT);
        CurImageIndex++;
    }

}
